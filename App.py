import os
import glob
import shutil
import re
from typing import List, Dict
import streamlit as st
import yt_dlp
import whisper
from google import genai
from streamlit_echarts import st_echarts
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv

# ================= LOAD ENV =================
# Loads variables from .env file into environment (safe, never committed to Git)
load_dotenv()

# ================= CONFIG =================
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

MODEL_NAME = "gemma-3-27b-it"
client = genai.Client(api_key=API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thread-safe lock
generation_lock = threading.Lock()

# Load embedding model for Q&A
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# ================= CLEAN =================
def clean_outputs():
    for folder in [AUDIO_DIR, OUTPUT_DIR]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except:
                pass

# ================= AUDIO =================
def get_audio_from_youtube(url):
    outtmpl = os.path.join(AUDIO_DIR, "temp.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
                "skip": ["hls", "dash"]
            }
        },
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-us,en;q=0.5",
            "Sec-Fetch-Mode": "navigate"
        }
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            ydl_opts["extractor_args"]["youtube"]["player_client"] = ["android"]
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
        else:
            raise e

    files = glob.glob(os.path.join(AUDIO_DIR, "temp.*"))
    if not files:
        raise Exception("No audio file was downloaded. Please check the URL.")

    audio_path = os.path.join(OUTPUT_DIR, "audio.mp3")
    shutil.copy(files[0], audio_path)
    return audio_path

# ================= TRANSCRIBE =================
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="translate", fp16=False)
    return result["text"].strip(), result["segments"]

# ================= GEMMA CALL WITH RETRY =================
def call_gemma_with_retry(prompt, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            res = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return res.text.strip()
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    wait = delay * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise Exception("❌ Model overloaded. Try again later.")
            else:
                raise e

# ================= SECONDS TO HH:MM:SS CONVERTER =================
def seconds_to_hhmmss(seconds: float):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

# ================= GENERATION FUNCTIONS =================
def generate_timeline_task(segments, session_id):
    """Generate timeline in background"""
    if not segments:
        return "00:00:00 – 00:00:00 | Introduction"

    formatted_transcript = ""
    chunk_size = 5
    for i in range(0, len(segments), chunk_size):
        group = segments[i:i+chunk_size]
        start_fmt = seconds_to_hhmmss(group[0]['start'])
        end_fmt = seconds_to_hhmmss(group[-1]['end'])
        text_content = " ".join([s['text'].strip() for s in group])
        formatted_transcript += f"[{start_fmt} - {end_fmt}] {text_content[:200]}...\n"

    prompt = f"""
You are generating a video timeline.

CRITICAL GOAL: Identify ONLY the MAJOR THEMATIC BLOCKS.

RULES:
1. Cover the FULL video duration.
2. CONTINUOUS time ranges (No gaps).
3. EXACT FORMAT: HH:MM:SS – HH:MM:SS | Topic Name
4. Topic Name: 2-5 words.
5. NO sub-topics. Merge details into broad blocks.

INPUT:
{formatted_transcript}

RETURN ONLY THE TIMELINE.
"""

    try:
        result = call_gemma_with_retry(prompt)
        return result
    except Exception as e:
        return f"Error generating timeline: {str(e)}"


def generate_summary_task(text, session_id):
    """Generate summary in background"""
    prompt = f"""
You are an expert summarizer.

GOAL: Write a concise EXECUTIVE SUMMARY of the video.

STRICT RULES:
- Write in **PARAGRAPHS ONLY**.
- **NO** bullet points.
- **NO** numbered lists.
- Focus on the main argument and conclusion.

Transcript:
{text}
"""
    return call_gemma_with_retry(prompt)


def generate_keypoints_task(text, session_id):
    """Generate key points in background"""
    prompt = f"""
TASK: specific Key Takeaways.

FORMAT:
1. Point One
2. Point Two

RULES:
- Numbered list only.
- No paragraphs.
- One sentence per point.

Transcript:
{text}
"""
    return call_gemma_with_retry(prompt)


def generate_mindmap_task(text, session_id):
    """Generate mindmap in background"""
    topic_prompt = f"""
Identify the MAIN TOPIC of this transcript in 2-5 words.

Transcript:
{text[:1000]}

Answer:
"""

    try:
        main_topic = call_gemma_with_retry(topic_prompt).strip()
        main_topic = main_topic.replace('*', '').replace('#', '').strip()
    except:
        main_topic = "Video Content"

    prompt = f"""
Create a hierarchial mind map.

MAIN TOPIC: {main_topic}

FORMAT:
- Use 2 spaces indentation.
- Plain text only. NO symbols (*, -).

STRUCTURE:
Main Topic
  Concept 1
    Detail A
    Detail B
  Concept 2
    Detail C

Transcript:
{text}
"""

    result = call_gemma_with_retry(prompt)

    cleaned = []
    for line in result.strip().split('\n'):
        clean_line = line
        clean_line = re.sub(r'[*#\-•○●►▸→├└│`]', '', clean_line)
        clean_line = re.sub(r'^\s*\d+[\.\)]\s*', '', clean_line)
        clean_line = clean_line.strip()

        if clean_line and len(clean_line) > 1:
            original_indent = len(line) - len(line.lstrip())
            normalized_indent = (original_indent // 2) * 2
            cleaned.append(' ' * normalized_indent + clean_line)

    return '\n'.join(cleaned)


def build_qa_index_task(text, session_id):
    """Build Q&A index in background"""
    def chunk_text(text, chunk_size=400, overlap=80):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            start = end - overlap
        return chunks

    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    return {"index": index, "chunks": chunks}

# ================= BACKGROUND PROCESSOR =================
def start_all_background_tasks():
    """Start all generation tasks in background"""
    session_id = st.session_state.get("session_id")
    transcript = st.session_state["raw_transcript"]
    segments = st.session_state["segments"]

    executor = ThreadPoolExecutor(max_workers=5)

    if "timeline_future" not in st.session_state:
        st.session_state["timeline_future"] = executor.submit(generate_timeline_task, segments, session_id)
        st.session_state["timeline_status"] = "processing"

    if "summary_future" not in st.session_state:
        st.session_state["summary_future"] = executor.submit(generate_summary_task, transcript, session_id)
        st.session_state["summary_status"] = "processing"

    if "keypoints_future" not in st.session_state:
        st.session_state["keypoints_future"] = executor.submit(generate_keypoints_task, transcript, session_id)
        st.session_state["keypoints_status"] = "processing"

    if "mindmap_future" not in st.session_state:
        st.session_state["mindmap_future"] = executor.submit(generate_mindmap_task, transcript, session_id)
        st.session_state["mindmap_status"] = "processing"

    if "qa_future" not in st.session_state:
        st.session_state["qa_future"] = executor.submit(build_qa_index_task, transcript, session_id)
        st.session_state["qa_status"] = "processing"


def check_and_update_status():
    """Check if background tasks are done and update status"""
    tasks = [
        ("timeline_future", "timeline", "timeline_status"),
        ("summary_future", "summary", "summary_status"),
        ("keypoints_future", "keypoints", "keypoints_status"),
        ("mindmap_future", "mindmap", "mindmap_status"),
        ("qa_future", "qa_data", "qa_status")
    ]

    for future_key, result_key, status_key in tasks:
        if future_key in st.session_state:
            future = st.session_state[future_key]
            if future.done() and result_key not in st.session_state:
                try:
                    result = future.result()
                    st.session_state[result_key] = result
                    st.session_state[status_key] = "completed"
                except Exception as e:
                    st.session_state[status_key] = f"error: {str(e)}"

# ================= MINDMAP HELPERS =================
def mindmap_to_edges(text):
    edges = []
    stack = []
    lines = text.strip().split('\n')

    for line_num, line in enumerate(lines):
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        level = indent // 2
        node = line.strip()

        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            edges.append({"parent": stack[-1][1], "child": node})

        stack.append((level, node))

    return edges


def get_theme():
    try:
        return st.get_option("theme.base") or "light"
    except:
        return "light"


def build_enhanced_tree(edges: List[Dict], is_dark_mode=False) -> Dict:
    if not edges:
        return {"name": "Error", "children": []}

    children_map = {}
    all_nodes = set()
    child_nodes = set()

    for e in edges:
        p, c = e["parent"], e["child"]
        all_nodes.add(p)
        all_nodes.add(c)
        child_nodes.add(c)
        children_map.setdefault(p, []).append(c)

    def count_leaves(name):
        childs = children_map.get(name, [])
        if not childs:
            return 1
        return sum(count_leaves(c) for c in childs)

    if is_dark_mode:
        colors = ["#a78bfa", "#60a5fa", "#34d399", "#fbbf24", "#f87171", "#f472b6"]
        text_color = "#f9fafb"
        border_color = "#374151"
    else:
        colors = ["#7c3aed", "#2563eb", "#059669", "#d97706", "#dc2626", "#db2777"]
        text_color = "#111827"
        border_color = "#f3f4f6"

    def make_node(name, depth=0):
        childs = children_map.get(name, [])
        color = colors[min(depth, len(colors)-1)]

        node_config = {
            "name": name,
            "value": count_leaves(name),
            "itemStyle": {
                "color": color,
                "borderColor": border_color,
                "borderWidth": 2,
                "shadowBlur": 10,
                "shadowColor": color
            },
            "label": {
                "fontSize": max(16 - depth * 2, 12),
                "fontWeight": "bold" if depth < 2 else "normal",
                "color": text_color,
                "padding": [4, 8],
                "backgroundColor": "rgba(255,255,255,0.85)" if not is_dark_mode else "rgba(0,0,0,0.65)",
                "borderRadius": 4
            }
        }

        if len(childs) > 0:
            node_config["children"] = [make_node(c, depth+1) for c in childs]
            node_config["collapsed"] = True

        return node_config

    roots = [n for n in all_nodes if n not in child_nodes]
    if not roots:
        return {}

    root_val = roots[0]
    return make_node(root_val, 0)


def get_tree_options(tree_data, is_dark_mode=False):
    tooltip_bg = "rgba(0, 0, 0, 0.9)" if is_dark_mode else "rgba(255, 255, 255, 0.95)"
    text_color = "#e5e7eb" if is_dark_mode else "#1f2937"

    return {
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove",
            "backgroundColor": tooltip_bg,
            "textStyle": {"color": text_color},
            "formatter": "{b}"
        },
        "series": [{
            "type": "tree",
            "data": [tree_data],
            "top": "5%",
            "left": "10%",
            "bottom": "5%",
            "right": "10%",
            "layout": "orthogonal",
            "orient": "LR",
            "roam": False,
            "symbol": "circle",
            "symbolSize": 16,
            "initialTreeDepth": 2,
            "nodeInterval": 25,
            "edgeLength": [150, 100, 80],
            "label": {
                "position": "left",
                "verticalAlign": "middle",
                "align": "right",
                "fontSize": 13,
                "lineHeight": 18,
                "distance": 8
            },
            "scaleLimit": {
                "min": 0.3,
                "max": 3
            },
            "leaves": {
                "label": {
                    "position": "right",
                    "verticalAlign": "middle",
                    "align": "left",
                    "fontSize": 13,
                    "lineHeight": 18,
                    "distance": 8
                }
            },
            "emphasis": {
                "focus": "descendant",
                "blurScope": "coordinateSystem"
            },
            "expandAndCollapse": True,
            "animationDuration": 450,
            "animationEasing": "cubicOut",
            "lineStyle": {
                "color": "#9ca3af",
                "width": 2,
                "curveness": 0.5
            }
        }]
    }


def answer_question(question, qa_data):
    """Answer question using Q&A index"""
    index = qa_data["index"]
    chunks = qa_data["chunks"]

    q_embedding = np.array(embed_model.encode([question])).astype("float32")
    distances, indices = index.search(q_embedding, 3)
    relevant = [chunks[idx] for idx in indices[0]]
    context = "\n\n".join(relevant)

    prompt = f"""
Answer based ONLY on context.

Context:
{context}

Question: {question}

Answer:
"""

    try:
        return call_gemma_with_retry(prompt), relevant
    except:
        return "Could not answer.", relevant

# ================= CUSTOM CSS =================
def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .tab-content {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-top: 1rem;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }
        .status-completed { background: #10b981; color: white; }
        .status-processing { background: #f59e0b; color: white; }
        .status-pending { background: #6b7280; color: white; }
    </style>
    """, unsafe_allow_html=True)


def get_status_badge(status_key):
    """Get status badge HTML for a task"""
    status = st.session_state.get(status_key, "pending")
    if status == "completed":
        return '<span class="status-badge status-completed">✓ Ready</span>'
    elif status == "processing":
        return '<span class="status-badge status-processing">⏳ Loading...</span>'
    else:
        return '<span class="status-badge status-pending">⏸ Pending</span>'

# ================= APP LOGIC =================
st.set_page_config(page_title="Insight in Video", layout="wide", page_icon="🎬")
apply_custom_css()

st.markdown('<div class="main-header"><h1>🎬 Insight in Video</h1><p>AI-Powered Video Summaries & Knowledge Maps</p></div>', unsafe_allow_html=True)

st.markdown("### 📹 Enter YouTube Link")
url = st.text_input("", placeholder="https://youtube.com/...", label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_btn = st.button("🚀 Generate Insights", use_container_width=True)

# ============= PROCESS AUDIO AND START BACKGROUND =============
if generate_btn and url:
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    clean_outputs()

    st.session_state["session_id"] = f"session_{int(time.time())}"

    with st.spinner("📥 Downloading audio..."):
        audio = get_audio_from_youtube(url)

    with st.spinner("📝 Transcribing audio..."):
        transcript, segments = transcribe_audio(audio)

    st.session_state["raw_transcript"] = transcript
    st.session_state["segments"] = segments
    st.session_state["audio_path"] = audio
    st.session_state["processed"] = True
    st.session_state["transcript_status"] = "completed"

    start_all_background_tasks()

    st.success("✅ Transcription complete! Analysis running in background...")
    time.sleep(1)
    st.rerun()

# ============= SHOW TABS =============
if st.session_state.get("processed", False):
    check_and_update_status()

    st.markdown("---")

    any_processing = any(
        st.session_state.get(key, "") == "processing"
        for key in ["timeline_status", "summary_status", "keypoints_status", "mindmap_status", "qa_status"]
    )

    if any_processing:
        time.sleep(2)
        st.rerun()

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = None

    st.markdown("### 📑 Select Analysis Type:")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"📄 Transcript {get_status_badge('transcript_status')}", unsafe_allow_html=True)
        if st.button("View", key="btn_transcript", use_container_width=True):
            st.session_state.active_tab = "transcript"
            st.rerun()

    with c2:
        st.markdown(f"⏱ Timeline {get_status_badge('timeline_status')}", unsafe_allow_html=True)
        if st.button("View", key="btn_timeline", use_container_width=True):
            st.session_state.active_tab = "timeline"
            st.rerun()

    with c3:
        st.markdown(f"📋 Summary {get_status_badge('summary_status')}", unsafe_allow_html=True)
        if st.button("View", key="btn_summary", use_container_width=True):
            st.session_state.active_tab = "summary"
            st.rerun()

    with c4:
        st.markdown(f"📌 Key Points {get_status_badge('keypoints_status')}", unsafe_allow_html=True)
        if st.button("View", key="btn_keypoints", use_container_width=True):
            st.session_state.active_tab = "keypoints"
            st.rerun()

    with c5:
        st.markdown(f"🧠 Mind Map {get_status_badge('mindmap_status')}", unsafe_allow_html=True)
        if st.button("View", key="btn_mindmap", use_container_width=True):
            st.session_state.active_tab = "mindmap"
            st.rerun()

    with c6:
        st.markdown(f"💬 Q&A {get_status_badge('qa_status')}", unsafe_allow_html=True)
        if st.button("View", key="btn_qa", use_container_width=True):
            st.session_state.active_tab = "qa"
            st.rerun()

    st.markdown("---")

    if st.session_state.active_tab:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        tab = st.session_state.active_tab

        if tab == "transcript":
            st.markdown("### 📄 Full Transcript")
            st.write(st.session_state["raw_transcript"])

        elif tab == "timeline":
            st.markdown("### ⏱ Video Timeline")
            if "timeline" in st.session_state:
                st.code(st.session_state["timeline"], language="text")
            else:
                st.info("⏳ Timeline is being generated in background... Please wait.")

        elif tab == "summary":
            st.markdown("### 📋 Executive Summary")
            if "summary" in st.session_state:
                st.write(st.session_state["summary"])
            else:
                st.info("⏳ Summary is being generated in background... Please wait.")

        elif tab == "keypoints":
            st.markdown("### 📌 Key Takeaways")
            if "keypoints" in st.session_state:
                st.write(st.session_state["keypoints"])
            else:
                st.info("⏳ Key points are being extracted in background... Please wait.")

        elif tab == "mindmap":
            st.markdown("### 🧠 Interactive Knowledge Map")
            st.caption("👆 Click nodes to expand/collapse")

            if "mindmap" in st.session_state:
                if "tree" not in st.session_state:
                    edges = mindmap_to_edges(st.session_state["mindmap"])
                    is_dark = get_theme() == "dark"
                    st.session_state["tree"] = build_enhanced_tree(edges, is_dark)

                options = get_tree_options(st.session_state["tree"], get_theme() == "dark")
                st_echarts(options, height="700px", key="mindmap_chart")
            else:
                st.info("⏳ Mind map is being created in background... Please wait.")

        elif tab == "qa":
            st.markdown("### 💬 Ask Questions About the Video")

            if "qa_data" in st.session_state:
                q = st.text_input("💭 Ask something about the video:", key="qa_input")
                if st.button("🔍 Get Answer", key="qa_submit") and q:
                    with st.spinner("🤔 Thinking..."):
                        ans, srcs = answer_question(q, st.session_state["qa_data"])
                    st.success(ans)
                    with st.expander("📚 View Source Context"):
                        for i, s in enumerate(srcs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.write(s)
                            st.markdown("---")
            else:
                st.info("⏳ Q&A index is being built in background... Please wait.")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("👆 **Click any 'View' button to see the content**")

        completed = sum(
            1 for key in ["timeline_status", "summary_status", "keypoints_status", "mindmap_status", "qa_status"]
            if st.session_state.get(key) == "completed"
        )
        total = 5
        st.progress(completed / total, text=f"Background Processing: {completed}/{total} completed")