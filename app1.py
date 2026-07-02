import os
import streamlit as st
from PIL import Image
import pytesseract
import tempfile
from gtts import gTTS
from deep_translator import GoogleTranslator
import pyttsx3
from textblob import TextBlob
import speech_recognition as sr
from pydub import AudioSegment
import base64
from datetime import datetime
import sympy as sp
import re
import pandas as pd
import json
from io import BytesIO
import html as html_escape
import mimetypes
import random
from openai import OpenAI
import httpx

# Optional libraries
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pronouncing
except Exception:
    pronouncing = None

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="AI Chatbot for Dyslexic Students", layout="wide")
st.markdown("""
<style>

@import url('https://fonts.cdnfonts.com/css/open-dyslexic');

/* Make assistant (Norman) messages white */
.chat-response-text {
    font-family: 'OpenDyslexic', sans-serif !important;
    color: white !important;              
    font-size: 20px !important;
    font-weight: 600 !important;
    background-color: #003399;   /* Dark blue background for contrast */
    padding: 14px;
    border-radius: 12px;
    line-height: 1.6 !important;
}

/* If using st.chat_message() system */
[data-testid="stChatMessageContent"] p {
    font-family: 'OpenDyslexic', sans-serif !important;
    color: white !important;
    font-size: 20px !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)
# ---------------------- SIDEBAR (defaults) ----------------------
st.sidebar.header("⚙️ Features")
feature = st.sidebar.radio(
    "Choose interaction mode:",
    [
        "💬 Chat",
        "🎤 Voice Input",
        "🖼️ Image to Text",
        "📹 Video to Text",
        "🧮 Math Assistant",
        "📄 Generate Report",
        "🌈 Customize Appearance",
        "🔤 Pronunciation",
        "🌍 Translate & Speak",
        "🎮 Gamified Learning",
        "🔎 Tone Detection",
        "📚 Grammar Correction",
        "♿ Accessibility",
        "📐 Geometry / Diagram Reader",
    ],
    index=0,
)

# Sidebar defaults (these can be overridden from the Customize Appearance page)
_sidebar_font_size = st.sidebar.slider("Font Size", 12, 36, 16)
_sidebar_font_color = st.sidebar.color_picker("Font Color", "#000000")
_sidebar_bg_color = st.sidebar.color_picker("Background Color", "#ffffff")
_sidebar_high_contrast = st.sidebar.checkbox("High Contrast Mode", value=False)
_sidebar_large_buttons = st.sidebar.checkbox("Large Buttons", value=False)
_sidebar_enable_highlight = st.sidebar.checkbox("Enable audio word-highlighting", value=True)

# ---------------------- PERSISTED UI SETTINGS ----------------------
# Initialize persisted UI settings from sidebar defaults if not present
if "ui_font_size" not in st.session_state:
    st.session_state["ui_font_size"] = _sidebar_font_size
if "ui_font_color" not in st.session_state:
    st.session_state["ui_font_color"] = _sidebar_font_color
if "ui_bg_color" not in st.session_state:
    st.session_state["ui_bg_color"] = _sidebar_bg_color
if "ui_high_contrast" not in st.session_state:
    st.session_state["ui_high_contrast"] = _sidebar_high_contrast
if "ui_large_buttons" not in st.session_state:
    st.session_state["ui_large_buttons"] = _sidebar_large_buttons
if "enable_highlight" not in st.session_state:
    st.session_state["enable_highlight"] = _sidebar_enable_highlight
if "ui_font_family" not in st.session_state:
    st.session_state["ui_font_family"] = "inherit"
if "ui_font_weight" not in st.session_state:
    st.session_state["ui_font_weight"] = "normal"

# apply effective values
font_size = st.session_state["ui_font_size"]
font_color = st.session_state["ui_font_color"]
bg_color = st.session_state["ui_bg_color"]
high_contrast = st.session_state["ui_high_contrast"]
large_buttons = st.session_state["ui_large_buttons"]
enable_highlight = st.session_state["enable_highlight"]
font_family = st.session_state["ui_font_family"]
font_weight = st.session_state["ui_font_weight"]

# ---------------------- PAGE HEADER & BASE CSS ----------------------
st.markdown(
    """
    <style>
      .card {background: #f8f9fa; padding:14px; border-radius:10px; margin-bottom:12px;}
      .badge {background:#ffd166; padding:6px 10px; border-radius:8px; display:inline-block; margin-right:6px;}
      .points {background:#06d6a0; padding:8px 12px; border-radius:12px; color:#fff; font-weight:600;}
      .section-title{font-size:20px; font-weight:700;}
      .response-container { padding:12px; border-radius:8px; }
      .norman-text { font-size:18px; line-height:1.6; }
      .highlight-word { padding:2px 4px; margin:1px; display:inline-block; background:transparent; border-radius:4px; transition: background-color 0.12s ease; }
      .highlight-word.active { background: #fff59d !important; }
      .audio-player { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align:center'>"
    "<h1>🧠 AI Chatbot for Dyslexic Students</h1>"
    "<p style='font-size:16px;color:#555'>Clear explanations, multisensory learning, and friendly practice.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------- LOAD API KEY ----------------------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")) or ""
OPENAI_KEY = OPENAI_KEY.strip()
if not OPENAI_KEY:
    st.error("❌ OpenAI API key not found! Please add it to `.streamlit/secrets.toml` or set environment variable OPENAI_API_KEY")
    st.stop()
client = OpenAI(api_key=OPENAI_KEY)

# optional import for video processing
try:
    from moviepy.editor import VideoFileClip
except Exception:
    VideoFileClip = None

# ====== Configuration ======
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

NORMAN_SYSTEM_PROMPT = (
    "You are Norman — a supportive, patient, and clear virtual assistant designed to help dyslexic "
    "students. Keep responses friendly, use short sentences, avoid complex jargon unless asked, and "
    "format mathematics clearly. When providing solutions to math problems, show steps and a final answer."
)

def validate_openai_key():
    try:
        # Try to make a simple API call instead of listing models (which might be rate-limited)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"}
            ],
            max_tokens=5
        )
        return True
    except Exception as e:
        print(f"OpenAI validation failed: {e}")
        return False

OPENAI_OK = validate_openai_key()

# --- Dynamic color & UI override ---
effective_text_color = "#ffffff" if high_contrast else font_color or "#000000"
effective_bg_for_response = "#000000" if high_contrast else bg_color or "#ffffff"
# ensure card background uses the response/background color so history boxes contrast with text
card_bg = effective_bg_for_response
st.markdown(
    f"""
    <style>
      body {{ background: {bg_color}; }}
      .response-container {{
        background: {effective_bg_for_response};
        color: {effective_text_color};
        font-family: {font_family};
        font-weight: {font_weight};
      }}
      .card {{
        background: {card_bg} !important;
        color: {effective_text_color} !important;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
      }}
      .section-title {{
        font-family: {font_family};
        font-weight: {font_weight};
        color: {effective_text_color} !important;
      }}
      .norman-text {{
        color: {effective_text_color};
        font-size: {font_size}px;
        font-family: {font_family};
        font-weight: {font_weight};
        line-height:1.6;
      }}
      .highlight-word {{ color: {effective_text_color}; }}
      .highlight-word.active {{ background: #fff59d !important; color: #000000 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== Helper functions =====
def call_ollama_chat(user_text, persona_prompt=NORMAN_SYSTEM_PROMPT):
    """Call Ollama local AI for chat responses (completely free!)"""
    try:
        # Ollama runs locally, or at http://ollama:11434 in Docker
        OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
        
        # Prepare the prompt
        prompt = f"{persona_prompt}\n\nUser: {user_text}\n\nNorman:"
        
        payload = {
            "model": "llama3.2",  # Small, fast model - great for dyslexia assistance!
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500
            }
        }
        
        # Call Ollama
        response = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
        
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                return result["response"].strip()
        
        return None
        
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

def call_huggingface_chat(user_text, persona_prompt=NORMAN_SYSTEM_PROMPT):
    """Call Hugging Face free API for chat responses"""
    try:
        # Use a free model from Hugging Face
        # We'll use a simple prompt format that works with most models
        prompt = f"{persona_prompt}\n\nUser: {user_text}\n\nNorman:"
        
        # Use the Hugging Face Inference API with a free model
        # You can get a free API key from https://huggingface.co/settings/tokens
        # For now, we'll try with a public endpoint or use a local model
        
        # First, try using a simple approach with a free model
        # Try using microsoft/Phi-3-mini-4k-instruct or similar
        
        # Let's use the free Hugging Face Inference API
        # For this to work best, you can add a HUGGING_FACE_API_KEY to secrets.toml
        
        HUGGING_FACE_API_KEY = st.secrets.get("HUGGING_FACE_API_KEY", "")
        
        if HUGGING_FACE_API_KEY:
            # Use the official Hugging Face API
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
            headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "return_full_text": False
                }
            }
            
            response = httpx.post(API_URL, headers=headers, json=payload, timeout=30.0)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
        
        # Fallback: Use a free model without API key (limited, but works for demonstration)
        # For this, let's use a different approach with a public model
        print("Trying Hugging Face without API key...")
        
        # Alternative approach: Use a simple, free model
        # Let's build a more capable fallback response system
        return None
        
    except Exception as e:
        print(f"Hugging Face API error: {e}")
        return None

def call_openai_chat(user_text, persona_prompt=NORMAN_SYSTEM_PROMPT, temperature=0.7, max_tokens=350):
    if not OPENAI_OK:
        return None
    try:
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": user_text},
        ]
        model_candidates = ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        for model in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Properly handle the OpenAI response object
                if hasattr(resp, 'choices') and len(resp.choices) > 0:
                    choice = resp.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        content = choice.message.content
                        if content:
                            return content.strip()
            except Exception as e:
                # Print the error for debugging
                print(f"Error with model {model}: {e}")
                continue
        return None
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

def fallback_ai_response(user_text):
    if not user_text.strip():
        return "Please enter some text."
    
    # Try to provide a direct, helpful response for common questions
    user_lower = user_text.lower()
    
    # Common questions about computers
    if any(keyword in user_lower for keyword in ['what is a computer', 'what is computer']):
        return ("— Norman: A computer is an electronic device that processes data and "
                "performs tasks according to instructions. It has hardware like a keyboard, "
                "screen, and processor, and software that tells it what to do!")
    
    elif any(keyword in user_lower for keyword in ['how', 'what', 'why', 'when', 'where']):
        # For question-like queries, give a helpful response
        return ("— Norman: That's a great question! I can help you with this when I'm "
                "connected to the AI service. Right now, you can use the other features "
                "like image-to-text, math assistant, or voice input!")
    
    sentiment = TextBlob(user_text).sentiment.polarity
    if sentiment > 0:
        reply = "That sounds positive! 😊 Let me know how I can help you."
    elif sentiment < 0:
        reply = "I'm here to help! � Let's work through this together."
    else:
        reply = random.choice(
            [
                "I'm here to help! Let's explore the other features available.",
                "That's interesting! Try using the math assistant or image-to-text features!",
                "Great! I can help with many things when connected to AI. Right now, try our other tools!",
                "Let's check out the gamified learning or pronunciation features!",
            ]
        )
    return "— Norman: " + reply

def format_detailed_reply(raw_text):
    if not raw_text:
        return raw_text
    txt = re.sub(r"^[—\-]+\s*Norman:\s*", "", raw_text, flags=re.I).strip()
    sentences = re.split(r'(?<=[.!?])\s+', txt)
    para_sentences = sentences[:4]
    paragraph = " ".join([s.strip() for s in para_sentences]).strip()
    rest = sentences[4:]
    bullets = []
    for i in range(3):
        if rest:
            bullets.append(rest.pop(0).strip())
        else:
            bullets.append("—")
    if not paragraph and sentences:
        paragraph = sentences[0]
    final = paragraph + "\n\n"
    for b in bullets:
        final += f"- {b}\n"
    return "— Norman: " + final.strip()

def get_builtin_response(user_text):
    """Get responses from a built-in knowledge base for common questions"""
    user_lower = user_text.lower().strip()
    
    # Built-in knowledge base
    knowledge_base = {
        # Computer & Technology
        "what is a computer": "A computer is an electronic device that processes data and performs tasks according to instructions. It has hardware like a keyboard, screen, and processor, and software that tells it what to do!",
        "what is computer": "A computer is an electronic device that processes data and performs tasks according to instructions. It has hardware like a keyboard, screen, and processor, and software that tells it what to do!",
        
        # Math
        "what is pi": "Pi (π) is a mathematical constant approximately equal to 3.14159. It's the ratio of a circle's circumference to its diameter!",
        "what is addition": "Addition is the mathematical operation of combining two or more numbers to get a total sum!",
        "what is multiplication": "Multiplication is a mathematical operation that's like repeated addition. For example, 3 × 4 means adding 3 four times (3+3+3+3=12)!",
        "what is division": "Division is the mathematical operation of splitting a number into equal parts. It's the opposite of multiplication!",
        
        # Science
        "what is gravity": "Gravity is the natural force that pulls objects toward each other. It's what keeps us on the ground and makes things fall!",
        "what is photosynthesis": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make their food (glucose) and release oxygen!",
        
        # English
        "what is a noun": "A noun is a word that names a person, place, thing, or idea. Examples: cat, school, happiness!",
        "what is a verb": "A verb is a word that describes an action, occurrence, or state of being. Examples: run, eat, is!",
        
        # General
        "hello": "Hello! I'm Norman, your friendly assistant. How can I help you today?",
        "hi": "Hi there! I'm Norman. What would you like to learn about?",
        "how are you": "I'm doing great, thanks for asking! How can I help you today?",
    }
    
    # Check for exact matches first
    if user_lower in knowledge_base:
        return knowledge_base[user_lower]
    
    # Check for partial matches
    for question, answer in knowledge_base.items():
        if question in user_lower:
            return answer
    
    return None

def get_ai_response(user_text, use_openai=True, persona="Norman", style="supportive"):
    if not user_text.strip():
        return "Please enter some text."
    
    # Try math solving first
    try:
        if re.search(r"\b(solve|integrate|differentiate|derivative)\b", user_text, flags=re.I):
            x = sp.symbols("x")
            lowered = user_text.lower().replace("^", "**")
            if "solve" in lowered:
                expr = lowered.split("solve", 1)[1].strip()
                if "=" in expr:
                    lhs, rhs = expr.split("=", 1)
                    sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
                else:
                    sol = sp.solve(sp.sympify(expr), x)
                sol_str = str(sol).replace("sqrt", "√")
                reply = f"— Norman: Solution: {sol_str}"
                if style == "detailed":
                    return format_detailed_reply(reply)
                return reply
    except Exception:
        pass
    
    # Try Ollama first! (local, free, no API key needed!)
    resp = call_ollama_chat(user_text, persona_prompt=NORMAN_SYSTEM_PROMPT)
    if resp:
        if not resp.lower().startswith("— norman"):
            resp = "— Norman: " + resp
        if style == "detailed":
            return format_detailed_reply(resp)
        return resp
    
    # Try OpenAI next
    if use_openai and OPENAI_OK:
        resp = call_openai_chat(user_text, persona_prompt=NORMAN_SYSTEM_PROMPT + f" Style: {style}.")
        if resp:
            if not resp.lower().startswith("— norman"):
                resp = "— Norman: " + resp
            if style == "detailed":
                return format_detailed_reply(resp)
            return resp
    
    # Try Hugging Face next
    resp = call_huggingface_chat(user_text, persona_prompt=NORMAN_SYSTEM_PROMPT)
    if resp:
        if not resp.lower().startswith("— norman"):
            resp = "— Norman: " + resp
        if style == "detailed":
            return format_detailed_reply(resp)
        return resp
    
    # Then check our built-in knowledge base
    builtin_resp = get_builtin_response(user_text)
    if builtin_resp:
        formatted = "— Norman: " + builtin_resp
        if style == "detailed":
            return format_detailed_reply(formatted)
        return formatted
    
    # Fall back to our improved fallback responses
    resp = fallback_ai_response(user_text)
    if style == "detailed":
        return format_detailed_reply(resp)
    return resp

# ------------------ TTS and Highlighting ------------------
def text_to_speech_with_duration(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        try:
            audio = AudioSegment.from_file(tmp.name)
            duration_ms = len(audio)
        except Exception:
            duration_ms = 0
        return tmp.name, duration_ms
    except Exception:
        try:
            engine = pyttsx3.init()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            engine.save_to_file(text, tmp.name)
            engine.runAndWait()
            try:
                audio = AudioSegment.from_file(tmp.name)
                duration_ms = len(audio)
            except Exception:
                duration_ms = 0
            return tmp.name, duration_ms
        except Exception:
            return None, 0

def make_word_timestamps(text, duration_ms):
    words = re.findall(r"\S+", text)
    if not words or not duration_ms or duration_ms <= 0:
        return []
    per_word = duration_ms / len(words)
    timestamps = []
    for i, w in enumerate(words):
        start = int(i * per_word)
        end = int((i + 1) * per_word)
        timestamps.append({"word": w, "start_ms": start, "end_ms": end})
    return timestamps

def build_highlighted_html_with_audio(text, audio_file_path, timestamps):
    if timestamps:
        spans = []
        for i, item in enumerate(timestamps):
            escaped = html_escape.escape(item["word"])
            spans.append(f"<span id='w{i}' class='highlight-word'>{escaped}</span>")
        spans_html = " ".join(spans)
    else:
        spans_html = f"<div class='norman-text'>{html_escape.escape(text)}</div>"
    try:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        audio_b64 = ""
    mime_type, _ = mimetypes.guess_type(audio_file_path) if audio_file_path else (None, None)
    if not mime_type:
        mime_type = "audio/mpeg" if (audio_file_path and audio_file_path.endswith(".mp3")) else "audio/wav"
    if timestamps and audio_b64:
        timestamp_arr = [[i, item["start_ms"] / 1000.0, item["end_ms"] / 1000.0] for i, item in enumerate(timestamps)]
        html = f"""
        <div class="response-container">
          <audio id="player" class="audio-player" controls preload="metadata">
            <source src="data:{mime_type};base64,{audio_b64}" type="{mime_type}">
            Your browser does not support the audio element.
          </audio>
          <div id="text-block" class="norman-text" style="margin-top:12px;">
            {spans_html}
          </div>
        </div>
        <script>
        const timestamps = {json.dumps(timestamp_arr)};
        const audio = document.getElementById('player');
        const words = document.querySelectorAll('.highlight-word');
        function clearHighlights() {{
          words.forEach(el => {{
            el.classList.remove('active');
            el.style.background = 'transparent';
          }});
        }}
        audio.ontimeupdate = function() {{
          const t = audio.currentTime;
          clearHighlights();
          for (let i=0;i<timestamps.length;i++) {{
            const start = timestamps[i][1];
            const end = timestamps[i][2];
            if (t >= start && t <= end) {{
              const el = document.getElementById('w'+timestamps[i][0]);
              if (el) {{
                el.classList.add('active');
                el.style.background = '#fff59d';
              }}
            }}
          }}
        }};
        audio.onended = function() {{ clearHighlights(); }};
        </script>
        """
    else:
        html = f"""
        <div class="response-container">
          <audio class="audio-player" controls preload="metadata">
            <source src="data:{mime_type};base64,{audio_b64}" type="{mime_type}">
            Your browser does not support the audio element.
          </audio>
          <div class="norman-text" style="margin-top:12px;">{spans_html}</div>
        </div>
        """
    return html

# ------------------ Other helpers ------------------
def text_to_speech_gtts(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def extract_text_from_image_bytes(image_bytes):
    try:
        if isinstance(image_bytes, (bytes, bytearray)):
            img = Image.open(BytesIO(image_bytes))
        elif hasattr(image_bytes, "read"):
            image_bytes.seek(0)
            img = Image.open(image_bytes)
        elif isinstance(image_bytes, Image.Image):
            img = image_bytes
        else:
            return ""
        img = img.convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def detect_tone(text):
    tb = TextBlob(text)
    polarity = tb.sentiment.polarity
    words = text.lower().split()
    if any(w in words for w in ("please", "could", "would", "thanks", "thank")):
        style = "polite"
    elif polarity > 0.4:
        style = "happy"
    elif polarity < -0.4:
        style = "angry/sad"
    else:
        style = "neutral"
    return {"polarity": polarity, "style": style}

def phonetic_spelling(word):
    if not word:
        return ""
    word_clean = re.sub(r'[^A-Za-z\']', '', word).lower()
    if not word_clean:
        return word
    if pronouncing:
        try:
            phones = pronouncing.phones_for_word(word_clean)
            if phones:
                return phones[0]
        except Exception:
            pass
    chunks = re.findall(r'[^aeiouy]*[aeiouy]+(?:[^aeiouy]+(?=[^aeiouy]|$))?', word_clean, flags=re.I)
    if not chunks:
        chunks = [word_clean[i:i+2] for i in range(0, len(word_clean), 2)]
    return "-".join(chunks)

def geometry_read_image(uploaded_file):
    img_bytes = uploaded_file.read()
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return "Could not open image."
    text = pytesseract.image_to_string(img)
    shapes = []
    if cv2 is not None:
        try:
            import numpy as np
            np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            _,thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
                if len(approx) == 3:
                    shapes.append("triangle")
                elif len(approx) == 4:
                    shapes.append("quadrilateral (rectangle/square)")
                elif len(approx) > 7:
                    shapes.append("circle/ellipse")
            shapes = list(dict.fromkeys(shapes))
        except Exception:
            shapes = []
    result = {"ocr_text": text.strip(), "shapes_detected": shapes}
    return result

def transcribe_audio_file_bytes(file_like):
    try:
        raw = file_like.read()
        suffix = os.path.splitext(getattr(file_like, "name", "audio.wav"))[1] or ".wav"
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_in.write(raw)
        tmp_in.flush()
        try:
            audio = AudioSegment.from_file(tmp_in.name)
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio.export(tmp_out.name, format="wav")
            wav_path = tmp_out.name
        except Exception:
            wav_path = tmp_in.name
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
        return text
    except Exception:
        return ""

def build_report(selected_sections):
    parts = []
    for sec in selected_sections:
        key = histories.get(sec)
        if not key:
            continue
        parts.append(f"==== {sec.upper()} ====")
        for sender, msg in st.session_state.get(key, []):
            parts.append(f"{sender}: {msg}")
        parts.append("\n")
    return "\n".join(parts).encode("utf-8")

# ====== Session state helpers ======
histories = {
    "chat": "chat_history_chat",
    "math": "chat_history_math",
    "image": "chat_history_image",
    "video": "chat_history_video",
    "audio": "chat_history_audio",
    "pron": "chat_history_pron",
    "translate": "chat_history_translate",
    "tone": "chat_history_tone",
    "geo": "chat_history_geo",
    "game": "chat_history_game",
    "grammar": "chat_history_grammar",
}
for h in histories.values():
    if h not in st.session_state:
        st.session_state[h] = []

for key, default in [
    ("chat_input", ""),
    ("style_select", "supportive"),
    ("use_openai", OPENAI_OK),
    ("image_upload", None),
    ("video_upload", None),
    ("audio_upload", None),
    ("math_input", ""),
    ("pron_input", ""),
    ("translate_input", ""),
    ("translate_lang", "en"),
    ("quiz_mode", "easy"),
    ("last_response_html", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "points" not in st.session_state:
    st.session_state["points"] = 0
if "badges" not in st.session_state:
    st.session_state["badges"] = []
if "current_quiz" not in st.session_state:
    st.session_state["current_quiz"] = None
if "quiz_progress" not in st.session_state:
    st.session_state["quiz_progress"] = 0

def display_history(history_key, title):
    # Render history with explicit text color so messages remain visible on any background
    hist = st.session_state.get(history_key, [])
    effective_text = globals().get("effective_text_color", "#000000")
    # outer card & title
    html = f"<div class='card'><div class='section-title'>{html_escape.escape(title)}</div>"
    if not hist:
        html += "<div style='color:#666'>No interactions yet.</div>"
    else:
        for sender, msg in hist[-50:]:
            label = "🧍 You" if sender.lower().startswith("you") else "🤖 Norman"
            # escape message and preserve simple line breaks
            safe_msg = html_escape.escape(str(msg)).replace("\n", "<br>")
            # use norman-text class so font-size / line-height apply, and inline color to ensure visibility
            html += (
                f"<div style='padding:6px;margin-bottom:6px;border-radius:6px;background:transparent;'>"
                f"<strong style='margin-right:6px;color:{effective_text};'>{label}:</strong>"
                f"<span class='norman-text' style='color:{effective_text};'>{safe_msg}</span>"
                f"</div>"
            )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def award_points(n=1):
    st.session_state["points"] += n
    if st.session_state["points"] >= 50 and "Master" not in st.session_state["badges"]:
        st.session_state["badges"].append("Master")
    elif st.session_state["points"] >= 20 and "Achiever" not in st.session_state["badges"]:
        st.session_state["badges"].append("Achiever")
    elif st.session_state["points"] >= 5 and "Explorer" not in st.session_state["badges"]:
        st.session_state["badges"].append("Explorer")

SAMPLE_QUIZZES = [
    {"q":"Which spelling is correct?","options":["acommodate","accommodate","acomodate"], "answer":1},
    {"q":"What is 6 × 7?","options":["42","36","48"], "answer":0},
    {"q":"Select the plural of 'child'","options":["childs","children","childes"], "answer":1},
]

def start_quiz():
    st.session_state["current_quiz"] = random.choice(SAMPLE_QUIZZES)
    st.session_state["quiz_progress"] = 0

def submit_quiz_answer(choice_index):
    q = st.session_state.get("current_quiz")
    if not q:
        return False
    correct = (choice_index == q["answer"])
    if correct:
        award_points(3)
    else:
        award_points(0)
    st.session_state["quiz_progress"] += 1
    st.session_state["current_quiz"] = None
    return correct

# ===== Handlers =====
def handle_get_answer():
    user_input = st.session_state.get("chat_input", "").strip()
    if not user_input:
        st.warning("Please enter a question.")
        return
    reply = get_ai_response(
        user_input,
        use_openai=st.session_state.get("use_openai", OPENAI_OK),
        style=st.session_state.get("style_select", "supportive"),
    )
    st.session_state[histories["chat"]].append(("You", user_input))
    st.session_state[histories["chat"]].append(("Norman", reply))
    award_points(1)
    if enable_highlight:
        audio_path, duration_ms = text_to_speech_with_duration(reply)
        timestamps = make_word_timestamps(reply, duration_ms)
        html = build_highlighted_html_with_audio(reply, audio_path, timestamps)
        st.session_state["last_response_html"] = html
    else:
        audio_path, _ = text_to_speech_with_duration(reply)
        st.session_state["last_response_html"] = ""
        if audio_path:
            st.audio(audio_path)

def handle_clear_chat():
    st.session_state[histories["chat"]] = []
    st.session_state["last_response_html"] = ""
    st.session_state["chat_input"] = ""

def handle_image_ocr():
    upload = st.session_state.get("image_upload")
    if not upload:
        st.warning("Please upload an image.")
        return
    upload.seek(0)
    text = extract_text_from_image_bytes(upload)
    st.session_state[histories["image"]].append(("You (image)", upload.name if hasattr(upload, "name") else "image"))
    st.session_state[histories["image"]].append(("Norman", text or "No text found"))
    award_points(1)
    reply = get_ai_response(text or "No text found", use_openai=st.session_state.get("use_openai", OPENAI_OK))
    if enable_highlight:
        audio_path, duration_ms = text_to_speech_with_duration(reply)
        timestamps = make_word_timestamps(reply, duration_ms)
        html = build_highlighted_html_with_audio(reply, audio_path, timestamps)
        st.components.v1.html(html, height=300 if timestamps else 180)
    else:
        audio_path, _ = text_to_speech_with_duration(reply)
        if audio_path:
            st.audio(audio_path)

def handle_video_transcribe():
    upload = st.session_state.get("video_upload")
    if not upload:
        st.warning("Please upload a video.")
        return
    try:
        tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.name)[1])
        upload.seek(0)
        tmp_vid.write(upload.read())
        tmp_vid.flush()
        text = ""
        if VideoFileClip is not None:
            try:
                clip = VideoFileClip(tmp_vid.name)
                tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                clip.audio.write_audiofile(tmp_audio.name, verbose=False, logger=None)
                r = sr.Recognizer()
                with sr.AudioFile(tmp_audio.name) as source:
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data)
            except Exception:
                text = ""
        st.session_state[histories["video"]].append(("You (video)", upload.name))
        st.session_state[histories["video"]].append(("Norman", text or "No transcription"))
        award_points(1)
        st.success("Transcription:")
        st.write(text or "No transcription")
        reply = get_ai_response(text or "No transcription", use_openai=st.session_state.get("use_openai", OPENAI_OK))
        if enable_highlight:
            audio_path, duration_ms = text_to_speech_with_duration(reply)
            timestamps = make_word_timestamps(reply, duration_ms)
            html = build_highlighted_html_with_audio(reply, audio_path, timestamps)
            st.components.v1.html(html, height=300 if timestamps else 180)
        else:
            audio_path, _ = text_to_speech_with_duration(reply)
            if audio_path:
                st.audio(audio_path)
    except Exception as e:
        st.error(f"Video processing failed: {e}")

def handle_audio_transcribe():
    upload = st.session_state.get("audio_upload")
    if not upload:
        st.warning("Please upload an audio file.")
        return
    upload.seek(0)
    text = transcribe_audio_file_bytes(upload)
    st.session_state[histories["audio"]].append(("You (audio)", upload.name))
    st.session_state[histories["audio"]].append(("Norman", text or "No transcription"))
    award_points(1)
    st.success("Transcription:")
    st.write(text or "No transcription")
    reply = get_ai_response(text or "No transcription", use_openai=st.session_state.get("use_openai", OPENAI_OK))
    if enable_highlight:
        audio_path, duration_ms = text_to_speech_with_duration(reply)
        timestamps = make_word_timestamps(reply, duration_ms)
        html = build_highlighted_html_with_audio(reply, audio_path, timestamps)
        st.components.v1.html(html, height=300 if timestamps else 180)
    else:
        audio_path, _ = text_to_speech_with_duration(reply)
        if audio_path:
            st.audio(audio_path)


def handle_math_solve():
    expr_text = st.session_state.get("math_input", "").strip()
    if not expr_text:
        st.warning("Enter a math expression or 'solve ...'.")
        return
    try:
        lowered = expr_text.lower().replace("^", "**")
        x = sp.symbols("x")
        if lowered.startswith("solve"):
            body = lowered.split("solve", 1)[1].strip()
            if "=" in body:
                lhs, rhs = body.split("=", 1)
                sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
            else:
                sol = sp.solve(sp.sympify(body), x)
            if isinstance(sol, (list, tuple)):
                formatted = [str(s).replace("sqrt", "√") for s in sol]
                result = f"{formatted}"
            else:
                result = str(sol).replace("sqrt", "√")
        else:
            res = sp.sympify(lowered)
            result = str(sp.simplify(res)).replace("sqrt", "√")
        st.session_state[histories["math"]].append(("You (math)", expr_text))
        st.session_state[histories["math"]].append(("Norman", result))
        # NOTE: do NOT mutate st.session_state["math_input"] here (avoids Streamlit widget state error)
        award_points(2)
        st.success("Math Result:")
        st.write(result)
        reply = get_ai_response("Explain: " + expr_text, use_openai=st.session_state.get("use_openai", OPENAI_OK))
        if enable_highlight:
            audio_path, duration_ms = text_to_speech_with_duration(reply)
            timestamps = make_word_timestamps(reply, duration_ms)
            html = build_highlighted_html_with_audio(reply, audio_path, timestamps)
            st.components.v1.html(html, height=300 if timestamps else 180)
        else:
            audio_path, _ = text_to_speech_with_duration(reply)
            if audio_path:
                st.audio(audio_path)
    except Exception as e:
        st.error(f"Could not compute expression: {e}")

# ------------------ UI: pages ------------------
if feature == "💬 Chat":
    st.header("💬 Chat")
    # Input at the top with a Send button, then show conversation below
    col_in, col_send = st.columns([8, 1])
    with col_in:
        st.text_input("Ask Norman anything:", key="chat_input", placeholder="Type a question and press Send")
    with col_send:
        if st.button("Send", key="send_chat"):
            handle_get_answer()
            st.rerun()

    # Render conversation history below the input
    display_history(histories["chat"], "Chat")

    # Clear chat button
    if st.button("Clear Chat"):
        handle_clear_chat()
        st.rerun()

elif feature == "🎤 Voice Input":
    st.header("🎤 Voice Input")
    st.file_uploader("Upload audio (wav/mp3/m4a):", type=["wav","mp3","m4a"], key="audio_upload")
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Transcribe Audio"):
            handle_audio_transcribe()
    with col2:
        if st.button("Clear Audio History"):
            st.session_state[histories["audio"]] = []
            st.rerun()
    display_history(histories["audio"], "Audio History")

elif feature == "🖼️ Image to Text":
    st.header("🖼️ Image to Text (OCR)")
    st.file_uploader("Upload an image (jpg/png):", type=["jpg","jpeg","png"], key="image_upload")
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Run OCR on Image"):
            handle_image_ocr()
    with col2:
        if st.button("Clear Image History"):
            st.session_state[histories["image"]] = []
            st.rerun()
    display_history(histories["image"], "Image OCR History")


elif feature == "📹 Video to Text":
    st.header("📹 Video to Text")
    st.file_uploader("Upload a video (mp4/mov):", type=["mp4","mov","mkv","webm"], key="video_upload")
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Transcribe Video"):
            handle_video_transcribe()
    with col2:
        if st.button("Clear Video History"):
            st.session_state[histories["video"]] = []
            st.rerun()
    display_history(histories["video"], "Video History")


elif feature == "🧮 Math Assistant":
    st.header("🧮 Math Assistant")
    st.text_area("Enter math question or 'solve ...':", key="math_input", height=100)
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Solve / Process"):
            handle_math_solve()
    with col2:
        if st.button("Clear Math History"):
            st.session_state[histories["math"]] = []
            st.session_state["math_input"] = ""
            st.rerun()
    display_history(histories["math"], "Math History")

elif feature == "🔤 Pronunciation":
    st.header("🔤 Pronunciation & Phonetic Spelling")
    st.text_input("Enter a word or phrase:", key="pron_input")
    col1, col2 = st.columns([2,1])
    with col1:
        if st.button("Show Phonetic & Play Pronunciation"):
            w = st.session_state.get("pron_input","").strip()
            if not w:
                st.warning("Please enter a word or phrase.")
            else:
                ph = phonetic_spelling(w)
                st.session_state[histories["pron"]].append(("You (pron)", w))
                st.session_state[histories["pron"]].append(("Norman", f"Phonetic: {ph}"))
                st.success(f"Phonetic (approx): {ph}")
                try:
                    audio_path = text_to_speech_gtts(w, lang="en")
                    st.audio(audio_path)
                except Exception:
                    st.error("Could not generate pronunciation audio.")
                award_points(1)
    with col2:
        if st.button("Clear Pronunciation History"):
            st.session_state[histories["pron"]] = []
    display_history(histories["pron"], "Pronunciation History")

# For Tone Detection section:
elif feature == "🔎 Tone Detection":
    st.header("🔎 Tone Detection")
    st.text_area("Paste text to analyze tone:", key="tone_input", height=120)
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Detect Tone"):
            t = st.session_state.get("tone_input", "")
            if not t:
                st.warning("Please paste some text.")
            else:
                res = detect_tone(t)
                st.session_state[histories["tone"]].append(("You (text)", t[:80]+"..."))
                st.session_state[histories["tone"]].append(("Norman", str(res)))
                st.write(res)
                award_points(1)
    with col2:
        if st.button("Clear Tone History"):
            st.session_state[histories["tone"]] = []
            st.session_state["tone_input"] = ""
            st.rerun()
    display_history(histories["tone"], "Tone History")

elif feature == "📚 Grammar Correction":
    st.header("📚 Grammar Correction")
    st.text_area("Paste text to correct:", key="grammar_input", height=180)
    style_gc = st.selectbox("Response style:", ["supportive", "concise", "detailed", "encouraging"], key="grammar_style")
    use_openai_gc = st.checkbox("Use OpenAI for correction", value=OPENAI_OK, key="use_openai_gc")
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Correct Grammar"):
            txt = st.session_state.get("grammar_input","").strip()
            if not txt:
                st.warning("Please paste some text to correct.")
            else:
                if use_openai_gc and OPENAI_OK:
                    prompt = f"Correct the grammar and punctuation of the following text and provide a short explanation:\n\n{txt}\n\nReturn the corrected text and explanation."
                    resp = call_openai_chat(prompt, persona_prompt=NORMAN_SYSTEM_PROMPT, temperature=0.2, max_tokens=400)
                    reply = resp or fallback_ai_response(txt)
                else:
                    try:
                        tb = TextBlob(txt)
                        corrected = str(tb.correct())
                        reply = f"— Norman: Corrected: {corrected}\nExplanation: Minor spelling/grammar corrections applied."
                    except Exception:
                        reply = fallback_ai_response(txt)
                if style_gc == "detailed":
                    reply = format_detailed_reply(reply)
                st.session_state[histories["grammar"]].append(("You (grammar)", txt[:120]+"..."))
                st.session_state[histories["grammar"]].append(("Norman", reply))
                st.markdown("### Correction:")
                st.write(reply)
                award_points(2)
    with col2:
        if st.button("Clear Grammar History"):
            st.session_state[histories["grammar"]] = []
            st.session_state["grammar_input"] = ""
            st.rerun()
    display_history(histories["grammar"], "Grammar History")

elif feature == "🌍 Translate & Speak":
    st.header("🌍 Translate & Speak")
    st.text_input("Enter text to translate:", key="translate_input")
    lang_map = {
        "English (en)": "en",
        "Hindi (hi)": "hi",
        "Spanish (es)": "es",
        "Tamil (ta)": "ta",
        "French (fr)": "fr",
    }
    lang_choice = st.selectbox("Target language:", options=list(lang_map.keys()), key="translate_lang")
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Translate & Speak"):
            txt = st.session_state.get("translate_input","").strip()
            if not txt:
                st.warning("Please enter text to translate.")
            else:
                code = lang_map.get(lang_choice, "en")
                try:
                    translated = GoogleTranslator(source='auto', target=code).translate(txt)
                except Exception:
                    translated = txt
                st.session_state[histories["translate"]].append(("You (translate)", txt[:80]+"..."))
                st.session_state[histories["translate"]].append(("Norman", translated))
                st.success("Translation:")
                st.write(translated)
                try:
                    audio_path = text_to_speech_gtts(translated, lang=code)
                    st.audio(audio_path)
                except Exception:
                    st.error("TTS for selected language not available.")
                award_points(1)
    with col2:
        if st.button("Clear Translation History"):
            st.session_state[histories["translate"]] = []
            st.session_state["translate_input"] = ""
            st.rerun()
    display_history(histories["translate"], "Translate History")

elif feature == "🎮 Gamified Learning":
    st.header("🎮 Gamified Learning")
    st.write("Earn points and badges by answering short quizzes.")
    st.markdown(f"**Points:** {st.session_state.get('points', 0)}  •  **Badges:** {', '.join(st.session_state.get('badges', [])) or 'None'}")
    if st.button("Start New Quiz"):
        start_quiz()
    current = st.session_state.get("current_quiz")
    if current:
        st.markdown(f"**Question:** {current.get('q')}")
        options = current.get("options", [])
        choice = st.radio("Select an answer:", options, key="quiz_choice")
        if st.button("Submit Answer"):
            try:
                idx = options.index(choice)
            except ValueError:
                idx = 0
            correct = submit_quiz_answer(idx)
            st.session_state[histories["game"]].append(("You (quiz)", current.get("q")))
            st.session_state[histories["game"]].append(("Norman", f"Answered: {choice} — {'Correct' if correct else 'Incorrect'}"))
            if correct:
                st.success("Correct! +3 points")
            else:
                st.error("Incorrect. Try another one.")
            st.rerun()
    else:
        st.info("No active quiz. Click 'Start New Quiz' to begin.")
    display_history(histories["game"], "Game History")

elif feature == "🔡 Phonetic Spelling":
    st.header("🔡 Phonetic Spelling (moved)")
    st.info("Phonetic spelling is now combined into the '🔤 Pronunciation' feature.")

elif feature == "📄 Generate Report":
    st.header("📄 Generate Report")
    sections = st.multiselect("Which sections to include in report?", options=list(histories.keys()), default=["chat","math"])
    if st.button("Build & Download Report"):
        data = build_report(sections)
        if data:
            st.download_button("Download .txt report", data=data, file_name="norman_session_report.txt")

elif feature == "🌈 Customize Appearance":
    st.header("🌈 Customize Appearance")
    st.write("Change font size, color, background and style for the app. These settings persist during the session.")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["ui_font_size"] = st.slider("Reading font size (px)", 12, 40, st.session_state["ui_font_size"], key="ui_font_size_slider")
        st.session_state["ui_font_color"] = st.color_picker("Font color", st.session_state["ui_font_color"], key="ui_color_picker")
        st.session_state["ui_bg_color"] = st.color_picker("Background color", st.session_state["ui_bg_color"], key="ui_bg_picker")
        st.session_state["ui_high_contrast"] = st.checkbox("High contrast mode", value=st.session_state["ui_high_contrast"], key="ui_high_contrast_checkbox")
    with col2:
        st.session_state["ui_font_family"] = st.selectbox("Font family", options=["inherit", "OpenDyslexic", "Arial", "Helvetica", "Times New Roman"], index=0, key="ui_font_family_select")
        st.session_state["ui_font_weight"] = st.selectbox("Font weight", options=["normal","bold","600","700"], index=0, key="ui_font_weight_select")
        st.session_state["enable_highlight"] = st.checkbox("Enable audio word-highlighting", value=st.session_state["enable_highlight"], key="ui_enable_highlight")
        if st.button("Reset to defaults"):
            st.session_state["ui_font_size"] = 16
            st.session_state["ui_font_color"] = "#000000"
            st.session_state["ui_bg_color"] = "#ffffff"
            st.session_state["ui_font_family"] = "inherit"
            st.session_state["ui_font_weight"] = "normal"
            st.session_state["ui_high_contrast"] = False
            st.session_state["enable_highlight"] = True
            st.rerun()
    st.markdown(
        f"""
        <style>
          .preview-card {{
            background: {st.session_state['ui_bg_color']};
            color: {('#fff' if st.session_state['ui_high_contrast'] else st.session_state['ui_font_color'])};
            font-size: {st.session_state['ui_font_size']}px;
            font-family: {st.session_state['ui_font_family']};
            font-weight: {st.session_state['ui_font_weight']};
            padding: 12px; border-radius:8px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='card preview-card'>Preview: Norman will display responses using these settings.</div>", unsafe_allow_html=True)

elif feature == "♿ Accessibility":
    st.header("♿ Accessibility")
    st.write("Accessibility settings: adjust line height, letter spacing and choose dyslexia-friendly font.")
    if "access_line_height" not in st.session_state:
        st.session_state["access_line_height"] = 1.6
    if "access_letter_spacing" not in st.session_state:
        st.session_state["access_letter_spacing"] = 0.0
    if "access_dyslexic_font" not in st.session_state:
        st.session_state["access_dyslexic_font"] = False

    st.session_state["access_line_height"] = st.slider("Line height", 1.0, 2.5, st.session_state["access_line_height"], step=0.1, key="access_line_slider")
    st.session_state["access_letter_spacing"] = st.slider("Letter spacing (px)", 0.0, 3.0, float(st.session_state["access_letter_spacing"]), step=0.1, key="access_letter_slider")
    st.session_state["access_dyslexic_font"] = st.checkbox("Use dyslexia-friendly font (OpenDyslexic)", value=st.session_state["access_dyslexic_font"], key="access_dys_font")

    chosen_family = "'OpenDyslexic', " + font_family if st.session_state["access_dyslexic_font"] else font_family
    st.markdown(
        f"""
        <style>
          .norman-text, .card, .response-container {{
            font-family: {chosen_family} !important;
            font-size: {st.session_state['ui_font_size']}px !important;
            line-height: {st.session_state['access_line_height']} !important;
            letter-spacing: {st.session_state['access_letter_spacing']}px !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Preview")
    sample = "Norman helps by breaking information into short clear sentences. Use the accessibility controls to adjust size, spacing and fonts."
    st.markdown(f"<div class='card'><div class='norman-text'>{sample}</div></div>", unsafe_allow_html=True)

# For Geometry Reader section:
elif feature == "📐 Geometry / Diagram Reader":
    st.header("📐 Geometry / Diagram Reader")
    st.write("Upload a diagram or geometry image. Norman will OCR the text and attempt to detect basic shapes.")
    uploaded = st.file_uploader("Upload diagram/image (jpg/png):", type=["jpg","jpeg","png"], key="geo_upload")
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("Analyze Diagram"):
            if not uploaded:
                st.warning("Please upload an image first.")
            else:
                try:
                    uploaded.seek(0)
                    result = geometry_read_image(uploaded)
                    st.session_state[histories["geo"]].append(("You (diagram)", getattr(uploaded, "name", "uploaded_image")))
                    st.session_state[histories["geo"]].append(("Norman", str(result)))
                    st.success("Analysis complete.")
                    st.markdown("**OCR Text:**")
                    st.write(result.get("ocr_text", ""))
                    st.markdown("**Shapes Detected:**")
                    shapes = result.get("shapes_detected", [])
                    if shapes:
                        for s in shapes:
                            st.write(f"- {s}")
                    else:
                        st.write("No clear shapes detected.")
                    award_points(2)
                except Exception as e:
                    st.error(f"Could not analyze image: {e}")
    with col2:
        if st.button("Clear Geometry History"):
            st.session_state[histories["geo"]] = []
            st.rerun()
    display_history(histories["geo"], "Geometry / Diagram History")

else:
    st.header("Toolbox")
    st.write("This section contains advanced or experimental tools (gamified bits)")

st.markdown("---")
st.caption("Developed by You — Empowering Dyslexic Students 🌟")