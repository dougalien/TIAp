import streamlit as st
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import html
import mimetypes

# =========================================================
# CONFIG
# =========================================================

OPENAI_MODEL = "gpt-4o"

# =========================================================
# PROMPTS
# =========================================================

SYSTEM = """
You are a thermal image analysis assistant.

Rules:
- Only use visible thermal patterns and plain context clues.
- If unsure, say "not observable".
- Be concise.
- Do not invent precise temperatures, hidden causes, diagnoses, or certainty that the image does not support.
- Stay grounded in relative hot and cool areas, shapes, gradients, and visible structure.
"""

OBSERVER = SYSTEM + """
Return JSON:
{
  "visible_evidence": ["", "", ""],
  "likely_identification": "",
  "alternative_identification": "",
  "confidence": 1,
  "reasoning": "",
  "next": ""
}
"""

LESSON_PROMPT = """
Create a short lesson plan based on this thermal image analysis.

Image label: {label}
Visible evidence: {evidence}
Likely identification: {likely}
Alternative identification: {alternative}
Reasoning: {reasoning}

Include:
- Learning objective
- Activity
- Assessment question
- Time estimate
"""

# =========================================================
# STATE
# =========================================================

def init_state():
    defaults = {
        "authenticated": False,
        "login_error": "",
        "username": "",
        "specimen_label": "",
        "student_observations": "",
        "student_best_answer": "",
        "known_name": "",
        "context_notes": "",
        "obs": None,
        "lesson_plan": None,
        "uploaded_name": None,
        "image_bytes": None,
        "image_mime": None,
        "image_data_uri": None,
        "show_zoom_preview": True,
        "zoom_fraction": 0.5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================================================
# HELPERS
# =========================================================

def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

def get_app_password() -> str:
    return get_secret("APP_PASSWORD", "")

def encode_image(file) -> str:
    return base64.b64encode(file.getvalue()).decode()

def display_name() -> str:
    name = st.session_state.username.strip()
    return name if name else "Friend"

def parse_observation(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return None

def file_to_data_uri(uploaded_file):
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    return raw, mime, f"data:{mime};base64,{b64}"

def pil_from_session_image():
    if not st.session_state.image_bytes:
        return None
    return Image.open(BytesIO(st.session_state.image_bytes))

def center_crop(img: Image.Image, frac: float):
    w, h = img.size
    frac = max(0.1, min(float(frac), 1.0))
    cw, ch = int(w * frac), int(h * frac)
    left = (w - cw) // 2
    top = (h - ch) // 2
    right = left + cw
    bottom = top + ch
    return img.crop((left, top, right, bottom))

def build_data_uri_from_pil(img: Image.Image):
    buf = BytesIO()
    fmt = img.format if getattr(img, "format", None) in ["JPEG", "PNG", "WEBP"] else "PNG"
    img.save(buf, format=fmt)
    raw = buf.getvalue()
    mime = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }.get(fmt, "image/png")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"

def build_context_block() -> str:
    label = st.session_state.specimen_label.strip() or "[not provided]"
    user_name = display_name()
    observations = st.session_state.student_observations.strip() or "[none entered]"
    guess = st.session_state.student_best_answer.strip() or "[none entered]"
    known = st.session_state.known_name.strip() or "[none provided]"
    notes = st.session_state.context_notes.strip() or "[none provided]"

    return f"""
Image label: {label}
User name: {user_name}
User observations: {observations}
User current guess: {guess}
Known identification: {known}
Extra notes: {notes}
""".strip()

def build_observer_prompt() -> str:
    return OBSERVER + f"""

Use this plain context if helpful:
{build_context_block()}
"""

def get_image_contents_for_openai():
    if not st.session_state.image_data_uri:
        return []

    contents = [{"type": "image_url", "image_url": {"url": st.session_state.image_data_uri}}]

    try:
        if st.session_state.image_bytes:
            img = pil_from_session_image()
            crop = center_crop(img, st.session_state.zoom_fraction)
            contents.append({"type": "image_url", "image_url": {"url": build_data_uri_from_pil(crop)}})
    except Exception:
        pass

    return contents

def call_openai_json(prompt: str) -> str:
    key = get_secret("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Analyze this thermal image."}, *get_image_contents_for_openai()]
            }
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_openai_text(prompt: str) -> str:
    key = get_secret("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful education assistant. Be concise, clear, and practical."
            },
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def render_audio_button(text_to_speak: str):
    safe_text = html.escape(text_to_speak).replace("\n", " ")
    st.components.v1.html(
        f"""
        <div>
          <button
            onclick="window.speechSynthesis.cancel(); window.speechSynthesis.speak(new SpeechSynthesisUtterance('{safe_text}'));"
            style="
              background:#1f1f1f;
              color:white;
              border:none;
              padding:0.65rem 1rem;
              border-radius:8px;
              cursor:pointer;
              font-size:14px;
              font-weight:600;
            ">
            Play Audio
          </button>
        </div>
        """,
        height=58,
    )

# =========================================================
# STYLES
# =========================================================

st.set_page_config(page_title="TIA", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 16px;
    color: #111111;
}

.stApp {
    background: #F6F7F9;
}

.main-card {
    background: #FFFFFF;
    border: 1px solid #D0D4DA;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.pro-card {
    background: #FAFAFA;
    border: 1px solid #C9CDD3;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.input-box {
    background: #EEF4FB;
    border: 1px solid #C9D8EA;
    border-radius: 10px;
    padding: 1rem;
    color: #111111;
}

.output-box {
    background: #FFF5F0;
    border: 1px solid #F0CBBE;
    border-radius: 10px;
    padding: 1rem;
    color: #111111;
}

.small-note {
    color: #4D5661;
    font-size: 0.95rem;
}

.brand-line {
    font-size: 1.02rem;
    color: #222222;
    margin-top: -0.2rem;
}

.brand-link {
    color: #2B5C88;
    font-size: 0.95rem;
    margin-top: 0.2rem;
}

.section-label {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.box-label {
    font-size: 0.92rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

div.stButton > button {
    border-radius: 8px;
    min-height: 44px;
    font-weight: 600;
    border: 1px solid #BFC6CF;
}

div.stButton > button:disabled {
    background: #E7EAEE !important;
    color: #666666 !important;
    border: 1px solid #C7CCD3 !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
textarea, input {
    border-radius: 8px !important;
}

hr {
    border: none;
    border-top: 1px solid #D8DCE2;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOGIN
# =========================================================

def render_login():
    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.markdown("""
        <div class="main-card" style="margin-top:3rem;">
            <div class="section-label">1. Sign In</div>
            <h1 style="margin-bottom:0.15rem;">TIA</h1>
            <div class="brand-line"><strong>T</strong>hermal <strong>I</strong>mage <strong>A</strong>nalysis by We are dougalien</div>
            <div class="brand-link">www.dougalien.com</div>
            <p class="small-note" style="margin-top:0.9rem;">
                Enter the app password to continue.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            entered = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Enter", use_container_width=True)

        if submitted:
            actual = get_app_password()
            if not actual:
                st.session_state.login_error = "APP_PASSWORD is missing from Streamlit secrets."
            elif entered == actual:
                st.session_state.authenticated = True
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "Incorrect password."

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

# =========================================================
# HEADER
# =========================================================

def render_header():
    st.markdown("""
    <div class="main-card">
        <h1 style="margin-bottom:0.15rem;">TIA</h1>
        <div class="brand-line"><strong>T</strong>hermal <strong>I</strong>mage <strong>A</strong>nalysis by We are dougalien</div>
        <div class="brand-link">www.dougalien.com</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PRO FEATURES
# =========================================================

def render_pro_features():
    st.markdown('<div class="section-label">2. Pro Features</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pro-card">
        <div class="small-note">
            These features are visible by design and remain locked until enabled in a future paid version.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.button("Use multiple APIs", disabled=True, use_container_width=True)
        st.button("Upload lesson materials", disabled=True, use_container_width=True)

    with c2:
        st.button("Create lesson plan", disabled=True, use_container_width=True)
        st.button("Save results", disabled=True, use_container_width=True)

    with c3:
        st.button("View analytics", disabled=True, use_container_width=True)
        st.button("See roadmap", disabled=True, use_container_width=True)

# =========================================================
# USER
# =========================================================

def render_user_section():
    st.markdown('<div class="section-label">3. User</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.session_state.username = st.text_input(
            "User name (optional)",
            value=st.session_state.username,
            placeholder="Enter a user name or leave blank"
        )

    with c2:
        st.session_state.specimen_label = st.text_input(
            "Image label or ID (optional)",
            value=st.session_state.specimen_label,
            placeholder="Enter image label or ID"
        )

    c3, c4 = st.columns(2)

    with c3:
        st.session_state.student_observations = st.text_input(
            "User observations (optional)",
            value=st.session_state.student_observations,
            placeholder="Brief observations"
        )

    with c4:
        st.session_state.student_best_answer = st.text_input(
            "Current guess (optional)",
            value=st.session_state.student_best_answer,
            placeholder="Current guess"
        )

# =========================================================
# UPLOAD
# =========================================================

def render_upload_section():
    st.markdown('<div class="section-label">4. Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload thermal image",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear thermal or infrared image for analysis."
    )

    if uploaded_file is not None:
        st.session_state.uploaded_name = uploaded_file.name
        raw, mime, data_uri = file_to_data_uri(uploaded_file)
        st.session_state.image_bytes = raw
        st.session_state.image_mime = mime
        st.session_state.image_data_uri = data_uri

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded thermal image", use_container_width=True)

        if st.session_state.show_zoom_preview:
            try:
                crop = center_crop(image, st.session_state.zoom_fraction)
                st.image(crop, caption="Center crop preview", use_container_width=True)
            except Exception:
                pass

    return uploaded_file

# =========================================================
# ANALYZE
# =========================================================

def render_analyze_section(uploaded_file):
    st.markdown('<div class="section-label">5. Analyze</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.button("Analyze", disabled=True, use_container_width=False)
        return

    if st.button("Analyze", use_container_width=False):
        try:
            raw = call_openai_json(build_observer_prompt())
            obs = parse_observation(raw)

            if not obs:
                st.error("The model returned an unreadable response.")
            else:
                st.session_state.obs = obs
                st.session_state.lesson_plan = None

        except requests.RequestException as e:
            st.error(f"Request error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# =========================================================
# RESULTS
# =========================================================

def render_result_section():
    if not st.session_state.obs:
        return

    obs = st.session_state.obs
    evidence = obs.get("visible_evidence", [])
    evidence_text = ", ".join([e for e in evidence if e]) if evidence else "No clear visible evidence returned."

    st.markdown('<div class="section-label">6. Result</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="input-box">
        <div class="box-label">{display_name()}</div>
        <div>Thermal image ready for analysis.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="output-box">
        <div class="box-label">TIA</div>
        <div><strong>Visible evidence:</strong> {evidence_text}</div>
        <div style="margin-top:0.45rem;"><strong>Likely identification:</strong> {obs.get('likely_identification', '')}</div>
        <div style="margin-top:0.45rem;"><strong>Alternative identification:</strong> {obs.get('alternative_identification', '')}</div>
        <div style="margin-top:0.45rem;"><strong>Confidence:</strong> {obs.get('confidence', '')} out of 5</div>
        <div style="margin-top:0.7rem;"><strong>Reasoning:</strong> {obs.get('reasoning', '')}</div>
        <div style="margin-top:0.7rem;"><strong>Next step:</strong> {obs.get('next', '')}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# AUDIO
# =========================================================

def render_audio_section():
    if not st.session_state.obs:
        return

    obs = st.session_state.obs
    evidence = obs.get("visible_evidence", [])
    evidence_text = ", ".join([e for e in evidence if e]) if evidence else "a few visible thermal features"

    st.markdown('<div class="section-label">7. Optional Audio</div>', unsafe_allow_html=True)

    speech_text = (
        f"Visible evidence: {evidence_text}. "
        f"Likely identification: {obs.get('likely_identification', '')}. "
        f"Alternative identification: {obs.get('alternative_identification', '')}. "
        f"Confidence: {obs.get('confidence', '')} out of 5. "
        f"Reasoning: {obs.get('reasoning', '')}. "
        f"Next step: {obs.get('next', '')}."
    )
    render_audio_button(speech_text)

# =========================================================
# DEVELOPER TOOLS
# =========================================================

def render_dev_tools():
    st.markdown('<div class="section-label">8. Developer Tools</div>', unsafe_allow_html=True)

    with st.expander("Open developer tools", expanded=False):
        st.write("These tools are for testing only and are not part of the standard user view.")

        if not st.session_state.obs:
            st.write("Analyze an image first to test lesson plan and save tools.")
            return

        obs = st.session_state.obs
        c1, c2 = st.columns(2)

        with c1:
            if st.button("Generate lesson plan now", use_container_width=True):
                prompt = LESSON_PROMPT.format(
                    label=st.session_state.specimen_label.strip() or "[not provided]",
                    evidence=", ".join(obs.get("visible_evidence", [])),
                    likely=obs.get("likely_identification", ""),
                    alternative=obs.get("alternative_identification", ""),
                    reasoning=obs.get("reasoning", "")
                )
                try:
                    st.session_state.lesson_plan = call_openai_text(prompt)
                except requests.RequestException as e:
                    st.error(f"Request error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        with c2:
            st.download_button(
                "Download current result as JSON",
                data=json.dumps(obs, indent=2),
                file_name="tia_result.json",
                use_container_width=True
            )

        if st.session_state.lesson_plan:
            st.markdown(f"""
            <div class="output-box">
                <div class="box-label">TIA</div>
                <div><strong>Lesson plan</strong></div>
                <div style="margin-top:0.7rem;">{st.session_state.lesson_plan}</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# MAIN
# =========================================================

if not st.session_state.authenticated:
    render_login()
    st.stop()

render_header()
render_pro_features()
render_user_section()
uploaded_file = render_upload_section()
render_analyze_section(uploaded_file)
render_result_section()
render_audio_section()
render_dev_tools()