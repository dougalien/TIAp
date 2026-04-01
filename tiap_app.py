import base64
import io
import json
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st
from PIL import Image, ImageOps

# Page config must be the first Streamlit command on the page.
st.set_page_config(page_title="TIAp", layout="centered")

# =========================================================
# APP CONFIG
# =========================================================

CONFIG = {
    "title": "TIAp",
    "short_name": "TIAp",
    "subtitle": "Thermal image analysis point-and-go by We are dougalien",
    "website": "www.dougalien.com",
    "image_label": "thermal or infrared image",
    "analyst_role": "careful thermal image analyst",
    "model": "gpt-4o-mini",
    "timeout": 60,
}

SYSTEM_PROMPT = f"""
You are a {CONFIG['analyst_role']} for a quick mobile workflow.

Rules:
- Use only visible image evidence.
- Keep the answer short and specific.
- Do not ask follow-up questions.
- Do not invent temperatures, emissivity, materials, diagnoses, or causes unless directly supported by visible evidence.
- If no temperature scale is visible, do not claim exact temperatures.
- If the image is not enough for a precise diagnosis, describe the thermal pattern or anomaly instead.
- Distinguish visible observations from interpretation.
- Prefer accurate, cautious wording over confident overreach.

Return valid JSON only:
{{
  "candidate": "best interpretation or most likely thermal pattern, anomaly, or issue visible in the image",
  "alternate": "brief alternate possibility or 'none'",
  "confidence": 1,
  "observations": ["visible feature 1", "visible feature 2", "visible feature 3"],
  "why": "one short explanation grounded in visible evidence",
  "limits": "what cannot be determined from image alone",
  "next_step": "single best next observation or simple real-world check"
}}

Confidence scale:
1 = weak guess
2 = plausible
3 = moderate
4 = strong
5 = very strong
""".strip()

# =========================================================
# STATE
# =========================================================


def init_state() -> None:
    defaults = {
        "authenticated": False,
        "login_error": "",
        "analysis": None,
        "analysis_error": "",
        "source_name": "",
        "focus_zone": "Full image",
        "last_image_b64": None,
        "input_mode": "Upload",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# =========================================================
# HELPERS
# =========================================================


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()



def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default



def get_app_password() -> str:
    return get_secret("APP_PASSWORD", "")



def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        return None
    return None



def normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        confidence = int(data.get("confidence", 1))
    except Exception:
        confidence = 1
    confidence = max(1, min(5, confidence))

    observations = data.get("observations", [])
    if not isinstance(observations, list):
        observations = []
    observations = [str(item).strip() for item in observations if str(item).strip()][:4]

    return {
        "candidate": str(data.get("candidate", "")).strip(),
        "alternate": str(data.get("alternate", "none")).strip(),
        "confidence": confidence,
        "observations": observations,
        "why": str(data.get("why", "")).strip(),
        "limits": str(data.get("limits", "")).strip(),
        "next_step": str(data.get("next_step", "")).strip(),
    }



def prepare_image(file_bytes: bytes, max_size: int = 1400) -> Image.Image:
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    return image



def crop_by_zone(image: Image.Image, zone: str) -> Image.Image:
    if zone == "Full image":
        return image

    width, height = image.size
    third_w = max(1, width // 3)
    third_h = max(1, height // 3)

    boxes = {
        "Top left": (0, 0, third_w, third_h),
        "Top center": (third_w, 0, third_w * 2, third_h),
        "Top right": (third_w * 2, 0, width, third_h),
        "Middle left": (0, third_h, third_w, third_h * 2),
        "Center": (third_w, third_h, third_w * 2, third_h * 2),
        "Middle right": (third_w * 2, third_h, width, third_h * 2),
        "Bottom left": (0, third_h * 2, third_w, height),
        "Bottom center": (third_w, third_h * 2, third_w * 2, height),
        "Bottom right": (third_w * 2, third_h * 2, width, height),
    }
    return image.crop(boxes.get(zone, (0, 0, width, height)))



def image_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")



def call_openai_json(image_b64: str) -> Dict[str, Any]:
    api_key = get_secret("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    payload = {
        "model": CONFIG["model"],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this thermal or infrared image carefully and identify the most likely "
                            "thermal pattern, anomaly, or issue supported by the visible evidence."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=CONFIG["timeout"],
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = safe_json_loads(content)
    if not parsed:
        raise RuntimeError("Model returned unreadable JSON.")
    return normalize_result(parsed)



def export_result_text(result: Dict[str, Any], source_name: str, focus_zone: str) -> str:
    obs_text = "\n".join(f"- {item}" for item in result.get("observations", []))
    return (
        f"{CONFIG['title']}\n"
        f"Source: {source_name or 'camera/upload'}\n"
        f"Focus area: {focus_zone}\n\n"
        f"Likely interpretation: {result.get('candidate', '')}\n"
        f"Alternate: {result.get('alternate', '')}\n"
        f"Confidence: {result.get('confidence', '')}/5\n\n"
        f"Visible observations:\n{obs_text}\n\n"
        f"Why this fit: {result.get('why', '')}\n"
        f"Limits: {result.get('limits', '')}\n"
        f"Next step: {result.get('next_step', '')}\n"
    )



def show_image_compat(image: Image.Image, caption: str) -> None:
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image, caption=caption, use_column_width=True)
        except TypeError:
            st.image(image, caption=caption)


# =========================================================
# STYLES
# =========================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px;
    }
    .stApp {
        background: #F7F8FA;
        color: #111111;
    }
    .card {
        background: white;
        border: 1px solid #D7DCE2;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.02rem;
        color: #2F3B48;
        margin-bottom: 0.25rem;
    }
    .site {
        color: #2B5C88;
        font-size: 0.96rem;
    }
    .small {
        color: #47515D;
        font-size: 0.96rem;
    }
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        min-height: 48px;
        border-radius: 10px;
        font-weight: 700;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# UI
# =========================================================


def render_login() -> None:
    st.markdown(
        f"""
        <div class="card" style="margin-top:2rem;">
            <div class="title">{CONFIG['title']}</div>
            <div class="subtitle">{CONFIG['subtitle']}</div>
            <div class="site">{CONFIG['website']}</div>
            <p class="small" style="margin-top:0.8rem;">Enter the app password to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        actual = get_app_password()
        if not actual:
            st.session_state.login_error = "APP_PASSWORD is missing from Streamlit secrets."
        elif entered == actual:
            st.session_state.authenticated = True
            st.session_state.login_error = ""
            rerun_app()
        else:
            st.session_state.login_error = "Incorrect password."

    if st.session_state.login_error:
        st.error(st.session_state.login_error)



def render_header() -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="title">{CONFIG['title']}</div>
            <div class="subtitle">{CONFIG['subtitle']}</div>
            <div class="site">{CONFIG['website']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_accessibility_note() -> None:
    st.info(
        "Phone-first layout, large controls, clear labels, short results, and no required follow-up chat. "
        "This version is designed for quick image capture and one-pass output."
    )



def get_image_input() -> Tuple[Optional[bytes], str]:
    image_bytes: Optional[bytes] = None
    source_name = ""

    input_mode = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    st.session_state.input_mode = input_mode

    if input_mode == "Camera":
        if hasattr(st, "camera_input"):
            camera_file = st.camera_input("Take a photo")
            if camera_file is not None:
                image_bytes = camera_file.getvalue()
                source_name = "camera_capture.png"
        else:
            st.warning("This Streamlit version does not support camera input. Use upload instead.")
    else:
        uploaded_file = st.file_uploader(
            f"Upload a {CONFIG['image_label']}",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            source_name = uploaded_file.name

    return image_bytes, source_name



def render_focus_controls() -> str:
    st.markdown("### Focus area")
    st.caption("Whole image is fastest. Zone focus is a simple substitute until exact tap-to-target is added.")
    zone = st.selectbox(
        "Analyze which part of the image?",
        [
            "Full image",
            "Top left", "Top center", "Top right",
            "Middle left", "Center", "Middle right",
            "Bottom left", "Bottom center", "Bottom right",
        ],
        index=0,
    )
    st.session_state.focus_zone = zone
    return zone



def render_result(result: Dict[str, Any]) -> None:
    st.markdown("### Result")
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{result.get('confidence', '')}/5")
    c2.metric("Mode", "One pass")

    st.write(f"**Likely interpretation:** {result.get('candidate', '')}")
    st.write(f"**Alternate:** {result.get('alternate', '')}")

    st.write("**Visible observations**")
    for item in result.get("observations", []):
        st.write(f"- {item}")

    st.write(f"**Why this fit:** {result.get('why', '')}")
    st.write(f"**Limits:** {result.get('limits', '')}")
    st.write(f"**Best next step:** {result.get('next_step', '')}")


# =========================================================
# MAIN
# =========================================================

if not st.session_state.authenticated:
    render_login()
    st.stop()

render_header()
render_accessibility_note()

image_bytes, source_name = get_image_input()

if image_bytes:
    try:
        base_image = prepare_image(image_bytes)
        show_image_compat(base_image, "Source image")
        focus_zone = render_focus_controls()
        focused_image = crop_by_zone(base_image, focus_zone)

        if focus_zone != "Full image":
            show_image_compat(focused_image, f"Focused view: {focus_zone}")

        if st.button("Run quick analysis"):
            st.session_state.analysis = None
            st.session_state.analysis_error = ""
            st.session_state.source_name = source_name
            with st.spinner("Analyzing image..."):
                image_b64 = image_to_b64(focused_image)
                st.session_state.last_image_b64 = image_b64
                st.session_state.analysis = call_openai_json(image_b64)
            st.success("Analysis complete.")
    except requests.RequestException as exc:
        st.session_state.analysis_error = f"Request error: {exc}"
    except Exception as exc:
        st.session_state.analysis_error = f"Unexpected error: {exc}"

if st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)

if st.session_state.analysis:
    render_result(st.session_state.analysis)
    export_text = export_result_text(
        st.session_state.analysis,
        st.session_state.source_name,
        st.session_state.focus_zone,
    )
    st.download_button(
        "Download result as text",
        data=export_text,
        file_name="tiap_result.txt",
        mime="text/plain",
    )
