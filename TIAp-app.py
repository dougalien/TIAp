import os
import json
import base64
import mimetypes
from io import BytesIO
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

# =========================================================
# TIAp — Thermal Image Analyzer
# Refactored with login, Free/Pro tiers, IR theme, zoom,
# cleaner answer UI, follow-up chat, and session download.
# =========================================================

st.set_page_config(
    page_title="TIAp – Thermal Image Analyzer",
    page_icon="🌡️",
    layout="wide",
)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"

# ---------------------------------------------------------
# State
# ---------------------------------------------------------

def init_state():
    defaults = {
        "authenticated": False,
        "login_error": "",
        "image_name": None,
        "image_bytes": None,
        "image_mime": None,
        "image_data_uri": None,
        "display_messages": [],
        "analysis_meta": {},
        "final_reply": "",
        "observer_json": None,
        "validator_json": None,
        "judge_json": None,
        "tier": "Free",
        "context_notes": "",
        "specimen_label": "",
        "student_observations": "",
        "student_best_answer": "",
        "known_name": "",
        "student_name": "",
        "include_auto_zoom": True,
        "zoom_fraction": 0.5,
        "show_zoom_preview": True,
        "openai_model": DEFAULT_OPENAI_MODEL,
        "claude_model": DEFAULT_CLAUDE_MODEL,
        "perplexity_model": DEFAULT_PERPLEXITY_MODEL,
        "session_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_analysis_state():
    st.session_state.final_reply = ""
    st.session_state.observer_json = None
    st.session_state.validator_json = None
    st.session_state.judge_json = None
    st.session_state.display_messages = []
    st.session_state.analysis_meta = {}


def get_secret_or_env(name: str, default: str = ""):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def app_password():
    return get_secret_or_env("APP_PASSWORD", "")


def get_openai_key():
    return get_secret_or_env("OPENAI_API_KEY", "")


def get_claude_key():
    return get_secret_or_env("CLAUDE_API_KEY", get_secret_or_env("ANTHROPIC_API_KEY", ""))


def get_perplexity_key():
    return get_secret_or_env("PERPLEXITY_API_KEY", "")


def sign_out():
    keep = {
        "openai_model": st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL),
        "claude_model": st.session_state.get("claude_model", DEFAULT_CLAUDE_MODEL),
        "perplexity_model": st.session_state.get("perplexity_model", DEFAULT_PERPLEXITY_MODEL),
    }
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()
    for k, v in keep.items():
        st.session_state[k] = v
    st.session_state.authenticated = False

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #140914 0%, #230D2B 32%, #41123A 55%, #6E1E31 76%, #A33B1F 90%, #D08D26 100%);
        color: #F6F4F7;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #120812 0%, #220C24 100%);
    }
    .main-card {
        background: rgba(18, 12, 24, 0.78);
        border: 1px solid rgba(255, 191, 73, 0.26);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .answer-box {
        background: rgba(255, 245, 214, 0.96);
        color: #241616;
        border-left: 6px solid #FFC24B;
        border-radius: 12px;
        padding: 1rem;
    }
    .next-box {
        background: rgba(255, 224, 210, 0.95);
        color: #241616;
        border-left: 6px solid #FF6B4A;
        border-radius: 12px;
        padding: 0.9rem;
    }
    .soft-label {
        font-size: 0.84rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #F1D58A;
        margin-bottom: 0.4rem;
    }
    .chat-user {
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.75rem;
        margin: 0.45rem 0;
    }
    .chat-ai {
        background: rgba(255, 194, 75, 0.13);
        border: 1px solid rgba(255, 194, 75, 0.18);
        border-radius: 12px;
        padding: 0.75rem;
        margin: 0.45rem 0;
    }
    .small-note {
        font-size: 0.92rem;
        color: #F9E6B7;
    }
    div[data-testid="stMetricValue"] {
        color: #FFF3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Image helpers
# ---------------------------------------------------------

def file_to_data_uri(uploaded_file):
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return raw, mime, data_uri


def update_uploaded_image(uploaded_file):
    raw, mime, data_uri = file_to_data_uri(uploaded_file)
    st.session_state.image_name = uploaded_file.name
    st.session_state.image_bytes = raw
    st.session_state.image_mime = mime
    st.session_state.image_data_uri = data_uri
    reset_analysis_state()


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


def get_image_contents_for_openai():
    if not st.session_state.image_data_uri:
        return []

    contents = [{"type": "image_url", "image_url": {"url": st.session_state.image_data_uri}}]

    if st.session_state.tier == "Pro" and st.session_state.include_auto_zoom and st.session_state.image_bytes:
        try:
            img = pil_from_session_image()
            crop = center_crop(img, st.session_state.zoom_fraction)
            contents.append({"type": "image_url", "image_url": {"url": build_data_uri_from_pil(crop)}})
        except Exception:
            pass
    return contents


def get_image_blocks_for_anthropic():
    if not st.session_state.image_bytes:
        return []

    blocks = []
    mime = st.session_state.get("image_mime") or "image/png"
    media_type = mime if mime in ["image/jpeg", "image/png", "image/webp", "image/gif"] else "image/png"
    source_data = base64.b64encode(st.session_state.image_bytes).decode("utf-8")
    blocks.append({
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": source_data},
    })

    if st.session_state.tier == "Pro" and st.session_state.include_auto_zoom:
        try:
            img = pil_from_session_image()
            crop = center_crop(img, st.session_state.zoom_fraction)
            buf = BytesIO()
            crop.save(buf, format="PNG")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                },
            })
        except Exception:
            pass
    return blocks

# ---------------------------------------------------------
# Prompting
# ---------------------------------------------------------

def specimen_context_block():
    label = st.session_state.specimen_label.strip() or "[no label]"
    notes = st.session_state.context_notes.strip() or "[no extra notes]"
    target = st.session_state.student_name.strip() or "[no main subject provided]"
    observations = st.session_state.student_observations.strip() or "[none entered yet]"
    best_answer = st.session_state.student_best_answer.strip() or "[none entered yet]"
    known_name = st.session_state.known_name.strip() or "[none provided]"

    return f"""
    Photo label / ID: {label}
    Main subject (user input): {target}
    User observations: {observations}
    User current guess: {best_answer}
    Known diagnosis / ground truth (optional): {known_name}
    Extra notes: {notes}
    """.strip()


def role_system_prompt(role: str):
    base = """
    You are helping analyze a single thermal / infrared image.
    Stay grounded in what the image can actually support: visible patterns, relative hot/cold areas,
    shapes, gradients, and plain context clues.
    Do not invent precise temperatures, hidden causes, diagnoses, or certainty that the image does not support.
    Be concise, careful, and honest.
    """.strip()

    if role == "observer":
        return f"""
        {base}
        You are the OBSERVER.

        Return valid JSON only with this schema:
        {{
          "visible_evidence": ["short item", "short item", "short item"],
          "likely_identification": "short phrase",
          "alternative_identification": "short phrase",
          "confidence": 1,
          "image_clarity": "clear / somewhat unclear / poor",
          "reasoning": "1-2 short sentences",
          "next_check": "short practical next check"
        }}
        """.strip()

    if role == "validator":
        return f"""
        {base}
        You are the VALIDATOR.

        Return valid JSON only with this schema:
        {{
          "visible_evidence": ["short item", "short item", "short item"],
          "likely_identification": "short phrase",
          "alternative_identification": "short phrase",
          "confidence": 1,
          "agreement_with_student_name": "probably right / close / not a good match / uncertain",
          "reasoning": "1-2 short sentences",
          "next_check": "short practical next check"
        }}
        """.strip()

    return f"""
    {base}
    You are the JUDGE.

    Return valid JSON only with this schema:
    {{
      "agreement_level": "high / medium / low",
      "winner": "observer / validator / blended",
      "final_identification": "short phrase",
      "confidence": 1,
      "why": "1-2 short sentences",
      "next_check": "short practical next check",
      "student_reply": "2-4 short sentences, plain language, concise, user-facing"
    }}
    """.strip()

# ---------------------------------------------------------
# API helpers
# ---------------------------------------------------------

def safe_json_loads(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def call_openai_json(system_prompt: str, user_text: str):
    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    payload = {
        "model": st.session_state.openai_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}, *get_image_contents_for_openai()],
            },
        ],
    }

    resp = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:1000]}")

    text = resp.json()["choices"][0]["message"]["content"]
    parsed = safe_json_loads(text)
    if parsed is None:
        raise RuntimeError("OpenAI returned non-JSON output.")
    return parsed


def call_claude_json(system_prompt: str, user_text: str):
    api_key = get_claude_key()
    if not api_key:
        raise RuntimeError("Missing CLAUDE_API_KEY or ANTHROPIC_API_KEY.")

    payload = {
        "model": st.session_state.claude_model,
        "max_tokens": 900,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}, *get_image_blocks_for_anthropic()]}],
    }

    resp = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Claude error {resp.status_code}: {resp.text[:1000]}")

    data = resp.json()
    text_parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    parsed = safe_json_loads("\n".join(text_parts).strip())
    if parsed is None:
        raise RuntimeError("Claude returned non-JSON output.")
    return parsed


def call_perplexity_judge(system_prompt: str, user_text: str):
    api_key = get_perplexity_key()
    if not api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY.")

    payload = {
        "model": st.session_state.perplexity_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }

    resp = requests.post(
        PERPLEXITY_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Perplexity error {resp.status_code}: {resp.text[:1000]}")

    parsed = safe_json_loads(resp.json()["choices"][0]["message"]["content"].strip())
    if parsed is None:
        raise RuntimeError("Perplexity returned non-JSON output.")
    return parsed

# ---------------------------------------------------------
# Analysis logic
# ---------------------------------------------------------

def disagreement_score(observer, validator):
    if not observer or not validator:
        return 0
    obs_id = (observer.get("likely_identification") or "").strip().lower()
    val_id = (validator.get("likely_identification") or "").strip().lower()
    obs_alt = (observer.get("alternative_identification") or "").strip().lower()
    val_alt = (validator.get("alternative_identification") or "").strip().lower()

    score = 0
    if obs_id and val_id and obs_id != val_id:
        score += 2
    if obs_id and val_alt and obs_id == val_alt:
        score -= 1
    if val_id and obs_alt and val_id == obs_alt:
        score -= 1
    try:
        if int(observer.get("confidence", 0)) <= 2 or int(validator.get("confidence", 0)) <= 2:
            score += 1
    except Exception:
        pass
    return max(score, 0)


def build_free_reply(observer):
    evidence = observer.get("visible_evidence", [])
    likely = observer.get("likely_identification", "uncertain")
    alt = observer.get("alternative_identification", "another possibility")
    conf = int(observer.get("confidence", 2))
    next_check = observer.get("next_check", "check another visible feature or retake the image")

    lead = f"I can see {', '.join(evidence[:3])}." if evidence else "I can see a few visible thermal patterns."
    if conf >= 4:
        body = f"The best fit is {likely}."
    elif conf == 3:
        body = f"The best fit may be {likely}, though {alt} is still plausible."
    else:
        body = f"This is still uncertain; {likely} is only a tentative fit."
    return f"{lead} {body} Next, {next_check}."


def analyze_image():
    if not st.session_state.image_bytes:
        st.warning("Please upload an image first.")
        return

    context = specimen_context_block()

    observer = call_openai_json(
        role_system_prompt("observer"),
        f"Analyze this thermal image with attention to the user's stated main subject. {context}",
    )
    st.session_state.observer_json = observer

    final_reply = ""
    validator = None
    judge = None

    if st.session_state.tier == "Free":
        final_reply = build_free_reply(observer)
        confidence = observer.get("confidence", 2)
        next_check = observer.get("next_check", "Retake the image or inspect one more visible feature.")
    else:
        validator = call_claude_json(
            role_system_prompt("validator"),
            f"Independently validate this thermal image and the main subject interpretation. {context}",
        )
        st.session_state.validator_json = validator

        use_judge = disagreement_score(observer, validator) >= 2
        try:
            if int(observer.get("confidence", 0)) <= 2 or int(validator.get("confidence", 0)) <= 2:
                use_judge = True
        except Exception:
            pass

        if use_judge:
            judge = call_perplexity_judge(
                role_system_prompt("judge"),
                (
                    "Compare these two structured analyses of the same image and main subject and produce a cautious user-facing result.\n\n"
                    f"Image context:\n{context}\n\n"
                    f"Observer JSON:\n{json.dumps(observer, ensure_ascii=False)}\n\n"
                    f"Validator JSON:\n{json.dumps(validator, ensure_ascii=False)}"
                ),
            )
            st.session_state.judge_json = judge

        if judge and judge.get("student_reply"):
            final_reply = judge["student_reply"].strip()
            confidence = judge.get("confidence", 2)
            next_check = judge.get("next_check", "Check another visible feature or confirm with direct inspection.")
        else:
            confidence = min(int(observer.get("confidence", 2)), int(validator.get("confidence", 2)))
            next_check = validator.get("next_check") or observer.get("next_check") or "Check another visible feature or retake the image."
            if disagreement_score(observer, validator) >= 2:
                final_reply = (
                    f"I can see useful thermal patterns, but there is still real uncertainty. "
                    f"One reading leans toward {observer.get('likely_identification', 'one interpretation')}, "
                    f"while another leans toward {validator.get('likely_identification', 'another interpretation')}. "
                    f"The safest next step is to {next_check}."
                )
            else:
                final_reply = (
                    f"I can see {', '.join((validator.get('visible_evidence') or [])[:3]) or 'several visible thermal features'}. "
                    f"The strongest fit is {validator.get('likely_identification', 'uncertain')}. "
                    f"Next, {next_check}."
                )

    st.session_state.final_reply = final_reply
    st.session_state.display_messages = [{"role": "assistant", "content": final_reply}]
    st.session_state.analysis_meta = {
        "tier": st.session_state.tier,
        "observer_model": st.session_state.openai_model,
        "validator_model": st.session_state.claude_model if validator else None,
        "judge_model": st.session_state.perplexity_model if judge else None,
        "confidence": confidence,
        "next_check": next_check,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.session_log.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tier": st.session_state.tier,
            "image_name": st.session_state.image_name,
            "main_subject": st.session_state.student_name,
            "result": final_reply,
            "confidence": confidence,
            "next_check": next_check,
        }
    )

# ---------------------------------------------------------
# Follow-up chat
# ---------------------------------------------------------

def followup_system_prompt():
    return """
    You are a concise tutor continuing the same thermal-image discussion.
    Stay consistent with the earlier analysis unless the new message gives a reason to revise it.
    Be honest about uncertainty.
    Use 2-4 short sentences.
    """.strip()


def send_followup(user_text: str):
    user_text = user_text.strip()
    if not user_text:
        return

    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    prior = st.session_state.get("final_reply", "[no earlier reply stored]")
    context = specimen_context_block()

    payload = {
        "model": st.session_state.openai_model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": followup_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Earlier reply:\n{prior}\n\n"
                    f"Image context:\n{context}\n\n"
                    f"User follow-up:\n{user_text}\n\n"
                    "Keep the answer short, grounded, and practical."
                ),
            },
        ],
    }

    resp = requests.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI follow-up error {resp.status_code}: {resp.text[:1000]}")

    reply = resp.json()["choices"][0]["message"]["content"].strip()
    st.session_state.display_messages.append({"role": "user", "content": user_text})
    st.session_state.display_messages.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------
# Auth
# ---------------------------------------------------------

init_state()

if not st.session_state.authenticated:
    st.markdown(
        """
        <div class="main-card" style="max-width:560px; margin:3rem auto; text-align:center;">
            <div class="soft-label">Login</div>
            <h2 style="margin-top:0.2rem; margin-bottom:0.2rem;">Welcome to TIAp</h2>
            <div class="small-note">Thermal Image Analyzer</div>
            <div class="small-note" style="margin-top:0.7rem;">Enter the app password to continue.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        entered = st.text_input("Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Enter", use_container_width=True)

    actual = app_password()
    if submitted:
        if not actual:
            st.session_state.login_error = "APP_PASSWORD is missing from secrets."
        elif entered == actual:
            st.session_state.authenticated = True
            st.session_state.login_error = ""
            st.rerun()
        else:
            st.session_state.login_error = "Incorrect password."

    if st.session_state.login_error:
        st.error(st.session_state.login_error)
    st.stop()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.markdown(
    """
    <div class="main-card">
        <h1 style="margin-bottom:0.2rem;">🌡️ TIAp</h1>
        <div class="small-note">Thermal Image Analyzer</div>
        <div class="small-note" style="margin-top:0.55rem;">IR colors are relative and may shift with device settings, palette choice, and scene conditions.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## TIAp")
    st.session_state.tier = st.radio("Tier", ["Free", "Pro"], index=0 if st.session_state.tier == "Free" else 1)

    if st.button("Sign out", use_container_width=True):
        sign_out()
        st.rerun()

    st.markdown("---")
    st.caption("Free: OpenAI only")
    st.caption("Pro: OpenAI + Claude + Perplexity, zoom, follow-up chat, details, session download")

    if st.session_state.tier == "Pro":
        st.markdown("### Pro Controls")
        st.session_state.include_auto_zoom = st.checkbox("Auto zoom center", value=st.session_state.include_auto_zoom)
        st.session_state.show_zoom_preview = st.checkbox("Show zoom preview", value=st.session_state.show_zoom_preview)
        st.session_state.zoom_fraction = st.slider("Zoom fraction", 0.2, 1.0, st.session_state.zoom_fraction, 0.1)

missing = []
if not get_openai_key():
    missing.append("OpenAI")
if st.session_state.tier == "Pro" and not get_claude_key():
    missing.append("Claude")
if st.session_state.tier == "Pro" and not get_perplexity_key():
    missing.append("Perplexity")
if missing:
    st.warning("Missing API keys: " + ", ".join(missing))

left, right = st.columns([1.05, 1])

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a thermal image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        update_uploaded_image(uploaded)

    if st.session_state.image_bytes:
        base_img = pil_from_session_image()
        st.image(base_img, caption=st.session_state.image_name or "Uploaded thermal image", use_container_width=True)

        if st.session_state.tier == "Pro" and st.session_state.show_zoom_preview:
            try:
                zoom_img = center_crop(base_img, st.session_state.zoom_fraction)
                st.image(zoom_img, caption="Zoomed center preview", use_container_width=True)
            except Exception:
                pass
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.text_input(
        "Main subject (object or scene)",
        key="student_name",
        placeholder="e.g. stream reach, breaker panel, north wall, person, device",
    )
    st.text_input("Short label / ID", key="specimen_label", placeholder="e.g. Office north wall – Mar 2026")
    st.text_area("Optional notes", key="context_notes", height=80)
    st.text_area("What do you notice?", key="student_observations", height=70)
    st.text_input("Your current guess", key="student_best_answer", placeholder="e.g. cold air leak, moisture, overheating")
    st.text_input("Known diagnosis / ground truth (optional)", key="known_name")
    st.markdown('</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("📸 Analyze image", use_container_width=True):
        try:
            analyze_image()
        except Exception as e:
            st.error(str(e))
with col_b:
    if st.button("Clear current result", use_container_width=True):
        reset_analysis_state()
        st.rerun()

if st.session_state.final_reply:
    meta = st.session_state.analysis_meta or {}
    st.markdown(
        f"""
        <div class="main-card">
            <div class="soft-label">Answer</div>
            <div class="answer-box">{st.session_state.final_reply}</div>
            <div class="next-box" style="margin-top:0.8rem;"><strong>Next check:</strong> {meta.get('next_check', 'Check another visible feature or confirm with direct inspection.')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Confidence", f"{meta.get('confidence', 2)}/5")
    st.caption("Use thermal images as screening tools. Confirm important building, electrical, mechanical, or medical concerns with direct inspection or qualified help.")

if st.session_state.tier == "Pro" and st.session_state.final_reply:
    st.markdown("---")
    st.subheader("Follow-up chat")
    followup = st.text_input("Ask about this same image", placeholder="e.g. Is this more like moisture or an air leak?")
    if st.button("Send follow-up") and followup.strip():
        try:
            send_followup(followup)
        except Exception as e:
            st.error(str(e))

    for msg in st.session_state.display_messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'><strong>TIAp:</strong> {msg['content']}</div>", unsafe_allow_html=True)

    with st.expander("Model details"):
        st.json(st.session_state.observer_json or {})
        if st.session_state.validator_json:
            st.json(st.session_state.validator_json)
        if st.session_state.judge_json:
            st.json(st.session_state.judge_json)

    if st.session_state.session_log:
        st.download_button(
            "Download session",
            data=json.dumps(st.session_state.session_log, indent=2),
            file_name="tiap_session.json",
            mime="application/json",
        )
