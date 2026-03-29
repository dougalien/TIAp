import os
import json
import base64
import mimetypes
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# =========================================================
# TIAP — Thermal Image Analyzer
# (Revashi iPhone IR adapter friendly)
# =========================================================

st.set_page_config(
    page_title="TIAP – Thermal Image Analyzer",
    page_icon="🌡️",
    layout="centered",
)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"

COUNSEL_MODES = {
    "Cheap": {
        "observer": True,
        "validator": False,
        "judge": False,
        "judge_on_disagreement_only": False,
        "label": "Low cost",
    },
    "Balanced": {
        "observer": True,
        "validator": True,
        "judge": True,
        "judge_on_disagreement_only": True,
        "label": "Best value",
    },
    "Max caution": {
        "observer": True,
        "validator": True,
        "judge": True,
        "judge_on_disagreement_only": False,
        "label": "Highest caution",
    },
}

# =========================================================
# Helpers
# =========================================================


def init_state():
    defaults = {
        "started": False,
        "image_name": None,
        "image_bytes": None,
        "image_mime": None,
        "image_data_uri": None,
        "last_uploaded_signature": None,
        "display_messages": [],
        "analysis_meta": {},
        "final_reply": "",
        "first_reply": "",
        "observer_json": None,
        "validator_json": None,
        "judge_json": None,
        "counsel_mode": "Balanced",
        "mode": "Auto",
        "context_notes": "",
        "specimen_label": "",
        "student_observations": "",
        "student_best_answer": "",
        "known_name": "",
        "student_name": "",  # now: main subject (stream, person, device, etc.)
        "include_auto_zoom": True,
        "zoom_fraction": 0.5,
        "clear_followup_next": False,
        "openai_model": DEFAULT_OPENAI_MODEL,
        "claude_model": DEFAULT_CLAUDE_MODEL,
        "perplexity_model": DEFAULT_PERPLEXITY_MODEL,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_secret_or_env(name: str, default: str = ""):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def get_openai_key():
    return get_secret_or_env("OPENAI_API_KEY", "")


def get_claude_key():
    return get_secret_or_env("CLAUDE_API_KEY", get_secret_or_env("ANTHROPIC_API_KEY", ""))


def get_perplexity_key():
    return get_secret_or_env("PERPLEXITY_API_KEY", "")


def file_to_data_uri(uploaded_file):
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return raw, mime, data_uri


def update_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return
    raw, mime, data_uri = file_to_data_uri(uploaded_file)
    st.session_state.image_name = uploaded_file.name
    st.session_state.image_bytes = raw
    st.session_state.image_mime = mime
    st.session_state.image_data_uri = data_uri


def ensure_image_data_uri():
    if st.session_state.get("image_bytes") and not st.session_state.get("image_data_uri"):
        try:
            mime = st.session_state.get("image_mime") or "image/png"
            b64 = base64.b64encode(st.session_state.image_bytes).decode("utf-8")
            st.session_state.image_data_uri = f"data:{mime};base64,{b64}"
        except Exception:
            st.session_state.image_bytes = None
            st.session_state.image_name = None
            st.session_state.image_mime = None
            st.session_state.image_data_uri = None


def get_image_contents_for_openai():
    if not st.session_state.image_bytes or not st.session_state.image_data_uri:
        return []

    contents = [
        {
            "type": "image_url",
            "image_url": {"url": st.session_state.image_data_uri},
        }
    ]

    if not st.session_state.include_auto_zoom:
        return contents

    try:
        img = Image.open(BytesIO(st.session_state.image_bytes))
        w, h = img.size
        frac = max(0.1, min(float(st.session_state.zoom_fraction), 1.0))
        cw, ch = int(w * frac), int(h * frac)
        left = (w - cw) // 2
        top = (h - ch) // 2
        right = left + cw
        bottom = top + ch

        crop_center = img.crop((left, top, right, bottom))
        buf = BytesIO()
        fmt = img.format if img.format in ["JPEG", "PNG", "WEBP"] else "PNG"
        crop_center.save(buf, format=fmt)
        crop_bytes = buf.getvalue()

        b64 = base64.b64encode(crop_bytes).decode("utf-8")
        mime = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(fmt, "image/png")
        crop_data_uri = f"data:{mime};base64,{b64}"

        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": crop_data_uri},
            }
        )
    except Exception:
        pass

    return contents


def get_image_blocks_for_anthropic():
    if not st.session_state.image_bytes:
        return []

    blocks = []
    try:
        mime = st.session_state.get("image_mime") or "image/png"
        media_type = mime if mime in ["image/jpeg", "image/png", "image/webp", "image/gif"] else "image/png"
        source_data = base64.b64encode(st.session_state.image_bytes).decode("utf-8")

        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": source_data,
                },
            }
        )

        if st.session_state.include_auto_zoom:
            img = Image.open(BytesIO(st.session_state.image_bytes))
            w, h = img.size
            frac = max(0.1, min(float(st.session_state.zoom_fraction), 1.0))
            cw, ch = int(w * frac), int(h * frac)
            left = (w - cw) // 2
            top = (h - ch) // 2
            right = left + cw
            bottom = top + ch

            crop_center = img.crop((left, top, right, bottom))
            buf = BytesIO()
            fmt = img.format if img.format in ["JPEG", "PNG", "WEBP"] else "PNG"
            crop_center.save(buf, format=fmt)
            crop_bytes = buf.getvalue()

            crop_media_type = {
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }.get(fmt, "image/png")

            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": crop_media_type,
                        "data": base64.b64encode(crop_bytes).decode("utf-8"),
                    },
                }
            )
    except Exception:
        pass

    return blocks


def specimen_context_block():
    label = st.session_state.specimen_label.strip() or "[no label]"
    notes = st.session_state.context_notes.strip() or "[no extra notes]"
    target = st.session_state.student_name.strip() or "[no main subject provided]"
    observations = st.session_state.student_observations.strip() or "[none entered yet]"
    best_answer = st.session_state.student_best_answer.strip() or "[none entered yet]"
    known_name = st.session_state.known_name.strip() or "[none provided]"
    mode = st.session_state.mode

    return f"""
    Photo label / ID: {label}
    Main subject (user input): {target}
    User observations: {observations}
    User current guess: {best_answer}
    Known diagnosis / ground truth (optional): {known_name}
    Extra notes: {notes}
    Mode: {mode}
    """.strip()


def role_system_prompt(role: str):
    base = """
    You are helping analyze a single image, often a thermal / infrared photo taken with a Revashi iPhone adapter.
    Stay grounded in what this image can actually show: patterns, shapes, relative hot/cold areas,
    obvious context clues, and safety or diagnostic concerns.
    The user has specified what the image is mainly about (for example: stream, rock outcrop, vegetation, person, device, room).
    Focus your analysis on that main subject, but be honest if the subject is not clearly visible.
    Do NOT invent precise numerical values, hidden causes, or properties that are not well supported by the image.
    Be concise, careful, and honest.
    """.strip()

    mode_hint = "Keep your comments centered on the user’s stated main subject in this image."

    if role == "observer":
        return f"""
        {base}
        You are the OBSERVER. Your job is to describe what is visibly present
        and propose the best tentative interpretation of the main subject.

        Return valid JSON only with this exact schema:
        {{
          "visible_evidence": ["short item", "short item", "short item"],
          "likely_identification": "short phrase",
          "alternative_identification": "short phrase",
          "confidence": 1,
          "image_clarity": "clear / somewhat unclear / poor",
          "reasoning": "1-2 short sentences",
          "next_check": "short practical next check"
        }}

        Rules:
        - confidence must be an integer from 1 to 5.
        - Keep items short and factual, tied to what you can see in the image.
        - If the known diagnosis is provided and the image is ambiguous, you may lean toward it but still note uncertainty.
        - {mode_hint}
        """.strip()

    if role == "validator":
        return f"""
        {base}
        You are the VALIDATOR. Act independently from the observer and check whether
        the likely interpretation of the main subject is well supported.

        Return valid JSON only with this exact schema:
        {{
          "visible_evidence": ["short item", "short item", "short item"],
          "likely_identification": "short phrase",
          "alternative_identification": "short phrase",
          "confidence": 1,
          "agreement_with_student_name": "probably right / close / not a good match / uncertain",
          "reasoning": "1-2 short sentences",
          "next_check": "short practical next check"
        }}

        Rules:
        - confidence must be an integer from 1 to 5.
        - Stay independent from the observer and focus on the main subject specified by the user.
        - {mode_hint}
        """.strip()

    return f"""
    {base}
    You are the JUDGE. You will compare the observer and validator outputs
    and produce a conservative, user-facing result about the main subject.

    Return valid JSON only with this exact schema:
    {{
      "agreement_level": "high / medium / low",
      "winner": "observer / validator / blended",
      "final_identification": "short phrase",
      "confidence": 1,
      "why": "1-2 short sentences",
      "next_check": "short practical next check",
      "student_reply": "2-4 short sentences, plain language, concise, user-facing"
    }}

    Rules:
    - confidence must be an integer from 1 to 5.
    - Prefer caution over confidence.
    - If disagreement is substantial, use blended and explain uncertainty.
    - If a known diagnosis is given and the image is ambiguous, weigh it appropriately.
    - {mode_hint}
    """.strip()


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
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def call_openai_json(system_prompt: str, user_text: str):
    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": st.session_state.openai_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    *get_image_contents_for_openai(),
                ],
            },
        ],
    }

    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:1000]}")

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    parsed = safe_json_loads(text)
    if parsed is None:
        raise RuntimeError("OpenAI returned non-JSON output.")
    return parsed


def call_claude_json(system_prompt: str, user_text: str):
    api_key = get_claude_key()
    if not api_key:
        raise RuntimeError("Missing CLAUDE_API_KEY or ANTHROPIC_API_KEY.")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    content_blocks = [{"type": "text", "text": user_text}, *get_image_blocks_for_anthropic()]

    payload = {
        "model": st.session_state.claude_model,
        "max_tokens": 900,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
    }

    resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Claude error {resp.status_code}: {resp.text[:1000]}")

    data = resp.json()
    text_parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    text = "\n".join(text_parts).strip()
    parsed = safe_json_loads(text)
    if parsed is None:
        raise RuntimeError("Claude returned non-JSON output.")
    return parsed


def call_perplexity_judge(system_prompt: str, user_text: str):
    api_key = get_perplexity_key()
    if not api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": st.session_state.perplexity_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_text,
            },
        ],
    }

    resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Perplexity error {resp.status_code}: {resp.text[:1000]}")

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    parsed = safe_json_loads(text)
    if parsed is None:
        raise RuntimeError("Perplexity returned non-JSON output.")
    return parsed


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
        oc = int(observer.get("confidence", 0))
        vc = int(validator.get("confidence", 0))
        if oc <= 2 or vc <= 2:
            score += 1
    except Exception:
        pass

    return max(score, 0)


def build_student_reply_from_observer(observer):
    evidence = observer.get("visible_evidence", [])
    likely = observer.get("likely_identification", "uncertain")
    alt = observer.get("alternative_identification", "another possibility")
    conf = int(observer.get("confidence", 2))
    clarity = observer.get("image_clarity", "clear")
    next_check = observer.get("next_check", "Check one more visible feature or repeat the image if needed.")

    sentence1 = f"I can see {', '.join(evidence[:3])}." if evidence else "I can see a few useful visible features."
    if conf >= 4:
        sentence2 = f"The best fit looks like {likely}."
    elif conf == 3:
        sentence2 = f"The best fit may be {likely}, but {alt} is still plausible."
    else:
        sentence2 = f"The image looks {clarity}, but the interpretation is still uncertain; {likely} is only a tentative fit."
    sentence3 = f"Next, {next_check}"
    return f"{sentence1} {sentence2} {sentence3}"


def build_student_reply_from_validator(validator):
    evidence = validator.get("visible_evidence", [])
    likely = validator.get("likely_identification", "uncertain")
    alt = validator.get("alternative_identification", "another possibility")
    conf = int(validator.get("confidence", 2))
    match = validator.get("agreement_with_student_name", "uncertain")
    next_check = validator.get("next_check", "check one more visible feature or repeat the image")

    sentence1 = (
        f"I notice {', '.join(evidence[:3])}."
        if evidence
        else "I notice a few visible features worth checking."
    )
    if conf >= 4:
        sentence2 = f"Your current idea looks {match}, and {likely} is the strongest fit."
    elif conf == 3:
        sentence2 = f"Your current idea looks {match}; {likely} may fit, but {alt} also deserves a look."
    else:
        sentence2 = f"Your current idea is {match}, but the image still leaves real uncertainty."
    sentence3 = f"Next, {next_check}."
    return f"{sentence1} {sentence2} {sentence3}"


def analyze_with_counsel():
    if st.session_state.image_bytes and not st.session_state.image_data_uri:
        st.warning("Image is still loading. Wait for the preview, then try again.")
        return

    if not st.session_state.image_bytes:
        st.warning("Please upload an image first.")
        return

    context = specimen_context_block()

    observer_prompt = role_system_prompt("observer")
    validator_prompt = role_system_prompt("validator")
    judge_prompt = role_system_prompt("judge")

    observer_user = f"""
    Analyze this image with special attention to the main subject specified by the user.
    Provide a cautious interpretation and practical next checks.
    {context}
    """.strip()

    observer = call_openai_json(observer_prompt, observer_user)
    st.session_state.observer_json = observer

    config = COUNSEL_MODES[st.session_state.counsel_mode]
    validator = None
    judge = None

    if config["validator"]:
        validator_user = f"""
        Independently validate this image and the main subject interpretation.
        Focus on whether the observer’s likely identification is well supported.
        {context}
        """.strip()
        validator = call_claude_json(validator_prompt, validator_user)
        st.session_state.validator_json = validator

    use_judge = False
    if config["judge"]:
        if not config["judge_on_disagreement_only"]:
            use_judge = True
        else:
            score = disagreement_score(observer, validator)
            if score >= 2:
                use_judge = True
            try:
                if int(observer.get("confidence", 0)) <= 2:
                    use_judge = True
            except Exception:
                pass
            try:
                if validator and int(validator.get("confidence", 0)) <= 2:
                    use_judge = True
            except Exception:
                pass

    if use_judge:
        judge_user = f"""
        Compare these two structured analyses of the SAME image and main subject
        and produce the most cautious user-facing reply.

        Image context:
        {context}

        Observer JSON:
        {json.dumps(observer, ensure_ascii=False)}

        Validator JSON:
        {json.dumps(validator, ensure_ascii=False) if validator else "{}"}
        """.strip()

        judge = call_perplexity_judge(judge_prompt, judge_user)
        st.session_state.judge_json = judge

    if judge and judge.get("student_reply"):
        final_reply = judge["student_reply"].strip()
    elif validator:
        if disagreement_score(observer, validator) >= 2:
            obs_id = observer.get("likely_identification", "an uncertain fit")
            val_id = validator.get("likely_identification", "another uncertain fit")
            next_check = (
                validator.get("next_check")
                or observer.get("next_check")
                or "check one more visible feature or repeat the image"
            )
            final_reply = (
                "I can see useful features in the image, but there is still some uncertainty. "
                f"One reading leans toward {obs_id}, while another leans toward {val_id}. "
                f"The safest next step is to {next_check}."
            )
        else:
            final_reply = build_student_reply_from_validator(validator)
    else:
        final_reply = build_student_reply_from_observer(observer)

    st.session_state.final_reply = final_reply
    st.session_state.first_reply = final_reply
    st.session_state.display_messages = [{"role": "assistant", "content": final_reply}]
    st.session_state.analysis_meta = {
        "observer_model": st.session_state.openai_model,
        "validator_model": st.session_state.claude_model if validator else None,
        "judge_model": st.session_state.perplexity_model if judge else None,
        "counsel_mode": st.session_state.counsel_mode,
    }
    st.session_state.started = True


def followup_system_prompt():
    return """
    You are a concise tutor continuing the SAME image discussion.
    Stay consistent with the prior analysis unless the new user message provides a reason to revise it.
    Be honest if the earlier answer was uncertain or likely wrong.
    Use 2-4 short sentences.
    Keep the answer grounded in visible evidence and simple next checks.
    """.strip()


def send_followup(user_text: str):
    user_text = user_text.strip()
    if not user_text:
        return

    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    prior = st.session_state.get("first_reply", "[no earlier reply stored]")
    context = specimen_context_block()

    followup_text = f"""
    You are continuing a tutoring session about one image and main subject.
    Earlier anchored reply:
    \"\"\"{prior}\"\"\"

    Image context:
    {context}

    User follow-up:
    {user_text}

    Instructions:
    - Stay short.
    - If the earlier answer was uncertain, say so plainly.
    - If new information changes the interpretation, admit the earlier guess may have been wrong.
    - Focus on what can be seen and what to check next.
    """.strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": st.session_state.openai_model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": followup_system_prompt()},
            {"role": "user", "content": followup_text},
        ],
    }

    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI follow-up error {resp.status_code}: {resp.text[:1000]}")

    data = resp.json()
    reply = data["choices"][0]["message"]["content"].strip()

    st.session_state.display_messages.append({"role": "user", "content": user_text})
    st.session_state.display_messages.append({"role": "assistant", "content": reply})


# =========================================================
# TIAP – Thermal Image Analyzer main UI
# =========================================================

init_state()
ensure_image_data_uri()

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 0.5rem;
    }
    .tiap-header {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .tiap-header h1 {
        font-size: 1.6rem;
        margin-bottom: 0.2rem;
    }
    .tiap-subtitle {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 0.6rem;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.0rem;
        padding: 0.6rem 0.2rem;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tiap-header">
      <h1>TIAP – Thermal Image Analyzer</h1>
      <div class="tiap-subtitle">
        Revashi iPhone IR photos → quick triage with three AI models.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = []
if not get_openai_key():
    missing.append("OpenAI (observer)")
if not get_claude_key():
    missing.append("Claude (validator)")
if not get_perplexity_key():
    missing.append("Perplexity (judge)")

if missing:
    st.info(
        "Add the missing API keys in `.streamlit/secrets.toml` to use all models: "
        + ", ".join(missing)
    )

st.write("### 1. Thermal image")

uploaded = st.file_uploader(
    "Upload a thermal or regular image (Revashi iPhone adapter compatible)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
)

if uploaded is not None:
    update_uploaded_image(uploaded)
    st.image(uploaded, caption=uploaded.name, use_column_width=True)

st.write("### 2. What is this image mainly about?")

st.session_state.student_name = st.text_input(
    "Main subject (object or scene)",
    value=st.session_state.student_name,
    placeholder="e.g. 'stream reach', 'rock outcrop', 'person', 'tree canopy', 'breaker panel', 'north wall outlet'",
)

st.session_state.specimen_label = st.text_input(
    "Short label / ID for this photo",
    value=st.session_state.specimen_label,
    placeholder="e.g. 'Bedroom north wall – Jan 2026'",
)

st.session_state.context_notes = st.text_area(
    "Optional notes (what you were checking, ambient conditions, any important context)",
    value=st.session_state.context_notes,
    height=80,
)

st.session_state.student_observations = st.text_area(
    "What do YOU notice in the image?",
    value=st.session_state.student_observations,
    height=60,
)

st.session_state.student_best_answer = st.text_input(
    "Your current guess (e.g. 'possible moisture', 'overheating breaker', 'cold air leak', 'healthy canopy')",
    value=st.session_state.student_best_answer,
)

st.session_state.known_name = st.text_input(
    "Known diagnosis / ground truth (optional)",
    value=st.session_state.known_name,
    placeholder="Leave blank if you are exploring.",
)

st.write("### 3. Analyze")

col_zoom, col_mode = st.columns(2)

with col_zoom:
    st.session_state.include_auto_zoom = st.checkbox(
        "Auto zoom center",
        value=st.session_state.include_auto_zoom,
        help="Include a zoomed crop of the central area for finer detail.",
    )
    st.session_state.zoom_fraction = st.slider(
        "Zoom fraction", 0.2, 1.0, st.session_state.zoom_fraction, 0.1
    )

with col_mode:
    st.session_state.counsel_mode = st.radio(
        "Model cost / caution",
        options=list(COUNSEL_MODES.keys()),
        index=list(COUNSEL_MODES.keys()).index(st.session_state.counsel_mode),
        help="Cheap = observer only, Balanced = all three with judge only on disagreement, Max caution = all three every time.",
    )

analyze_clicked = st.button("📸 Analyze image")

if analyze_clicked:
    analyze_with_counsel()

if st.session_state.final_reply:
    st.write("### TIAP result")
    st.markdown(st.session_state.final_reply)

    with st.expander("Observer / validator / judge JSON details"):
        st.json(st.session_state.observer_json or {})
        if st.session_state.validator_json:
            st.json(st.session_state.validator_json)
        if st.session_state.judge_json:
            st.json(st.session_state.judge_json)

st.write("### Follow‑up question")

followup_text = st.text_input(
    "Ask a short follow‑up about this same image and subject",
    value="",
    placeholder="e.g. 'Is this pattern more like moisture or air leak?'",
)

if st.button("Send follow‑up") and followup_text.strip():
    send_followup(followup_text)
    for msg in st.session_state.display_messages:
        role = "You" if msg["role"] == "user" else "TIAP"
        st.markdown(f"**{role}:** {msg['content']}")