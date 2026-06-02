import os
from flask import Flask, render_template, request, jsonify
import anthropic

# Vercel runs this from the api/ directory — templates live one level up
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
app = Flask(__name__, template_folder=template_dir)
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a real-time conversation translator for live speech. Output ONLY the translation — no explanations, no quotes, no metadata.

Rules:
- Preserve tone, emotion, and intent (casual stays casual, urgent stays urgent)
- Translate idioms and slang by meaning, not word-for-word
- Argentine Spanish specifics:
  * voseo: "vos/andás/tenés/querés/sabés" = you/you are/you have/you want/you know
  * "che" = hey / man / dude (attention-getter or filler)
  * "boludo/a" = dude / idiot (friendly or insulting depending on context)
  * "quilombo" = mess / chaos / trouble
  * "laburo/laburar" = work
  * "pibe/piba" = kid / guy / girl
  * "re-" prefix = very/really (re-bueno = really good)
  * "copado/a" = cool / awesome
  * "fiaca" = laziness / can't be bothered
  * "guita" = money
  * "flashar" = to imagine / to trip out
  * "zarpado/a" = crazy / extreme (positive or negative)
- Keep translations concise — this is live speech, brevity matters"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    source_lang = data.get("source_lang") or "Argentine Spanish"
    target_lang = data.get("target_lang") or "English"
    context = data.get("context") or []

    if not text:
        return jsonify({"translation": ""})

    messages = []
    for entry in context[-4:]:
        messages.append({"role": "user", "content": f"[{source_lang}] {entry['original']}"})
        messages.append({"role": "assistant", "content": entry["translation"]})
    messages.append({
        "role": "user",
        "content": f"Translate from {source_lang} to {target_lang}: {text}",
    })

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return jsonify({"translation": response.content[0].text.strip()})
