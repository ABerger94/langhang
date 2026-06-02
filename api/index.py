import os
from flask import Flask, render_template, request, jsonify
import anthropic

# Vercel runs this from the api/ directory — templates live one level up
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
app = Flask(__name__, template_folder=template_dir)
try:
    client = anthropic.Anthropic()
except Exception:
    client = None

SYSTEM_PROMPT = """You are a live conversation interpreter — like a professional interpreter whispering in someone's ear during a real conversation.

Your output is heard directly by the listener. It must sound like how a native speaker of the target language would naturally say it — correct idioms, natural rhythm, right register. A fluent speaker of the target language should not be able to tell it was translated.

Output ONLY the translation. No explanations, no alternatives, no notes, no quotes.

Match the speaker's register exactly: casual stays casual, excited stays excited, frustrated stays frustrated, vulgar stays vulgar. Use contractions naturally. Drop filler sounds (uh, um, eh, mmm).

Argentine Spanish guidance:
- Voseo is informal "you" — never translate it as formal; "¿cómo andás?" = "how's it going?", "¿qué hacés?" = "what are you up to?"
- "che / boludo / flaco / pibe / viejo" are friend-to-friend address — use "hey / dude / man / buddy / bro" matching the warmth or bite in the tone
- "dale" = "sure" / "okay" / "let's go" / "come on" — pick whichever fits the moment
- "re-[adj]" = really/so — "re-cansado" = "so tired", "re-copado" = "so cool"
- "quilombo" = mess / chaos / shitstorm — scale with intensity
- "me tiene harto/podrido" = "I'm sick of this" / "I can't take it anymore"
- "buena onda / mala onda" = "good vibes / bad vibes", or "chill / not cool"
- "mirá vos" = "no way" / "huh, would you look at that"
- "copado/a" = cool / awesome; "zarpado/a" = wild / insane / crazy
- "laburo" = work/job; "guita" = money/cash; "fiaca" = laziness/"can't be bothered"
- "la concha de tu madre" / "la puta madre" — match the expletive intensity in the target language, don't translate literally
- Lunfardo varies wildly — always translate the *feeling and intensity*, never the literal words"""


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

    if client is None:
        return jsonify({"error": "ANTHROPIC_API_KEY is not set. Add it to your Vercel project → Settings → Environment Variables, then redeploy."}), 500

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return jsonify({"translation": response.content[0].text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
