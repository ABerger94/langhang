import os
from flask import Flask, render_template, request, jsonify
import anthropic
from openai import OpenAI

app = Flask(__name__)

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

MODELS = {
    "claude-sonnet-4-6":         {"provider": "anthropic", "env": "ANTHROPIC_API_KEY"},
    "claude-haiku-4-5-20251001": {"provider": "anthropic", "env": "ANTHROPIC_API_KEY"},
    "gpt-4o":                    {"provider": "openai",    "env": "OPENAI_API_KEY"},
    "gpt-4o-mini":               {"provider": "openai",    "env": "OPENAI_API_KEY"},
    "llama-3.3-70b-versatile":   {"provider": "groq",      "env": "GROQ_API_KEY"},
    "llama-3.1-8b-instant":      {"provider": "groq",      "env": "GROQ_API_KEY"},
    "gemma2-9b-it":              {"provider": "groq",      "env": "GROQ_API_KEY"},
}


def build_context_messages(context, source_lang, target_lang, text):
    msgs = []
    for entry in context[-4:]:
        msgs.append({"role": "user",      "content": f"[{source_lang}] {entry['original']}"})
        msgs.append({"role": "assistant", "content": entry["translation"]})
    msgs.append({"role": "user", "content": f"Translate from {source_lang} to {target_lang}: {text}"})
    return msgs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    data        = request.get_json()
    text        = (data.get("text")        or "").strip()
    source_lang =  data.get("source_lang") or "Argentine Spanish"
    target_lang =  data.get("target_lang") or "English"
    context     =  data.get("context")     or []
    model_id    =  data.get("model")       or "claude-sonnet-4-6"

    if not text:
        return jsonify({"translation": ""})

    model_info = MODELS.get(model_id)
    if not model_info:
        return jsonify({"error": f"Unknown model '{model_id}'."}), 400

    api_key = os.environ.get(model_info["env"])
    if not api_key:
        return jsonify({"error": f"{model_info['env']} is not set."}), 500

    context_msgs = build_context_messages(context, source_lang, target_lang, text)

    try:
        provider = model_info["provider"]

        if provider == "anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model_id,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=context_msgs,
            )
            translation = resp.content[0].text.strip()

        else:
            base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
            client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=500,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *context_msgs],
            )
            translation = resp.choices[0].message.content.strip()

        return jsonify({"translation": translation})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
