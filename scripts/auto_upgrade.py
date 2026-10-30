
import json
import os
import re
import urllib.request
from pathlib import Path

HTML_PATH = Path("backend/app/index.html")
START = "/* THESIS_AUTO_UPGRADE_START */"
END = "/* THESIS_AUTO_UPGRADE_END */"

API_KEY = os.environ.get("GEMINI_UPGRADE_KEY", "")

if not API_KEY:
    raise SystemExit("Missing GEMINI_UPGRADE_KEY")

html = HTML_PATH.read_text(encoding="utf-8")

if "</style>" not in html:
    raise SystemExit("HTML style section not found")

# The AI can choose only approved visual settings.
PALETTES = {
    "mint": "#74edbe",
    "blue": "#82baff",
    "violet": "#b9a0ff",
    "cyan": "#76dedf",
    "gold": "#f0ca86",
}

ROUNDS = {
    "soft": "12px",
    "modern": "16px",
    "smooth": "20px",
}

SPACING = {
    "compact": "11px",
    "balanced": "15px",
    "spacious": "19px",
}

current = ""

if START in html and END in html:
    current = html.split(START, 1)[1].split(END, 1)[0]

prompt = f"""
You are the visual design agent for THESIS AI,
a premium, dark Vietnamese investment research app.

Improve the interface style using approved options only.

Select:
- accent: mint, blue, violet, cyan, gold
- radius: soft, modern, smooth
- spacing: compact, balanced, spacious

Current CSS:
{current}

Prefer a sophisticated and accessible design.
Make a different combination from the current CSS.

Return ONLY valid JSON, with exactly these keys:
"accent", "radius", "spacing"

No markdown. No explanation.
"""

payload = {
    "contents": [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ],
    "generationConfig": {
        "temperature": 0.8,
        "responseMimeType": "application/json"
    }
}

url = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/gemini-3.1-flash-lite:"
    "generateContent"
)

request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    },
    method="POST"
)

with urllib.request.urlopen(request, timeout=45) as response:
    data = json.loads(response.read().decode("utf-8"))

answer = data["candidates"][0]["content"]["parts"][0]["text"]
settings = json.loads(answer)

if not isinstance(settings, dict):
    raise SystemExit("Invalid AI response")

if set(settings) != {"accent", "radius", "spacing"}:
    raise SystemExit("Unexpected AI fields")

accent = settings["accent"]
radius = settings["radius"]
spacing = settings["spacing"]

if accent not in PALETTES:
    raise SystemExit("Unsupported accent")

if radius not in ROUNDS:
    raise SystemExit("Unsupported radius")

if spacing not in SPACING:
    raise SystemExit("Unsupported spacing")

def make_css(a, r, s):
    return f"""
{START}
:root {{
  --accent: {PALETTES[a]};
  --mint: {PALETTES[a]};
}}
.suggestion {{
  border-radius: {ROUNDS[r]};
  padding: {SPACING[s]};
}}
.composer {{
  border-radius: {ROUNDS[r]};
}}
.action {{
  border-radius: {ROUNDS[r]};
}}
{END}
"""

css = make_css(accent, radius, spacing)

if START in html and END in html:
    old = re.escape(START) + r".*?" + re.escape(END)
    html = re.sub(old, lambda _: css.strip(), html, count=1, flags=re.S)
else:
    html = html.replace("</style>", css + "\n</style>", 1)

# Ensure each run produces a real visual change.
if html == HTML_PATH.read_text(encoding="utf-8"):
    for a in PALETTES:
        for r in ROUNDS:
            for s in SPACING:
                candidate = make_css(a, r, s)
                alternate = re.sub(
                    re.escape(START) + r".*?" + re.escape(END),
                    lambda _: candidate.strip(),
                    html,
                    count=1,
                    flags=re.S
                )
                if alternate != html:
                    html = alternate
                    break
            else:
                continue
            break
        else:
            continue
        break

HTML_PATH.write_text(html, encoding="utf-8")

print("THESIS AI visual upgrade generated successfully")
print("Accent:", accent)
print("Radius:", radius)
print("Spacing:", spacing)
