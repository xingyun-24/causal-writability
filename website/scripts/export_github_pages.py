from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs"


def main() -> None:
    with urllib.request.urlopen("http://localhost:3000/", timeout=20) as response:
        html = response.read().decode("utf-8")

    css = (ROOT / "app" / "globals.css").read_text(encoding="utf-8")
    css = re.sub(r'^@import\s+"tailwindcss";\s*', "", css)

    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.S | re.I)
    html = re.sub(r'<link\b[^>]*rel="stylesheet"[^>]*>', "", html, flags=re.I)
    html = re.sub(r'<link\b[^>]*rel="modulepreload"[^>]*>', "", html, flags=re.I)
    html = html.replace("</head>", f"<style>{css}</style></head>", 1)
    html = html.replace('href="/paper/', 'href="./paper/')
    html = html.replace('src="/paper/', 'src="./paper/')
    html = html.replace('href="/og.png"', 'href="./og.png"')
    html = html.replace('src="/og.png"', 'src="./og.png"')

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "index.html").write_text(html, encoding="utf-8")
    (OUTPUT / "404.html").write_text(html, encoding="utf-8")
    (OUTPUT / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copytree(ROOT / "public" / "paper", OUTPUT / "paper")
    shutil.copy2(ROOT / "public" / "og.png", OUTPUT / "og.png")
    print(f"Static export written to {OUTPUT}")


if __name__ == "__main__":
    main()
