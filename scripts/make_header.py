from pathlib import Path

THEMES = {
    "light": {"bg": "#fafaf5", "ink": "#1a1a1a", "muted": "#7a7a78", "accent": "#6b4a3a", "dot": "#1a1a1a"},
    "dark": {"bg": "#1a1714", "ink": "#ece6dc", "muted": "#a39a90", "accent": "#c4a48c", "dot": "#ece6dc"},
}

LAYERS = [
    ("PYTHON · LINEAR ALGEBRA", "PHASE 00 · 01"),
    ("CALCULUS · PROBABILITY", "PHASE 02 · 03"),
    ("ML FUNDAMENTALS", "PHASE 04"),
    ("REGRESSION · CLASSIFICATION", "PHASE 05 · 06"),
    ("TREES · ENSEMBLES", "PHASE 07"),
    ("CLUSTERING · PCA", "PHASE 08 · 09"),
    ("PRACTICAL ML", "PHASE 10"),
]

BOTTOM, STEP, LEFT, MID, RIGHT = 330, 25, 834, 924, 1014


def slab(y, i, n, accent, bg):
    top = i == n - 1
    t, s1, s2 = 0.05 + 0.13 * i / (n - 1), 0.13 + 0.13 * i / (n - 1), 0.19 + 0.13 * i / (n - 1)
    w = 1.5 if top else 1.1
    return f"""    <g>
      <polygon points="{LEFT},{y} {MID},{y-20} {RIGHT},{y} {RIGHT},{y+8} {MID},{y+28} {LEFT},{y+8}" fill="{bg}"/>
      <polygon points="{LEFT},{y} {MID},{y-20} {RIGHT},{y} {MID},{y+20}" fill="{accent}" fill-opacity="{t:.2f}"/>
      <polygon points="{LEFT},{y} {MID},{y+20} {MID},{y+28} {LEFT},{y+8}" fill="{accent}" fill-opacity="{s1:.2f}"/>
      <polygon points="{MID},{y+20} {RIGHT},{y} {RIGHT},{y+8} {MID},{y+28}" fill="{accent}" fill-opacity="{s2:.2f}"/>
      <polygon class="bp" points="{LEFT},{y} {MID},{y-20} {RIGHT},{y} {MID},{y+20}" stroke-width="{w}"/>
      <polyline class="bp" points="{LEFT},{y} {LEFT},{y+8} {MID},{y+28} {RIGHT},{y+8} {RIGHT},{y}" stroke-width="{w}"/>
      <line class="bp" x1="{MID}" y1="{y+20}" x2="{MID}" y2="{y+28}" stroke-width="{w}"/>
    </g>"""


def header(c):
    n = len(LAYERS)
    ys = [BOTTOM - STEP * i for i in range(n)]
    slabs = "\n".join(slab(y, i, n, c["accent"], c["bg"]) for i, y in enumerate(ys))
    ticks = "\n".join(f'    <line x1="1018" y1="{y}" x2="1036" y2="{y}"/>' for y in ys)
    names = "\n".join(f'    <text x="1044" y="{y+3.5}">{name}</text>' for y, (name, _) in zip(ys, LAYERS))
    nums = "\n".join(f'    <text x="1044" y="{y+14.5}">{num}</text>' for y, (_, num) in zip(ys, LAYERS))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="420" viewBox="0 0 1280 420" role="img" aria-label="Classic ML from Scratch: a reference manual in 11 phases, stacked from Python and math up to supervised, unsupervised and practical machine learning">
  <defs>
    <pattern id="paper" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
      <circle cx="0" cy="0" r="1" fill="{c['dot']}" fill-opacity="0.08"/>
    </pattern>
    <pattern id="paperRule" x="0" y="0" width="100" height="6" patternUnits="userSpaceOnUse">
      <rect width="4" height="3" fill="{c['accent']}"/>
      <rect x="8" y="3" width="6" height="3" fill="{c['accent']}" fill-opacity="0.35"/>
    </pattern>
    <style>
      .bp {{ stroke: {c['accent']}; fill: none; stroke-linejoin: round; stroke-linecap: round; }}
      .mono {{ font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
      .display {{ font-family: 'VT323', ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 400; }}
      .serif {{ font-family: 'Source Serif 4', Georgia, serif; }}
    </style>
  </defs>

  <rect width="1280" height="420" fill="{c['bg']}"/>
  <rect width="1280" height="420" fill="url(#paper)"/>

  <g class="mono" font-size="11" letter-spacing="2.4" fill="{c['muted']}" text-anchor="end">
    <text x="1224" y="44">© 2026  ·  OPEN SOURCE  ·  MIT LICENSE</text>
  </g>
  <line x1="56" y1="64" x2="1224" y2="64" stroke="{c['accent']}" stroke-width="0.6" stroke-opacity="0.4"/>

  <g transform="translate(56, 110)">
    <text class="display" x="0" y="58" font-size="68" letter-spacing="1.5" fill="{c['accent']}">CLASSIC ML</text>
    <text class="display" x="0" y="124" font-size="68" letter-spacing="1.5" fill="{c['ink']}">FROM SCRATCH.</text>
  </g>

  <g class="serif" font-size="17" fill="{c['ink']}">
    <text x="56" y="296">A reference manual for people who want to understand machine learning from first principles.</text>
  </g>

  <line x1="780" y1="110" x2="780" y2="362" stroke="{c['accent']}" stroke-width="0.6" stroke-opacity="0.4" stroke-dasharray="3 3"/>

  <g class="mono" font-size="9" letter-spacing="2" fill="{c['accent']}">
    <text x="820" y="120">FIG_000.A  ·  CURRICULUM STACK</text>
  </g>

{slabs}

  <g class="bp" stroke-width="0.8" stroke-dasharray="3 3">
{ticks}
  </g>
  <g class="mono" font-size="10.5" letter-spacing="1" fill="{c['accent']}" font-weight="600">
{names}
  </g>
  <g class="mono" font-size="9" letter-spacing="1" fill="{c['muted']}">
{nums}
  </g>

  <rect x="0" y="408" width="1280" height="6" fill="url(#paperRule)"/>
  <line x1="0" y1="404" x2="1280" y2="404" stroke="{c['ink']}" stroke-opacity="0.16" stroke-width="0.8"/>
</svg>
"""


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "assets" / "readme"
    for name, colors in THEMES.items():
        (out / f"header-{name}.svg").write_text(header(colors))
