from pathlib import Path

THEMES = {
    "light": {"bg": "#fafaf5", "ink": "#1a1a1a", "muted": "#7a7a78", "accent": "#6b4a3a", "dot": "#1a1a1a"},
    "dark": {"bg": "#1a1714", "ink": "#ece6dc", "muted": "#a39a90", "accent": "#c4a48c", "dot": "#ece6dc"},
}

LAYERS = [
    ("PYTHON · LINEAR ALGEBRA", "PHASE 00 · 01"),
    ("CALCULUS · PROBABILITY", "PHASE 02 · 03"),
    ("ML FUNDAMENTALS", "PHASE 04"),
    ("SUPERVISED LEARNING", "PHASE 05 · 06"),
    ("TREES · ENSEMBLES", "PHASE 07"),
    ("CLUSTERING · PCA", "PHASE 08 · 09"),
    ("PRACTICAL ML", "PHASE 10"),
]

THOTH = ("M 77 410 H 204 V 460 H 166 V 665 H 116 V 460 H 77 Z "
         "M 230 410 H 281 V 512 H 333 V 410 H 384 V 665 H 333 V 563 H 281 V 665 H 230 Z "
         "M 410 410 H 614 V 665 H 410 Z M 512 458 L 443 591 L 512 617 L 581 591 Z "
         "M 640 410 H 768 V 460 H 729 V 665 H 679 V 460 H 640 Z "
         "M 794 410 H 845 V 512 H 897 V 410 H 947 V 665 H 897 V 563 H 845 V 665 H 794 Z")

CX, BASE, APEX, HALF, DEPTH_X, DEPTH_Y, GAP = 880, 352, 122, 112, 54, 22, 0.014
LABEL_X = 1062


def lerp(p, q, t):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def pts(*points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def wordmark(x, y, height, fill):
    """The THOTH logo as paths (no font needed), its top-left corner at (x, y)."""
    scale = height / 255
    return (f'<g transform="translate({x - 77 * scale:.2f}, {y - 410 * scale:.2f}) scale({scale:.5f})">'
            f'<path fill="{fill}" fill-rule="evenodd" d="{THOTH}"/></g>')


def pyramid(c):
    """A wireframe stepped pyramid, one tier per group of phases, with its hidden back edges dashed."""
    accent, bg = c["accent"], c["bg"]
    apex = (CX + DEPTH_X / 2, APEX)
    left, right, back = (CX - HALF, BASE), (CX + HALF, BASE), (CX + HALF + DEPTH_X, BASE - DEPTH_Y)
    hidden = (CX - HALF + DEPTH_X, BASE - DEPTH_Y)
    n = len(LAYERS)
    parts = [f'    <g stroke="{accent}" stroke-width="0.8" stroke-dasharray="3 3" stroke-opacity="0.55" fill="none">',
             f'      <polyline points="{pts(left, hidden, back)}"/>',
             f'      <line x1="{hidden[0]:.1f}" y1="{hidden[1]:.1f}" x2="{apex[0]:.1f}" y2="{apex[1]:.1f}"/>',
             "    </g>"]
    anchors = []
    for i in range(n):
        top = i == n - 1
        t0, t1 = i / n + (GAP if i else 0), (i + 1) / n - (0 if top else GAP)
        l0, l1, r0, r1 = lerp(left, apex, t0), lerp(left, apex, t1), lerp(right, apex, t0), lerp(right, apex, t1)
        b0, b1 = lerp(back, apex, t0), lerp(back, apex, t1)
        h1 = lerp(hidden, apex, t1)
        front = [l0, r0, apex] if top else [l0, r0, r1, l1]
        side = [r0, b0, apex] if top else [r0, b0, b1, r1]
        width = 1.5 if top else 1.1
        tier = [f'      <polygon class="bp" points="{pts(*front)}" fill="{accent}" fill-opacity="{0.03 + 0.02 * i:.2f}" stroke-width="{width}"/>',
                f'      <polygon class="bp" points="{pts(*side)}" fill="{accent}" fill-opacity="{0.10 + 0.03 * i:.2f}" stroke-width="{width}"/>']
        if not top:
            tier.append(f'      <polygon class="bp" points="{pts(l1, r1, b1, h1)}" fill="{bg}" fill-opacity="0.6" stroke-width="0.8" stroke-opacity="0.7"/>')
        parts += ["    <g>", *tier, "    </g>"]
        edge = lerp(back, apex, (t0 + t1) / 2)
        anchors.append((edge[0] + 6, (r0[1] + r1[1]) / 2))
    return "\n".join(parts), anchors


def header(c):
    tiers, anchors = pyramid(c)
    ticks = "\n".join(f'    <line x1="{x:.1f}" y1="{y:.1f}" x2="{LABEL_X - 8}" y2="{y:.1f}"/>' for x, y in anchors)
    names = "\n".join(f'    <text x="{LABEL_X}" y="{y + 3.5:.1f}">{name}</text>' for (_, y), (name, _) in zip(anchors, LAYERS))
    nums = "\n".join(f'    <text x="{LABEL_X}" y="{y + 14:.1f}">{num}</text>' for (_, y), (_, num) in zip(anchors, LAYERS))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="420" viewBox="0 0 1280 420" role="img" aria-label="THOTH · Classic ML from Scratch: learning from pyramids. 11 phases built from Python and math at the base up to supervised, unsupervised and practical machine learning">
  <defs>
    <pattern id="paper" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
      <circle cx="0" cy="0" r="1" fill="{c['dot']}" fill-opacity="0.08"/>
    </pattern>
    <pattern id="paperRule" x="0" y="0" width="100" height="6" patternUnits="userSpaceOnUse">
      <rect width="4" height="3" fill="{c['accent']}"/>
      <rect x="8" y="3" width="6" height="3" fill="{c['accent']}" fill-opacity="0.35"/>
    </pattern>
    <style>
      .bp {{ stroke: {c['accent']}; stroke-linejoin: round; stroke-linecap: round; }}
      .mono {{ font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
      .display {{ font-family: 'VT323', ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 400; }}
      .serif {{ font-family: 'Source Serif 4', Georgia, serif; }}
    </style>
  </defs>

  <rect width="1280" height="420" fill="{c['bg']}"/>
  <rect width="1280" height="420" fill="url(#paper)"/>

  {wordmark(56, 30, 22, c['ink'])}
  <g class="mono" font-size="11" letter-spacing="2.4" fill="{c['muted']}" text-anchor="end">
    <text x="1224" y="47">© 2026  ·  OPEN SOURCE  ·  MIT LICENSE</text>
  </g>
  <line x1="56" y1="64" x2="1224" y2="64" stroke="{c['accent']}" stroke-width="0.6" stroke-opacity="0.4"/>

  <g transform="translate(56, 110)">
    <text class="display" x="0" y="58" font-size="68" letter-spacing="1.5" fill="{c['accent']}">CLASSIC ML</text>
    <text class="display" x="0" y="124" font-size="68" letter-spacing="1.5" fill="{c['ink']}">FROM SCRATCH.</text>
  </g>

  <g class="serif" font-size="17" fill="{c['ink']}">
    <text x="56" y="296">Learning from pyramids: machine learning from first principles, one phase at a time.</text>
  </g>

  <line x1="720" y1="110" x2="720" y2="362" stroke="{c['accent']}" stroke-width="0.6" stroke-opacity="0.4" stroke-dasharray="3 3"/>

  <g class="mono" font-size="9" letter-spacing="2" fill="{c['accent']}">
    <text x="752" y="104">FIG_000.A  ·  THE PYRAMID</text>
  </g>

{tiers}

  <g class="bp" stroke-width="0.8" stroke-dasharray="3 3">
{ticks}
  </g>
  <g class="mono" font-size="10" letter-spacing="0.8" fill="{c['accent']}" font-weight="600">
{names}
  </g>
  <g class="mono" font-size="8.5" letter-spacing="0.8" fill="{c['muted']}">
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
