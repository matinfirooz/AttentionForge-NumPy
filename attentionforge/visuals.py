"""Deterministic SVG graphics using only the Python standard library.

These are presentation utilities; all attention calculations live in NumPy.
"""
from html import escape
from pathlib import Path

BG, PANEL, FG, MUTED = "#091220", "#101f32", "#e7f0fa", "#91a6be"
CYAN, GOLD, PURPLE = "#52e5c3", "#ffc46b", "#b2a1ff"


class SVG:
    def __init__(self, width, height, title):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
                      f'<title>{escape(title)}</title>',
                      f'<rect width="{width}" height="{height}" rx="20" fill="{BG}"/>']

    def rect(self, x, y, w, h, fill=PANEL, radius=8, **extra):
        attrs = " ".join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in extra.items())
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" {attrs}/>')

    def text(self, x, y, value, size=16, fill=FG, weight=400, anchor="start", **extra):
        attrs = " ".join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in extra.items())
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Inter,DejaVu Sans,Arial,sans-serif" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" {attrs}>{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color=MUTED, width=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def path(self, points, color=CYAN, width=3):
        self.parts.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts) + "\n</svg>\n", encoding="utf-8")


def color(value):
    value = max(0.0, min(1.0, float(value)))
    low, high = (18, 39, 60), (82, 229, 195)
    return "#" + "".join(f"{round(a+(b-a)*value):02x}" for a, b in zip(low, high))


def heatmap(svg, matrix, x, y, cell=48, *, labels=None, numbers=True):
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            svg.rect(x + j * cell, y + i * cell, cell - 3, cell - 3,
                     color(value), 5)
            if numbers:
                svg.text(x+j*cell+(cell-3)/2, y+i*cell+cell*0.57,
                         f"{float(value):.2f}", 12,
                         BG if value > 0.45 else FG, anchor="middle")
    if labels:
        for j, label in enumerate(labels):
            svg.text(x+j*cell+(cell-3)/2, y-10, label, 12, MUTED, anchor="middle")


def hero(path):
    svg = SVG(1280, 490, "AttentionForge: NumPy-only attention from forward to learning")
    svg.text(52, 51, "MATINFIROOZ / ATTENTIONFORGE-NUMPY", 14, CYAN, 700, letter_spacing=2)
    svg.text(50, 125, "Attention, with nothing hidden.", 50, FG, 700)
    svg.text(52, 166, "Read the math. Check the gradients. Train the memory. Inspect every head.", 20, MUTED)
    stages = [("01", "PROJECT", "Q, K, V"), ("02", "COMPARE", "Q × Kᵀ / √d"),
              ("03", "NORMALIZE", "masked softmax"), ("04", "RETRIEVE", "P × V")]
    for i, (num, title, detail) in enumerate(stages):
        x = 52 + i * 302
        svg.rect(x, 212, 272, 143)
        svg.text(x+18, 242, num, 14, CYAN, 700)
        svg.text(x+18, 276, title, 21, FG, 700)
        svg.text(x+18, 318, detail, 19, MUTED)
        if i < 3:
            svg.line(x+275, 282, x+295, 282, CYAN, 2)
            svg.path([(x+290, 277), (x+296, 282), (x+290, 287)], CYAN, 2)
    chips = [("NUMPY ONLY", 178), ("MANUAL BACKWARD", 246),
             ("KV CACHE + RoPE", 243), ("TILED ONLINE SOFTMAX", 319)]
    x = 52
    for text, width in chips:
        svg.rect(x, 396, width, 42, "#152c3b", 21)
        svg.text(x+width/2, 423, text, 14, CYAN, 600, anchor="middle")
        x += width+14
    svg.save(path)


def training_plot(history, path):
    svg = SVG(1100, 420, "Measured recall-task validation accuracy and loss during training")
    svg.text(36, 42, "A memory that learns where to look", 25, FG, 700)
    svg.text(36, 69, "Synthetic key–value retrieval · fixed validation set · NumPy + analytical gradients", 14, MUTED)
    panels = [("Validation accuracy", "val_accuracy", 65, CYAN, 1.0),
              ("Validation cross entropy", "val_loss", 615, GOLD,
               max(float(h["val_loss"]) for h in history) * 1.1)]
    for title, field, x0, stroke, ymax in panels:
        svg.text(x0, 109, title, 17, FG, 600)
        width, height, top = 415, 215, 130
        xmax = max(h["step"] for h in history) or 1
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            yy = top + height * (1-fraction)
            svg.line(x0, yy, x0+width, yy, "#22344a")
            svg.text(x0-10, yy+5, f"{fraction*ymax:.2f}", 11, MUTED, anchor="end")
        points = [(x0+width*h["step"]/xmax, top+height*(1-h[field]/ymax)) for h in history]
        svg.path(points, stroke)
        for fraction in (0, 0.5, 1):
            svg.text(x0+width*fraction, 372, round(xmax*fraction), 12, MUTED, anchor="middle")
        svg.text(x0+width/2, 399, "optimizer steps", 12, MUTED, anchor="middle")
    svg.save(path)


def recall_heads(before, after, info, prediction, path):
    svg = SVG(1100, 425, "Attention heads before and after learning key-value retrieval")
    keys, values, chosen = info["keys"][0], info["values"][0], int(info["chosen"][0])
    svg.text(35, 43, "Watch a query find its value", 26, FG, 700)
    svg.text(35, 75, f"Query: key {keys[chosen]}    Target: value {values[chosen]}    Prediction: value {prediction}", 17, MUTED)
    for i, (title, weights, x0) in enumerate((("Before training", before, 110), ("After training", after, 655))):
        svg.text(x0, 113, title, 18, FG, 600)
        labels = [f"k{k}:v{v}" for k, v in zip(keys, values)]
        matrix = weights[0, :, 0, :]
        heatmap(svg, matrix, x0, 150, 55, labels=labels)
        for head in range(matrix.shape[0]):
            svg.text(x0-12, 182+head*55, f"Head {head}", 12, MUTED, anchor="end")
        svg.rect(x0+chosen*55-2, 147, 54, 221, "none", 7, stroke=GOLD, stroke_width=2)
    svg.text(35, 402, "Gold column = matching key. Brighter cells = larger attention probability.", 14, MUTED)
    svg.save(path)
