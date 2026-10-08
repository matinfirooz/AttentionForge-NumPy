"""Generate exact SVG heatmaps and a self-contained offline HTML explorer."""
import json
from pathlib import Path
import numpy as np
from attentionforge import scaled_dot_product_attention, attention_entropy, causal_mask
from attentionforge.visuals import SVG, hero, heatmap, FG, MUTED, CYAN, GOLD


def make_data():
    rng = np.random.default_rng(22)
    # Constructed projections illustrate attention; these are not language embeddings.
    q, k, v = [rng.normal(size=(1, 4, 6, 8)) for _ in range(3)]
    q[0, 0] = k[0, 0]  # A head with a visible diagonal preference.
    q[0, 1] = k[0, 1, 2]  # A head that repeatedly retrieves the same key.
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(8)
    variants = []
    for mode in ("bidirectional", "causal", "padding"):
        for temperature in (0.25, 0.5, 1.0, 2.0):
            mask = np.array([True, True, True, True, False, False]) if mode == "padding" else None
            output, weights = scaled_dot_product_attention(q, k, v, mask=mask,
                causal=mode == "causal", temperature=temperature)
            allowed = causal_mask(6) if mode == "causal" else np.ones((6, 6), dtype=bool)
            if mask is not None:
                allowed &= mask
            variants.append({"mode": mode, "temperature": temperature,
                             "weights": weights[0].tolist(), "output": output[0].tolist(),
                             "entropy": attention_entropy(weights)[0].tolist(),
                             "scores": (scores[0]/temperature).tolist(),
                             "allowed": allowed.tolist()})
    return {"tokens": ["I", "build", "attention", "with", "NumPy", "."],
            "q": q[0].tolist(), "k": k[0].tolist(), "v": v[0].tolist(),
            "variants": variants,
            "note": "Constructed Q/K/V tensors, not learned language semantics. Every tensor was computed in NumPy; browser controls select precomputed cases."}


def main():
    destination = Path("docs")
    (destination/"assets").mkdir(parents=True, exist_ok=True)
    hero(destination/"assets"/"hero.svg")
    data = make_data()
    (destination/"results"/"explorer_data.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    variant = next(v for v in data["variants"] if v["mode"] == "causal" and v["temperature"] == 1)
    svg = SVG(1130, 470, "Four causal attention heads from a reproducible NumPy computation")
    svg.text(35, 42, "One sequence. Four ways to attend.", 25, FG, 700)
    svg.text(35, 71, "Constructed projections · causal mask · rows=query, columns=key · temperature=1", 14, MUTED)
    for head in range(4):
        x0 = 52+head*273
        svg.text(x0, 111, f"HEAD {head}", 14, CYAN, 700)
        heatmap(svg, variant["weights"][head], x0, 137, 40, numbers=False)
        svg.text(x0, 400, "earlier ← keys → later", 13, MUTED)
    svg.text(35, 444, "Masked future positions are exactly zero. Each visible query row sums to one.", 14, MUTED)
    svg.save(destination/"assets"/"attention_heads.svg")
    template = (destination/"explorer_template.html").read_text(encoding="utf-8")
    output = template.replace("__ATTENTION_DATA__", json.dumps(data, separators=(",", ":")))
    (destination/"attention_explorer.html").write_text(output, encoding="utf-8")
    print("Generated docs/assets/*.svg and docs/attention_explorer.html (no server needed).")


if __name__ == "__main__":
    main()
