"""Fusion: combine what the review SAYS with what the photo SHOWS into one decision per order.

Inputs per order:  text probabilities (negative/neutral/positive), issue tags, product category,
                   and the photo model's P(bad) (or no photo).
Output:            status OK / REVIEW / URGENT, a risk score 0..1, and human-readable reason codes.

Step 1, risk score (weights in config.yaml -> fusion, tuned on simulated orders):
    risk = w_text * P(negative) + w_img * P(bad photo) + w_tag * (1 if any issue tag else 0)
    Without a photo the image weight is shared out over text and tags (text-only decision).
    risk >= urgent threshold -> URGENT, >= review threshold -> REVIEW, else OK.

Step 2, rules that override the score when text and photo can be compared (the "2x2" logic):
    negative text + bad photo                    -> URGENT  both show a problem
    positive text + bad photo                    -> REVIEW  photo contradicts the positive review
    damage claim in text + clean photo           -> REVIEW  claim not visible: check, possible false claim
    negative text + clean photo, no damage claim -> REVIEW  a service/product problem a photo can't show
    any other bad photo                          -> at least REVIEW (a damaged/rotten photo is never OK)

Why rules on top of a score: a weighted sum can't express *disagreement*. "Customer is happy but the
photo shows a crushed box" or "customer claims damage but the photo looks fine" are exactly the
cases a human should look at, and a single number would average them away.

Which text claims a photo can check depends on the photo model: the package model sees box damage,
the produce model sees rot. A "doesn't turn on" complaint is not visible on a package photo, so it
must not count as a "false claim" when the box looks fine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import load_config

# Text claims each photo model can confirm or contradict.
VERIFIABLE_TAGS = {"packages": {"damaged_packaging"}, "produce": {"spoiled_food"}}

REASONS = {
    "BOTH_PROBLEM": "Text and photo both show a problem.",
    "PHOTO_CONTRADICTS_REVIEW": "Photo shows {bad_label} although the review is positive.",
    "CLAIM_NOT_VISIBLE": "Review claims {claim}, but the photo looks {good_label}: check it (possible false claim).",
    "ISSUE_NOT_VISUAL": "Negative review, photo looks {good_label}: a service or product problem a photo can't show.",
    "PHOTO_PROBLEM": "Photo shows {bad_label} (P = {p_bad:.0%}).",
    "NEGATIVE_TEXT": "Review is negative (P = {p_neg:.0%}).",
    "ISSUE_TAGS": "Review mentions: {tags}.",
    "NO_PHOTO": "No photo: decision based on the text only.",
    "NO_PHOTO_MODEL": "No photo model for category '{category}': decision based on the text only.",
    "LOOKS_FINE": "No problem found.",
}
STATUS_ORDER = ["OK", "REVIEW", "URGENT"]


@dataclass
class Decision:
    status: str
    risk: float
    reasons: list[dict[str, str]] = field(default_factory=list)
    used_photo: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reason(code: str, **kw) -> dict[str, str]:
    return {"code": code, "text": REASONS[code].format(**kw)}


def fuse(text_probs: dict[str, float], tags: dict[str, str] | list[str], category: str,
         image: dict[str, Any] | None = None, settings: dict | None = None) -> Decision:
    """Combine one order's text and photo results into a Decision.

    `image` is the output of Predictor.image(): {"domain", "p_bad", "is_bad", "threshold", ...},
    or None when there is no photo. `settings` defaults to config.yaml -> fusion.
    """
    cfg = settings or load_config()["fusion"]
    w, th = cfg["weights"], cfg["thresholds"]
    tags = list(tags)
    p_neg = float(text_probs["negative"])
    text_label = max(text_probs, key=text_probs.get)
    has_tag = 1.0 if tags else 0.0

    # ---- step 1: weighted risk score
    if image is None:
        total = w["text"] + w["tag"]
        risk = (w["text"] * p_neg + w["tag"] * has_tag) / total
    else:
        risk = w["text"] * p_neg + w["image"] * float(image["p_bad"]) + w["tag"] * has_tag
    status = "URGENT" if risk >= th["urgent"] else "REVIEW" if risk >= th["review"] else "OK"

    reasons: list[dict[str, str]] = []
    if image is None:
        reasons.append(_reason("NO_PHOTO"))
    if text_label == "negative" or p_neg >= th["review"]:
        reasons.append(_reason("NEGATIVE_TEXT", p_neg=p_neg))
    if tags:
        reasons.append(_reason("ISSUE_TAGS", tags=", ".join(t.replace("_", " ") for t in tags)))

    # ---- step 2: rules when text and photo can be compared
    if image is not None:
        domain = image["domain"]
        bad = bool(image["is_bad"])
        bad_label, good_label = ("damage" if domain == "packages" else "rot"), \
                                ("intact" if domain == "packages" else "fresh")
        claims = [t for t in tags if t in VERIFIABLE_TAGS.get(domain, set())]
        if bad:
            reasons.append(_reason("PHOTO_PROBLEM", bad_label=bad_label, p_bad=float(image["p_bad"])))
        if bad and text_label == "negative":
            status = "URGENT"
            reasons.insert(0, _reason("BOTH_PROBLEM"))
        elif bad and text_label == "positive":
            status = "REVIEW"
            reasons.insert(0, _reason("PHOTO_CONTRADICTS_REVIEW", bad_label=bad_label))
        elif bad:
            status = max(status, "REVIEW", key=STATUS_ORDER.index)
        elif claims:
            status = "REVIEW"
            reasons.insert(0, _reason("CLAIM_NOT_VISIBLE", claim=claims[0].replace("_", " "),
                                      good_label=good_label))
        elif text_label == "negative":
            status = "REVIEW"
            reasons.insert(0, _reason("ISSUE_NOT_VISUAL", good_label=good_label))

    if status == "OK" and all(r["code"] == "NO_PHOTO" for r in reasons):
        reasons.append(_reason("LOOKS_FINE"))
    return Decision(status=status, risk=round(float(risk), 4), reasons=reasons, used_photo=image is not None,
                    details={"text_label": text_label, "p_negative": round(p_neg, 4), "tags": tags,
                             "image": image, "category": category})


def flag_order(review_text: str, category: str, image=None, predictor=None) -> dict[str, Any]:
    """One-call API for outsiders: review text (+ optional PIL image) -> decision dict.

        from src.fusion import flag_order
        flag_order("Box arrived crushed, want a refund", "electronics", Image.open("photo.jpg"))

    Categories: "electronics" (package photo model), "grocery" (produce photo model), anything else
    uses the package model for a photo, or text only without one. Pass a shared `Predictor` when
    flagging many orders, so the models are loaded only once.
    """
    if predictor is None:
        from src.inference import Predictor

        predictor = Predictor()
    text = predictor.text(review_text)
    img = predictor.image(image, category) if image is not None else None
    decision = fuse(text["probs"], text["tags"], category, img)
    out = decision.to_dict()
    out["text"] = text
    return out


# ------------------------------------------------------------ simulated orders --
# There is no public dataset of orders with BOTH a review and a customer photo, so we SIMULATE
# orders by pairing a test review with a test photo of the matching product type. The pairs are
# artificial (the review and the photo come from different customers); this is stated wherever
# the results are shown. Only TEST splits are used, so no model has seen any of these inputs.
# Ground truth "problematic order" = the photo's true label is bad OR the review's true label is negative.

def build_simulated_orders(seed: int | None = None):
    """Two disjoint sets of simulated orders: `val` (to tune weights) and `test` (to report results).

    Each set has n/2 electronics orders (electronics reviews + package photos) and n/2 grocery orders
    (grocery reviews + produce photos). The mix of (review label, photo label) combinations is set in
    config.yaml -> fusion.pairing_mix, so every 2x2 case appears often enough to measure.
    """
    import numpy as np
    import pandas as pd

    from src.config import path

    cfg = load_config()
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    n = cfg["fusion"]["n_simulated_orders"]
    mix = cfg["fusion"]["pairing_mix"]
    reviews = pd.read_csv(path("data_processed") / "text_test.csv")
    domains = {"electronics": ("packages", reviews[reviews["category"] == "electronics"]),
               "grocery": ("produce", reviews[reviews["category"] == "grocery"])}

    sets = {"val": [], "test": []}
    for category, (domain, revs) in domains.items():
        manifest = pd.read_csv(path("data_processed", "images") / f"{domain}_manifest.csv")
        imgs = manifest[manifest["split"] == "test"]
        bad_label = {"packages": "damaged", "produce": "rotten"}[domain]
        used_reviews, used_images = set(), set()
        for name in ("val", "test"):
            for combo, share in mix.items():
                text_label, photo = combo.split("+")
                k = int(round(share * n / 2))
                r_pool = revs[(revs["label"] == text_label) & ~revs["review_id"].isin(used_reviews)]
                is_bad = imgs["label"] == bad_label
                i_pool = imgs[(is_bad if photo == "bad" else ~is_bad) & ~imgs["path"].isin(used_images)]
                r = r_pool.iloc[rng.choice(len(r_pool), k, replace=False)]
                i = i_pool.iloc[rng.choice(len(i_pool), k, replace=False)]
                used_reviews.update(r["review_id"])
                used_images.update(i["path"])
                sets[name].append(pd.DataFrame({
                    "category": category, "review_id": r["review_id"].to_numpy(), "text": r["text"].to_numpy(),
                    "true_text": text_label, "image_path": i["path"].to_numpy(), "image_source": i["source"].to_numpy(),
                    "true_photo": photo}))
    out = {}
    for name, parts in sets.items():
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        df.insert(0, "order_id", [f"SIM-{name[0].upper()}{i:03d}" for i in range(len(df))])
        df["true_problem"] = (df["true_text"] == "negative") | (df["true_photo"] == "bad")
        out[name] = df
    return out["val"], out["test"]


def score_orders(orders, predictor):
    """Run the deployed models (ONNX) on every simulated order: text probabilities, tags, photo P(bad)."""
    import numpy as np
    from PIL import Image

    from src.config import path

    orders = orders.copy()
    probs = predictor.text_model.predict_proba(orders["text"].tolist())
    for j, lab in enumerate(predictor.text_model.labels):
        orders[f"p_{lab}"] = probs[:, j]
    orders["pred_text"] = np.array(predictor.text_model.labels)[probs.argmax(1)]
    from src.issue_tags import tag_list

    orders["tags"] = orders["text"].map(tag_list)
    root = path("data_processed", "images")
    images = [predictor.image(Image.open(root / p), c) for p, c in zip(orders["image_path"], orders["category"])]
    orders["image_domain"] = [i["domain"] for i in images]
    orders["p_bad"] = [i["p_bad"] for i in images]
    orders["photo_bad"] = [i["is_bad"] for i in images]
    orders["photo_threshold"] = [i["threshold"] for i in images]
    return orders


def decide_all(scored, settings: dict | None = None, use_photo: bool = True):
    """Fusion decision for every scored order (optionally ignoring the photos)."""
    out = []
    for r in scored.itertuples(index=False):
        probs = {"negative": r.p_negative, "neutral": r.p_neutral, "positive": r.p_positive}
        image = ({"domain": r.image_domain, "p_bad": r.p_bad, "is_bad": r.photo_bad, "threshold": r.photo_threshold}
                 if use_photo else None)
        out.append(fuse(probs, r.tags, r.category, image, settings))
    return out


def flag_metrics(y_true, flagged) -> dict[str, float]:
    """How well 'flagged for a human' (REVIEW or URGENT) matches 'really problematic'."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {"precision": round(float(precision_score(y_true, flagged, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, flagged, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, flagged, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_true, flagged)), 4),
            "flag_rate": round(float(sum(flagged) / len(flagged)), 4)}


def compare_strategies(scored, settings: dict | None = None):
    """Text only vs photo only vs combined (score only, and score + rules) on the same orders."""
    import pandas as pd

    y = scored["true_problem"].to_numpy()
    s = settings or load_config()["fusion"]
    w = s["weights"]
    score_only = (w["text"] * scored["p_negative"] + w["image"] * scored["p_bad"]
                  + w["tag"] * scored["tags"].map(bool).astype(float)) >= s["thresholds"]["review"]
    rows = {
        "text only (review negative)": flag_metrics(y, (scored["pred_text"] == "negative").to_numpy()),
        "text only (fusion without photo)": flag_metrics(y, [d.status != "OK" for d in decide_all(scored, s, False)]),
        "photo only (photo model says bad)": flag_metrics(y, scored["photo_bad"].to_numpy()),
        "combined: weighted score only": flag_metrics(y, score_only.to_numpy()),
        "combined: score + rules (ours)": flag_metrics(y, [d.status != "OK" for d in decide_all(scored, s)]),
    }
    return pd.DataFrame(rows).T


def tune_weights(scored_val, tolerance: float = 0.01) -> tuple[dict, float, dict]:
    """Choose weights and the REVIEW threshold on the simulated VALIDATION orders.

    1. Grid search: flag F1 for every combination of weights (step 0.05) and review threshold.
    2. Among all combinations within `tolerance` of the best F1, take the most BALANCED weights
       (largest smallest weight), then the review threshold closest to 0.40.
    Why step 2: because the rules already use the photo's yes/no decision, many weightings score
    almost the same, and the single best one tends to be extreme (e.g. image weight 0). With 300
    orders such small differences are noise, and a balanced score keeps every signal visible in
    the risk number shown on the website.
    The URGENT threshold stays fixed: there is no ground truth for "how urgent", only for
    "problematic or not". Returns (settings, val F1 of the chosen settings, info about the search).
    """
    import numpy as np

    base = load_config()["fusion"]
    y = scored_val["true_problem"].to_numpy()
    results = []
    for w_text in np.arange(0.0, 1.0001, 0.05):
        for w_tag in np.arange(0.0, 0.3001, 0.05):
            w_img = round(1.0 - w_text - w_tag, 2)
            if w_img < 0 or round(w_text + w_tag, 2) == 0:
                continue  # weights must add up to 1, and the no-photo decision needs some text weight
            for review in np.arange(0.20, 0.6001, 0.05):
                s = {"weights": {"text": round(float(w_text), 2), "image": float(w_img), "tag": round(float(w_tag), 2)},
                     "thresholds": {"review": round(float(review), 2), "urgent": base["thresholds"]["urgent"]}}
                f1 = flag_metrics(y, [d.status != "OK" for d in decide_all(scored_val, s)])["f1"]
                results.append((f1, s))
    best_f1 = max(f for f, _ in results)
    near = [(f, s) for f, s in results if f >= best_f1 - tolerance]
    f1, chosen = max(near, key=lambda fs: (min(fs[1]["weights"].values()), -abs(fs[1]["thresholds"]["review"] - 0.4), fs[0]))
    single_best = max(results, key=lambda fs: fs[0])[1]
    return chosen, f1, {"best_val_f1": best_f1, "single_best_settings": single_best, "n_within_tolerance": len(near),
                        "tolerance": tolerance}


# ------------------------------------------------------------ experiment + plots --

def run_fusion_experiment(predictor=None, save: bool = True) -> dict:
    """Whole Phase 6 experiment: build simulated orders, score them with the deployed models,
    tune weights on the validation orders, and compare strategies on the test orders.

    Saves results/fusion_simulated_test.csv (every test order with its decision) and the numbers
    to results/metrics.json -> fusion. Returns everything for the notebook.
    """
    import pandas as pd

    from src.config import path
    from src.evaluate import save_metrics

    if predictor is None:
        from src.inference import Predictor

        predictor = Predictor(threads=4)
    val, test = build_simulated_orders()
    sv, st = score_orders(val, predictor), score_orders(test, predictor)
    tuned, val_f1, info = tune_weights(sv)
    comparison = compare_strategies(st, tuned)
    decisions = decide_all(st, tuned)
    st["status"] = [d.status for d in decisions]
    st["risk"] = [d.risk for d in decisions]
    st["reasons"] = [" | ".join(r["text"] for r in d.reasons) for d in decisions]
    status_table = pd.crosstab([st["true_text"], st["true_photo"]], st["status"]).reindex(columns=STATUS_ORDER,
                                                                                          fill_value=0)
    if save:
        st.drop(columns=["tags"]).assign(tags=st["tags"].map(", ".join)).to_csv(
            path("results", create=True) / "fusion_simulated_test.csv", index=False)
        save_metrics("fusion", "simulated_orders", {
            "n_val": len(sv), "n_test": len(st), "problematic_share_test": round(float(st["true_problem"].mean()), 4),
            "tuned_settings": tuned, "val_f1_chosen": val_f1, "tuning_info": info,
            "comparison_test": comparison.to_dict(orient="index"),
            "note": "Orders are SIMULATED: each pairs a test review with a test photo from different customers."})
    return {"val": sv, "test": st, "tuned": tuned, "val_f1": val_f1, "tuning_info": info,
            "comparison": comparison, "status_table": status_table}


def plot_strategy_comparison(comparison, filename: str = "fusion_strategy_comparison"):
    """Precision, recall and F1 of the 'flag this order' decision for each strategy."""
    import matplotlib.pyplot as plt
    import numpy as np

    from src.plots import INK, apply_style, save

    apply_style()
    metrics = [("precision", "#2a78d6"), ("recall", "#eb6834"), ("f1", "#1baf7a")]  # validated categorical slots 1-3
    names = list(comparison.index)
    y = np.arange(len(names))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(names) + 1.2))
    for k, (m, color) in enumerate(metrics):
        vals = comparison[m].to_numpy()
        ax.barh(y + (k - 1) * h, vals, height=h - 0.03, color=color, label=m)
        for yy, v in zip(y + (k - 1) * h, vals):
            ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_title("Flagging problematic orders (simulated test orders)", loc="left", pad=28)
    fig.tight_layout()
    save(fig, filename)
    return fig


def plot_demo_orders(filename: str = "fusion_demo_orders"):
    """Grid of the demo orders: photo (if any), status and the main reason."""
    import textwrap

    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image

    from src.config import ROOT
    from src.plots import INK, apply_style, save

    apply_style()
    orders = pd.read_csv(ROOT / "demo_data" / "orders.csv").fillna("")
    expected = pd.read_csv(ROOT / "demo_data" / "expected.csv").set_index("order_id")
    badge = {"OK": "#0ca30c", "REVIEW": "#fab219", "URGENT": "#d03b3b"}   # status colours, always with the word
    cols = 4
    rows = -(-len(orders) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 4.1))
    for ax, r in zip(axes.ravel(), orders.itertuples()):
        e = expected.loc[r.order_id]
        ax.axis("off")
        if r.image:
            ax.imshow(Image.open(ROOT / "demo_data" / r.image))
        else:
            ax.text(0.5, 0.5, "no photo", ha="center", va="center", color="#8a8984", transform=ax.transAxes)
        ax.set_title(f"{r.order_id} · {r.category}", loc="left", fontsize=9, color=INK)
        ax.text(0.02, 0.97, f" {e.status} ", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold",
                color="white" if e.status != "REVIEW" else INK, bbox=dict(facecolor=badge[e.status], edgecolor="none"))
        review = textwrap.shorten(r.review_text, 90, placeholder="…")
        reason = e.reasons.split(" | ")[0]
        ax.text(0, -0.04, "\n".join(textwrap.wrap(f"“{review}”", 42) + [""] + textwrap.wrap("→ " + reason, 42)),
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
    for ax in axes.ravel()[len(orders):]:
        ax.axis("off")
    fig.tight_layout(h_pad=7)
    save(fig, filename)
    return fig
