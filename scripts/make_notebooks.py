"""Generate the project notebooks from plain Python, so they stay short and only call src/.

Usage:  python scripts/make_notebooks.py 01          # write notebooks/01_text_data.ipynb
Then run it:  jupyter nbconvert --to notebook --execute --inplace notebooks/01_text_data.ipynb

Why generate them: notebook JSON is hard to review in git. Keeping the cells here makes
every change readable, and the executed notebook (with outputs) is what we show on video.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SETUP = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # make `src` importable from notebooks/

import pandas as pd
from IPython.display import Image, display
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def nb_01_text_data():
    return [
        md("""
# 01 · Text data: from 5 raw sources to one clean, balanced dataset

**Goal:** build the dataset for the 3-class sentiment models (negative / neutral / positive).

**Inputs:** `data/raw/text/` (downloaded by `scripts/download_data.py text`)
- Lecturer: Amazon Review (one memory card), Customer Feedback (external test only)
- Amazon Reviews 2023: Electronics and Grocery samples (300k reviews each, random chunks)
- Flipkart product reviews

**Outputs:** `data/processed/text_{train,val,test,external_test}.csv`, count tables in `results/`, figures in `results/figures/`.

All logic lives in `src/data_text.py` and `src/preprocess_text.py`; this notebook only calls it and shows results.
"""),
        code(SETUP),
        md("## 1. What does each raw source look like?"),
        code("""from src.data_text import describe_raw_sources
describe_raw_sources()"""),
        md("""
Every source with stars is mostly **positive** (60–80% five stars). Without balancing, a model
could score ~75% accuracy by always answering "positive", so balancing matters.
"""),
        md("""
## 2. Two traps found while inspecting the data

**Trap 1: Flipkart's `Review` column is not written by customers.** Flipkart fills it in from
the star rating. The table shows each common value almost always goes with one rating:
"""),
        code("""from src.data_text import flipkart_title_leak
flipkart_title_leak()"""),
        md("""
If we trained on it, the model would learn to *read the star rating*, not the sentiment, and
look great on paper. So for Flipkart we use only the `Summary` column (the customer's own words).

**Trap 2: Amazon's automatic titles.** ~11% of Amazon 2023 reviews have titles like "Five Stars",
generated from the rating. We drop those titles and keep the body text.
"""),
        code("""from src.data_text import combine_title_body
print(combine_title_body("Five Stars", "Works fine, fast shipping"))
print(combine_title_body("Does not work", "Tried several configurations"))"""),
        md("""
## 3. Labelling, cleaning, deduplication, balancing, splitting

One rule for every source with stars: **1–2 = negative, 3 = neutral, 4–5 = positive**.
Customer Feedback has no stars, so it keeps its own labels; it is used only as an external test.

Cleaning steps (in order): drop rows without a valid label → drop texts under 3 words →
drop texts that appear with *different* labels (ambiguous) → drop duplicates (on normalised
text, across all sources, so a review can't be in both train and test).
"""),
        code("""from src.data_text import build_text_dataset
data = build_text_dataset(save=True, verbose=False)
print("Raw counts per source and label:")
display(data["raw_counts"])
print("What each cleaning step removed:")
display(data["clean_log"])"""),
        md("""
**Balancing:** downsample to 15,000 reviews per class, shared fairly across sources, and within
each source no class may be more than 3× its smallest class (so the model can't learn shortcuts
like "Flipkart style = negative" or "memory card = positive").
"""),
        code("""print("After cleaning:")
display(data["clean_counts"])
print("After balancing:")
display(data["balanced_counts"])
print("Split sizes (70/15/15, stratified by label and source):")
display(data["split_counts"])"""),
        md("""
## 4. Two preprocessing paths

- **Classical models** (Naive Bayes, TF-IDF + LR) only count words → lowercase, remove punctuation,
  remove stopwords **but keep negations** ("not good" must not become "good"), lemmatise.
- **Transformers** (MiniLM, DistilBERT, RoBERTa) read full sentences → light cleaning only (HTML, URLs).
"""),
        code("""data["train"][["label", "text", "text_classical"]].sample(6, random_state=42)"""),
        md("## 5. Exploratory figures"),
        code("""from src.eda import plot_class_balance
plot_class_balance(data["clean_counts"], data["balanced_counts"]);"""),
        code("""from src.eda import plot_length_distribution
plot_length_distribution(data["train"]);"""),
        code("""from src.eda import plot_distinctive_words
plot_distinctive_words(data["train"]);"""),
        md("## 6. Spot-check: example reviews per source and class"),
        code("""from src.eda import sample_reviews
sample_reviews(data["train"], n=1)"""),
        md("## 7. External test set (Customer Feedback)"),
        code("""ext = data["external_test"]
print(len(ext), "reviews after removing duplicates")
display(ext["label"].value_counts())
ext[["label", "text"]].head(8)"""),
        md("""
## Key findings

- **5 sources, 810k raw reviews → 641k after cleaning → 45,000 balanced** (15k per class): 31,499 train / 6,750 val / 6,750 test.
- **Two label leaks were found and removed**: Flipkart's rating-generated `Review` titles and Amazon's automatic "Five Stars" titles. Keeping them would have inflated accuracy without the model really understanding sentiment.
- Raw data is heavily skewed to positive (60–80% five stars), so balancing is essential; neutral (3 stars) is the rarest and hardest class.
- Negative and neutral reviews are longer than positive ones (median 23–24 vs 17 words): unhappy customers explain what went wrong.
- The most typical negative words (waste, threw, garbage, expired, moldy, returnable) point at concrete problems, which is exactly what the issue tagger and the fusion step build on. Neutral reviews are recognisable by hedging words (ok, meh, average, decent, somewhat, kinda).
- The external Customer Feedback set is tiny (75 reviews after removing duplicates: 42 positive, 33 negative), has only positive/negative labels and is mostly not about products (music, films, restaurants), so it is a test of how well the model handles **unfamiliar text**, scored on positive vs negative only.
"""),
    ]


NOTEBOOKS = {"01": ("01_text_data.ipynb", nb_01_text_data)}


def main() -> None:
    keys = sys.argv[1:] or list(NOTEBOOKS)
    for key in keys:
        filename, builder = NOTEBOOKS[key]
        nb = nbf.v4.new_notebook()
        nb.cells = builder()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        out = ROOT / "notebooks" / filename
        nbf.write(nb, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
