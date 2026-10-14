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

- **5 sources, 810k raw reviews → 641k after cleaning → 45,000 balanced** (15k per class): 31,499 train / 6,750 val / 6,751 test.
- **Two label leaks were found and removed**: Flipkart's rating-generated `Review` titles and Amazon's automatic "Five Stars" titles. Keeping them would have inflated accuracy without the model really understanding sentiment.
- Raw data is heavily skewed to positive (60–80% five stars), so balancing is essential; neutral (3 stars) is the rarest and hardest class.
- Negative and neutral reviews are longer than positive ones (median 23–24 vs 17 words): unhappy customers explain what went wrong.
- The most typical negative words (waste, threw, garbage, expired, moldy, returnable) point at concrete problems, which is exactly what the issue tagger and the fusion step build on. Neutral reviews are recognisable by hedging words (ok, meh, average, decent, somewhat, kinda).
- The external Customer Feedback set is tiny (75 reviews after removing duplicates: 42 positive, 33 negative), has only positive/negative labels and is mostly not about products (music, films, restaurants), so it is a test of how well the model handles **unfamiliar text**, scored on positive vs negative only.
"""),
    ]


def nb_02_text_models():
    return [
        md("""
# 02 · Text models: which sentiment model works best?

**Goal:** compare all sentiment models on the same test set and explain what they learned.

**Models:**
- *Ours, trained locally:* TextBlob (rules, course baseline), CountVectorizer + Naive Bayes, TF-IDF + Logistic Regression, MiniLM embeddings + Logistic Regression.
- *Ready-made from the internet, used as they are:* Cardiff RoBERTa (3 classes, trained on tweets), nlptown BERT (1–5 stars), the default Hugging Face `pipeline` model (positive/negative only).
- *Fine-tuned on our data (Colab GPU):* DistilBERT, and the Cardiff RoBERTa improved on our reviews.

**Inputs:** `data/processed/`, `results/metrics.json` (written by `scripts/train_text_models.py` and the Colab notebook).
**Outputs:** comparison figures in `results/figures/`.

Main metric: **macro-F1** (average of the three class F1 scores, so neutral counts as much as the others).
Every model picked its settings on the validation set; the test set was used once, at the end.
"""),
        code(SETUP),
        md("## 1. Comparison table"),
        code("""from src.evaluate import comparison_table, import_colab_metrics
print("Imported from Colab:", import_colab_metrics() or "nothing yet")
table = comparison_table("text")
table"""),
        code("""from src.evaluate import plot_model_comparison
plot_model_comparison(table, "test macro-F1");"""),
        md("""
**External test (Customer Feedback, 75 short reviews, positive vs negative only).** These are short, very
clear sentences, mostly not about products. It checks whether a model still works on unfamiliar text.
"""),
        code("""plot_model_comparison(table, "external F1 (pos/neg)", filename="text_model_comparison_external");"""),
        md("## 2. Results per source (test set macro-F1)"),
        code("""from src.evaluate import load_metrics
per_source = pd.DataFrame({name: {src: v["macro_f1"] for src, v in m["test_per_source"].items()}
                           for name, m in load_metrics()["text"].items()}).T
per_source.loc[table.index]"""),
        md("## 3. Confusion matrices: where do the models get confused?"),
        code("""from pathlib import Path
figs = Path("../results/figures")
for name in table.index:
    f = figs / f"cm_text_{name}.png"
    if f.exists():
        display(Image(filename=str(f), width=330))"""),
        md("""
## 4. Explainability: what does TF-IDF + Logistic Regression look for?

Logistic Regression gives every word (and word pair) a weight per class. The biggest weights are
the words that push a review towards that class. Word pairs like "not worth" or "stopped working"
are the reason we used 1–2 grams.
"""),
        code("""import joblib
from src.config import path
from src.text_models import TfidfLogRegModel
from src.evaluate import plot_top_words
tfidf = TfidfLogRegModel()
tfidf.pipe = joblib.load(path("models_train") / "text_tfidf_logreg.joblib")
plot_top_words(tfidf.top_words(15));"""),
        md("""
## 5. Error analysis: the most confident mistakes

Reviews where TF-IDF + LR was *sure* and still wrong. These show the limits of a word-counting model and of star-based labels.
"""),
        code("""from src.evaluate import misclassified_examples
test = pd.read_csv("../data/processed/text_test.csv")
misclassified_examples(test, "tfidf_logreg", n=10)"""),
        md("""
## 6. Issue tagger: *what* went wrong

Sentiment says how unhappy a customer is; the tagger (`src/issue_tags.py`) finds concrete problems
and shows the exact phrase that triggered each tag. It handles negation ("the box was **not** damaged" → no tag).
"""),
        code("""from src.issue_tags import tag_issues
examples = [
    "The box arrived crushed and the jar inside leaked everywhere. I want a refund.",
    "Wrong size, not what I ordered. Returned it.",
    "Stopped working after a week, and the replacement never arrived.",
    "The bread was moldy and past its best before date.",
    "Box was not damaged at all, great packaging, fast delivery!",
]
pd.DataFrame({"review": examples, "tags found (phrase)": [tag_issues(e) for e in examples]})"""),
        md("""
There are no ground-truth tags, so as a sanity check we look at how often each tag fires per sentiment class.
A good tagger fires far more on negative reviews than on positive ones.
"""),
        code("""from src.issue_tags import tag_rates
(tag_rates(test) * 100).round(1).rename(columns=lambda c: c + " %")"""),
        md("""
## Key findings

- **Best model: RoBERTa fine-tuned on our reviews (macro-F1 0.790)**, then fine-tuned DistilBERT (0.768). Both also score 1.00 on the external set.
- **Fine-tuning is what makes the difference:** the same Cardiff RoBERTa scores 0.609 used as it is and **0.790 after 2 epochs on our data** (+0.18);
  its neutral F1 goes from 0.31 to 0.69. Downloading a strong model is a good start, but it has to learn *our* labels.
- **Among models without fine-tuning, simple beats big:** TF-IDF + Logistic Regression (**0.731**) and Naive Bayes (0.724) beat every
  ready-made internet model, including a 670 MB BERT (nlptown, 0.715), while being ~10,000× faster (0.01 vs 125 ms per review) and 200× smaller.
- **Accuracy vs cost:** the fine-tuned transformers are ~6 points better than TF-IDF but ~100–500 MB and ~100–150 ms per review on CPU (same
  architectures as the HF benchmarks timed here), which matters for a small web server.
- **Ready-made models struggle with "neutral":** the Cardiff RoBERTa reaches F1 0.31 on neutral, and the default HF pipeline
  can't predict neutral at all (macro-F1 0.48). What counts as "neutral" depends on the data, so models trained on someone else's data don't know our definition.
- **But ready-made models are strong on clear, unfamiliar text:** on the external Customer Feedback set, both RoBERTa and nlptown score 1.00 and MiniLM 0.99,
  while TF-IDF drops to 0.88: word counts only know the vocabulary of our training data.
- **Neutral is the hardest class for every model** (TF-IDF: 62% recall). Negative and positive are almost never swapped (3–4%); the mistakes are mostly "neighbouring" classes.
- **Many "confident mistakes" are label noise**: "pent are perfect" or "very bad" rated 3 stars. Star ratings are a cheap but imperfect label, which caps every model's score.
- **Explainability:** TF-IDF + LR looks for sensible words; the negation phrases we kept on purpose ("not good", "not worth", "not work") are among the strongest negative signals, and hedging words ("but", "ok", "however", "though") drive neutral.
- **Issue tagger:** 27% of negative reviews get at least one tag vs 1.8% of positive ones; refunds/returns and defective products are the most common problems.
"""),
    ]


def nb_04_image_data():
    return [
        md("""
# 04 · Image data: packages (damaged vs intact) and produce (rotten vs fresh)

**Goal:** build clean, leak-free image datasets for the two photo models.

| Domain | Business question | Sources |
|---|---|---|
| packages | was the order damaged in shipping? | Kaggle *Industrial Quality Control of Packages* (3D renders) + Roboflow *Damaged vs good packages* (CC BY 4.0) + Roboflow *BoxCondition* (CC BY 4.0, added because of the bias described below) |
| produce | is the food spoiled? | Kaggle *Fruits fresh and rotten for classification* (apple, banana, orange) |

**Inputs:** `data/raw/images/` (from `scripts/download_data.py images`).
**Outputs:** `data/processed/images/<domain>/{train,val,test}/<class>/` (224×224 JPEG), a manifest CSV per domain, figures in `results/figures/`.

Steps (all in `src/data_images.py`): index → remove unreadable files → **remove copies** → resize to 224×224 → `splitfolders` 70/10/20 (seed 42) → final check that no val/test image is a copy of a train image.
"""),
        code(SETUP),
        md("""
## 1. The hidden problem: copies

If a photo and an edited copy of it land in train and test, the model has effectively seen the
test answer and its accuracy is inflated. These datasets are full of copies: the fruit set was
"augmented" by its author (rotated by random angles, flipped, cropped, with noise added).

**Attempt 1: perceptual hash** (a 64-bit fingerprint, compared in all 8 orientations). It failed both ways:
- on the Kaggle package renders (one box on one conveyor belt) it calls *different* boxes copies,
  even a damaged and an intact one;
- on the fruits, copies rotated by e.g. 20° look as "different" as genuinely different fruits.
"""),
        code("""from src.data_images import index_domain, closest_pairs, plot_pairs
raw_packages = index_domain("packages")
kaggle_only = raw_packages[raw_packages["source"] == "packages_kaggle"]
pairs = closest_pairs(kaggle_only, method="phash", n=3, same_label=False, band=(0, 4))
plot_pairs(pairs, "Hash: different renders (and labels) look like copies", "image_phash_kaggle_lookalikes", "hash distance");"""),
        md("""
**Attempt 2: CLIP embeddings** grouped *similar-looking* fruits (hundreds of different bananas
into one group), not *copies*.

**Final method: SSCD** (Meta AI, MIT licence), a model trained specifically for **copy detection**.
It gives an image and its edited copies nearly the same fingerprint. Two images count as copies if
their SSCD similarity is ≥ 0.65 (chosen by looking at pairs: between 0.6 and 0.75 almost all pairs
are rotated/flipped/noised copies). The Kaggle renders get a stricter rule (only exact copies, ≥ 0.99),
because SSCD also sees every render of that one scene as similar.

Copies are grouped with "leader" clustering (each image joins a group only if it is directly similar
to that group's first image), so copy-of-a-copy chains can't merge different photos. One image per group is kept.
"""),
        code("""from src.data_images import copy_group_example, plot_images
raw_produce = index_domain("produce")
plot_images(copy_group_example(raw_produce), "One original fruit photo and its copies in the raw dataset", "image_produce_copy_group");"""),
        code("""pairs = closest_pairs(raw_produce.sample(3000, random_state=42), method="sscd", n=4, band=(0.65, 0.8))
plot_pairs(pairs, "SSCD: pairs just above the copy threshold", "image_produce_sscd_pairs");"""),
        md("## 2. Packages: build the dataset"),
        code("""from src.data_images import build_image_dataset
packages = build_image_dataset("packages", verbose=False)
print("Raw images per source and class:"); display(packages["raw_counts"])
print("Unreadable files:", len(packages["broken"]))
print("Copies:", packages["duplicate_report"])
print("Val/test copies of train images found after the split (removed):", packages["cross_split_copies_removed"])
print("Final counts per source, class and split:"); display(packages["split_counts"])"""),
        code("""from src.data_images import plot_sample_grid
plot_sample_grid(packages["manifest"], "packages", group_col="source");"""),
        md("## 3. Produce: build the dataset"),
        code("""produce = build_image_dataset("produce", verbose=False)
print("Raw images per class:"); display(produce["raw_counts"])
print("Unreadable files:", len(produce["broken"]))
print("Copies:", produce["duplicate_report"])
print("Val/test copies of train images found after the split (removed):", produce["cross_split_copies_removed"])
print("Final counts per class and split:"); display(produce["split_counts"])
print("Images per fruit and class:"); display(produce["subtype_counts"])"""),
        code("""plot_sample_grid(produce["manifest"], "produce", group_col="subtype");"""),
        md("""
## Key findings

- **Copies were the main data problem.** 84% of the fruit images (11,372 of 13,599) were rotated, flipped,
  cropped or noised copies of other images. Kept in, copies of training photos would sit in the test set and
  inflate accuracy. After removing them: **2,227 different fruit photos** (930 fresh, 1,297 rotten), split
  1,558 / 222 / 447. The Kaggle zip also contained the whole dataset twice.
- **Choosing the right copy detector mattered:** the perceptual hash missed rotated copies and wrongly flagged
  different renders; CLIP grouped similar-looking fruits; SSCD, a model made for copy detection, separated copies
  from different photos (checked by eye at several similarity levels).
- **Packages: 4,953 → 2,195 images** (2,758 copies removed, mostly inside BoxCondition; 80 photos appeared in both
  Roboflow sets and were merged; 4 groups with conflicting labels dropped), split 1,535 / 218 / 442.
  No unreadable files in either domain, and no val/test image is a copy of a train image.
- **Dataset bias, and the fix:** in the first Roboflow set "intact" is mostly shiny product boxes on studio backgrounds
  and "damaged" mostly brown shipping cardboard (plus some label noise); the Kaggle packages are 3D renders of one box
  on one conveyor belt. A model could learn "brown cardboard = damaged". We therefore added **BoxCondition**, where both
  classes are brown cardboard boxes (650 damaged / 649 intact after copy removal), so the model has to learn the damage
  itself. Grad-CAM in Phase 4 will show where the model looks, and results are reported per source.
"""),
    ]


COLAB_SETUP = '''# ---- Setup: GPU check, Google Drive, code, packages ----
import os, subprocess, sys
from pathlib import Path

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout.strip()
assert gpu, "No GPU! Menu: Runtime -> Change runtime type -> T4 GPU, then run again."
print("GPU:", gpu)

from google.colab import drive, userdata
drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/rrf")          # project folder on your Drive
REPO = Path("/content/application-of-text-qualification")

# Get the project code. Two ways, because the GitHub repo is private:
#  A) code.zip on Drive (MyDrive/rrf/code.zip, made with `python scripts/make_code_zip.py`): no token needed
#  B) git clone with a token saved in Colab "Secrets" (key icon, left sidebar) as GITHUB_TOKEN
CODE_ZIP = DRIVE / "code.zip"
if CODE_ZIP.exists():
    import shutil, zipfile
    shutil.rmtree(REPO, ignore_errors=True)
    zipfile.ZipFile(CODE_ZIP).extractall(REPO)
    print("code from", CODE_ZIP)
elif not REPO.exists():
    token = userdata.get("GITHUB_TOKEN")
    subprocess.run(["git", "clone", "-q",
                    f"https://{token}@github.com/Flaxxxy/application-of-text-qualification.git", str(REPO)],
                   check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull", "-q"], check=True)
os.chdir(REPO)
sys.path.insert(0, str(REPO))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
print("ready:", REPO)'''


def nb_colab_image(domain: str, number: str):
    bad, good = {"packages": ("damaged", "intact"), "produce": ("rotten", "fresh")}[domain]
    question = {"packages": "was the package damaged in shipping?", "produce": "is the fruit rotten?"}[domain]
    return [
        md(f"""
# Colab {number} · Image model: {domain} ({bad} vs {good})

**Business question:** {question}

**Goal:** train and compare two models on the {domain} photos:
- **small CNN from scratch** (course architecture: Conv2D, MaxPooling, Dropout, Flatten, Dense softmax);
- **MobileNetV2 transfer learning** (ImageNet weights): base frozen first, then the top 30 layers fine-tuned. This one goes to the website.

Both use augmentation (flip, rotation, zoom, brightness, contrast), class weights and early stopping. The decision
threshold is tuned on the validation set so that at least 90% of {bad} photos are caught. Grad-CAM shows where
the model looks.

**Runtime:** GPU (Runtime → Change runtime type → T4 GPU). About 15–30 minutes.

**Inputs (Google Drive):** `MyDrive/rrf/code.zip`, `MyDrive/rrf/images.zip`
**Outputs:** copied to `MyDrive/rrf/` (metrics, predictions, figures, and the trained MobileNetV2 model).
"""),
        code(COLAB_SETUP),
        md("## 1. Get the images from Drive"),
        code(f'''import zipfile
zipfile.ZipFile(DRIVE / "images.zip").extractall(REPO)
import pandas as pd
m = pd.read_csv(REPO / "data/processed/images/{domain}_manifest.csv")
pd.crosstab(m["label"], m["split"], margins=True)'''),
        md("## 2. Small CNN from scratch (baseline)"),
        code(f'''from src.config import set_seed
from src.image_models import run_experiment
set_seed()
cnn = run_experiment("{domain}", "small_cnn")'''),
        md("## 3. MobileNetV2 transfer learning"),
        code(f'''mnv2 = run_experiment("{domain}", "mobilenetv2")'''),
        md("## 4. Comparison (test set, tuned threshold)"),
        code(f'''import pandas as pd
rows = {{}}
for name, mtr in [("small_cnn", cnn), ("mobilenetv2", mnv2)]:
    t = mtr["test"]; bad = t["per_class"]["{bad}"]
    rows[name] = {{"accuracy": t["accuracy"], "macro-F1": t["macro_f1"], "recall {bad}": bad["recall"],
                  "precision {bad}": bad["precision"], "threshold": mtr["threshold"],
                  "macro-F1 at 0.5": mtr["test_default"]["macro_f1"], "size MB": mtr["size_mb"]}}
display(pd.DataFrame(rows).T)
display(pd.DataFrame({{n: {{g: v["macro_f1"] for g, v in mtr["test_per_group"].items()}}
                      for n, mtr in [("small_cnn", cnn), ("mobilenetv2", mnv2)]}}))'''),
        md("## 5. Copy results to Drive and list what to bring back"),
        code(f'''import shutil
files = []
for name in ["small_cnn", "mobilenetv2"]:
    files += [f"results/colab/metrics_image_{domain}__{{name}}.json",
              f"results/predictions/image_{domain}_{{name}}_test.csv",
              f"results/figures/cm_image_{domain}_{{name}}.png",
              f"results/figures/history_{domain}_{{name}}.png",
              f"results/figures/gradcam_{domain}_{{name}}.png"]
files.append("models/train/image_{domain}_mobilenetv2.keras")   # needed for the website (Phase 5)
for f in files:
    dst = DRIVE / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / f, dst)

print("Copy these files from Google Drive (MyDrive/rrf/...) into the SAME paths in the project:\\n")
for f in files:
    print("  ", f)'''),
        md(FINDINGS_IMAGE[domain]),
    ]


FINDINGS_IMAGE = {
    "packages": """
## Key findings (run of 2026-10-14; test set of 442 photos, threshold tuned on validation)

| model | macro-F1 | recall damaged | precision damaged |
|---|---|---|---|
| **MobileNetV2** | **0.839** | 0.872 | 0.807 |
| small CNN | 0.509 | 0.910 | 0.520 |
| CLIP zero-shot (no training) | 0.348 | 0.967 | 0.477 |

- **Transfer learning is essential here:** with ~1,500 mixed training photos the CNN from scratch barely beats guessing;
  MobileNetV2's ImageNet features make the difference. A general model with no training (CLIP) is no better than guessing.
- **Per source:** real cardboard photos (BoxCondition) 0.894, Roboflow 0.800, Kaggle 3D renders 0.656. Grad-CAM shows the
  model looks at crushed corners, holes and torn flaps on real photos, but at the conveyor belt on the renders.
- **Fine-tuning the top layers did not help** (validation loss rose from 0.40 to 0.49), so the model from the frozen-base stage
  is kept. The code now checks this automatically.
- Two small CNN and MobileNetV2 runs were repeated on the Mac (CPU) with the same code and seed after improving the threshold
  rule and the stage selection; the MobileNetV2 predictions are identical on CPU and GPU.
""",
    "produce": """
## Key findings (run of 2026-10-14; test set of 447 photos, threshold tuned on validation)

| model | macro-F1 | recall rotten | precision rotten |
|---|---|---|---|
| **MobileNetV2** | **0.977** | 0.962 | 1.000 |
| small CNN | 0.940 | 0.966 | 0.937 |
| CLIP zero-shot (no training) | 0.716 | 0.969 | 0.715 |

- **Rot is easy to see:** even the CNN from scratch reaches 0.94, and MobileNetV2 0.977 with no false alarms on the test set.
- **Per fruit (MobileNetV2):** banana 1.000, orange 0.978, apple 0.952. Grad-CAM highlights the brown/mouldy patches.
- **Fine-tuning helped** (validation loss 0.06 → 0.045), so the fine-tuned model is kept.
- These scores are honest only because 84% of the original images (copies) were removed in Phase 3; with copies in the
  test set the numbers would look even higher but mean less.
""",
}


def nb_colab_03_finetune_text():
    return [
        md("""
# Colab 03 · Fine-tune transformers for review sentiment (DistilBERT + RoBERTa)

**Goal:** fine-tune two transformers on our 31.5k training reviews and score them exactly like the local models.
- **DistilBERT** (`distilbert-base-uncased`): a general language model with a new 3-class head.
- **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`): already a sentiment model (trained on tweets). We tested it *as it is* locally; here we **improve it on our review data**, so the slides can show before vs after.

**Runtime:** GPU (Runtime → Change runtime type → T4 GPU). About 25–35 minutes in total.

**Inputs (on your Google Drive):** `MyDrive/rrf/data/processed/text_{train,val,test,external_test}.csv`

**Outputs:** metrics, predictions and confusion matrices are copied to `MyDrive/rrf/results/`; the fine-tuned models are saved to `MyDrive/rrf/models/` (too big for git).
"""),
        code(COLAB_SETUP),
        md("## 1. Get the processed data from Drive"),
        code('''import shutil
src_dir, dst_dir = DRIVE / "data" / "processed", REPO / "data" / "processed"
dst_dir.mkdir(parents=True, exist_ok=True)
for name in ["train", "val", "test", "external_test"]:
    shutil.copy(src_dir / f"text_{name}.csv", dst_dir)

import pandas as pd
data = {s: pd.read_csv(dst_dir / f"text_{s}.csv") for s in ["train", "val", "test", "external_test"]}
{k: len(v) for k, v in data.items()}'''),
        md("""
## 2. Fine-tune and evaluate both models

The training code is `finetune_transformer()` in `src/text_models.py`; scoring uses the same
`run_and_score()` as every local model, so the numbers are directly comparable.
Speed is not reported here, because a GPU is not comparable to the CPU timings of the local models.
"""),
        code('''import json
from src.config import load_config, path, set_seed
from src.evaluate import plot_confusion_matrix
from src.text_models import FineTunedModel, finetune_transformer, run_and_score, save_predictions

set_seed()
ft = load_config()["text"]["finetune"]
results = {}
for name, base_model in ft["models"].items():
    print(f"\\n===== {name}  ({base_model}) =====")
    out_dir = DRIVE / "models" / f"text_{name}"
    trainer = finetune_transformer(base_model, data["train"], data["val"], f"/content/tmp_{name}",
                                   epochs=ft["epochs"], max_length=ft["max_length"],
                                   batch_size=ft["batch_size"], lr=ft["learning_rate"])
    trainer.save_model(str(out_dir))          # best epoch (by validation macro-F1) + tokenizer
    model = FineTunedModel(name, out_dir, max_length=ft["max_length"])
    metrics, proba = run_and_score(model, data["test"], data["external_test"])
    metrics.update({"ms_per_review": None, "base_model": base_model,
                    "chosen_on_val": {"epochs": ft["epochs"], "lr": ft["learning_rate"],
                                      "best_val_macro_f1": round(trainer.state.best_metric, 4)}})
    path("results", "colab", create=True).joinpath(f"metrics_text_{name}.json").write_text(json.dumps(metrics, indent=2))
    save_predictions(name, data["test"], proba)
    plot_confusion_matrix(metrics["test"]["confusion_matrix"], metrics["test"]["labels"],
                          f"{name}  (macro-F1 {metrics['test']['macro_f1']:.3f})", f"cm_text_{name}")
    results[name] = metrics
    print(f"{name}: test macro-F1 {metrics['test']['macro_f1']:.4f} | external F1 {metrics['external']['macro_f1']:.4f}")'''),
        code('''pd.DataFrame({n: {"test macro-F1": m["test"]["macro_f1"], "test accuracy": m["test"]["accuracy"],
                  "external F1 (pos/neg)": m["external"]["macro_f1"]} for n, m in results.items()}).T'''),
        md("## 3. Copy results to Drive and list what to bring back"),
        code('''files = []
for name in results:
    files += [f"results/colab/metrics_text_{name}.json",
              f"results/predictions/text_{name}_test.csv",
              f"results/figures/cm_text_{name}.png"]
for f in files:
    dst = DRIVE / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / f, dst)

print("Copy these files from Google Drive (MyDrive/rrf/...) into the SAME paths in the repo:\\n")
for f in files:
    print("  ", f)
print("\\nThe fine-tuned models stay on Drive in MyDrive/rrf/models/ (not needed in the repo).")'''),
        md("""
## Key findings

Results of the run on 2026-10-13 (T4 GPU, 2 epochs, lr 2e-5, max 256 tokens), test set of 6,751 reviews:

- **RoBERTa fine-tuned: macro-F1 0.790** (validation 0.793). Used as it is, the same model scored 0.609, so fine-tuning added +0.18; neutral F1 rose from 0.31 to 0.69.
- **DistilBERT fine-tuned: macro-F1 0.768** (validation 0.780), starting from a general language model with a new classification head.
- Both are the best text models in the project and score 1.00 on the external Customer Feedback set. Full comparison: `notebooks/02_text_models.ipynb`.
"""),
    ]


NOTEBOOKS = {
    "01": ("01_text_data.ipynb", nb_01_text_data),
    "02": ("02_text_models.ipynb", nb_02_text_models),
    "04": ("04_image_data.ipynb", nb_04_image_data),
    "colab03": ("colab_03_finetune_text.ipynb", nb_colab_03_finetune_text),
    "colab05": ("colab_05_image_packages.ipynb", lambda: nb_colab_image("packages", "05")),
    "colab06": ("colab_06_image_produce.ipynb", lambda: nb_colab_image("produce", "06")),
}


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
