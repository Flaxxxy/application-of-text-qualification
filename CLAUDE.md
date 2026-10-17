# Project brief: Retail Review Flagging (text + image classification)

You are helping me (Denis, student) build a course project for "AI in Accounting & Finance" (NTU).
Read this whole file before doing anything. It is the single source of truth for the project.

---

## 1. The assignment (what will be graded)

Business case: an online retailer gets thousands of reviews a day, with text and customer photos. The company wants to:
- classify review text into **positive / neutral / negative**,
- detect **product defects or misdeliveries** from customer photos,
- **combine both** to automatically flag problematic orders.

I chose the coding option. Deliverables:
- Python code (scikit-learn, TensorFlow/Keras, PyTorch, HuggingFace allowed)
- a text classification model (e.g. Naive Bayes, BERT)
- a 3–5 minute screen-recorded video walkthrough (notebooks + live website demo)

Grading criteria — keep these in mind for every decision:
1. Accuracy and performance of models
2. Data preprocessing pipeline
3. Explainability and commentary in code
4. Creativity in combining text insights
5. Usability and reusability of code

## 2. Course alignment (important)

The lecturer's course used these tools and terms. Use them where they fit, so the project clearly matches the course:
- Text: NLTK cleaning (lowercase, stopwords, tokenise, lemmatise), CountVectorizer, TF-IDF, TextBlob polarity, BERT embeddings + logistic regression, HuggingFace `pipeline`
- Images: skimage / OpenCV / PIL preprocessing, `splitfolders` (70/10/20), `ImageDataGenerator` (rescale + augmentation), Keras CNN (Conv2D, MaxPooling, Dropout, Flatten, Dense softmax, adam, categorical cross-entropy), pretrained ImageNet backbones (ResNet, MobileNet, ViT embeddings)
- Evaluation: confusion matrix heatmaps, accuracy, precision/recall/F1
- Deployment: save models (joblib), Flask app with routes + Jinja templates + `static/styles.css`, SQLite logging of predictions, environment variables for secrets, deploy on **Render**

## 3. Final design (already decided — do not change without asking me)

| Part | Compared in notebooks | Deployed in the website |
|---|---|---|
| Text sentiment (3 classes) | TextBlob baseline, CountVectorizer + Naive Bayes, TF-IDF + Logistic Regression, MiniLM sentence embeddings + Logistic Regression, fine-tuned DistilBERT | MiniLM embeddings + LR, exported to ONNX (quantized) |
| Issue tags | keyword/regex lexicon | same |
| Image model A: packages (damaged vs intact) — misdeliveries / shipping damage | small Keras CNN vs transfer learning (MobileNetV2, optionally ResNet50) | MobileNetV2 → ONNX |
| Image model B: produce (fresh vs rotten) — product defects | small Keras CNN vs transfer learning (MobileNetV2, optionally ResNet50) | MobileNetV2 → ONNX |
| Fusion | rule + weighted risk score, tested on simulated orders | same |
| Website | — | Flask, local first, then Render free tier |

**Routing:** the order's product category decides which image model runs.
`electronics` → package model. `grocery` → produce model. Other categories → text only (image optional, package model).

## 4. Hard constraints

- **Deployment target: Render free tier = 512 MB RAM, 0.1 CPU, sleeps after 15 min idle.** Design for this from day one:
  - The app must NOT import `torch`, `tensorflow` or `transformers`. Inference uses only `onnxruntime`, `tokenizers`, `numpy`, `Pillow` (+ `scikit-learn` only if really needed; prefer saving LR weights as `.npy`).
  - Two requirement files: `requirements-train.txt` (everything) and `requirements-app.txt` (slim, app only).
  - Load all models once at app start. gunicorn with 1 worker.
  - Before deploying, measure peak RAM locally (`scripts/check_memory.py` with psutil). Target < 450 MB. If above, tell me; fallback is TF-IDF + LR for the deployed text model.
- **Heavy training runs on Google Colab (free GPU), not locally.** This means the image models and DistilBERT. The light text models (TextBlob, NB, TF-IDF, MiniLM embeddings + LR) can run locally on CPU.
- Random seed 42 everywhere. No hard-coded absolute paths; use `config.yaml`.
- `data/` and large training artifacts are git-ignored. Only small deploy models (`models/deploy/`) are committed (must stay < 100 MB total).

## 5. Data

### 5.1 Text sources (combine them)
Verify every slug/URL before downloading. If one fails, stop and ask me.

| Source | Where | Role | Category |
|---|---|---|---|
| Lecturer: Amazon Review | Kaggle `mehmetisik/amazon-review` | train/val/test | electronics |
| Lecturer: Customer Feedback | Kaggle `vishweshsalodkar/customer-feedback-dataset` | **external test only, never train on it** (very small) | other |
| Amazon Reviews 2023 (McAuley Lab) — Electronics slice | HuggingFace `McAuley-Lab/Amazon-Reviews-2023`, raw review jsonl | train/val/test | electronics |
| Amazon Reviews 2023 — Grocery_and_Gourmet_Food slice | same | train/val/test | grocery |
| Flipkart product reviews | Kaggle `niraliivaghani/flipkart-product-customer-reviews-dataset` | train/val/test (extra negatives/neutrals) | other |

Rules:
- Amazon 2023 files are huge. **Stream** them and keep only a random sample (e.g. first 300k lines, then sample). Never download a full category file. The old HF loading script no longer works; read the raw jsonl directly.
- **One labelling rule for all sources, based on star ratings:** 1–2 = negative, 3 = neutral, 4–5 = positive. For Flipkart use the `Rate` column (clean non-numeric values), NOT its Vader-based `Sentiment` column. For the Customer Feedback set use its own labels (it has no stars) — that is fine because it's only an external test.
- Unified schema: `review_id, source, category, text, rating, label`. Combine title + body where both exist.
- Clean: drop duplicates (exact + normalised text), drop empty/very short texts (< 3 words), strip HTML/URLs.
- Balance: downsample positives so classes are roughly equal. Target size per class configurable in `config.yaml` (start with ~15k per class).
- Split 70/15/15, stratified by `label` and `source`. Save to `data/processed/`.
- Report per-source and per-class counts before and after cleaning.

### 5.2 Image sources
| Model | Source | Notes |
|---|---|---|
| Packages | Kaggle `christianvorhemus/industrial-quality-control-of-packages` | small (side/top views, intact/damaged) |
| Packages | Roboflow Universe `aadhavs-first-workspace/damaged-vs-good-packages` (classification, CC BY 4.0) | needs my Roboflow API key |
| Produce | Kaggle `sriramr/fruits-fresh-and-rotten-for-classification` | collapse fruit classes to binary fresh/rotten, keep fruit type as metadata |
| Demo only | a few customer photos from Amazon Reviews 2023 `images` field | unlabelled, only for the website demo, never for training |

Rules:
- Merge the two package sources into one binary dataset, keeping a `source` column. Report results per source.
- Remove corrupt/unreadable files. **Check for near-duplicates across splits** (perceptual hash, e.g. `imagehash`). Some of these datasets contain augmented copies, which would leak into the test set and inflate the accuracy.
- Resize to 224×224, one folder per class, `splitfolders` 70/10/20 (seed 42).

## 6. Phases (do them in order; stop at every checkpoint)

**Phase 0 — Setup**
- Create the repo structure (section 9), `config.yaml`, both requirement files, `.gitignore`, `PROGRESS.md`, git init.
- Write `scripts/download_data.py` (Kaggle API, Roboflow, HF streaming). Tell me exactly which keys/tokens you need and where to put them (env vars / `.env`, never committed).
- ✅ Checkpoint: show the structure and the list of what you need from me.

**Phase 1 — Text data**
- Download, then inspect each source (shape, columns, sample rows, rating distribution).
- Label mapping, cleaning, combining, balancing, splitting as in 5.1.
- Two preprocessing paths, and explain why in comments:
  - classical models (NB/LR): lowercase, remove punctuation/URLs, NLTK stopwords **but keep negations** (not, no, never, n't), lemmatise;
  - transformer models (MiniLM/DistilBERT): light cleaning only, no stopword removal (the models need full context).
- EDA figures: class balance per source, review length distribution, top words per class.
- ✅ Checkpoint: show the counts tables and figures.

**Phase 2 — Text models**
- TextBlob (polarity thresholds tuned on val), CountVectorizer + MultinomialNB, TF-IDF (1–2 grams) + Logistic Regression (class_weight balanced), MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) embeddings + LR — all locally.
- DistilBERT fine-tuning as a Colab notebook (section 7).
- Main metric: **macro-F1**. Also accuracy, per-class report, confusion matrix. Evaluate on the internal test set, per source, AND on the external Customer Feedback set.
- Explainability: top positive/negative words from the TF-IDF + LR coefficients; examples of misclassified reviews with a short comment on why.
- Issue tagger (`src/issue_tags.py`): regex lexicon with tags `damaged_packaging, wrong_item, missing_item, defective_product, spoiled_food, late_delivery, refund_request`. Simple negation handling ("not broken" ≠ broken). Unit tests.
- Save all results to `results/metrics.json` and figures to `results/figures/`.
- ✅ Checkpoint: comparison table of all text models.

**Phase 3 — Image data**
- Download, clean, dedupe, merge, split as in 5.2. Show sample grids per class.
- ✅ Checkpoint: counts per class/split/source + duplicate check result.

**Phase 4 — Image models (Colab)**
- For each domain (packages, produce): baseline small Keras CNN vs MobileNetV2 transfer learning (frozen base, then fine-tune the top layers). ResNet50 is optional if time allows.
- Augmentation (flip, rotation, brightness, zoom), class weights if imbalanced, early stopping.
- Evaluate: accuracy, precision/recall/F1, confusion matrix. **Focus on recall of the bad class** (damaged / rotten); tune the decision threshold on validation.
- Explainability: Grad-CAM heatmaps on a few correct and wrong examples.
- ✅ Checkpoint: comparison table per domain.

**Phase 5 — Export for deployment**
- Export the deployed models to ONNX: MiniLM (with dynamic int8 quantization), both MobileNetV2 models (tf2onnx). Save tokenizer.json, LR weights, thresholds and preprocessing settings (image size, normalisation) to `models/deploy/`.
- `src/inference.py`: pure numpy/onnxruntime inference (MiniLM mean pooling + L2 normalise done in numpy). No torch/tf imports.
- **Parity test:** the ONNX predictions must match the original models on ~50 samples (report the max difference). Mismatched preprocessing is the most common bug here.
- ✅ Checkpoint: parity results + model file sizes.

**Phase 6 — Fusion (the "creativity" part)**
- `src/fusion.py` takes: text probabilities, issue tags, category, image bad-probability (or none).
- Weighted risk score: `risk = w_text * p_negative + w_img * p_bad + w_tag * has_issue_tag` (weights in config, tuned on simulated-orders validation).
- Status: OK / REVIEW / URGENT, with human-readable **reason codes**. Rule overrides (2×2 logic):
  - negative text + bad image → URGENT ("text and photo both show a problem")
  - positive text + bad image → REVIEW ("photo contradicts positive review")
  - damage/defect tag in text + clean image → REVIEW ("claim not visible in photo — check, possible false claim")
  - negative text + clean image, no damage tag → REVIEW ("service/product issue, not visible damage")
  - no image → text-only decision, says so in the reason.
- Simulated orders: build ~300 orders (half electronics, half grocery) by pairing reviews and images **from the test splits only**. Ground truth "problematic" = true image label is bad OR true text label is negative. State clearly in code and README that the pairs are simulated.
- Compare text-only vs image-only vs combined flagging (precision/recall/F1 on the flag).
- Also create `demo_data/`: ~10 hand-picked orders covering every rule above, for the video.
- ✅ Checkpoint: comparison table + the demo cases.

**Phase 7 — Flask app (local)**
- Pages:
  1. **Home**: short explanation of the business problem and how the system works (simple diagram).
  2. **Check an order**: review text + category dropdown + optional photo upload → sentiment with probabilities, issue tags, image result, final status (colour badge), reason codes.
  3. **Batch**: upload a CSV (`order_id, category, review_text`) plus an optional ZIP of images named by order_id; max 100 rows; results table with filtering by status, and a CSV download. Also a button to run the bundled demo orders.
  4. **History**: the last 50 predictions from SQLite.
  5. **Models**: metrics table and confusion matrices from `results/`.
- Clean, simple design in `static/styles.css`, Jinja templates, no heavy JS frameworks. Upload size limit, input validation, friendly error messages.
- Run `scripts/check_memory.py` and report peak RAM.
- ✅ Checkpoint: screenshots or a description of each page + the RAM number.

**Phase 8 — Deploy to Render**
- `render.yaml`, start command `gunicorn app.app:app --workers 1 --timeout 120`, a `/health` route, env vars. Note in the README that SQLite resets on redeploy (fine for a demo).
- Walk me through the steps I must do myself (GitHub repo, Render account, connecting them).
- ✅ Checkpoint: live URL works for all pages.

**Phase 9 — Polish**
- README: problem, architecture diagram, datasets + licences, how to run (local, Colab, Render), results tables, limitations (simulated pairing, domain shift, small package dataset, star-based labels).
- Re-read all code comments for clarity (see section 8).
- Suggest a 3–5 min video outline based on the final results.

## 7. Colab handoff (how heavy training works)

Colab notebooks (`notebooks/colab_*.ipynb`) must:
- start with a setup cell: mount Google Drive, clone the GitHub repo, `pip install -r requirements-train.txt`, copy processed data from Drive (or download it);
- train, evaluate and save outputs (models, metrics json, figures) to a Drive folder;
- end with a cell that prints exactly which files I need to copy back into the repo and where.

Before each Colab phase, tell me step by step what to do: what to upload to Drive, which notebook to open, the runtime type (GPU), and what to bring back.

## 8. Code quality rules

- Reusable code lives in `src/` as functions with docstrings and type hints. Notebooks only call `src/` and show results. No copy-pasted logic between notebooks.
- Comments explain **why**, not only what, in plain simple English. I will walk through this code on video, so a classmate should be able to follow it.
- Each notebook starts with a short markdown intro (goal, inputs, outputs) and ends with a "key findings" cell.
- Light `pytest` tests for: label mapping, text cleaning, issue tagger, fusion rules, inference parity.
- Never invent or estimate metrics. Every number in the README or app must come from a real run saved in `results/`.

## 9. Repo structure

```
retail-review-flagging/
├── CLAUDE.md
├── README.md
├── PROGRESS.md                # your running log: what was done, decisions, open issues
├── config.yaml
├── requirements-train.txt
├── requirements-app.txt
├── render.yaml
├── data/                      # git-ignored: raw/, interim/, processed/
├── notebooks/
│   ├── 01_text_data.ipynb
│   ├── 02_text_models.ipynb
│   ├── colab_03_distilbert.ipynb
│   ├── 04_image_data.ipynb
│   ├── colab_05_image_packages.ipynb
│   ├── colab_06_image_produce.ipynb
│   └── 07_fusion.ipynb
├── src/
│   ├── config.py
│   ├── data_text.py
│   ├── preprocess_text.py
│   ├── text_models.py
│   ├── issue_tags.py
│   ├── data_images.py
│   ├── image_models.py
│   ├── export_onnx.py
│   ├── inference.py           # deploy-safe: no torch/tf/transformers
│   ├── fusion.py
│   └── evaluate.py
├── models/
│   ├── train/                 # git-ignored
│   └── deploy/                # committed, small ONNX + weights + settings
├── results/                   # metrics.json, figures/
├── app/
│   ├── app.py
│   ├── db.py
│   ├── templates/
│   └── static/styles.css
├── demo_data/
├── scripts/
│   ├── download_data.py
│   └── check_memory.py
└── tests/
```

## 10. How to work with me

- Work one phase at a time. At each ✅ checkpoint, stop and give a short summary: what you did, key numbers, problems, and what's next. Wait for my "go".
- Keep `PROGRESS.md` updated, and make a git commit at the end of each phase.
- Ask before: downloading anything > 2 GB, installing large packages, changing the design in section 3, or dropping a dataset.
- If something fails (broken link, bad labels, memory too high), explain the problem in plain words and propose 1–2 options. Don't silently work around it.
- Write short and simple. I prefer step-by-step explanations alongside the code.

## 11. Additions agreed during the project

- **Repo:** folder and GitHub repo are both `application-of-text-qualification` (personal GitHub, private). The `retail-review-flagging/` name in section 9 just means the repo root.
- **Presentation-ready.** The final result must support a clear live demo: one command to start the app, bundled demo orders that show every fusion rule, and screens a non-technical audience understands.
- **Reusable by outsiders.** Someone who clones the repo must be able to use it without reading the code: a clear README quick start, a simple Python API (e.g. `flag_order(text, category, image)`), a command-line script for single reviews and CSV batches, and pinned app requirements.
- **Compare against existing models.** The presentation should show how our models compare to ready-made models from the internet (see PROGRESS.md for the agreed list).
- **Fusion weights** get a final review after everything else is built; they can be adjusted then.
- **Packages data (2026-10-13):** added Roboflow `sih-yna9n/boxcondition-onzuw` (BoxCondition, CC BY 4.0, classification) as a third package source, because the two planned sources are biased (intact = product boxes, damaged = brown cardboard; Kaggle = 3D renders).
- **Website text model (2026-10-13):** fine-tuned DistilBERT exported to ONNX int8 replaces MiniLM + LR (section 3), if its measured size/RAM/speed fit Render in Phase 5; otherwise fall back to TF-IDF + LR.
- **Copy detection (Phase 3):** SSCD copy-detection embeddings instead of perceptual hashes (hashes missed rotated copies and wrongly merged different renders); see PROGRESS.md.
- **Hosting (2026-10-17):** Heroku (Basic dyno, paid by the GitHub Student Developer Pack credit) instead of Render; `render.yaml` kept as a backup. Same start command (`Procfile`).
