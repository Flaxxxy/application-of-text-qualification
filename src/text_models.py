"""All text sentiment models behind one small interface, so they can be compared fairly.

Every model has:
    fit(train, val)        -> learns from train, picks its settings on val (never on test)
    predict_proba(df)      -> array (n, 3) with P(negative), P(neutral), P(positive)
    predict(df)            -> array of labels
    predict_binary(df)     -> "negative"/"positive" only, for the external set (no neutral there)

Models, from simplest to most complex:
    TextBlobModel       rule-based polarity score, no training (course baseline)
    NaiveBayesModel     word counts + Multinomial Naive Bayes
    TfidfLogRegModel    TF-IDF of words and word pairs + Logistic Regression
    MiniLMLogRegModel   MiniLM sentence embeddings + Logistic Regression  <- deployed in the app
    HFBenchmarkModel    ready-made sentiment models from Hugging Face, used as they are
    finetune_transformer()  fine-tunes DistilBERT / RoBERTa on our data (Colab, GPU)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import load_config, path

LABELS = np.array(["negative", "neutral", "positive"])


def _macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)


class TextModel:
    """Base class: subclasses implement fit() and predict_proba()."""

    name = "base"
    kind = ""            # short description for the comparison table
    chosen: dict = {}    # settings picked on the validation set, for the report

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "TextModel":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return LABELS[self.predict_proba(df).argmax(axis=1)]

    def predict_binary(self, df: pd.DataFrame) -> np.ndarray:
        """Forced choice between negative and positive: whichever of the two is more likely."""
        p = self.predict_proba(df)
        return np.where(p[:, 2] >= p[:, 0], "positive", "negative")

    def size_mb(self) -> float | None:
        return None


# ------------------------------------------------------------------ TextBlob --

class TextBlobModel(TextModel):
    """TextBlob gives a polarity score from -1 (negative) to +1 (positive) using a word list.

    It has no "neutral" output, so we need two cut-offs: below `low` = negative, above
    `high` = positive, in between = neutral. We try many pairs on the validation set and
    keep the pair with the best macro-F1. No learning from our data beyond these 2 numbers.
    """

    name, kind = "textblob", "rule-based baseline"

    def __init__(self):
        self.low, self.high = -0.05, 0.25

    @staticmethod
    def polarity(df: pd.DataFrame) -> np.ndarray:
        from textblob import TextBlob

        return np.array([TextBlob(t).sentiment.polarity for t in df["text"]])

    def _label(self, pol: np.ndarray) -> np.ndarray:
        return np.where(pol < self.low, "negative", np.where(pol > self.high, "positive", "neutral"))

    def fit(self, train, val):
        pol = self.polarity(val)
        best = (-1.0, self.low, self.high)
        for low in np.arange(-0.4, 0.21, 0.025):
            for high in np.arange(low, 0.61, 0.025):
                self.low, self.high = low, high
                score = _macro_f1(val["label"], self._label(pol))
                if score > best[0]:
                    best = (score, low, high)
        _, self.low, self.high = best
        self.chosen = {"low": round(float(self.low), 3), "high": round(float(self.high), 3)}
        return self

    def predict(self, df):
        return self._label(self.polarity(df))

    def predict_proba(self, df):
        # TextBlob has no probabilities; a one-hot of the decision keeps the interface uniform.
        pred = self.predict(df)
        return (pred[:, None] == LABELS[None, :]).astype(float)

    def predict_binary(self, df):
        return np.where(self.polarity(df) >= 0, "positive", "negative")


# --------------------------------------------------------- classical sklearn --

class _SklearnModel(TextModel):
    """Shared code for the two bag-of-words models (they read the `text_classical` column)."""

    grid: dict[str, list] = {}

    def _build(self, **params) -> Pipeline:
        raise NotImplementedError

    def fit(self, train, val):
        # Small grid search on the validation set: the test set stays untouched until the end.
        best_score, best_params = -1.0, None
        for params in _grid(self.grid):
            pipe = self._build(**params).fit(train["text_classical"].fillna(""), train["label"])
            score = _macro_f1(val["label"], pipe.predict(val["text_classical"].fillna("")))
            if score > best_score:
                best_score, best_params, self.pipe = score, params, pipe
        self.chosen = {**best_params, "val_macro_f1": round(best_score, 4)}
        return self

    def predict_proba(self, df):
        proba = self.pipe.predict_proba(df["text_classical"].fillna(""))
        order = [list(self.pipe.classes_).index(lab) for lab in LABELS]
        return proba[:, order]

    def save(self, file: Path) -> None:
        joblib.dump(self.pipe, file)
        self._file = file

    def size_mb(self):
        f = getattr(self, "_file", None)
        return round(f.stat().st_size / 1e6, 1) if f and f.exists() else None


def _grid(grid: dict[str, list]) -> list[dict]:
    combos = [{}]
    for key, values in grid.items():
        combos = [{**c, key: v} for c in combos for v in values]
    return combos


class NaiveBayesModel(_SklearnModel):
    """CountVectorizer (how often each word appears) + Multinomial Naive Bayes.

    Naive Bayes asks: given these word counts, which class most likely wrote this review?
    `alpha` is smoothing for words never seen with a class; tuned on validation.
    """

    name, kind = "naive_bayes", "CountVectorizer + MultinomialNB"
    grid = {"alpha": [0.1, 0.3, 1.0]}

    def _build(self, alpha):
        return Pipeline([
            ("counts", CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
            ("nb", MultinomialNB(alpha=alpha)),
        ])


class TfidfLogRegModel(_SklearnModel):
    """TF-IDF of single words and word pairs + Logistic Regression.

    TF-IDF lowers the weight of words that appear in almost every review ("product") and
    raises distinctive ones ("refund"). Word pairs (bigrams) catch phrases like "not good"
    or "stopped working". class_weight="balanced" protects the classes equally.
    Its coefficients show which words push a review to each class (explainability).
    """

    name, kind = "tfidf_logreg", "TF-IDF (1-2 grams) + Logistic Regression"
    grid = {"C": [0.5, 1.0, 2.0, 4.0]}

    def _build(self, C):
        cfg = load_config()["text"]["models"]["tfidf"]
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=tuple(cfg["ngram_range"]), min_df=cfg["min_df"],
                                      max_features=cfg["max_features"], sublinear_tf=True)),
            ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                      random_state=load_config()["seed"])),
        ])

    def top_words(self, n: int = 15) -> dict[str, pd.Series]:
        """Words with the largest coefficient for each class: what the model 'looks for'."""
        vocab = np.array(self.pipe.named_steps["tfidf"].get_feature_names_out())
        lr = self.pipe.named_steps["lr"]
        out = {}
        for i, lab in enumerate(lr.classes_):
            idx = np.argsort(lr.coef_[i])[::-1][:n]
            out[lab] = pd.Series(lr.coef_[i][idx], index=vocab[idx])
        return out


# ------------------------------------------------------------ MiniLM + LR --

class MiniLMLogRegModel(TextModel):
    """MiniLM turns each review into 384 numbers that capture its meaning; LR classifies them.

    MiniLM (a small, distilled BERT) is used frozen: we don't train it, we only use its
    sentence embeddings. That makes it cheap to train (seconds) and small enough to run on
    the 512 MB website after ONNX export. It reads the light-cleaned `text` column, because
    it needs full sentences.
    """

    name, kind = "minilm_logreg", "MiniLM embeddings + Logistic Regression"

    def __init__(self):
        cfg = load_config()["text"]["models"]
        self.model_name = cfg["minilm"]
        self.max_length = cfg["minilm_max_length"]
        self._encoder = None

    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, device="cpu")
            self._encoder.max_seq_length = self.max_length
        return self._encoder

    def embed(self, df: pd.DataFrame, cache_name: str | None = None) -> np.ndarray:
        """Sentence embeddings, L2-normalised. Cached in data/interim/ because encoding 30k
        reviews takes a few minutes on CPU."""
        cache = path("data_interim", create=True) / f"minilm_{cache_name}_{len(df)}.npy" if cache_name else None
        if cache and cache.exists():
            return np.load(cache)
        emb = self.encoder().encode(df["text"].tolist(), batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=bool(cache_name), convert_to_numpy=True)
        if cache:
            np.save(cache, emb)
        return emb

    def fit(self, train, val, train_emb: np.ndarray | None = None, val_emb: np.ndarray | None = None):
        Xtr = train_emb if train_emb is not None else self.embed(train, "train")
        Xva = val_emb if val_emb is not None else self.embed(val, "val")
        best = (-1.0, None)
        for C in [0.5, 1.0, 2.0, 4.0, 8.0]:
            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                    random_state=load_config()["seed"]).fit(Xtr, train["label"])
            score = _macro_f1(val["label"], lr.predict(Xva))
            if score > best[0]:
                best, self.lr = (score, C), lr
        self.chosen = {"C": best[1], "val_macro_f1": round(best[0], 4)}
        return self

    def predict_proba(self, df, emb: np.ndarray | None = None):
        X = emb if emb is not None else self.embed(df)
        proba = self.lr.predict_proba(X)
        return proba[:, [list(self.lr.classes_).index(lab) for lab in LABELS]]

    def save(self, file: Path) -> None:
        joblib.dump(self.lr, file)
        self._file = file

    def size_mb(self):
        # MiniLM itself (~90 MB as float32) + the tiny LR; ONNX int8 export later shrinks this.
        f = getattr(self, "_file", None)
        lr_mb = f.stat().st_size / 1e6 if f and f.exists() else 0
        n_params = sum(p.numel() for p in self.encoder()[0].auto_model.parameters())
        return round(n_params * 4 / 1e6 + lr_mb, 1)


# --------------------------------------------------- Hugging Face benchmarks --

class HFBenchmarkModel(TextModel):
    """A ready-made sentiment model from Hugging Face, used as it is (no training on our data).

    Each model has its own labels (e.g. "1 star".."5 stars"); config.yaml maps them to ours,
    and probabilities of labels that map to the same class are added up.
    A binary model (positive/negative only) can never say "neutral" — that is a real
    limitation worth showing in the comparison.
    """

    kind = "ready-made (no training)"

    def __init__(self, key: str, batch_size: int = 32):
        bench = load_config()["text"]["benchmarks"][key]
        self.name = f"bench_{key}"
        self.model_name = bench["model"]
        self.label_map = bench["labels"]
        self.batch_size = batch_size
        self._pipe = None

    def pipe(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("text-classification", model=self.model_name, top_k=None,
                                  truncation=True, max_length=512, device="cpu")
        return self._pipe

    def predict_proba(self, df):
        outputs = self.pipe()(df["text"].tolist(), batch_size=self.batch_size)
        proba = np.zeros((len(df), 3))
        col = {lab: i for i, lab in enumerate(LABELS)}
        for row, scores in enumerate(outputs):
            for s in scores:
                proba[row, col[self.label_map[s["label"]]]] += s["score"]
        return proba

    def size_mb(self):
        n_params = sum(p.numel() for p in self.pipe().model.parameters())
        return round(n_params * 4 / 1e6, 1)


# ------------------------------------------------------- evaluation helper --

def run_and_score(model: TextModel, test: pd.DataFrame, external: pd.DataFrame,
                  test_proba: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray]:
    """Predict on test + external, time it, and return (metrics, test probabilities).

    Speed is measured on CPU over the whole test set, as milliseconds per review, because
    the website runs on a small CPU server.
    """
    from src.evaluate import evaluate_text_model

    t0 = time.perf_counter()
    proba = test_proba if test_proba is not None else model.predict_proba(test)
    ms = (time.perf_counter() - t0) * 1000 / len(test)
    pred = model.predict(test) if isinstance(model, TextBlobModel) else LABELS[proba.argmax(axis=1)]
    metrics = evaluate_text_model(model.name, test, pred, external, model.predict_binary(external))
    metrics.update({"type": model.kind, "chosen_on_val": model.chosen,
                    "ms_per_review": None if test_proba is not None else round(ms, 2),
                    "size_mb": model.size_mb()})
    return metrics, proba


def save_predictions(name: str, test: pd.DataFrame, proba: np.ndarray, pred: np.ndarray | None = None) -> Path:
    """Test-set predictions (without the text) for error analysis and the fusion step."""
    out = path("results", "predictions", create=True) / f"text_{name}_test.csv"
    pred = pred if pred is not None else LABELS[proba.argmax(axis=1)]
    pd.DataFrame({"review_id": test["review_id"], "source": test["source"], "label": test["label"],
                  "pred": pred, "p_negative": proba[:, 0].round(4), "p_neutral": proba[:, 1].round(4),
                  "p_positive": proba[:, 2].round(4)}).to_csv(out, index=False)
    return out


# --------------------------------------------- fine-tuning (Colab, on a GPU) --

def finetune_transformer(model_name: str, train: pd.DataFrame, val: pd.DataFrame, out_dir: str | Path,
                         epochs: int = 2, max_length: int = 256, batch_size: int = 32, lr: float = 2e-5,
                         max_steps: int = -1) -> Any:
    """Fine-tune a transformer (DistilBERT, RoBERTa, ...) for our 3 classes with HF Trainer.

    Fine-tuning = keep all the language knowledge from pre-training and adjust every weight a
    little on our reviews. Runs on Colab's GPU (minutes); on a CPU it would take hours.
    The best epoch is chosen by validation macro-F1. `max_steps` > 0 is only for a quick
    smoke test. Returns the trained HF Trainer.
    """
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    seed = load_config()["seed"]
    label2id = {lab: i for i, lab in enumerate(LABELS)}
    id2label = {i: lab for lab, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(model_name)
    # If the model already has a 3-class sentiment head in our order (the Cardiff RoBERTa
    # does: negative, neutral, positive), it is kept, so training starts from its existing
    # sentiment knowledge. Otherwise (DistilBERT) a new head is created.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({"text": df["text"].tolist(), "label": df["label"].map(label2id).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_length), batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"macro_f1": f1_score(labels, preds, average="macro"),
                "accuracy": float((preds == labels).mean())}

    # Warm-up = start with a tiny learning rate for the first 6% of steps, so the new
    # classification head doesn't wreck the pre-trained weights early on.
    # transformers v4 calls it warmup_ratio; v5 removed it and takes a fraction in warmup_steps.
    import inspect

    warmup = ({"warmup_ratio": 0.06} if "warmup_ratio" in inspect.signature(TrainingArguments).parameters
              else {"warmup_steps": 0.06})
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr, weight_decay=0.01, **warmup,
        eval_strategy="epoch" if max_steps < 0 else "steps", eval_steps=None if max_steps < 0 else max_steps,
        save_strategy="epoch" if max_steps < 0 else "steps", save_steps=None if max_steps < 0 else max_steps,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", save_total_limit=1,
        fp16=torch.cuda.is_available(), seed=seed, report_to="none", logging_steps=100,
    )
    trainer = Trainer(model=model, args=args, train_dataset=to_ds(train), eval_dataset=to_ds(val),
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    trainer.train()
    return trainer


class FineTunedModel(TextModel):
    """Wraps a fine-tuned HF model (from finetune_transformer or a saved folder) for prediction."""

    kind = "fine-tuned on our data"

    def __init__(self, name: str, model_dir: str | Path, max_length: int = 256, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size

    def predict_proba(self, df):
        import torch

        texts, out = df["text"].tolist(), []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.vstack(out)

    def size_mb(self):
        return round(sum(p.numel() for p in self.model.parameters()) * 4 / 1e6, 1)
