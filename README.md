# Retail Review Flagging

Course project for "AI in Accounting & Finance" (NTU): classify customer review text (positive / neutral / negative), detect damaged packages and rotten produce in customer photos, and combine both to flag problematic orders.

Work in progress — see [PROGRESS.md](PROGRESS.md). The full README (architecture, datasets, results, how to run) comes in the final phase.

## Quick start (local)

```bash
conda create -n rrf python=3.11 -y
conda activate rrf
pip install -r requirements-train.txt
cp .env.example .env          # then fill in your Kaggle + Roboflow keys
python scripts/download_data.py --check
```
