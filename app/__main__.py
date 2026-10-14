"""Start the website locally:  python -m app   (then open http://127.0.0.1:5000)."""
import os

from app.app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
