"""Website tests with Flask's test client: every page loads, forms validate input, batch works."""

import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "deploy" / "settings.json").exists(),
                                reason="models/deploy not exported yet")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import app.app as webapp

    webapp.DB = tmp_path_factory.mktemp("db") / "test.db"   # never touch the real history
    webapp.init_db(webapp.DB)
    return webapp.app.test_client()


@pytest.mark.parametrize("url", ["/", "/check", "/check?demo=DEMO-05", "/batch", "/history", "/models", "/health"])
def test_pages_load(client, url):
    assert client.get(url).status_code == 200


def test_unknown_page_is_friendly_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404 and b"page doesn" in r.data


def test_figure_route_only_serves_png(client):
    assert client.get("/figures/../config.yaml").status_code == 404


def test_check_order_with_demo_photo(client):
    r = client.post("/check", data={"review_text": "Box arrived crushed and the mug was broken, refund please",
                                    "category": "electronics", "demo": "DEMO-01"})
    assert r.status_code == 200 and b"URGENT" in r.data and b"<mark" in r.data


def test_check_rejects_too_short_review(client):
    r = client.post("/check", data={"review_text": "ok", "category": "electronics"})
    assert b"at least 3 words" in r.data


def test_check_rejects_a_non_image_upload(client):
    r = client.post("/check", data={"review_text": "the box was fine thanks", "category": "grocery",
                                    "photo": (io.BytesIO(b"not an image"), "x.jpg")}, content_type="multipart/form-data")
    assert b"not a readable image" in r.data


def test_batch_demo_orders(client):
    r = client.post("/batch", data={"action": "demo"})
    assert r.data.count(b"data-status=") == 12


def test_batch_csv_upload_with_zip(client):
    import zipfile

    csv = b"order_id,category,review_text\nA1,electronics,box arrived crushed and torn\nA2,grocery,lovely fresh fruit thanks\n"
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.write(ROOT / "demo_data" / "images" / "DEMO-01.jpg", "A1.jpg")
    zbuf.seek(0)
    r = client.post("/batch", data={"action": "upload", "orders": (io.BytesIO(csv), "o.csv"),
                                    "photos": (zbuf, "p.zip")}, content_type="multipart/form-data")
    assert r.data.count(b"data-status=") == 2 and b"URGENT" in r.data


def test_batch_reports_csv_problems(client):
    r = client.post("/batch", data={"action": "upload", "orders": (io.BytesIO(b"id,text\n1,x\n"), "o.csv")},
                    content_type="multipart/form-data")
    assert b"missing" in r.data


def test_history_shows_logged_orders(client):
    client.post("/batch", data={"action": "demo"})
    assert client.get("/history").data.count(b"badge-") >= 12
