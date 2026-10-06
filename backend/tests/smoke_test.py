"""Exercises the real HTTP API end-to-end with a fake (non-ML) embedding
function standing in for sentence-transformers, so this runs without a
model download. Validates: file upload + text extraction, DB upsert,
cross-section duplicate detection, per-component breakdown, filtering."""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embeddings  # noqa: E402


def fake_embed_text(text: str) -> np.ndarray:
    """Bag-of-hashed-words vector — not semantic, but shares the same
    interface and gives sensible relative similarities for this test."""
    vec = np.zeros(embeddings.EMBEDDING_DIM, dtype=np.float32)
    text = (text or "").strip().lower()
    if not text:
        return vec
    for word in text.split():
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % embeddings.EMBEDDING_DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


embeddings.embed_text = fake_embed_text  # patch before main.py imports it by reference

import main  # noqa: E402

main.embed_text = fake_embed_text  # main did `from embeddings import embed_text`

from fastapi.testclient import TestClient  # noqa: E402

main.init_db()  # TestClient doesn't fire startup events unless used as a context manager
client = TestClient(main.app)


def submit(section, team_number, subject, abstract, doc, proto_desc, title=""):
    data = {
        "section": section,
        "team_number": str(team_number),
        "subject": subject,
        "title": title,
        "members": "Test Student",
        "abstract_text": abstract,
        "prototype_description": proto_desc,
    }
    files = {"documentation_file": ("doc.txt", doc.encode(), "text/plain")}
    res = client.post("/api/projects", data=data, files=files)
    assert res.status_code == 200, res.text
    return res.json()


def main_test():
    # Two near-duplicate traffic projects in DIFFERENT sections.
    # NOTE: the fake test embedding below is bag-of-hashed-words (literal overlap only) —
    # unlike the real sentence-transformer model it has no notion of synonyms, so these
    # two texts are written to share vocabulary directly. This test validates the
    # pipeline's mechanics (upload -> extract -> embed -> compare -> filter -> rank),
    # not the semantic quality of the real model.
    a = submit(
        "A", 1, "Machine Learning",
        "traffic monitoring system using camera feeds to detect vehicle congestion in real time",
        "we use a detection model on camera feeds to count vehicles and estimate congestion in real time",
        "a real time system that watches camera feeds detects vehicles and flags congestion in real time",
        title="Traffic Vision",
    )
    b = submit(
        "C", 9, "Machine Learning",
        "traffic monitoring system using camera feeds to detect vehicle congestion levels",
        "we use a detection model on camera feeds to estimate vehicle congestion and traffic levels",
        "a system that watches camera feeds detects vehicles and flags traffic congestion levels",
        title="Road Sense",
    )
    # An unrelated project, also Machine Learning, also a different section
    c = submit(
        "B", 4, "Machine Learning",
        "Plant disease detection from leaf images using a convolutional neural network.",
        "We trained a CNN classifier on a leaf image dataset to identify common crop diseases.",
        "A mobile app where a farmer photographs a leaf and a CNN classifies the likely disease.",
        title="Crop Vision",
    )
    # Same idea as A, but a different SUBJECT — should not show up unless cross_subject=true
    d = submit(
        "D", 2, "Computer Networks",
        "traffic monitoring system using camera feeds to detect vehicle congestion in real time",
        "network protocol design for a distributed sensor mesh",
        "a real time system that watches camera feeds detects vehicles and flags congestion in real time",
        title="Traffic Vision Clone (different subject)",
    )

    assert a["section"] == "A" and a["team_number"] == 1

    # --- duplicate detection within Machine Learning, across sections A/B/C ---
    res = client.get("/api/duplicates", params={"subject": "Machine Learning", "threshold": 10})
    assert res.status_code == 200, res.text
    matches = res.json()
    pairs = {frozenset((m["project_a"]["id"], m["project_b"]["id"])) for m in matches}
    assert frozenset((a["id"], b["id"])) in pairs, "A and B (same idea, different sections) should be flagged"

    top = max(matches, key=lambda m: m["overall_similarity_pct"])
    assert frozenset((top["project_a"]["id"], top["project_b"]["id"])) == frozenset((a["id"], b["id"])), \
        "the near-duplicate pair should rank highest"
    assert top["overall_similarity_pct"] > 60, f"expected high similarity, got {top}"
    assert set(top["matched_parts"]) >= {"abstract", "prototype"}, top["matched_parts"]

    # unrelated C should NOT be flagged against A at a moderate threshold
    res_c = client.get("/api/duplicates", params={"subject": "Machine Learning", "threshold": 40})
    pairs_c = {frozenset((m["project_a"]["id"], m["project_b"]["id"])) for m in res_c.json()}
    assert frozenset((a["id"], c["id"])) not in pairs_c, "unrelated project wrongly flagged"

    # --- subject scoping: A vs D share text but different subjects ---
    res_same_subj = client.get("/api/duplicates", params={"subject": "Machine Learning", "threshold": 10})
    pairs_same = {frozenset((m["project_a"]["id"], m["project_b"]["id"])) for m in res_same_subj.json()}
    assert frozenset((a["id"], d["id"])) not in pairs_same, "different-subject pair leaked into same-subject results"

    res_cross = client.get("/api/duplicates", params={"threshold": 10, "cross_subject": "true"})
    pairs_cross = {frozenset((m["project_a"]["id"], m["project_b"]["id"])) for m in res_cross.json()}
    assert frozenset((a["id"], d["id"])) in pairs_cross, "cross_subject=true should surface the cross-subject match"

    # --- per-project similar endpoint ---
    res_similar = client.get(f"/api/projects/{a['id']}/similar", params={"threshold": 10})
    assert res_similar.status_code == 200
    assert any(m["project_b"]["id"] == b["id"] or m["project_a"]["id"] == b["id"] for m in res_similar.json())

    # --- project listing / detail / stats ---
    assert client.get("/api/projects").status_code == 200
    detail = client.get(f"/api/projects/{a['id']}")
    assert detail.status_code == 200
    assert "camera" in detail.json()["abstract_text"].lower()
    stats = client.get("/api/stats").json()
    assert stats["total_projects"] == 4

    # --- resubmission upserts rather than duplicating ---
    a2 = submit(
        "A", 1, "Machine Learning",
        "Updated abstract about traffic monitoring.",
        "updated doc",
        "updated prototype description",
        title="Traffic Vision v2",
    )
    assert a2["id"] == a["id"], "resubmitting same section+team+subject should update, not create a new row"
    assert client.get("/api/stats").json()["total_projects"] == 4

    # --- metadata-only resubmission (no new abstract/doc) should NOT wipe existing text ---
    res_meta = client.post(
        "/api/projects",
        data={
            "section": "A", "team_number": "1", "subject": "Machine Learning",
            "title": "Traffic Vision v3", "members": "Test Student",
            "abstract_text": "", "prototype_description": "same prototype, just renaming",
        },
    )
    assert res_meta.status_code == 200, res_meta.text
    detail_a3 = client.get(f"/api/projects/{a['id']}").json()
    assert detail_a3["title"] == "Traffic Vision v3"
    assert detail_a3["abstract_text"], "abstract text should be preserved, not wiped, on metadata-only resubmission"

    # --- pages render ---
    for path in ["/", "/dashboard", "/projects", f"/projects/{a['id']}"]:
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}"

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main_test()
