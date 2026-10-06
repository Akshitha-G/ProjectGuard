# ProjectGuard

Detects when two teams' final-year projects are conceptually the same idea,
even when the wording is completely different — by comparing sentence
embeddings, not keywords. Built for a department with sections A–D, three
subjects (Computer Networks, Machine Learning, Introduction to Data Science),
and up to 16 teams per section.

## How it works

1. A professor submits a team's abstract (file or pasted text), documentation
   (file), and prototype zip + a short **prototype description** (required —
   see "Why a description, not code diffing" below).
2. Each of the three text pieces is turned into a 384-dimensional sentence
   embedding using `all-MiniLM-L6-v2` (via `sentence-transformers`).
3. Every submission is compared against every other submission **in the same
   subject, across all four sections** using cosine similarity — not just
   within one section.
4. The dashboard lists pairs above a similarity threshold, showing both
   teams' section + number, an overall similarity %, a category (Low /
   Moderate / High / Very High), and which specific parts (abstract /
   documentation / prototype) matched.

### Why a prototype description, not code diffing

Diffing raw source code tells you if two zips share text, not whether two
teams built the same idea. Two teams can implement the identical concept in
completely different code (different language, structure, variable names) —
literal code comparison would miss that entirely, and would also flag
unrelated teams who both used the same boilerplate/starter template. So the
zip is stored for the professor to open by hand, but the **description** of
what the prototype does is what gets embedded and compared, consistent with
how the abstract and documentation are compared.

If your department later wants literal code-similarity detection too (e.g.
via `python-Levenshtein`/AST comparison as a *second*, separate signal), that
can be added as an additional column — ask and I'll build it in.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`. That's it — no separate frontend build step,
no database server to install. Everything (including uploaded files and the
SQLite database) is stored under `../data/`.

The first request that needs to generate an embedding will download the
`all-MiniLM-L6-v2` model (~90MB) from Hugging Face — needs internet access
once; it's cached locally after that.

### Running on a shared machine for the whole department

`uvicorn main:app --host 0.0.0.0 --port 8000` makes it reachable from other
machines on the same network at `http://<that-machine's-IP>:8000`. For
anything beyond a LAN demo (e.g. a public URL), it should sit behind a proper
process manager and reverse proxy (gunicorn+nginx, or a platform like
Railway/Render) — ask if you want that set up.

## Verifying it works

```bash
cd backend
pip install httpx   # test-only dependency
python3 tests/smoke_test.py
```

This runs the real API end-to-end (upload → extract → embed → compare →
filter → rank) using a fake embedding function, so it doesn't need to
download the real model. `ALL SMOKE TESTS PASSED` means the pipeline works;
it isn't a substitute for trying real submissions through the UI.

## Project structure

```
backend/
  main.py          FastAPI app: page routes + /api/* endpoints
  models.py         SQLAlchemy model (one row per team per subject)
  database.py       SQLite engine/session setup
  embeddings.py      Embedding generation + cosine similarity (the ML core)
  extract.py         Text extraction from PDF/DOCX/TXT, zip file listing
  schemas.py         Pydantic response shapes
  tests/smoke_test.py  End-to-end pipeline test
frontend/
  templates/         Jinja2 pages (upload, dashboard, all-submissions, detail)
  static/            style.css, app.js
data/                SQLite DB + uploaded files (created on first run)
```

## Notable design decisions / things worth knowing

- **Comparison scope**: by default, teams are only compared against other
  teams in the *same subject* (a Computer Networks project and an ML project
  aren't really comparable). The dashboard has a "compare across subjects
  too" toggle if you want it.
- **Similarity is a percentage, not a plagiarism score.** The dashboard
  language is deliberately "review recommended," not "duplicate found" — a
  high score is a starting point for a human to look at both submissions,
  same as the original spec intended.
- **No vector database.** At 4 sections × ≤16 teams × 3 subjects (≤192
  projects), brute-force cosine similarity over all pairs is instant. If the
  department scales far beyond that, swap in `pgvector` or FAISS — the
  `embeddings.py` interface is already isolated for that.
- **Resubmission**: uploading again for the same section + team + subject
  updates that row rather than creating a duplicate entry. Leaving a field
  blank on resubmission keeps the previously saved value rather than
  wiping it.

## Not yet built (natural next steps, kept out of this first pass on purpose)

- Faculty login / auth (currently anyone with the URL can submit or view —
  fine for a LAN classroom tool, not for anything public)
- CSV/PDF export of the duplicate-review table
- Clustering / topic discovery / similarity network graph (from the original
  wishlist) — genuinely useful, but adds real complexity for not much extra
  value at ~190 projects; worth doing once the core detection is proven out
- Bulk-importing team rosters instead of one submission at a time

Happy to build any of these next — just say which.
