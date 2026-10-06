from __future__ import annotations

import shutil
from itertools import combinations
from pathlib import Path

from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import get_db, init_db, UPLOAD_DIR
from models import Project
from extract import extract_text, list_zip_contents
from embeddings import (
    embed_text,
    bytes_to_vec,
    vec_to_bytes,
    cosine_similarity,
    similarity_to_percent,
    category_for,
)
from schemas import ProjectOut, ProjectDetailOut, MatchOut, ComponentSimilarity

BASE_DIR = Path(__file__).resolve().parent.parent

ALLOWED_SECTIONS = ["A", "B", "C", "D"]
ALLOWED_SUBJECTS = [
    "Computer Networks",
    "Machine Learning",
    "Introduction to Data Science",
]
COMPONENTS = ["abstract", "documentation", "prototype"]
DEFAULT_PART_THRESHOLD = 55.0  # a single component >= this % counts as "duplicated"

app = FastAPI(title="ProjectGuard")
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "frontend" / "templates")


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------- pages ----

@app.get("/", response_class=HTMLResponse)
def page_upload(request: Request):
    return templates.TemplateResponse(
        request, "upload.html",
        {"sections": ALLOWED_SECTIONS, "subjects": ALLOWED_SUBJECTS},
    )


@app.get("/dashboard", response_class=HTMLResponse)
def page_dashboard(request: Request):
    return templates.TemplateResponse(
        request, "dashboard.html",
        {"sections": ALLOWED_SECTIONS, "subjects": ALLOWED_SUBJECTS},
    )


@app.get("/projects", response_class=HTMLResponse)
def page_projects(request: Request):
    return templates.TemplateResponse(
        request, "projects.html",
        {"sections": ALLOWED_SECTIONS, "subjects": ALLOWED_SUBJECTS},
    )


@app.get("/projects/{project_id}", response_class=HTMLResponse)
def page_project_detail(request: Request, project_id: int):
    return templates.TemplateResponse(
        request, "project_detail.html", {"project_id": project_id}
    )


# --------------------------------------------------------------- helpers ---

def _project_dir(section: str, team_number: int, subject: str) -> Path:
    slug = subject.lower().replace(" ", "_")
    d = UPLOAD_DIR / f"{section}_{team_number}_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_upload(upload: UploadFile, dest_dir: Path) -> str:
    dest = dest_dir / upload.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest)


def _to_out(p: Project) -> ProjectOut:
    return ProjectOut.model_validate(p)


def _build_match(pa: Project, pb: Project, part_threshold: float) -> MatchOut | None:
    pairs = [
        ("abstract", pa.abstract_text, pa.abstract_embedding, pb.abstract_text, pb.abstract_embedding),
        ("documentation", pa.documentation_text, pa.documentation_embedding, pb.documentation_text, pb.documentation_embedding),
        ("prototype", pa.prototype_description, pa.prototype_embedding, pb.prototype_description, pb.prototype_embedding),
    ]

    components: list[ComponentSimilarity] = []
    matched_parts: list[str] = []
    valid_pcts: list[float] = []

    for name, text_a, emb_a, text_b, emb_b in pairs:
        has_both = bool((text_a or "").strip()) and bool((text_b or "").strip())
        if has_both:
            sim = cosine_similarity(bytes_to_vec(emb_a), bytes_to_vec(emb_b))
            pct = similarity_to_percent(sim)
            valid_pcts.append(pct)
            if pct >= part_threshold:
                matched_parts.append(name)
        else:
            pct = 0.0
        components.append(ComponentSimilarity(component=name, similarity_pct=pct))

    if not valid_pcts:
        return None  # nothing comparable between these two submissions yet

    overall = round(sum(valid_pcts) / len(valid_pcts), 1)
    return MatchOut(
        project_a=_to_out(pa),
        project_b=_to_out(pb),
        overall_similarity_pct=overall,
        category=category_for(overall),
        components=components,
        matched_parts=matched_parts,
    )


# ------------------------------------------------------------------- API ---

@app.post("/api/projects", response_model=ProjectOut)
async def submit_project(
    section: str = Form(...),
    team_number: int = Form(...),
    subject: str = Form(...),
    title: str = Form(""),
    members: str = Form(""),
    abstract_text: str = Form(""),
    prototype_description: str = Form(...),
    abstract_file: UploadFile | None = File(None),
    documentation_file: UploadFile | None = File(None),
    prototype_zip: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    section = section.strip().upper()
    if section not in ALLOWED_SECTIONS:
        raise HTTPException(400, f"section must be one of {ALLOWED_SECTIONS}")
    if subject not in ALLOWED_SUBJECTS:
        raise HTTPException(400, f"subject must be one of {ALLOWED_SUBJECTS}")
    if team_number < 1:
        raise HTTPException(400, "team_number must be positive")
    if not prototype_description.strip():
        raise HTTPException(400, "prototype_description is required — a few sentences on what it does")

    dest_dir = _project_dir(section, team_number, subject)

    resolved_abstract_text = abstract_text.strip()
    abstract_filename = ""
    if abstract_file is not None and abstract_file.filename:
        path = _save_upload(abstract_file, dest_dir)
        abstract_filename = Path(path).name
        extracted = extract_text(path)
        if extracted:
            resolved_abstract_text = extracted

    documentation_text = ""
    documentation_filename = ""
    if documentation_file is not None and documentation_file.filename:
        path = _save_upload(documentation_file, dest_dir)
        documentation_filename = Path(path).name
        documentation_text = extract_text(path)

    prototype_zip_filename = ""
    prototype_zip_contents = ""
    if prototype_zip is not None and prototype_zip.filename:
        path = _save_upload(prototype_zip, dest_dir)
        prototype_zip_filename = Path(path).name
        prototype_zip_contents = "\n".join(list_zip_contents(path))

    existing = db.execute(
        select(Project).where(
            Project.section == section,
            Project.team_number == team_number,
            Project.subject == subject,
        )
    ).scalar_one_or_none()

    would_have_abstract = bool(resolved_abstract_text or (existing and existing.abstract_text))
    would_have_documentation = bool(documentation_text or (existing and existing.documentation_text))
    if not would_have_abstract and not would_have_documentation:
        raise HTTPException(
            400, "Provide at least an abstract (text or file) or a documentation file"
        )

    project = existing or Project(section=section, team_number=team_number, subject=subject)
    project.title = title.strip()
    project.members = members.strip()
    # On resubmission, an empty field means "unchanged", not "erase this" — only
    # overwrite when new content was actually provided.
    project.abstract_text = resolved_abstract_text or (project.abstract_text if existing else "")
    project.documentation_text = documentation_text or (project.documentation_text if existing else "")
    project.prototype_description = prototype_description.strip()
    if abstract_filename:
        project.abstract_filename = abstract_filename
    if documentation_filename:
        project.documentation_filename = documentation_filename
    if prototype_zip_filename:
        project.prototype_zip_filename = prototype_zip_filename
        project.prototype_zip_contents = prototype_zip_contents

    project.abstract_embedding = vec_to_bytes(embed_text(project.abstract_text))
    project.documentation_embedding = vec_to_bytes(embed_text(project.documentation_text))
    project.prototype_embedding = vec_to_bytes(embed_text(project.prototype_description))

    db.add(project)
    db.commit()
    db.refresh(project)
    return _to_out(project)


@app.get("/api/projects", response_model=list[ProjectOut])
def list_projects(
    section: str | None = None,
    subject: str | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(Project)
    if section:
        stmt = stmt.where(Project.section == section.upper())
    if subject:
        stmt = stmt.where(Project.subject == subject)
    stmt = stmt.order_by(Project.subject, Project.section, Project.team_number)
    projects = db.execute(stmt).scalars().all()
    return [_to_out(p) for p in projects]


@app.get("/api/projects/{project_id}", response_model=ProjectDetailOut)
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "project not found")
    return ProjectDetailOut(
        id=project.id,
        section=project.section,
        team_number=project.team_number,
        subject=project.subject,
        title=project.title,
        members=project.members,
        created_at=project.created_at,
        abstract_text=project.abstract_text,
        documentation_text=project.documentation_text,
        prototype_description=project.prototype_description,
        abstract_filename=project.abstract_filename,
        documentation_filename=project.documentation_filename,
        prototype_zip_filename=project.prototype_zip_filename,
        prototype_zip_contents=(
            project.prototype_zip_contents.splitlines() if project.prototype_zip_contents else []
        ),
    )


@app.get("/api/projects/{project_id}/similar", response_model=list[MatchOut])
def similar_projects(
    project_id: int,
    threshold: float = Query(30.0),
    part_threshold: float = Query(DEFAULT_PART_THRESHOLD),
    cross_subject: bool = Query(False),
    db: Session = Depends(get_db),
):
    target = db.get(Project, project_id)
    if not target:
        raise HTTPException(404, "project not found")

    stmt = select(Project).where(Project.id != project_id)
    if not cross_subject:
        stmt = stmt.where(Project.subject == target.subject)
    others = db.execute(stmt).scalars().all()

    matches = []
    for other in others:
        m = _build_match(target, other, part_threshold)
        if m and m.overall_similarity_pct >= threshold:
            matches.append(m)
    matches.sort(key=lambda m: m.overall_similarity_pct, reverse=True)
    return matches


@app.get("/api/duplicates", response_model=list[MatchOut])
def duplicates(
    subject: str | None = Query(None, description="Restrict to one subject"),
    section: str | None = Query(None, description="Only pairs involving this section"),
    threshold: float = Query(30.0, description="Minimum overall similarity %% to include"),
    part_threshold: float = Query(DEFAULT_PART_THRESHOLD),
    cross_subject: bool = Query(
        False, description="Compare across different subjects too (off by default)"
    ),
    db: Session = Depends(get_db),
):
    """Every team is compared against every other team across all 4 sections
    (within the same subject, by default) — not just within one section."""
    stmt = select(Project)
    if subject:
        stmt = stmt.where(Project.subject == subject)
    projects = db.execute(stmt).scalars().all()

    results: list[MatchOut] = []
    if cross_subject:
        candidate_pairs = combinations(projects, 2)
    else:
        by_subject: dict[str, list[Project]] = {}
        for p in projects:
            by_subject.setdefault(p.subject, []).append(p)
        candidate_pairs = (
            pair for group in by_subject.values() for pair in combinations(group, 2)
        )

    for pa, pb in candidate_pairs:
        if section and section.upper() not in (pa.section, pb.section):
            continue
        m = _build_match(pa, pb, part_threshold)
        if m and m.overall_similarity_pct >= threshold:
            results.append(m)

    results.sort(key=lambda m: m.overall_similarity_pct, reverse=True)
    return results


@app.get("/api/stats")
def stats(db: Session = Depends(get_db)):
    projects = db.execute(select(Project)).scalars().all()
    by_subject: dict[str, int] = {}
    by_section: dict[str, int] = {}
    for p in projects:
        by_subject[p.subject] = by_subject.get(p.subject, 0) + 1
        by_section[p.section] = by_section.get(p.section, 0) + 1
    return {
        "total_projects": len(projects),
        "by_subject": by_subject,
        "by_section": by_section,
    }
