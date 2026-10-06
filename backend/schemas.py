from datetime import datetime

from pydantic import BaseModel


class ProjectOut(BaseModel):
    id: int
    section: str
    team_number: int
    subject: str
    title: str
    members: str
    created_at: datetime

    class Config:
        from_attributes = True


class ProjectDetailOut(ProjectOut):
    abstract_text: str
    documentation_text: str
    prototype_description: str
    abstract_filename: str
    documentation_filename: str
    prototype_zip_filename: str
    prototype_zip_contents: list[str] = []


class ComponentSimilarity(BaseModel):
    component: str            # "abstract" | "documentation" | "prototype"
    similarity_pct: float


class MatchOut(BaseModel):
    project_a: ProjectOut
    project_b: ProjectOut
    overall_similarity_pct: float
    category: str
    components: list[ComponentSimilarity]
    matched_parts: list[str]
