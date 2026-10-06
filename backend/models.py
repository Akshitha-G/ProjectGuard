import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, LargeBinary, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Project(Base):
    """One team's submission for one subject.

    A given (section, team_number) will typically have three rows here —
    one per subject (Computer Networks / Machine Learning / Intro to Data Science).
    """

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)

    section = Column(String(1), nullable=False)       # "A" | "B" | "C" | "D"
    team_number = Column(Integer, nullable=False)      # 1..16
    subject = Column(String(64), nullable=False)
    title = Column(String(255), default="")
    members = Column(Text, default="")                 # free text, comma-separated names

    abstract_text = Column(Text, default="")
    documentation_text = Column(Text, default="")
    prototype_description = Column(Text, default="")   # what the prototype does (required)

    abstract_filename = Column(String(255), default="")
    documentation_filename = Column(String(255), default="")
    prototype_zip_filename = Column(String(255), default="")
    prototype_zip_contents = Column(Text, default="")  # newline-joined file listing, for display

    # Embeddings stored as raw float32 bytes (numpy .tobytes()); 384-dim for all-MiniLM-L6-v2
    abstract_embedding = Column(LargeBinary)
    documentation_embedding = Column(LargeBinary)
    prototype_embedding = Column(LargeBinary)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("section", "team_number", "subject", name="uq_team_section_subject"),
    )
