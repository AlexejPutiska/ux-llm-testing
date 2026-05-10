"""
SQLAlchemy ORM models for the UX LLM Testing application.

Database schema overview:
  - PersonaBatch  : a named group of simulated user personas with demographic distributions
  - Persona       : a single simulated user generated from a PersonaBatch
  - Test          : top-level container for all test tasks (only one global test is used)
  - TestTask      : a single UX task (preference / first_click / feedback)
  - TestSession   : one execution run of a batch against a set of tasks
  - TestRun       : one LLM run within a session (either aggregate or per-persona)
  - TestRunItem   : result of one task within a TestRun, storing raw and parsed LLM output
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PersonaBatch(Base):
    """
    A named group of simulated user personas.

    Stores both the raw demographic distributions (used to generate individual personas)
    and the pre-computed aggregate persona text (used for the aggregate simulation mode).
    """

    __tablename__ = "persona_batch"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Human-readable name for the batch (e.g. "Preference test group")
    name: Mapped[str] = mapped_column(Text, nullable=False)

    # Total number of simulated users in this batch
    count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Age range derived from the age distribution (used for display purposes)
    age_from: Mapped[int] = mapped_column(Integer, nullable=False)
    age_to: Mapped[int] = mapped_column(Integer, nullable=False)

    # Flat lists of unique values present in each demographic dimension
    genders: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    similar_apps_experience: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    decision_style: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    frustration_tolerance: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)

    # Full distributions as JSON arrays: [{"value": "18-24", "count": 5}, ...]
    age_distribution: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    genders_distribution: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    similar_apps_experience_distribution: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    decision_style_distribution: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    frustration_tolerance_distribution: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)

    # Optional free-text description appended to every persona in this batch
    extra_description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Pre-built natural language summary of the group used in aggregate simulation prompts
    aggregate_persona_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Persona(Base):
    """
    A single simulated user generated from a PersonaBatch.

    Each persona has a unique set of characteristics (age, gender, decision style, etc.)
    drawn from the batch distributions, along with a pre-built natural language description
    used in N-person simulation prompts.
    """

    __tablename__ = "persona"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Reference to the parent batch; deletion is restricted if personas exist
    batch_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("persona_batch.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # 1-based index of this persona within its batch (used for display ordering)
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Raw demographic characteristics as a JSON object, e.g.:
    # {"age": 23, "gender": "ženy", "decision_style": "analytický", ...}
    characteristics: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Copy of the batch's extra_description at the time of persona creation
    extra_description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Pre-built natural language description of this individual user,
    # used directly in LLM prompts for N-person simulation mode
    persona_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class Test(Base):
    """
    Top-level container for UX test tasks.

    In the current implementation a single global test instance is created on startup
    and all tasks belong to it. The model is kept generic to allow future multi-test support.
    """

    __tablename__ = "test"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="Global LLM UX Test")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")


class TestTask(Base):
    """
    A single UX test task presented to both real users and the LLM.

    Three task types are supported:
      - preference  : choose between two or more UI design variants
      - first_click : identify the first element the user would click to complete a goal
      - feedback    : provide open-ended positive/negative observations about a screenshot

    The config JSONB field stores task-type-specific data:
      - preference : {"options": [{"label": "Design A", "image": "data:image/..."}, ...]}
      - first_click: {"image": "data:image/..."}
      - feedback   : {"image": "data:image/..."}
    """

    __tablename__ = "test_task"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    test_id: Mapped[int] = mapped_column(
        ForeignKey("test.id", ondelete="CASCADE"),
        nullable=False,
    )

    # One of: "preference", "first_click", "feedback"
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Ordering index within tasks of the same type; used to keep tasks in creation order
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Short display title (auto-generated from task_text if not provided)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Full task instruction shown to the user / included in the LLM prompt
    task_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Optional follow-up open-ended question (not applicable to feedback tasks)
    follow_up_question: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Inactive tasks are excluded from new test runs but preserved for historical data
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    # Task-type-specific configuration (images, option labels); see class docstring
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class TestSession(Base):
    """
    One execution of a PersonaBatch against a selected set of TestTasks.

    A session groups together one aggregate run and N individual persona runs
    (one per persona in the batch). Snapshot fields preserve the batch state
    at the time the session was created so that results remain interpretable
    even if the batch is later modified.

    Status lifecycle: created -> finished | failed
    """

    __tablename__ = "test_session"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    test_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("test.id", ondelete="CASCADE"),
        nullable=False,
    )

    batch_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("persona_batch.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # "created" | "finished" | "failed"
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="created")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # LLM model identifier and additional parameters (temperature, top_p, response_format)
    llm_model: Mapped[str] = mapped_column(Text, nullable=False, default="gpt-4o")
    llm_params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # IDs of the tasks included in this session (preserved for reproducibility)
    selected_task_ids: Mapped[list[int] | dict] = mapped_column(JSONB, nullable=False, default=list)

    # Snapshots of batch metadata at session creation time
    batch_name_snapshot: Mapped[str] = mapped_column(Text, nullable=False, default="")
    batch_count_snapshot: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    aggregate_persona_text_snapshot: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class TestRun(Base):
    """
    One LLM simulation run within a TestSession.

    Each session produces exactly one aggregate run (subject_type="aggregate") and
    one persona run (subject_type="persona") per persona in the batch.

    - Aggregate run  : the LLM receives the group profile and simulates all users at once
    - Persona run    : the LLM receives a single persona profile and simulates one user

    Status lifecycle: created -> finished | failed
    """

    __tablename__ = "test_run"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    session_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("test_session.id", ondelete="CASCADE"),
        nullable=False,
    )

    # "aggregate" or "persona"
    subject_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # NULL for aggregate runs; set to the specific persona ID for persona runs
    persona_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("persona.id", ondelete="RESTRICT"),
        nullable=True,
    )

    batch_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("persona_batch.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # "created" | "finished" | "failed"
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="created")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    llm_model: Mapped[str] = mapped_column(Text, nullable=False, default="gpt-4o")
    llm_params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class TestRunItem(Base):
    """
    The result of executing one TestTask within a TestRun.

    Stores both the raw LLM output and the parsed/normalized response so that
    the experiment can be re-analysed without calling the API again.

    Fields:
      answer_json      : summarised result (e.g. vote counts for preference tasks)
      raw_responses    : individual per-simulated-user responses before aggregation
      raw_output_text  : raw LLM text output (set only when JSON parsing fails)
      result_mode      : "aggregate_direct" | "persona_single" | "persona_group_aggregated"
      usage            : OpenAI token usage statistics for cost tracking
      error_message    : set if the LLM call or parsing failed for this item
    """

    __tablename__ = "test_run_item"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    run_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("test_run.id", ondelete="CASCADE"),
        nullable=False,
    )

    task_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("test_task.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # Snapshot of the task at execution time; ensures results remain interpretable
    # even if the original task is later edited or deleted
    task_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Aggregated summary of responses (vote counts, click targets, note counts)
    answer_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Individual per-simulated-user responses in normalised form
    raw_responses: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Raw LLM output text; populated only when JSON parsing fails
    raw_output_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Indicates how the result was produced (aggregate vs. individual persona mode)
    result_mode: Mapped[str | None] = mapped_column(Text, nullable=True)

    # OpenAI API token usage for this item (prompt_tokens, completion_tokens, total_tokens)
    usage: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Error message if the LLM call or response parsing failed for this specific item
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
