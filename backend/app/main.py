"""
FastAPI backend for the UX LLM Testing application.

This module implements the REST API that:
  1. Manages UX test tasks (preference, first_click, feedback) and their configurations.
  2. Manages persona batches — groups of simulated users with defined demographic distributions.
  3. Runs LLM simulations against test tasks in two modes:
       - Aggregate mode  : the model receives a group profile and simulates all users at once.
       - N-person mode   : each persona is sent as a separate request, simulating one user at a time.
  4. Stores all responses persistently in PostgreSQL so that the experiment can be
     re-analysed without re-calling the OpenAI API.

Automatic API documentation is available at http://localhost:8000/docs when the server is running.

Environment variables:
  OPENAI_API_KEY  (required) : OpenAI API key used for all LLM calls.
  DATABASE_URL    (optional) : PostgreSQL connection string; defaults to localhost development DB.
"""

import os
import json
import random
from collections import Counter
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session
from openai import OpenAI

from db.session import SessionLocal, engine
from db.models import (
    Base,
    Persona,
    PersonaBatch,
    Test,
    TestSession,
    TestTask,
    TestRun,
    TestRunItem,
)

# Load environment variables from .env file if present.
# This allows running the backend without manually setting env variables in the shell.
load_dotenv()

app = FastAPI(
    title="UX LLM Testing API",
    description="Backend API for simulating UX tests using Large Language Models.",
)

# Allow requests from the Next.js frontend running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create all database tables on startup if they do not already exist
Base.metadata.create_all(bind=engine)

# Initialise the OpenAI client; the API key must be set as an environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file or environment.")

client = OpenAI(api_key=api_key)

# --- Constants ---

DEFAULT_TEST_NAME = "Global LLM UX Test"

# Allowed values for task_type and subject_type fields
VALID_TASK_TYPES = {"preference", "first_click", "feedback"}
VALID_SUBJECT_TYPES = {"persona", "aggregate"}

# Ordering for task types when sorting results (preference first, then first_click, then feedback)
TYPE_ORDER = {"preference": 1, "first_click": 2, "feedback": 3}

# Mapping of age band labels to (min_age, max_age) integer ranges
AGE_BANDS: dict[str, tuple[int, int]] = {
    "18-24": (18, 24),
    "25-34": (25, 34),
    "35-44": (35, 44),
    "45-54": (45, 54),
    "55-64": (55, 64),
    "65+":   (65, 80),
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    """FastAPI dependency that provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def ensure_default_test():
    """
    Create the default global test record on application startup if it does not exist.

    The application uses a single shared Test instance. This startup hook ensures
    it is always present without requiring manual database seeding.
    """
    db = SessionLocal()
    try:
        try:
            existing = db.query(Test).first()
        except ProgrammingError:
            raise RuntimeError(
                "DB schema for table 'test' does not match the model. Check column names."
            )

        if not existing:
            row = Test(name=DEFAULT_TEST_NAME, description="")
            db.add(row)
            db.commit()
    finally:
        db.close()


def get_default_test(db: Session) -> Test:
    """
    Return the single global Test record, creating it if necessary.

    All test tasks are associated with this record.
    """
    test = db.query(Test).order_by(Test.id.asc()).first()
    if not test:
        test = Test(name=DEFAULT_TEST_NAME, description="")
        db.add(test)
        db.commit()
        db.refresh(test)
    return test


def next_order_for_type(db: Session, test_id: int, task_type: str) -> int:
    """
    Return the next available order_index for a new task of the given type.

    Tasks are ordered independently within each task type so that preference tasks,
    first_click tasks, and feedback tasks each have their own sequential numbering.
    """
    last = (
        db.query(TestTask)
        .filter(TestTask.test_id == test_id, TestTask.task_type == task_type)
        .order_by(TestTask.order_index.desc())
        .first()
    )
    if not last:
        return 1
    return int(last.order_index) + 1


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def validate_task_payload(
    task_type: str,
    task_text: str,
    config: dict[str, Any],
    follow_up_question: str | None,
):
    """
    Validate the fields of a task creation or update request.

    Raises HTTPException 400 if any required field is missing or has an invalid value.
    Each task type has its own config requirements:
      - preference  : config.options must be a list of 2–4 dicts with 'label' and 'image'
      - first_click : config.image must be a non-empty string
      - feedback    : config.image must be a non-empty string
    """
    if task_type not in VALID_TASK_TYPES:
        raise HTTPException(status_code=400, detail="Invalid task_type")

    if not isinstance(task_text, str) or not task_text.strip():
        raise HTTPException(status_code=400, detail="task_text must not be empty")

    if follow_up_question is not None and not isinstance(follow_up_question, str):
        raise HTTPException(status_code=400, detail="follow_up_question must be a string or null")

    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")

    if task_type == "preference":
        options = config.get("options")
        if not isinstance(options, list) or len(options) < 2 or len(options) > 4:
            raise HTTPException(status_code=400, detail="Preference: options must contain 2 to 4 items")

        for i, opt in enumerate(options):
            if not isinstance(opt, dict):
                raise HTTPException(status_code=400, detail=f"Preference: option {i} is not an object")

            label = str((opt.get("label") or "")).strip()
            image = str((opt.get("image") or "")).strip()
            if not label:
                raise HTTPException(status_code=400, detail=f"Preference: option {i} is missing a label")
            if not image:
                raise HTTPException(status_code=400, detail=f"Preference: option {i} is missing an image")

    elif task_type == "first_click":
        image = str((config.get("image") or "")).strip()
        if not image:
            raise HTTPException(status_code=400, detail="First-click: image is required")

    elif task_type == "feedback":
        image = str((config.get("image") or "")).strip()
        if not image:
            raise HTTPException(status_code=400, detail="Feedback: image is required")


# ---------------------------------------------------------------------------
# Type aliases used in Pydantic models
# ---------------------------------------------------------------------------

Gender = Literal["muži", "ženy"]
Level = Literal["nízka", "stredná", "vysoká"]
Experience = Literal[
    "Časté (aspoň niekoľkokrát do mesiaca)",
    "Občasné (niekoľkokrát ročne)",
    "Zriedkavé (menej ako raz ročne)",
]
DecisionStyle = Literal["rýchly", "analytický", "exploratívny"]
AgeBand = Literal["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
TaskType = Literal["preference", "first_click", "feedback"]


# ---------------------------------------------------------------------------
# Pydantic schemas (request / response models)
# ---------------------------------------------------------------------------

class DistributionItemIn(BaseModel):
    """One entry in a demographic distribution: a value label and how many personas have it."""
    value: str
    count: int


class DistributionItemOut(BaseModel):
    value: str
    count: int


class PersonaBatchCreate(BaseModel):
    """Request body for creating a new persona batch with demographic distributions."""
    name: str
    count: int

    age_distribution: list[DistributionItemIn] = []
    genders_distribution: list[DistributionItemIn] = []
    similar_apps_experience_distribution: list[DistributionItemIn] = []
    decision_style_distribution: list[DistributionItemIn] = []
    frustration_tolerance_distribution: list[DistributionItemIn] = []

    extra_description: str = ""


class PersonaBatchOut(BaseModel):
    """Full representation of a persona batch returned by the API."""
    id: int
    name: str
    count: int
    age_from: int
    age_to: int

    genders: list[str]
    similar_apps_experience: list[str]
    decision_style: list[str]
    frustration_tolerance: list[str]

    age_distribution: list[DistributionItemOut] = []
    genders_distribution: list[DistributionItemOut] = []
    similar_apps_experience_distribution: list[DistributionItemOut] = []
    decision_style_distribution: list[DistributionItemOut] = []
    frustration_tolerance_distribution: list[DistributionItemOut] = []

    extra_description: str
    aggregate_persona_text: str
    created_at: Any
    model_config = ConfigDict(from_attributes=True)


class PersonaOut(BaseModel):
    """Representation of a single simulated user persona."""
    id: int
    batch_id: int
    batch_index: int
    characteristics: dict[str, Any]
    extra_description: str
    persona_text: str
    model_config = ConfigDict(from_attributes=True)


class PersonaBatchCreateOut(BaseModel):
    """Response returned after successfully creating a persona batch."""
    batch_id: int
    aggregate_persona_text: str
    created_persona_ids: list[int]


class PersonaBatchDetailOut(BaseModel):
    """Full batch detail including all individual personas."""
    batch: PersonaBatchOut
    personas: list[PersonaOut]


class TestTaskCreate(BaseModel):
    """Request body for creating a new UX test task."""
    task_type: TaskType
    title: str = ""
    task_text: str
    follow_up_question: str | None = None
    config: dict[str, Any] = {}
    is_active: bool = True


class TestTaskUpdate(BaseModel):
    """Request body for partially updating an existing task (all fields optional)."""
    title: str | None = None
    task_text: str | None = None
    follow_up_question: str | None = None
    config: dict[str, Any] | None = None
    is_active: bool | None = None


class TestTaskOut(BaseModel):
    """Full representation of a test task returned by the API."""
    id: int
    test_id: int
    task_type: str
    order_index: int
    title: str
    task_text: str
    follow_up_question: str | None
    is_active: bool
    config: dict[str, Any] = {}
    model_config = ConfigDict(from_attributes=True)


class LLMParamsIn(BaseModel):
    """LLM configuration parameters for a test run."""
    model: str = "gpt-5.4"
    temperature: float | None = None   # None = use model default
    top_p: float | None = None         # None = use model default
    response_format: Literal["json_object", "text"] = "json_object"


class RunCreate(BaseModel):
    """Request body for creating a single-persona test run."""
    persona_id: int
    task_ids: list[int] | None = None  # None = use all active tasks
    llm: LLMParamsIn = LLMParamsIn()


class BatchRunsCreate(BaseModel):
    """Request body for creating a full test session (aggregate + all persona runs)."""
    batch_id: int
    task_ids: list[int] | None = None  # None = use all active tasks
    llm: LLMParamsIn = LLMParamsIn()


class RunCreateOut(BaseModel):
    run_id: int


class BatchRunsCreateOut(BaseModel):
    """Response returned after creating a full test session."""
    session_id: int
    batch_id: int
    aggregate_run_id: int | None
    persona_run_ids: list[int]
    total_runs: int


class RunOut(BaseModel):
    """Summary representation of a test run."""
    id: int
    session_id: int
    subject_type: str
    persona_id: int | None
    batch_id: int
    persona_name: str
    status: str
    llm_model: str
    llm_params: dict[str, Any]
    model_config = ConfigDict(from_attributes=True)


class RunItemOut(BaseModel):
    """Result of one task within a run, including raw and parsed LLM output."""
    id: int
    task_id: int
    task_snapshot: dict[str, Any]
    answer_json: dict[str, Any] | None = None
    raw_responses: dict[str, Any] | None = None
    raw_output_text: str | None = None
    result_mode: str | None = None
    usage: dict[str, Any] | None = None
    error_message: str | None = None
    model_config = ConfigDict(from_attributes=True)


class RunDetailOut(BaseModel):
    run: RunOut
    items: list[RunItemOut]


class BatchRunsDetailOut(BaseModel):
    batch: PersonaBatchOut
    aggregate_run: RunOut | None
    persona_runs: list[RunOut]


class TestSessionOut(BaseModel):
    """Summary of a test session including run counts and status."""
    id: int
    test_id: int
    batch_id: int
    batch_name_snapshot: str
    batch_count_snapshot: int
    aggregate_persona_text_snapshot: str
    status: str
    llm_model: str
    llm_params: dict[str, Any]
    selected_task_ids: list[int] | dict[str, Any]
    created_at: Any
    total_runs: int
    aggregate_run_id: int | None = None
    persona_run_count: int = 0
    selected_task_count: int = 0
    model_config = ConfigDict(from_attributes=True)


class SessionTaskResultOut(BaseModel):
    """Result of one task within a session, used in session detail responses."""
    id: int | None = None
    task_id: int
    task_snapshot: dict[str, Any]
    answer_json: dict[str, Any] | None = None
    raw_responses: dict[str, Any] | None = None
    raw_output_text: str | None = None
    result_mode: str | None = None
    usage: dict[str, Any] | None = None
    error_message: str | None = None


class AggregateBlockOut(BaseModel):
    """Aggregate simulation results for a session."""
    run: RunOut | None
    items: list[SessionTaskResultOut]


class PersonaGroupOut(BaseModel):
    """N-person simulation results for a session, merged across all individual persona runs."""
    total_personas: int
    run_count: int
    items: list[SessionTaskResultOut]


class TestSessionDetailOut(BaseModel):
    """Full detail of a test session including both aggregate and persona results."""
    session: TestSessionOut
    aggregate: AggregateBlockOut | None
    persona_group: PersonaGroupOut


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    """Return a simple status response to confirm the API is running."""
    return {"ok": True, "msg": "API is running"}


# ---------------------------------------------------------------------------
# Persona helper functions
# ---------------------------------------------------------------------------

def _rand_age(age_from: int, age_to: int) -> int:
    """Return a random integer age within the given inclusive range."""
    if age_from == age_to:
        return age_from
    return random.randint(age_from, age_to)


def _age_range_from_band(band: str) -> tuple[int, int]:
    """Convert an age band label (e.g. '18-24') to a (min, max) integer tuple."""
    rng = AGE_BANDS.get(str(band).strip())
    if not rng:
        raise HTTPException(status_code=400, detail=f"Invalid age band: {band}")
    return rng


def _min_max_from_age_distribution(items: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Compute the overall minimum and maximum age from a list of age band distribution items.

    Used to populate the age_from / age_to summary fields on the PersonaBatch record.
    """
    mins: list[int] = []
    maxs: list[int] = []

    for item in items:
        value = str(item.get("value") or "").strip()
        count = int(item.get("count") or 0)
        if count <= 0:
            continue

        lo, hi = _age_range_from_band(value)
        mins.append(lo)
        maxs.append(hi)

    if not mins or not maxs:
        raise HTTPException(status_code=400, detail="Age distribution must not be empty")

    return min(mins), max(maxs)


def _rand_age_from_band(band: str) -> int:
    """Return a random age within the range corresponding to an age band label."""
    lo, hi = _age_range_from_band(band)
    return _rand_age(lo, hi)


def _as_nonempty_str(v: Any) -> str | None:
    """Return the string value of v if it is non-empty after stripping, otherwise None."""
    s = str(v or "").strip()
    return s if s else None


def _normalize_notes(v: Any) -> list[str]:
    """
    Normalise the 'notes' field from an LLM feedback response.

    Accepts either a list of strings or a single string; returns a list of
    non-empty strings. Used to standardise feedback task output.
    """
    if isinstance(v, list):
        out = []
        for x in v:
            sx = _as_nonempty_str(x)
            if sx:
                out.append(sx)
        return out

    sx = _as_nonempty_str(v)
    return [sx] if sx else []


def _normalize_distribution_items(items: list[DistributionItemIn] | None) -> list[dict[str, Any]]:
    """
    Convert a list of DistributionItemIn objects to a plain list of dicts,
    validating that values are non-empty, unique, and counts are non-negative.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items or []:
        value = str(item.value or "").strip()
        count = int(item.count or 0)

        if not value:
            raise HTTPException(status_code=400, detail="Distribution contains an empty value")

        if value in seen:
            raise HTTPException(status_code=400, detail=f"Distribution contains duplicate value: {value}")

        if count < 0:
            raise HTTPException(status_code=400, detail=f"Distribution has negative count for value: {value}")

        seen.add(value)
        out.append({"value": value, "count": count})

    return out


def _sum_distribution(items: list[dict[str, Any]]) -> int:
    """Sum the 'count' values of all items in a distribution list."""
    total = 0
    for item in items:
        total += int(item.get("count") or 0)
    return total


def _nonzero_distribution_values(items: list[dict[str, Any]]) -> list[str]:
    """Return the list of value labels that have a count greater than zero."""
    out: list[str] = []
    for item in items:
        count = int(item.get("count") or 0)
        value = str(item.get("value") or "").strip()
        if count > 0 and value:
            out.append(value)
    return out


def _nonzero_distribution_items(items: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Return (value, count) tuples for all items with a count greater than zero."""
    out: list[tuple[str, int]] = []

    for item in items:
        count = int(item.get("count") or 0)
        value = str(item.get("value") or "").strip()

        if count > 0 and value:
            out.append((value, count))

    return out


def _format_age_band_label(value: str) -> str:
    """
    Convert an age band key (e.g. '18-24') to a human-readable Slovak phrase
    (e.g. '18 až 24 rokov'). Used when generating persona description texts.
    """
    value = str(value).strip()

    if value == "65+":
        return "65 a viac rokov"

    if "-" in value:
        lo, hi = value.split("-", 1)
        return f"{lo.strip()} až {hi.strip()} rokov"

    return value


def _join_with_and(items: list[str]) -> str:
    """
    Join a list of strings with commas and 'a' (Slovak for 'and') before the last item.
    Examples: ["A"] -> "A", ["A", "B"] -> "A a B", ["A", "B", "C"] -> "A, B a C"
    """
    items = [str(x).strip() for x in items if str(x).strip()]

    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} a {items[1]}"

    return ", ".join(items[:-1]) + f" a {items[-1]}"


def _sentence_lower_first(text: str) -> str:
    """Lower-case the first character of a string (used for Slovak grammar in persona text)."""
    text = str(text or "").strip()
    if not text:
        return text
    return text[:1].lower() + text[1:]


def _user_count_text(count: int) -> str:
    """Return a correctly declined Slovak noun phrase for a user count (1/2-4/5+)."""
    if count == 1:
        return "1 používateľ"
    if count in {2, 3, 4}:
        return f"{count} používatelia"
    return f"{count} používateľov"


def _expand_distribution_to_slots(
    items: list[dict[str, Any]],
    total_count: int,
    *,
    exact: bool,
) -> list[str | None]:
    """
    Expand a distribution into a randomised list of value slots with length total_count.

    When exact=True, the distribution counts must sum to exactly total_count.
    When exact=False, unspecified slots are filled with None (attribute left unassigned).
    The resulting list is shuffled so that persona characteristics are randomly mixed.
    """
    values: list[str | None] = []

    for item in items:
        value = str(item.get("value") or "").strip()
        count = int(item.get("count") or 0)

        if not value or count <= 0:
            continue

        values.extend([value] * count)

    if exact:
        if len(values) != total_count:
            raise HTTPException(
                status_code=400,
                detail="For required distributions, the sum must equal the persona count exactly.",
            )
    else:
        if len(values) > total_count:
            raise HTTPException(
                status_code=400,
                detail="The sum of an optional distribution must not exceed the persona count.",
            )
        values.extend([None] * (total_count - len(values)))

    random.shuffle(values)
    return values


def validate_distribution_values(
    items: list[dict[str, Any]],
    allowed_values: set[str],
    label: str,
    *,
    total_count: int,
    exact: bool,
):
    """
    Validate that all values in a distribution belong to the allowed set
    and that the total count satisfies the exact/maximum constraint.

    Args:
        items:          List of {"value": ..., "count": ...} dicts.
        allowed_values: Set of permitted value strings.
        label:          Human-readable name of the distribution (used in error messages).
        total_count:    Expected total (exact match or upper bound depending on 'exact').
        exact:          If True, counts must sum to total_count; if False, must not exceed it.
    """
    for item in items:
        value = str(item.get("value") or "").strip()
        if value not in allowed_values:
            raise HTTPException(
                status_code=400,
                detail=f"{label}: invalid value '{value}'",
            )

    total = _sum_distribution(items)
    if exact and total != total_count:
        raise HTTPException(
            status_code=400,
            detail=f"{label}: sum must be exactly {total_count}",
        )

    if not exact and total > total_count:
        raise HTTPException(
            status_code=400,
            detail=f"{label}: sum must not exceed {total_count}",
        )


def aggregate_persona_text_from_payload(payload: PersonaBatchCreate) -> str:
    """
    Build the natural language group profile text used in aggregate simulation prompts.

    The text describes the demographic composition of the group in Slovak,
    for example: "Skupinu tvorí 17 používateľov. Vo veku 18 až 24 rokov je 10 používateľov..."
    This text is sent verbatim to the LLM as the 'Profil skupiny používateľov' section.
    """
    age_items = _nonzero_distribution_items(_normalize_distribution_items(payload.age_distribution))
    gender_items = _nonzero_distribution_items(_normalize_distribution_items(payload.genders_distribution))
    exp_items = _nonzero_distribution_items(
        _normalize_distribution_items(payload.similar_apps_experience_distribution)
    )
    style_items = _nonzero_distribution_items(
        _normalize_distribution_items(payload.decision_style_distribution)
    )
    fr_items = _nonzero_distribution_items(
        _normalize_distribution_items(payload.frustration_tolerance_distribution)
    )

    parts: list[str] = [f"Skupinu tvorí {payload.count} používateľov."]

    if age_items:
        age_phrases = []
        for value, count in age_items:
            if count == 1:
                age_phrases.append(
                    f"vo veku {_format_age_band_label(value)} je 1 používateľ"
                )
            elif count in {2, 3, 4}:
                age_phrases.append(
                    f"vo veku {_format_age_band_label(value)} sú {count} používatelia"
                )
            else:
                age_phrases.append(
                    f"vo veku {_format_age_band_label(value)} je {count} používateľov"
                )

        parts.append(_join_with_and(age_phrases) + ".")

    if gender_items:
        gender_map = {value: count for value, count in gender_items}
        men = int(gender_map.get("muži", 0))
        women = int(gender_map.get("ženy", 0))

        if men > 0 and women > 0:
            parts.append(f"V skupine je {men} mužov a {women} žien.")
        elif men > 0:
            parts.append(f"V skupine je {men} mužov.")
        elif women > 0:
            parts.append(f"V skupine je {women} žien.")

    if exp_items:
        exp_phrases = [
            f"{value} skúsenosti s podobnými aplikáciami má {_user_count_text(count)}"
            for value, count in exp_items
        ]
        parts.append(_join_with_and(exp_phrases) + ".")

    if style_items:
        style_phrases = [
            f"{value} štýl rozhodovania má {_user_count_text(count)}"
            for value, count in style_items
        ]
        parts.append(_join_with_and(style_phrases) + ".")

    if fr_items:
        fr_phrases = [
            f"{value} toleranciu frustrácie má {_user_count_text(count)}"
            for value, count in fr_items
        ]
        parts.append(_join_with_and(fr_phrases) + ".")

    extra = (payload.extra_description or "").strip()
    if extra:
        parts.append(f"Doplňujúci opis: {extra}")

    return " ".join(parts)


def persona_text_from_characteristics(ch: dict[str, Any], batch: PersonaBatch | None = None) -> str:
    """
    Build the natural language description of a single persona used in N-person prompts.

    The text is generated from the persona's characteristics dict and, optionally,
    the batch's extra_description. Example output:
    "Používateľ má 23 rokov a je žena. S podobnými aplikáciami má časté skúsenosti.
     Štýl rozhodovania je analytický a tolerancia frustrácie je stredná."
    """
    parts: list[str] = []

    age = ch.get("age")
    gender = _as_nonempty_str(ch.get("gender"))
    exp = _as_nonempty_str(ch.get("similar_apps_experience"))
    style = _as_nonempty_str(ch.get("decision_style"))
    fr = _as_nonempty_str(ch.get("frustration_tolerance"))

    if age is not None and gender == "ženy":
        parts.append(f"Používateľ má {age} rokov a je žena.")
    elif age is not None and gender == "muži":
        parts.append(f"Používateľ má {age} rokov a je muž.")
    elif age is not None:
        parts.append(f"Používateľ má {age} rokov.")
    elif gender == "ženy":
        parts.append("Používateľ je žena.")
    elif gender == "muži":
        parts.append("Používateľ je muž.")

    if exp:
        parts.append(
            f"S podobnými aplikáciami má {_sentence_lower_first(exp)} skúsenosti."
        )

    if style and fr:
        parts.append(
            f"Štýl rozhodovania je {style} a tolerancia frustrácie je {fr}."
        )
    elif style:
        parts.append(f"Štýl rozhodovania je {style}.")
    elif fr:
        parts.append(f"Tolerancia frustrácie je {fr}.")

    extra = ""
    if batch and (batch.extra_description or "").strip():
        extra = f"Doplňujúci opis: {batch.extra_description.strip()}"

    if not parts and not extra:
        return "Používateľ nemá bližšie určené charakteristiky."

    if extra:
        parts.append(extra)

    return " ".join(parts)


def build_persona_characteristics_list(payload: PersonaBatchCreate) -> list[dict[str, Any]]:
    """
    Generate a list of individual persona characteristic dicts from the batch distributions.

    Each dict contains the demographic attributes for one simulated user:
    age_band, age (randomly sampled within the band), gender, similar_apps_experience,
    decision_style, and frustration_tolerance.

    Age and gender distributions are mandatory (exact=True); the other three
    are optional (exact=False) — personas may have None for those attributes
    if the distribution does not cover all slots.
    """
    total_count = int(payload.count)

    age_distribution = _normalize_distribution_items(payload.age_distribution)
    genders_distribution = _normalize_distribution_items(payload.genders_distribution)
    experience_distribution = _normalize_distribution_items(payload.similar_apps_experience_distribution)
    decision_distribution = _normalize_distribution_items(payload.decision_style_distribution)
    frustration_distribution = _normalize_distribution_items(payload.frustration_tolerance_distribution)

    # Validate that each distribution only contains allowed values and correct counts
    validate_distribution_values(
        age_distribution,
        set(AGE_BANDS.keys()),
        "Vek",
        total_count=total_count,
        exact=True,
    )
    validate_distribution_values(
        genders_distribution,
        {"muži", "ženy"},
        "Pohlavie",
        total_count=total_count,
        exact=True,
    )
    validate_distribution_values(
        experience_distribution,
        {
            "Časté (aspoň niekoľkokrát do mesiaca)",
            "Občasné (niekoľkokrát ročne)",
            "Zriedkavé (menej ako raz ročne)",
        },
        "Skúsenosti s podobnými aplikáciami",
        total_count=total_count,
        exact=False,
    )
    validate_distribution_values(
        decision_distribution,
        {"rýchly", "analytický", "exploratívny"},
        "Štýl rozhodovania",
        total_count=total_count,
        exact=False,
    )
    validate_distribution_values(
        frustration_distribution,
        {"nízka", "stredná", "vysoká"},
        "Tolerancia frustrácie",
        total_count=total_count,
        exact=False,
    )

    # Expand each distribution into a shuffled list of length total_count
    age_slots = _expand_distribution_to_slots(age_distribution, total_count, exact=True)
    gender_slots = _expand_distribution_to_slots(genders_distribution, total_count, exact=True)
    exp_slots = _expand_distribution_to_slots(experience_distribution, total_count, exact=False)
    decision_slots = _expand_distribution_to_slots(decision_distribution, total_count, exact=False)
    frustration_slots = _expand_distribution_to_slots(frustration_distribution, total_count, exact=False)

    out: list[dict[str, Any]] = []

    for i in range(total_count):
        age_band = str(age_slots[i])
        ch: dict[str, Any] = {
            "age_band": age_band,
            "age": _rand_age_from_band(age_band),  # random age within the band
        }

        # Only include attributes that were explicitly assigned (non-None slots)
        if gender_slots[i] is not None:
            ch["gender"] = gender_slots[i]

        if exp_slots[i] is not None:
            ch["similar_apps_experience"] = exp_slots[i]

        if decision_slots[i] is not None:
            ch["decision_style"] = decision_slots[i]

        if frustration_slots[i] is not None:
            ch["frustration_tolerance"] = frustration_slots[i]

        out.append(ch)

    return out


def build_persona_batch_out(b: PersonaBatch) -> PersonaBatchOut:
    """Construct a PersonaBatchOut response object from a PersonaBatch ORM record."""
    age_distribution = b.age_distribution if isinstance(b.age_distribution, list) else []
    genders_distribution = b.genders_distribution if isinstance(b.genders_distribution, list) else []
    similar_apps_experience_distribution = (
        b.similar_apps_experience_distribution
        if isinstance(b.similar_apps_experience_distribution, list)
        else []
    )
    decision_style_distribution = (
        b.decision_style_distribution
        if isinstance(b.decision_style_distribution, list)
        else []
    )
    frustration_tolerance_distribution = (
        b.frustration_tolerance_distribution
        if isinstance(b.frustration_tolerance_distribution, list)
        else []
    )

    return PersonaBatchOut(
        id=int(b.id),
        name=b.name,
        count=int(b.count),
        age_from=int(b.age_from),
        age_to=int(b.age_to),
        genders=list(b.genders or []),
        similar_apps_experience=list(b.similar_apps_experience or []),
        decision_style=list(b.decision_style or []),
        frustration_tolerance=list(b.frustration_tolerance or []),
        age_distribution=age_distribution,
        genders_distribution=genders_distribution,
        similar_apps_experience_distribution=similar_apps_experience_distribution,
        decision_style_distribution=decision_style_distribution,
        frustration_tolerance_distribution=frustration_tolerance_distribution,
        extra_description=b.extra_description or "",
        aggregate_persona_text=b.aggregate_persona_text or "",
        created_at=b.created_at,
    )


# ---------------------------------------------------------------------------
# Persona batch endpoints
# ---------------------------------------------------------------------------

@app.post("/personas/batch", response_model=PersonaBatchCreateOut, summary="Create persona batch")
def create_personas_batch(payload: PersonaBatchCreate, db: Session = Depends(get_db)):
    """
    Create a new persona batch from demographic distributions.

    Validates all distributions, generates individual persona records with randomised
    but distribution-compliant characteristics, and builds the aggregate persona text
    used for aggregate simulation mode.

    Returns the batch ID, the generated aggregate persona text, and the list of created persona IDs.
    """
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")

    if payload.count < 5 or payload.count > 50:
        raise HTTPException(status_code=400, detail="count must be between 5 and 50")

    age_distribution = _normalize_distribution_items(payload.age_distribution)
    genders_distribution = _normalize_distribution_items(payload.genders_distribution)
    similar_apps_experience_distribution = _normalize_distribution_items(
        payload.similar_apps_experience_distribution
    )
    decision_style_distribution = _normalize_distribution_items(payload.decision_style_distribution)
    frustration_tolerance_distribution = _normalize_distribution_items(
        payload.frustration_tolerance_distribution
    )

    validate_distribution_values(
        age_distribution, set(AGE_BANDS.keys()), "Vek",
        total_count=int(payload.count), exact=True,
    )
    validate_distribution_values(
        genders_distribution, {"muži", "ženy"}, "Pohlavie",
        total_count=int(payload.count), exact=True,
    )
    validate_distribution_values(
        similar_apps_experience_distribution,
        {
            "Časté (aspoň niekoľkokrát do mesiaca)",
            "Občasné (niekoľkokrát ročne)",
            "Zriedkavé (menej ako raz ročne)",
        },
        "Skúsenosti s podobnými aplikáciami",
        total_count=int(payload.count), exact=False,
    )
    validate_distribution_values(
        decision_style_distribution, {"rýchly", "analytický", "exploratívny"},
        "Štýl rozhodovania", total_count=int(payload.count), exact=False,
    )
    validate_distribution_values(
        frustration_tolerance_distribution, {"nízka", "stredná", "vysoká"},
        "Tolerancia frustrácie", total_count=int(payload.count), exact=False,
    )

    age_from, age_to = _min_max_from_age_distribution(age_distribution)

    # Build the aggregate persona text that will be sent to the LLM in aggregate mode
    agg_text = aggregate_persona_text_from_payload(payload)

    batch = PersonaBatch(
        name=name,
        count=int(payload.count),
        age_from=int(age_from),
        age_to=int(age_to),
        genders=_nonzero_distribution_values(genders_distribution),
        similar_apps_experience=_nonzero_distribution_values(similar_apps_experience_distribution),
        decision_style=_nonzero_distribution_values(decision_style_distribution),
        frustration_tolerance=_nonzero_distribution_values(frustration_tolerance_distribution),
        age_distribution=age_distribution,
        genders_distribution=genders_distribution,
        similar_apps_experience_distribution=similar_apps_experience_distribution,
        decision_style_distribution=decision_style_distribution,
        frustration_tolerance_distribution=frustration_tolerance_distribution,
        extra_description=(payload.extra_description or "").strip(),
        aggregate_persona_text=agg_text,
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    # Generate individual persona records from the distributions
    all_characteristics = build_persona_characteristics_list(payload)

    created_ids: list[int] = []
    for i, ch in enumerate(all_characteristics, start=1):
        p_text = persona_text_from_characteristics(ch, batch=batch)

        row = Persona(
            batch_id=int(batch.id),
            batch_index=i,
            characteristics=ch,
            extra_description=batch.extra_description or "",
            persona_text=p_text,
        )
        db.add(row)
        db.flush()
        created_ids.append(int(row.id))

    db.commit()

    return PersonaBatchCreateOut(
        batch_id=int(batch.id),
        aggregate_persona_text=agg_text,
        created_persona_ids=created_ids,
    )


@app.get("/persona-batches", response_model=list[PersonaBatchOut], summary="List all persona batches")
def list_persona_batches(db: Session = Depends(get_db)):
    """Return all persona batches ordered by creation date (newest first)."""
    rows = db.query(PersonaBatch).order_by(PersonaBatch.id.desc()).all()
    return [build_persona_batch_out(x) for x in rows]


@app.get("/persona-batches/{batch_id}", response_model=PersonaBatchDetailOut, summary="Get persona batch detail")
def get_persona_batch(batch_id: int, db: Session = Depends(get_db)):
    """Return a persona batch together with all its individual personas."""
    b = db.query(PersonaBatch).filter(PersonaBatch.id == batch_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Batch not found")

    persons = (
        db.query(Persona)
        .filter(Persona.batch_id == batch_id)
        .order_by(Persona.batch_index.asc())
        .all()
    )

    return PersonaBatchDetailOut(
        batch=build_persona_batch_out(b),
        personas=[PersonaOut.model_validate(p) for p in persons],
    )


@app.delete("/persona-batches/{batch_id}", summary="Delete persona batch")
def delete_persona_batch(batch_id: int, db: Session = Depends(get_db)):
    """
    Delete a persona batch and all its personas.

    Returns 409 if any test sessions reference this batch, to prevent data loss.
    """
    b = db.query(PersonaBatch).filter(PersonaBatch.id == batch_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Prevent deletion if sessions exist that reference this batch
    any_session = db.query(TestSession.id).filter(TestSession.batch_id == batch_id).first()
    if any_session:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete batch: existing test sessions reference it.",
        )

    db.query(Persona).filter(Persona.batch_id == batch_id).delete(synchronize_session=False)
    db.delete(b)
    db.commit()
    return {"ok": True}


@app.get("/personas", response_model=list[PersonaOut], summary="List all personas")
def list_personas(db: Session = Depends(get_db)):
    """Return all individual personas across all batches."""
    rows = db.query(Persona).order_by(Persona.id.asc()).all()
    return [PersonaOut.model_validate(x) for x in rows]


@app.delete("/personas/{persona_id}", summary="Delete persona")
def delete_persona(persona_id: int, db: Session = Depends(get_db)):
    """
    Delete an individual persona.

    Returns 409 if any test runs are associated with this persona.
    """
    row = db.query(Persona).filter(Persona.id == persona_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Persona not found")

    has_runs = db.query(TestRun.id).filter(TestRun.persona_id == persona_id).first()
    if has_runs:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete persona: it is referenced by existing test runs.",
        )

    db.delete(row)
    db.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Test task endpoints
# ---------------------------------------------------------------------------

@app.get("/test/tasks", response_model=list[TestTaskOut], summary="List test tasks")
def list_tasks(task_type: str | None = None, db: Session = Depends(get_db)):
    """
    Return all test tasks, optionally filtered by task_type.

    Tasks are returned in ascending order_index order within their type.
    """
    test = get_default_test(db)

    q = db.query(TestTask).filter(TestTask.test_id == test.id)

    if task_type is not None:
        if task_type not in VALID_TASK_TYPES:
            raise HTTPException(status_code=400, detail="Invalid task_type")
        q = q.filter(TestTask.task_type == task_type)

    return q.order_by(TestTask.order_index.asc()).all()


@app.post("/test/tasks", response_model=TestTaskOut, summary="Create test task")
def create_task(payload: TestTaskCreate, db: Session = Depends(get_db)):
    """
    Create a new UX test task.

    The task is assigned the next available order_index within its type.
    If no title is provided, the first 60 characters of task_text are used.
    Note: feedback tasks cannot have a follow_up_question.
    """
    test = get_default_test(db)

    task_type = payload.task_type
    task_text = (payload.task_text or "").strip()

    # Feedback tasks do not support follow-up questions
    follow_up_question = payload.follow_up_question
    if task_type == "feedback":
        follow_up_question = None

    config = payload.config or {}
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")

    # Remove follow_up_question from config if accidentally included
    if "follow_up_question" in config:
        config = dict(config)
        config.pop("follow_up_question", None)

    validate_task_payload(task_type, task_text, config, follow_up_question)

    order_index = next_order_for_type(db, test.id, task_type)

    title = (payload.title or "").strip()
    if not title:
        title = task_text[:60] + ("…" if len(task_text) > 60 else "")

    row = TestTask(
        test_id=test.id,
        task_type=task_type,
        order_index=order_index,
        title=title,
        task_text=task_text,
        follow_up_question=follow_up_question,
        config=config,
        is_active=payload.is_active,
    )

    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@app.put("/test/tasks/{task_id}", response_model=TestTaskOut, summary="Update test task")
def update_task(task_id: int, payload: TestTaskUpdate, db: Session = Depends(get_db)):
    """
    Partially update an existing test task.

    Only fields included in the request body are updated; omitted fields retain their current values.
    The task type cannot be changed after creation.
    """
    test = get_default_test(db)

    row = db.query(TestTask).filter(TestTask.id == task_id, TestTask.test_id == test.id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    new_title = row.title
    new_task_text = row.task_text
    new_follow = row.follow_up_question
    new_config = row.config
    new_active = row.is_active

    if payload.title is not None:
        new_title = payload.title.strip()

    if payload.task_text is not None:
        new_task_text = payload.task_text.strip()

    if payload.is_active is not None:
        new_active = bool(payload.is_active)

    if payload.config is not None:
        if not isinstance(payload.config, dict):
            raise HTTPException(status_code=400, detail="config must be an object")
        new_config = payload.config

    # Enforce: feedback tasks never have a follow-up question
    if row.task_type == "feedback":
        new_follow = None
    else:
        if payload.follow_up_question is not None:
            f = payload.follow_up_question.strip()
            new_follow = f if f else None

    if isinstance(new_config, dict) and "follow_up_question" in new_config:
        new_config = dict(new_config)
        new_config.pop("follow_up_question", None)

    if not new_title:
        new_title = new_task_text[:60] + ("…" if len(new_task_text) > 60 else "")

    validate_task_payload(row.task_type, new_task_text, new_config, new_follow)

    row.title = new_title
    row.task_text = new_task_text
    row.follow_up_question = new_follow
    row.config = new_config
    row.is_active = new_active

    db.commit()
    db.refresh(row)
    return row


@app.delete("/test/tasks/{task_id}", summary="Delete test task")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    """Delete a test task. This is a hard delete; historical run data is preserved via snapshots."""
    test = get_default_test(db)

    row = db.query(TestTask).filter(TestTask.id == task_id, TestTask.test_id == test.id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    db.delete(row)
    db.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

def build_task_snapshot(t: TestTask) -> dict[str, Any]:
    """
    Create an immutable snapshot of a task's state at the time a run is created.

    The snapshot is stored in TestRunItem.task_snapshot so that results can be
    interpreted even if the original task record is later modified or deleted.
    """
    return {
        "task_id": t.id,
        "task_type": t.task_type,
        "order_index": t.order_index,
        "title": t.title,
        "task_text": t.task_text,
        "follow_up_question": t.follow_up_question,
        "config": t.config or {},
    }


def _try_parse_json(s: str) -> dict[str, Any] | None:
    """Attempt to parse a string as JSON; return None if parsing fails."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def resolve_tasks_for_run(db: Session, task_ids: list[int] | None) -> list[TestTask]:
    """
    Resolve the list of tasks to be used in a run.

    If task_ids is None, all active tasks are used.
    If task_ids is provided, the tasks are returned in the requested order.
    Raises 400 if any requested task ID is not found or not active.
    """
    test = get_default_test(db)

    base_q = (
        db.query(TestTask)
        .filter(
            TestTask.test_id == test.id,
            TestTask.is_active == True,
            TestTask.task_type.in_(list(VALID_TASK_TYPES)),
        )
    )

    if task_ids is not None:
        ids = [int(x) for x in task_ids]
        ids = list(dict.fromkeys(ids))  # deduplicate while preserving order
        if len(ids) == 0:
            raise HTTPException(status_code=400, detail="task_ids is empty – select at least one task")

        tasks = base_q.filter(TestTask.id.in_(ids)).all()
        found_ids = {t.id for t in tasks}
        missing = [i for i in ids if i not in found_ids]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Some task_ids are not available (not found / inactive / wrong test): {missing}",
            )

        # Preserve the caller-requested ordering
        order_map = {tid: idx for idx, tid in enumerate(ids)}
        tasks.sort(key=lambda t: order_map.get(t.id, 10**9))
    else:
        tasks = base_q.all()

    if not tasks:
        raise HTTPException(status_code=400, detail="No active tasks found in the test")

    return tasks


def sort_task_results(items: list[SessionTaskResultOut]) -> list[SessionTaskResultOut]:
    """
    Sort task result items by task type order (preference, first_click, feedback)
    and then by order_index within each type.
    """
    def _sort_key(x: SessionTaskResultOut):
        snap = x.task_snapshot or {}
        ttype = str(snap.get("task_type") or "")
        o = int(snap.get("order_index") or 0)
        return (TYPE_ORDER.get(ttype, 99), o, int(x.task_id))

    return sorted(items, key=_sort_key)


def create_test_session_row(
    db: Session,
    *,
    test_id: int,
    batch: PersonaBatch,
    llm: LLMParamsIn,
    selected_task_ids: list[int],
) -> TestSession:
    """
    Insert a new TestSession record into the database.

    Snapshots the current batch metadata (name, count, aggregate text) so that
    the session result can be interpreted independently of future batch changes.
    """
    session = TestSession(
        test_id=test_id,
        batch_id=int(batch.id),
        status="created",
        llm_model=llm.model,
        llm_params=llm.model_dump(exclude_none=True),
        selected_task_ids=selected_task_ids,
        batch_name_snapshot=batch.name or "",
        batch_count_snapshot=int(batch.count or 0),
        aggregate_persona_text_snapshot=batch.aggregate_persona_text or "",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def create_run_with_items(
    db: Session,
    *,
    session_id: int,
    subject_type: str,
    batch_id: int,
    persona_id: int | None,
    llm: LLMParamsIn,
    tasks: list[TestTask],
) -> TestRun:
    """
    Create a TestRun and pre-allocate one TestRunItem per task.

    Items are created with answer_json=None; they are populated when
    execute_single_run() is called. This allows tracking which items
    succeeded and which failed independently.
    """
    if subject_type not in VALID_SUBJECT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid subject_type")

    if subject_type == "persona" and persona_id is None:
        raise HTTPException(status_code=400, detail="subject_type='persona' requires a persona_id")

    if subject_type == "aggregate" and persona_id is not None:
        raise HTTPException(status_code=400, detail="subject_type='aggregate' must have persona_id=null")

    run = TestRun(
        session_id=session_id,
        subject_type=subject_type,
        persona_id=persona_id,
        batch_id=batch_id,
        status="created",
        llm_model=llm.model,
        llm_params=llm.model_dump(exclude_none=True),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Pre-allocate one item per task with null results
    for t in tasks:
        snap = build_task_snapshot(t)
        item = TestRunItem(
            run_id=run.id,
            task_id=t.id,
            task_snapshot=snap,
            answer_json=None,
            raw_responses=None,
            raw_output_text=None,
            result_mode=None,
            usage=None,
            error_message=None,
        )
        db.add(item)

    db.commit()
    return run


def get_run_subject_text_and_count(db: Session, run: TestRun) -> tuple[str, int]:
    """
    Return the persona/group description text and the number of users it represents.

    For aggregate runs: returns the pre-built group profile text and the batch count.
    For persona runs: returns the individual persona text and count=1.
    """
    batch = db.query(PersonaBatch).filter(PersonaBatch.id == run.batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    if run.subject_type == "aggregate":
        agg = (batch.aggregate_persona_text or "").strip()
        # Fall back to a minimal description if the aggregate text is missing
        text = agg or (
            f"Agregovaná persona skupiny {batch.name}. "
            f"Vekový rozsah: {batch.age_from}–{batch.age_to}."
        )
        return text, int(batch.count)

    if run.subject_type == "persona":
        if run.persona_id is None:
            raise HTTPException(status_code=400, detail="Persona run is missing persona_id")

        persona = db.query(Persona).filter(Persona.id == run.persona_id).first()
        if not persona:
            raise HTTPException(status_code=404, detail="Persona not found")

        text = persona.persona_text or persona_text_from_characteristics(
            persona.characteristics or {},
            batch=batch,
        )
        return text, 1

    raise HTTPException(status_code=400, detail="Invalid subject_type in run")


def build_messages_for_task(
    subject_text: str,
    subject_type: str,
    represented_count: int,
    snap: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Construct the OpenAI chat messages (system + user) for a single task.

    The prompt structure varies by task type and simulation mode:

    Aggregate mode:
      - System: instructs the model to simulate a group of users
      - User: includes the group profile, task instruction, expected response count,
              and a JSON schema example showing the required output format.
              For preference/first_click tasks, images are appended as image_url content blocks.

    N-person mode (persona):
      - System: instructs the model to simulate one specific user
      - User: includes the individual persona profile, task instruction,
              and a JSON schema example for a single response.

    All prompts explicitly instruct the model to:
      - Decide based only on the profile and what is visibly present in the screenshot
      - Not invent UI elements that are not shown
      - Respond in Slovak
      - Return JSON
    """
    task_type = str(snap.get("task_type") or "")
    task_text = str(snap.get("task_text") or "").strip()
    follow = str(snap.get("follow_up_question") or "").strip()
    config = snap.get("config") or {}

    # --- System message: defines the simulation role ---
    if subject_type == "aggregate":
        subject_intro = "Profil skupiny používateľov"
        system = (
            "Simuluješ skupinu používateľov podľa agregovaného profilu. "
            "Každá odpoveď má reprezentovať jedného člena skupiny. "
            "Rozhoduj sa len podľa profilu skupiny, zadania a toho, čo je jasne viditeľné na screenshote. "
            "Nevymýšľaj prvky, texty ani funkcionalitu, ktoré na obrázku nie sú. "
            "Píš po slovensky. "
            "Vráť JSON."
        )
    else:
        subject_intro = "Profil používateľa"
        system = (
            "Simuluješ jedného používateľa podľa zadanej persony. "
            "Rozhoduj sa len podľa persony, zadania a toho, čo je jasne viditeľné na screenshote. "
            "Nevymýšľaj prvky, texty ani funkcionalitu, ktoré na obrázku nie sú. "
            "Píš po slovensky. "
            "Vráť JSON."
        )

    # --- User message: task-type specific prompt + images ---

    if task_type == "preference":
        options = config.get("options") if isinstance(config, dict) else None
        if not isinstance(options, list) or len(options) < 2:
            options = []

        option_labels = [
            str((opt or {}).get("label") or "").strip()
            for opt in options
            if str((opt or {}).get("label") or "").strip()
        ]

        if subject_type == "aggregate":
            # Aggregate: model returns an array of responses, one per simulated user
            if follow:
                json_hint = {
                    "responses": [
                        {
                            "simulated_index": 1,
                            "chosen_label": "jedna z labels",
                            "follow_up_response": "odpoveď na doplňujúcu otázku",
                        }
                    ]
                }
            else:
                json_hint = {
                    "responses": [
                        {
                            "simulated_index": 1,
                            "chosen_label": "jedna z labels",
                        }
                    ]
                }

            prompt_text = (
                f"{subject_intro}:\n{subject_text}\n\n"
                f"Úloha preference:\n{task_text}\n\n"
                f"Simuluj presne {represented_count} používateľov z tejto skupiny.\n"
                f"Každý objekt v poli responses predstavuje jedného simulovaného používateľa.\n"
                f"Pole responses musí mať presne {represented_count} položiek.\n"
                "Pre každého vyber jednu možnosť, ktorú by si najpravdepodobnejšie zvolil ako používateľ.\n"
                "Rozhoduj sa prirodzene podľa profilu skupiny\n"
                f"chosen_label musí byť presne jedna z týchto možností: {', '.join(option_labels)}.\n"
            )

            if follow:
                prompt_text += (
                    f"Doplňujúca otázka:\n{follow}\n"
                    "Ak je doplňujúca otázka zadaná, vyplň follow_up_response.\n\n"
                )
            else:
                prompt_text += "\n"

            prompt_text += (
                "Vráť JSON v tvare:\n"
                f"{json.dumps(json_hint, ensure_ascii=False)}"
            )
        else:
            # N-person: model returns a single response for one persona
            if follow:
                json_hint = {
                    "chosen_label": "jedna z labels",
                    "follow_up_response": "odpoveď na doplňujúcu otázku",
                }
            else:
                json_hint = {
                    "chosen_label": "jedna z labels",
                }

            prompt_text = (
                f"{subject_intro}:\n{subject_text}\n\n"
                f"Úloha preference:\n{task_text}\n\n"
                "Vyber jednu možnosť, ktorú by si ako tento používateľ najpravdepodobnejšie zvolil.\n"
                f"chosen_label musí byť presne jedna z týchto možností: {', '.join(option_labels)}.\n"
            )

            if follow:
                prompt_text += (
                    f"Doplňujúca otázka:\n{follow}\n"
                    "Ak je doplňujúca otázka zadaná, vyplň follow_up_response.\n\n"
                )
            else:
                prompt_text += "\n"

            prompt_text += (
                "Vráť JSON v tvare:\n"
                f"{json.dumps(json_hint, ensure_ascii=False)}"
            )

        # Build content array: text prompt followed by labelled images for each option
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        for opt in options:
            label = str((opt or {}).get("label") or "").strip()
            img = str((opt or {}).get("image") or "").strip()
            if label:
                content.append({"type": "text", "text": f"Label možnosti: {label}"})
            if img.startswith("data:image/"):
                content.append({"type": "image_url", "image_url": {"url": img}})

        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    if task_type == "first_click":
        img = ""
        if isinstance(config, dict):
            img = str(config.get("image") or "").strip()

        if subject_type == "aggregate":
            if follow:
                json_hint = {
                    "responses": [
                        {
                            "simulated_index": 1,
                            "click_target": "textový popis miesta prvého kliku",
                            "follow_up_response": "odpoveď na doplňujúcu otázku",
                        }
                    ]
                }
            else:
                json_hint = {
                    "responses": [
                        {
                            "simulated_index": 1,
                            "click_target": "textový popis miesta prvého kliku",
                        }
                    ]
                }

            prompt_text = (
                f"{subject_intro}:\n{subject_text}\n\n"
                f"Úloha first click:\n{task_text}\n\n"
                f"Simuluj presne {represented_count} používateľov z tejto skupiny.\n"
                f"Pole responses musí mať presne {represented_count} položiek.\n"
                "Pre každého uveď, kam by smeroval prvý klik ako prvý prirodzený pokus splniť úlohu.\n"
                "click_target opíš podľa toho, čo je na obrazovke reálne viditeľné.\n"
            )

            if follow:
                prompt_text += (
                    f"Doplňujúca otázka:\n{follow}\n"
                    "Ak je doplňujúca otázka zadaná, vyplň follow_up_response.\n\n"
                )
            else:
                prompt_text += "\n"

            prompt_text += (
                "Vráť JSON v tvare:\n"
                f"{json.dumps(json_hint, ensure_ascii=False)}"
            )
        else:
            if follow:
                json_hint = {
                    "click_target": "textový popis miesta prvého kliku",
                    "follow_up_response": "odpoveď na doplňujúcu otázku",
                }
            else:
                json_hint = {
                    "click_target": "textový popis miesta prvého kliku",
                }

            prompt_text = (
                f"{subject_intro}:\n{subject_text}\n\n"
                f"Úloha first click:\n{task_text}\n\n"
                "Uveď, kam by si ako tento používateľ klikol ako prvý prirodzený pokus splniť úlohu.\n"
                "click_target opíš podľa toho, čo je na obrazovke reálne viditeľné.\n"
            )

            if follow:
                prompt_text += (
                    f"Doplňujúca otázka:\n{follow}\n"
                    "Ak je doplňujúca otázka zadaná, vyplň follow_up_response.\n\n"
                )
            else:
                prompt_text += "\n"

            prompt_text += (
                "Vráť JSON:\n"
                f"{json.dumps(json_hint, ensure_ascii=False)}"
            )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        if img.startswith("data:image/"):
            content.append({"type": "image_url", "image_url": {"url": img}})

        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    # --- Feedback task ---
    img = ""
    if isinstance(config, dict):
        img = str(config.get("image") or "").strip()

    if subject_type == "aggregate":
        json_hint = {
            "responses": [
                {
                    "simulated_index": 1,
                    "notes": [],   # list of feedback observations
                }
            ]
        }

        prompt_text = (
            f"{subject_intro}:\n{subject_text}\n\n"
            f"Úloha spätnej väzby:\n{task_text}\n\n"
            f"Pole responses musí mať presne {represented_count} položiek.\n"
            "Každý objekt predstavuje jedného simulovaného používateľa.\n"
            "Do notes uveď body spätnej väzby z pohľadu používateľa.\n"
            "Zameraj sa len na to, čo by si daný používateľ reálne všimol na základe screenshotu a zadania.\n"
            "Vráť JSON v tvare:\n"
            f"{json.dumps(json_hint, ensure_ascii=False)}"
        )
    else:
        json_hint = {
            "notes": [],   # list of feedback observations for this single persona
        }

        prompt_text = (
            f"{subject_intro}:\n{subject_text}\n\n"
            f"Úloha spätnej väzby:\n{task_text}\n\n"
            "Do notes uveď body spätnej väzby z pohľadu používateľa.\n"
            "Zameraj sa len na to, čo by si si ako tento používateľ reálne všimol na základe screenshotu a zadania.\n"
            "Vráť JSON v tvare:\n"
            f"{json.dumps(json_hint, ensure_ascii=False)}"
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    if img.startswith("data:image/"):
        content.append({"type": "image_url", "image_url": {"url": img}})

    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# LLM API call
# ---------------------------------------------------------------------------

def call_openai_for_task(llm: LLMParamsIn, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    """
    Call the OpenAI Chat Completions API for a single task.

    Passes temperature and top_p only if they are explicitly set (None = use model default).
    When response_format is "json_object", the model is instructed to return valid JSON.

    Returns:
        text  : The raw text content of the model's response.
        usage : Token usage statistics dict (prompt_tokens, completion_tokens, total_tokens),
                or None if usage data is unavailable.
    """
    kwargs: dict[str, Any] = {
        "model": llm.model,
        "messages": messages,
    }

    if llm.temperature is not None:
        kwargs["temperature"] = llm.temperature

    if llm.top_p is not None:
        kwargs["top_p"] = llm.top_p

    if llm.response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()

    # Extract token usage for cost tracking; gracefully handle missing data
    usage = None
    try:
        if getattr(resp, "usage", None):
            usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
    except Exception:
        usage = None

    return text, usage


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------

def normalize_task_responses(
    task_type: str,
    parsed: dict[str, Any] | None,
    subject_type: str,
) -> list[dict[str, Any]] | None:
    """
    Normalise the parsed LLM JSON response into a uniform list of response dicts.

    Aggregate responses are expected to have a 'responses' array; persona responses
    may be a flat dict or wrapped in a single-item 'responses' array.

    Returns None if the input is None (indicating a JSON parse failure upstream).
    """
    if parsed is None:
        return None

    if subject_type == "aggregate":
        # Aggregate: expect {"responses": [...]}
        raw_list = parsed.get("responses")
        if isinstance(raw_list, list):
            source_list = raw_list
        else:
            source_list = [parsed]
    else:
        # Persona: expect a flat dict or {"responses": [single_item]}
        if isinstance(parsed.get("responses"), list) and parsed["responses"]:
            source_list = [parsed["responses"][0]]
        else:
            source_list = [parsed]

    out: list[dict[str, Any]] = []

    if task_type == "preference":
        for idx, item in enumerate(source_list, start=1):
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "simulated_index": int(item.get("simulated_index") or idx),
                    "chosen_label": _as_nonempty_str(item.get("chosen_label")) or "",
                    "follow_up_response": _as_nonempty_str(item.get("follow_up_response")),
                }
            )
        return out

    if task_type == "first_click":
        for idx, item in enumerate(source_list, start=1):
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "simulated_index": int(item.get("simulated_index") or idx),
                    "click_target": _as_nonempty_str(item.get("click_target")) or "",
                    "follow_up_response": _as_nonempty_str(item.get("follow_up_response")),
                }
            )
        return out

    if task_type == "feedback":
        for idx, item in enumerate(source_list, start=1):
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "simulated_index": int(item.get("simulated_index") or idx),
                    "notes": _normalize_notes(item.get("notes")),
                }
            )
        return out

    return None


def build_summary_for_task(task_type: str, snap: dict[str, Any], responses: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build an aggregated summary of all responses for a single task.

    The summary is stored as answer_json on TestRunItem and displayed in the frontend.

    - preference  : vote counts per option label, total responses, open answer count
    - first_click : click counts per target description, total responses
    - feedback    : total responses, number with notes, total note count
    """
    if task_type == "preference":
        config = snap.get("config") or {}
        options = config.get("options") if isinstance(config, dict) else []
        ordered_labels = [
            str((opt or {}).get("label") or "").strip()
            for opt in options
            if str((opt or {}).get("label") or "").strip()
        ]

        counts = Counter()
        for r in responses:
            chosen = _as_nonempty_str(r.get("chosen_label"))
            if chosen:
                counts[chosen] += 1

        # Preserve the original option ordering; add unexpected labels at the end
        selected_counts = [{"label": label, "count": int(counts.get(label, 0))} for label in ordered_labels]

        for extra_label, cnt in counts.items():
            if extra_label not in ordered_labels:
                selected_counts.append({"label": extra_label, "count": int(cnt)})

        open_answer_count = sum(
            1 for r in responses if _as_nonempty_str(r.get("follow_up_response")) is not None
        )

        return {
            "task_type": "preference",
            "selected_counts": selected_counts,
            "total_responses": len(responses),
            "open_answer_count": int(open_answer_count),
            "has_follow_up": bool(str(snap.get("follow_up_question") or "").strip()),
        }

    if task_type == "first_click":
        counts = Counter()
        for r in responses:
            target = _as_nonempty_str(r.get("click_target"))
            if target:
                counts[target] += 1

        click_counts = [{"target": k, "count": int(v)} for k, v in counts.items()]

        open_answer_count = sum(
            1 for r in responses if _as_nonempty_str(r.get("follow_up_response")) is not None
        )

        return {
            "task_type": "first_click",
            "click_counts": click_counts,
            "total_responses": len(responses),
            "open_answer_count": int(open_answer_count),
            "has_follow_up": bool(str(snap.get("follow_up_question") or "").strip()),
        }

    # Feedback summary
    responses_with_notes = 0
    note_count = 0
    for r in responses:
        notes = r.get("notes")
        if isinstance(notes, list) and len(notes) > 0:
            responses_with_notes += 1
            note_count += len(notes)

    return {
        "task_type": "feedback",
        "total_responses": len(responses),
        "responses_with_notes": int(responses_with_notes),
        "note_count": int(note_count),
    }


def build_raw_responses_payload(
    task_type: str,
    subject_type: str,
    represented_count: int,
    responses: list[dict[str, Any]],
    *,
    grouped: bool = False,
) -> dict[str, Any]:
    """
    Wrap normalised per-user responses in a metadata envelope stored as raw_responses.

    The 'grouped' flag indicates whether the payload contains merged results
    from multiple persona runs (used when building persona_group results).
    """
    return {
        "task_type": task_type,
        "subject_type": subject_type,
        "represented_count": int(represented_count),
        "response_count": len(responses),
        "grouped": grouped,
        "responses": responses,
    }


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def execute_single_run(run: TestRun, db: Session) -> str:
    """
    Execute all pending tasks for a single TestRun by calling the OpenAI API.

    For each task item:
      1. Build the prompt messages (system + user with optional images).
      2. Call the OpenAI API.
      3. Parse and normalise the JSON response.
      4. Build summary (answer_json) and raw responses payload.
      5. Persist the result to the database immediately.

    If any item fails, its error_message is set and the run is marked as 'failed'.
    Successfully completed items are not re-executed on retry (idempotent).

    Returns the final run status: "finished" or "failed".
    """
    if run.status == "finished":
        return "finished"

    subject_text, represented_count = get_run_subject_text_and_count(db, run)

    items = db.query(TestRunItem).filter(TestRunItem.run_id == run.id).all()
    if not items:
        raise HTTPException(status_code=400, detail="Run has no items")

    def sort_key(it: TestRunItem):
        snap = it.task_snapshot or {}
        ttype = str(snap.get("task_type") or "")
        o = int(snap.get("order_index") or 0)
        return (TYPE_ORDER.get(ttype, 99), o, int(it.id))

    items_sorted = sorted(items, key=sort_key)
    llm = LLMParamsIn(**(run.llm_params or {"model": run.llm_model}))

    try:
        for it in items_sorted:
            # Skip already completed items (enables safe retry on partial failure)
            if it.answer_json is not None and it.error_message is None:
                continue

            snap = it.task_snapshot or {}
            task_type = str(snap.get("task_type") or "")
            if task_type not in VALID_TASK_TYPES:
                it.answer_json = None
                it.raw_responses = None
                it.raw_output_text = None
                it.result_mode = None
                it.usage = None
                it.error_message = f"Invalid task_type in snapshot: {task_type}"
                db.add(it)
                db.commit()
                continue

            # Build the prompt and call the LLM
            messages = build_messages_for_task(
                subject_text=subject_text,
                subject_type=run.subject_type,
                represented_count=represented_count,
                snap=snap,
            )

            try:
                content_text, usage = call_openai_for_task(llm, messages)
                parsed = _try_parse_json(content_text)
                normalized = normalize_task_responses(task_type, parsed, run.subject_type)

                it.result_mode = "aggregate_direct" if run.subject_type == "aggregate" else "persona_single"
                it.usage = usage
                it.error_message = None

                if normalized is None:
                    # JSON parsing failed; store the raw text for manual inspection
                    it.answer_json = {"task_type": task_type, "text": content_text}
                    it.raw_responses = None
                    it.raw_output_text = content_text
                else:
                    it.answer_json = build_summary_for_task(task_type, snap, normalized)
                    it.raw_responses = build_raw_responses_payload(
                        task_type=task_type,
                        subject_type=run.subject_type,
                        represented_count=represented_count,
                        responses=normalized,
                    )
                    it.raw_output_text = None

            except Exception as e:
                # Per-item error: record the error but continue with remaining items
                it.answer_json = None
                it.raw_responses = None
                it.raw_output_text = None
                it.result_mode = "aggregate_direct" if run.subject_type == "aggregate" else "persona_single"
                it.usage = None
                it.error_message = f"{type(e).__name__}: {str(e)}"

            db.add(it)
            db.commit()

        # Set the run status based on whether any items failed
        refreshed_items = db.query(TestRunItem).filter(TestRunItem.run_id == run.id).all()
        any_err = any((x.error_message for x in refreshed_items))
        run.status = "failed" if any_err else "finished"
        run.error_message = "Some tasks failed" if any_err else None
        db.add(run)
        db.commit()

        return run.status

    except Exception as e:
        run.status = "failed"
        run.error_message = f"{type(e).__name__}: {str(e)}"
        db.add(run)
        db.commit()
        raise HTTPException(status_code=502, detail="Run execution failed on the server")


def refresh_session_status(db: Session, session_id: int) -> str:
    """
    Recompute and persist the status of a TestSession based on its runs.

    Rules:
      - Any failed run  -> session is "failed"
      - All finished    -> session is "finished"
      - Otherwise       -> session is "created"
    """
    session = db.query(TestSession).filter(TestSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    runs = db.query(TestRun).filter(TestRun.session_id == session_id).all()
    if not runs:
        session.status = "created"
        session.error_message = None
        db.add(session)
        db.commit()
        return session.status

    statuses = [str(r.status or "created") for r in runs]

    if any(s == "failed" for s in statuses):
        session.status = "failed"
        session.error_message = "Some runs in the session failed"
    elif all(s == "finished" for s in statuses):
        session.status = "finished"
        session.error_message = None
    else:
        session.status = "created"
        session.error_message = None

    db.add(session)
    db.commit()
    return session.status


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def build_run_out(db: Session, r: TestRun) -> RunOut:
    """Construct a RunOut response object from a TestRun ORM record."""
    batch = db.query(PersonaBatch).filter(PersonaBatch.id == r.batch_id).first()
    session = db.query(TestSession).filter(TestSession.id == r.session_id).first()

    if r.subject_type == "aggregate":
        batch_name = batch.name if batch else (session.batch_name_snapshot if session else f"batch#{r.batch_id}")
        persona_name = f"{batch_name} (agregovaná persona)"
    else:
        p = db.query(Persona).filter(Persona.id == r.persona_id).first()
        persona_name = f"persona#{r.persona_id}"
        if p:
            batch_name = batch.name if batch else (session.batch_name_snapshot if session else f"batch#{p.batch_id}")
            persona_name = f"{batch_name} #{p.batch_index}"

    return RunOut(
        id=int(r.id),
        session_id=int(r.session_id),
        subject_type=r.subject_type,
        persona_id=int(r.persona_id) if r.persona_id is not None else None,
        batch_id=int(r.batch_id),
        persona_name=persona_name,
        status=r.status,
        llm_model=r.llm_model,
        llm_params=r.llm_params or {},
    )


def build_session_out(db: Session, s: TestSession) -> TestSessionOut:
    """Construct a TestSessionOut response object from a TestSession ORM record."""
    runs = db.query(TestRun).filter(TestRun.session_id == s.id).all()
    aggregate_run = next((r for r in runs if r.subject_type == "aggregate"), None)
    persona_run_count = sum(1 for r in runs if r.subject_type == "persona")

    selected_task_ids = s.selected_task_ids if isinstance(s.selected_task_ids, (list, dict)) else []
    selected_task_count = len(selected_task_ids) if isinstance(selected_task_ids, list) else 0

    return TestSessionOut(
        id=int(s.id),
        test_id=int(s.test_id),
        batch_id=int(s.batch_id),
        batch_name_snapshot=s.batch_name_snapshot or "",
        batch_count_snapshot=int(s.batch_count_snapshot or 0),
        aggregate_persona_text_snapshot=s.aggregate_persona_text_snapshot or "",
        status=s.status,
        llm_model=s.llm_model,
        llm_params=s.llm_params or {},
        selected_task_ids=selected_task_ids,
        created_at=s.created_at,
        total_runs=len(runs),
        aggregate_run_id=int(aggregate_run.id) if aggregate_run else None,
        persona_run_count=persona_run_count,
        selected_task_count=selected_task_count,
    )


def build_session_task_item_from_run_item(it: TestRunItem) -> SessionTaskResultOut:
    """Convert a TestRunItem ORM record to a SessionTaskResultOut response object."""
    return SessionTaskResultOut(
        id=int(it.id),
        task_id=int(it.task_id),
        task_snapshot=it.task_snapshot or {},
        answer_json=it.answer_json,
        raw_responses=it.raw_responses,
        raw_output_text=it.raw_output_text,
        result_mode=it.result_mode,
        usage=it.usage,
        error_message=it.error_message,
    )


def get_session_runs(db: Session, session_id: int) -> tuple[TestRun | None, list[TestRun]]:
    """
    Return the aggregate run and the list of persona runs for a session.

    Persona runs are sorted by their persona's batch_index for consistent ordering.
    """
    runs = (
        db.query(TestRun)
        .filter(TestRun.session_id == session_id)
        .order_by(TestRun.id.asc())
        .all()
    )

    aggregate_run = next((r for r in runs if r.subject_type == "aggregate"), None)
    persona_runs = [r for r in runs if r.subject_type == "persona"]

    # Build a mapping from persona_id to batch_index for sorting
    persona_map = {
        int(p.id): int(p.batch_index)
        for p in db.query(Persona).filter(Persona.batch_id == (runs[0].batch_id if runs else -1)).all()
    }

    persona_runs_sorted = sorted(
        persona_runs,
        key=lambda x: persona_map.get(int(x.persona_id or 0), 10**9),
    )

    return aggregate_run, persona_runs_sorted


def get_latest_session_for_batch(db: Session, batch_id: int) -> TestSession | None:
    """Return the most recently created TestSession for a given batch, or None."""
    return (
        db.query(TestSession)
        .filter(TestSession.batch_id == batch_id)
        .order_by(TestSession.id.desc())
        .first()
    )


def build_aggregate_block_out(db: Session, session_id: int) -> AggregateBlockOut | None:
    """Build the aggregate simulation result block for a session detail response."""
    aggregate_run, _ = get_session_runs(db, session_id)
    if not aggregate_run:
        return None

    items = (
        db.query(TestRunItem)
        .filter(TestRunItem.run_id == aggregate_run.id)
        .order_by(TestRunItem.id.asc())
        .all()
    )

    item_outs = [build_session_task_item_from_run_item(x) for x in items]
    item_outs = sort_task_results(item_outs)

    return AggregateBlockOut(
        run=build_run_out(db, aggregate_run),
        items=item_outs,
    )


def build_persona_group_out(db: Session, session_id: int) -> PersonaGroupOut:
    """
    Build the N-person simulation result block for a session detail response.

    Merges individual persona run items grouped by task_id so that the frontend
    can display aggregated statistics across all personas for each task.
    Each merged item contains all individual responses annotated with persona metadata.
    """
    _, persona_runs = get_session_runs(db, session_id)

    if not persona_runs:
        return PersonaGroupOut(total_personas=0, run_count=0, items=[])

    persona_name_by_run_id: dict[int, str] = {
        int(run.id): build_run_out(db, run).persona_name for run in persona_runs
    }

    run_ids = [int(r.id) for r in persona_runs]
    all_items = (
        db.query(TestRunItem)
        .filter(TestRunItem.run_id.in_(run_ids))
        .order_by(TestRunItem.id.asc())
        .all()
    )

    run_by_id = {int(r.id): r for r in persona_runs}

    # Group items by task_id and merge individual responses
    grouped: dict[int, dict[str, Any]] = {}

    for it in all_items:
        snap = it.task_snapshot or {}
        task_id = int(it.task_id)

        if task_id not in grouped:
            grouped[task_id] = {
                "task_id": task_id,
                "task_snapshot": snap,
                "responses": [],
                "errors": [],
            }

        run = run_by_id.get(int(it.run_id))
        if run is None:
            continue

        if it.error_message:
            grouped[task_id]["errors"].append(
                f"{persona_name_by_run_id.get(int(run.id), f'run#{run.id}')}: {it.error_message}"
            )

        # Enrich each response with the persona's identity for traceability
        payload = it.raw_responses or {}
        response_list = payload.get("responses") if isinstance(payload, dict) else None
        if isinstance(response_list, list):
            for resp in response_list:
                if not isinstance(resp, dict):
                    continue
                enriched = dict(resp)
                enriched["persona_id"] = int(run.persona_id) if run.persona_id is not None else None
                enriched["persona_name"] = persona_name_by_run_id.get(int(run.id), f"run#{run.id}")
                grouped[task_id]["responses"].append(enriched)

    items_out: list[SessionTaskResultOut] = []

    for task_id, data in grouped.items():
        snap = data["task_snapshot"] or {}
        task_type = str(snap.get("task_type") or "")
        responses = data["responses"] or []
        errors = data["errors"] or []

        answer_json = None
        raw_responses = None
        if task_type in VALID_TASK_TYPES and responses:
            answer_json = build_summary_for_task(task_type, snap, responses)
            raw_responses = build_raw_responses_payload(
                task_type=task_type,
                subject_type="persona",
                represented_count=len(persona_runs),
                responses=responses,
                grouped=True,
            )

        items_out.append(
            SessionTaskResultOut(
                id=None,
                task_id=int(task_id),
                task_snapshot=snap,
                answer_json=answer_json,
                raw_responses=raw_responses,
                raw_output_text=None,
                result_mode="persona_group_aggregated",
                usage=None,
                error_message=" | ".join(errors) if errors else None,
            )
        )

    items_out = sort_task_results(items_out)

    return PersonaGroupOut(
        total_personas=len(persona_runs),
        run_count=len(persona_runs),
        items=items_out,
    )


# ---------------------------------------------------------------------------
# Run and session endpoints
# ---------------------------------------------------------------------------

@app.post("/runs", response_model=RunCreateOut, summary="Create single-persona run")
def create_run(payload: RunCreate, db: Session = Depends(get_db)):
    """
    Create a test session with a single persona run (N-person mode, one persona).

    Used for testing a single persona against a task set without the full batch workflow.
    """
    persona = db.query(Persona).filter(Persona.id == payload.persona_id).first()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    batch = db.query(PersonaBatch).filter(PersonaBatch.id == persona.batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    tasks = resolve_tasks_for_run(db, payload.task_ids)
    test = get_default_test(db)

    session = create_test_session_row(
        db,
        test_id=int(test.id),
        batch=batch,
        llm=payload.llm,
        selected_task_ids=[int(t.id) for t in tasks],
    )

    run = create_run_with_items(
        db,
        session_id=int(session.id),
        subject_type="persona",
        batch_id=int(persona.batch_id),
        persona_id=int(persona.id),
        llm=payload.llm,
        tasks=tasks,
    )

    return RunCreateOut(run_id=int(run.id))


@app.post("/test-sessions", response_model=BatchRunsCreateOut, summary="Create full test session")
@app.post("/runs/batch", response_model=BatchRunsCreateOut)
def create_test_session(payload: BatchRunsCreate, db: Session = Depends(get_db)):
    """
    Create a full test session for a persona batch.

    Creates one aggregate run and one persona run per persona in the batch.
    The session groups all runs together for coordinated execution and result viewing.
    Runs are not executed immediately; call /test-sessions/{session_id}/execute to start.
    """
    batch = db.query(PersonaBatch).filter(PersonaBatch.id == payload.batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    personas = (
        db.query(Persona)
        .filter(Persona.batch_id == batch.id)
        .order_by(Persona.batch_index.asc())
        .all()
    )
    if not personas:
        raise HTTPException(status_code=400, detail="Batch has no personas")

    tasks = resolve_tasks_for_run(db, payload.task_ids)
    test = get_default_test(db)

    session = create_test_session_row(
        db,
        test_id=int(test.id),
        batch=batch,
        llm=payload.llm,
        selected_task_ids=[int(t.id) for t in tasks],
    )

    # Create the aggregate run (one prompt simulates the whole group)
    aggregate_run = create_run_with_items(
        db,
        session_id=int(session.id),
        subject_type="aggregate",
        batch_id=int(batch.id),
        persona_id=None,
        llm=payload.llm,
        tasks=tasks,
    )

    # Create one persona run per individual persona
    persona_run_ids: list[int] = []
    for p in personas:
        run = create_run_with_items(
            db,
            session_id=int(session.id),
            subject_type="persona",
            batch_id=int(batch.id),
            persona_id=int(p.id),
            llm=payload.llm,
            tasks=tasks,
        )
        persona_run_ids.append(int(run.id))

    return BatchRunsCreateOut(
        session_id=int(session.id),
        batch_id=int(batch.id),
        aggregate_run_id=int(aggregate_run.id),
        persona_run_ids=persona_run_ids,
        total_runs=1 + len(persona_run_ids),
    )


@app.post("/runs/{run_id}/execute", summary="Execute a single run")
def execute_run(run_id: int, db: Session = Depends(get_db)):
    """Execute all pending tasks for a single run by calling the OpenAI API."""
    run = db.query(TestRun).filter(TestRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    status = execute_single_run(run, db)
    refresh_session_status(db, int(run.session_id))
    return {"ok": True, "status": status}


@app.post("/test-sessions/{session_id}/execute", summary="Execute all runs in a session")
def execute_test_session(session_id: int, db: Session = Depends(get_db)):
    """
    Execute all runs in a test session sequentially.

    The aggregate run is executed first, followed by all persona runs in order.
    Each run's result is persisted immediately so that partial results are available
    even if execution is interrupted. Progress is logged to stdout.
    """
    session = db.query(TestSession).filter(TestSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    aggregate_run, persona_runs = get_session_runs(db, session_id)
    if aggregate_run is None and not persona_runs:
        raise HTTPException(status_code=404, detail="This session has no runs")

    # Execute aggregate run first, then persona runs in order
    ordered_runs: list[TestRun] = []
    if aggregate_run is not None:
        ordered_runs.append(aggregate_run)
    ordered_runs.extend(persona_runs)

    total_runs = len(ordered_runs)
    print(f"[SESSION {session_id}] starting, total runs: {total_runs}")

    results: list[dict[str, Any]] = []
    any_failed = False

    for idx, run in enumerate(ordered_runs, start=1):
        remaining = total_runs - idx
        print(
            f"[SESSION {session_id}] executing run {idx}/{total_runs} "
            f"(run_id={run.id}, type={run.subject_type}, remaining={remaining})"
        )

        status = execute_single_run(run, db)
        if status == "failed":
            any_failed = True

        results.append(
            {
                "run_id": int(run.id),
                "session_id": int(run.session_id),
                "subject_type": run.subject_type,
                "persona_id": int(run.persona_id) if run.persona_id is not None else None,
                "status": status,
            }
        )

    session_status = refresh_session_status(db, session_id)
    print(f"[SESSION {session_id}] done, final status: {session_status}")

    return {
        "ok": True,
        "session_id": int(session_id),
        "status": session_status,
        "results": results,
        "any_failed": any_failed,
    }


@app.post("/runs/batch/{batch_id}/execute", summary="Execute latest session for a batch")
def execute_latest_session_for_batch(batch_id: int, db: Session = Depends(get_db)):
    """Execute the most recently created test session for the given batch."""
    session = get_latest_session_for_batch(db, batch_id)
    if not session:
        raise HTTPException(status_code=404, detail="No session found for this batch")

    return execute_test_session(int(session.id), db)


@app.get("/test-sessions", response_model=list[TestSessionOut], summary="List all test sessions")
def list_test_sessions(db: Session = Depends(get_db)):
    """Return all test sessions ordered by creation date (newest first)."""
    rows = db.query(TestSession).order_by(TestSession.id.desc()).all()
    return [build_session_out(db, x) for x in rows]


@app.get("/test-sessions/{session_id}", response_model=TestSessionDetailOut, summary="Get test session detail")
def get_test_session_detail(session_id: int, db: Session = Depends(get_db)):
    """
    Return full detail of a test session including aggregate and persona group results.

    The response contains:
      - session  : metadata, status, LLM parameters
      - aggregate: aggregate simulation results (one response set per task)
      - persona_group: N-person simulation results merged across all personas
    """
    session = db.query(TestSession).filter(TestSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return TestSessionDetailOut(
        session=build_session_out(db, session),
        aggregate=build_aggregate_block_out(db, session_id),
        persona_group=build_persona_group_out(db, session_id),
    )


@app.delete("/test-sessions/{session_id}", summary="Delete test session")
def delete_test_session(session_id: int, db: Session = Depends(get_db)):
    """Delete a test session and all associated runs and run items (cascade)."""
    session = db.query(TestSession).filter(TestSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()
    return {"ok": True}


@app.get("/runs", response_model=list[RunOut], summary="List all runs")
def list_runs(db: Session = Depends(get_db)):
    """Return all test runs ordered by creation date (newest first)."""
    rows = db.query(TestRun).order_by(TestRun.id.desc()).all()
    return [build_run_out(db, r) for r in rows]


@app.get("/runs/batch/{batch_id}", response_model=BatchRunsDetailOut, summary="Get latest batch runs detail")
def get_latest_batch_runs_detail(batch_id: int, db: Session = Depends(get_db)):
    """Return the aggregate and persona runs from the most recent session for a batch."""
    batch = db.query(PersonaBatch).filter(PersonaBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    session = get_latest_session_for_batch(db, batch_id)
    if not session:
        return BatchRunsDetailOut(
            batch=build_persona_batch_out(batch),
            aggregate_run=None,
            persona_runs=[],
        )

    aggregate_run, persona_runs = get_session_runs(db, int(session.id))

    return BatchRunsDetailOut(
        batch=build_persona_batch_out(batch),
        aggregate_run=build_run_out(db, aggregate_run) if aggregate_run else None,
        persona_runs=[build_run_out(db, r) for r in persona_runs],
    )


@app.get("/runs/{run_id}", response_model=RunDetailOut, summary="Get run detail")
def get_run_detail(run_id: int, db: Session = Depends(get_db)):
    """Return full detail of a single run including all task results."""
    r = db.query(TestRun).filter(TestRun.id == run_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")

    items = (
        db.query(TestRunItem)
        .filter(TestRunItem.run_id == r.id)
        .order_by(TestRunItem.id.asc())
        .all()
    )

    run_out = build_run_out(db, r)
    items_out = [RunItemOut.model_validate(x) for x in items]
    return {"run": run_out, "items": items_out}


@app.delete("/runs/{run_id}", summary="Delete run")
def delete_run(run_id: int, db: Session = Depends(get_db)):
    """Delete a run and all its items. Refreshes the parent session status afterwards."""
    run = db.query(TestRun).filter(TestRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    session_id = int(run.session_id)
    db.delete(run)
    db.commit()
    refresh_session_status(db, session_id)
    return {"ok": True}
