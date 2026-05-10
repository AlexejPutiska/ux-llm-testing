# UX Testing with Large Language Models

The system allows creating and running three types of UX tests (preference test, first-click test, feedback test) using screenshots. Tests are executed by simulating user responses via a large language model (OpenAI API). Results are stored in a database and can be reviewed in the test history.

---

## Project Structure

```
web/
├── backend/          # Python / FastAPI REST API
│   ├── app/
│   │   └── main.py   # All API endpoints and LLM logic
│   ├── db/
│   │   ├── models.py # SQLAlchemy database models
│   │   └── session.py# Database connection setup
│   ├── .env          # API key (not committed to git)
│   └── requirements.txt
├── frontend/         # Next.js / TypeScript UI
│   └── app/
│       ├── o_projekte/       # About page
│       ├── sprava_testov/    # Test management
│       ├── sprava_person/    # Persona batch management
│       ├── historia_testov/  # Test history & results
│       └── spustit_test/     # Run tests
├── docker-compose.yml  # PostgreSQL database container
├── .env.example        # Example environment configuration
└── .gitignore
```

---

## Requirements

- **Python 3.10+**
- **Node.js 18+**
- **Docker** — used to run the PostgreSQL database
- **OpenAI API key**

---

## Database Setup

The database runs in a Docker container. Make sure **Docker Desktop is installed and running** before proceeding.

Start the database with:

```bash
docker-compose up -d
```

---

## Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp ../.env.example .env
# Edit .env and set your OpenAI API key
```

### Environment Variables (`backend/.env`)

```
OPENAI_API_KEY=sk-...
# DATABASE_URL=postgresql://postgres:aaa123@127.0.0.1:5432/bp
```

`OPENAI_API_KEY` is required — you can obtain one at [platform.openai.com](https://platform.openai.com).  
`DATABASE_URL` is optional — if not set, it defaults to the Docker database with the credentials above.

### Run the Backend

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive API docs: `http://127.0.0.1:8000/docs`

---

## Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The application will be available at `http://localhost:3000`.

By default the frontend connects to `http://127.0.0.1:8000`. To use a different backend URL, set the `NEXT_PUBLIC_API_URL` environment variable.

---

## Application Pages

| Page | Route | Description |
|---|---|---|
| O projekte | `/o_projekte` | Project description |
| Správa testov | `/sprava_testov` | Create and manage UX test tasks |
| Správa persón | `/sprava_person` | Create and manage persona batches |
| História testovania | `/historia_testov` | Browse past test sessions and results |
| Spustiť testovanie | `/spustit_test` | Configure and launch a test session |

---

## Supported Test Types

- **Preference test** — the model chooses between two or more screenshot variants (A/B test)
- **First-click test** — the model identifies which UI element it would click first to complete a task
- **Feedback test** — the model provides free-form usability feedback on a screenshot

---

## LLM Simulation Modes

Each test session runs in two parallel modes:

- **Aggregate persona** — a single prompt representing the whole group profile
- **N-person** — one prompt per individual persona in the batch, results are aggregated
