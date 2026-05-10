/**
 * Run tests page — Spustiť testovanie.
 *
 * Allows the user to configure and launch a full LLM-powered UX test session.
 * The workflow has four steps:
 *   1. Select a persona batch (group of simulated users)
 *   2. Select which test tasks to include (grouped by type: preference, first_click, feedback)
 *   3. Optionally override LLM parameters (temperature, top_p)
 *   4. Launch the session — creates it on the backend, executes all runs, and shows results
 *
 * After execution, results are displayed inline in two switchable views:
 *   - Aggregate persona: responses generated from the single group-level prompt
 *   - N-person: responses aggregated across individual persona prompts
 */
"use client";

import { useEffect, useMemo, useState } from "react";

// Backend API base URL — can be overridden via NEXT_PUBLIC_API_URL environment variable
const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type PersonaBatch = {
  id: number;
  name: string;
  count: number;
  age_from: number;
  age_to: number;
  genders: string[];
  similar_apps_experience: string[];
  decision_style: string[];
  frustration_tolerance: string[];
  extra_description: string;
  aggregate_persona_text: string;
  created_at: string;
};

type TaskType = "preference" | "first_click" | "feedback";
type ResultView = "aggregate" | "n_person";

type TestTask = {
  id: number;
  task_type: TaskType;
  order_index: number;
  title: string;
  task_text: string;
  follow_up_question: string | null;
  config: Record<string, any>;
};

type LLMParams = {
  temperature: string;
  top_p: string;
};

const DEFAULT_PARAMS: LLMParams = {
  temperature: "",
  top_p: "",
};

type CreateSessionOut = {
  session_id: number;
  batch_id: number;
  aggregate_run_id: number | null;
  persona_run_ids: number[];
  total_runs: number;
};

type SessionInfo = {
  id: number;
  test_id: number;
  batch_id: number;
  batch_name_snapshot: string;
  batch_count_snapshot: number;
  aggregate_persona_text_snapshot: string;
  status: string;
  llm_model: string;
  llm_params: Record<string, any>;
  selected_task_ids: number[] | Record<string, any>;
  created_at: string;
  total_runs: number;
  aggregate_run_id: number | null;
  persona_run_count: number;
  selected_task_count: number;
};

type RunInfo = {
  id: number;
  session_id: number;
  subject_type: "persona" | "aggregate";
  persona_id: number | null;
  batch_id: number;
  persona_name: string;
  status: string;
  llm_model: string;
  llm_params: Record<string, any>;
};

type RunSummaryPreference = {
  task_type: "preference";
  selected_counts: { label: string; count: number }[];
  total_responses: number;
  open_answer_count: number;
  has_follow_up: boolean;
};

type RunSummaryFirstClick = {
  task_type: "first_click";
  click_counts: { target: string; count: number }[];
  total_responses: number;
  open_answer_count: number;
  has_follow_up: boolean;
};

type RunSummaryFeedback = {
  task_type: "feedback";
  total_responses: number;
  responses_with_notes: number;
  note_count: number;
};

type RunAnswerSummary =
  | RunSummaryPreference
  | RunSummaryFirstClick
  | RunSummaryFeedback
  | { task_type?: string; text?: string }
  | Record<string, any>;

type BaseResponse = {
  simulated_index?: number;
  persona_id?: number | null;
  persona_name?: string;
};

type PreferenceResponse = BaseResponse & {
  chosen_label: string;
  follow_up_response: string | null;
};

type FirstClickResponse = BaseResponse & {
  click_target: string;
  follow_up_response: string | null;
};

type FeedbackResponse = BaseResponse & {
  notes: string[];
};

type RawResponsesPayload = {
  task_type: TaskType;
  subject_type: "persona" | "aggregate";
  represented_count: number;
  response_count: number;
  grouped?: boolean;
  responses: Array<PreferenceResponse | FirstClickResponse | FeedbackResponse>;
};

type SessionTaskResult = {
  id: number | null;
  task_id: number;
  task_snapshot: {
    task_id: number;
    task_type: TaskType;
    order_index: number;
    title: string;
    task_text: string;
    follow_up_question: string | null;
    config: Record<string, any>;
  };
  answer_json: RunAnswerSummary | null;
  raw_responses: RawResponsesPayload | null;
  raw_output_text: string | null;
  result_mode: string | null;
  usage: Record<string, any> | null;
  error_message: string | null;
};

type AggregateBlock = {
  run: RunInfo | null;
  items: SessionTaskResult[];
};

type PersonaGroupBlock = {
  total_personas: number;
  run_count: number;
  items: SessionTaskResult[];
};

type SessionDetail = {
  session: SessionInfo;
  aggregate: AggregateBlock | null;
  persona_group: PersonaGroupBlock;
};

function typeLabel(t: TaskType) {
  if (t === "preference") return "Preferenčný test";
  if (t === "first_click") return "First-click test";
  return "Spätná väzba";
}

function formatLLMParamsText(params: Record<string, any>) {
  const temperature =
    typeof params.temperature === "number" ? String(params.temperature) : "predvolené";
  const topP =
    typeof params.top_p === "number" ? String(params.top_p) : "predvolené";

  return [`temperature: ${temperature}`, `top_p: ${topP}`].join("\n");
}

function formatDate(value?: string) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleTimeString("sk-SK", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderSummary(taskType: TaskType, summary: RunAnswerSummary | null) {
  if (!summary) return <div>(bez výstupu)</div>;

  if (typeof summary === "object" && "text" in summary && typeof summary.text === "string") {
    return <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{summary.text}</div>;
  }

  if (taskType === "preference") {
    const s = summary as RunSummaryPreference;
    const counts = Array.isArray(s.selected_counts) ? s.selected_counts : [];

    return (
      <div style={{ lineHeight: 1.6 }}>
        <div>
          <strong>Počet odpovedí:</strong> {s.total_responses ?? 0}
        </div>
        <div>
          <strong>Počet odpovedí na otvorenú otázku:</strong> {s.open_answer_count ?? 0}
        </div>

        <div style={{ marginTop: 8, fontWeight: 600 }}>Rozdelenie výberov</div>
        {counts.length === 0 ? (
          <div>(bez rozdelenia)</div>
        ) : (
          <ul style={{ marginTop: 6 }}>
            {counts.map((x, i) => (
              <li key={`${x.label}-${i}`}>
                <strong>{x.label}</strong>: {x.count}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  if (taskType === "first_click") {
    const s = summary as RunSummaryFirstClick;
    const counts = Array.isArray(s.click_counts) ? s.click_counts : [];

    return (
      <div style={{ lineHeight: 1.6 }}>
        <div>
          <strong>Počet odpovedí:</strong> {s.total_responses ?? 0}
        </div>
        <div>
          <strong>Počet odpovedí na otvorenú otázku:</strong> {s.open_answer_count ?? 0}
        </div>

        <div style={{ marginTop: 8, fontWeight: 600 }}>Rozdelenie prvých klikov</div>
        {counts.length === 0 ? (
          <div>(bez rozdelenia)</div>
        ) : (
          <ul style={{ marginTop: 6 }}>
            {counts.map((x, i) => (
              <li key={`${x.target}-${i}`}>
                <strong>{x.target}</strong>: {x.count}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  const s = summary as RunSummaryFeedback;
  return (
    <div style={{ lineHeight: 1.6 }}>
      <div>
        <strong>Počet odpovedí:</strong> {s.total_responses ?? 0}
      </div>
      <div>
        <strong>Počet odpovedí s poznámkami:</strong> {s.responses_with_notes ?? 0}
      </div>
      <div>
        <strong>Celkový počet bodov spätnej väzby:</strong> {s.note_count ?? 0}
      </div>
    </div>
  );
}

function renderRawResponses(taskType: TaskType, raw: RawResponsesPayload | null) {
  if (!raw || !Array.isArray(raw.responses) || raw.responses.length === 0) {
    return <div>(bez detailných odpovedí)</div>;
  }

  return (
    <div>
      <div style={{ marginBottom: 8, lineHeight: 1.6 }}>
        <div>
          <strong>Režim:</strong>{" "}
          {raw.subject_type === "aggregate"
            ? "Agregovaná persona"
            : raw.grouped
              ? "N-person"
              : "Jedna persona"}
        </div>
        <div>
          <strong>Reprezentovaný počet používateľov:</strong> {raw.represented_count}
        </div>
        <div>
          <strong>Počet detailných odpovedí:</strong> {raw.response_count}
        </div>
      </div>

      <div style={{ display: "grid", gap: 10 }}>
        {raw.responses.map((r, idx) => {
          const key = `resp-${idx}`;
          const personaName =
            "persona_name" in r && typeof r.persona_name === "string" && r.persona_name.trim()
              ? r.persona_name
              : null;

          if (taskType === "preference") {
            const x = r as PreferenceResponse;
            return (
              <div key={key} style={{ border: "1px solid #eee", padding: 10 }}>
                <div>
                  <strong>{personaName ? personaName : `Odpoveď #${idx + 1}`}</strong>
                </div>
                <div>
                  <strong>Vybraná možnosť:</strong> {x.chosen_label || "—"}
                </div>
                <div style={{ whiteSpace: "pre-wrap" }}>
                  <strong>Doplňujúca odpoveď:</strong> {x.follow_up_response?.trim() ? x.follow_up_response : "—"}
                </div>
              </div>
            );
          }

          if (taskType === "first_click") {
            const x = r as FirstClickResponse;
            return (
              <div key={key} style={{ border: "1px solid #eee", padding: 10 }}>
                <div>
                  <strong>{personaName ? personaName : `Odpoveď #${idx + 1}`}</strong>
                </div>
                <div>
                  <strong>Prvý klik:</strong> {x.click_target || "—"}
                </div>
                <div style={{ whiteSpace: "pre-wrap" }}>
                  <strong>Doplňujúca odpoveď:</strong> {x.follow_up_response?.trim() ? x.follow_up_response : "—"}
                </div>
              </div>
            );
          }

          const x = r as FeedbackResponse;
          return (
            <div key={key} style={{ border: "1px solid #eee", padding: 10 }}>
              <div>
                <strong>{personaName ? personaName : `Odpoveď #${idx + 1}`}</strong>
              </div>
              {!Array.isArray(x.notes) || x.notes.length === 0 ? (
                <div>—</div>
              ) : (
                <ul style={{ marginTop: 6 }}>
                  {x.notes.map((note, noteIdx) => (
                    <li key={`${key}-note-${noteIdx}`} style={{ whiteSpace: "pre-wrap" }}>
                      {note}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TaskResultCard({ item }: { item: SessionTaskResult }) {
  const s = item.task_snapshot;

  return (
    <div style={{ border: "1px solid #eee", padding: 12, marginBottom: 12, maxWidth: 980 }}>
      <div style={{ marginBottom: 6 }}>
        <strong>#{s.order_index}</strong> {s.title || s.task_text}{" "}
        <span style={{ opacity: 0.7 }}>
          ({s.task_type})
        </span>
      </div>

      <div style={{ marginBottom: 8 }}>
        <div style={{ fontWeight: 600 }}>Zadanie</div>
        <div style={{ whiteSpace: "pre-wrap" }}>{s.task_text}</div>
      </div>

      {s.follow_up_question && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600 }}>Doplňujúca otázka</div>
          <div style={{ whiteSpace: "pre-wrap" }}>{s.follow_up_question}</div>
        </div>
      )}

      {s.task_type === "preference" && Array.isArray(s.config?.options) && (
        <div style={{ marginTop: 6 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            Možnosti ({s.config.options.length})
          </div>
          <ul>
            {s.config.options.map((opt: any, i: number) => (
              <li key={i} style={{ marginBottom: 12 }}>
                <div style={{ marginBottom: 6 }}>
                  <strong>{opt?.label || `Možnosť ${i + 1}`}</strong>
                </div>
                {opt?.image && (
                  <img
                    src={opt.image}
                    alt={`opt-${i + 1}`}
                    style={{ maxWidth: 700, width: "100%", border: "1px solid #ccc" }}
                  />
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {(s.task_type === "first_click" || s.task_type === "feedback") && s.config?.image && (
        <div style={{ marginTop: 6 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Screenshot</div>
          <img
            src={s.config.image}
            alt={`task-${item.task_id}-img`}
            style={{ maxWidth: 700, width: "100%", border: "1px solid #ccc" }}
          />
        </div>
      )}

      <div style={{ marginTop: 10 }}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>Súhrn výsledku</div>

        {item.error_message ? (
          <div style={{ color: "crimson", whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
            Chyba pri úlohe: {item.error_message}
          </div>
        ) : item.answer_json ? (
          <div style={{ background: "#fafafa", padding: 10 }}>
            {renderSummary(s.task_type, item.answer_json)}
          </div>
        ) : item.raw_output_text ? (
          <div style={{ whiteSpace: "pre-wrap", background: "#fafafa", padding: 10, lineHeight: 1.5 }}>
            {item.raw_output_text}
          </div>
        ) : (
          <div>(bez výstupu)</div>
        )}
      </div>

      {!item.error_message && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Jednotlivé odpovede na otvorené otázky</div>
          <div style={{ background: "#fafafa", padding: 10 }}>
            {renderRawResponses(s.task_type, item.raw_responses)}
          </div>
        </div>
      )}
    </div>
  );
}

function ResultSection({
  title,
  subtitle,
  items,
}: {
  title: string;
  subtitle?: React.ReactNode;
  items: SessionTaskResult[];
}) {
  const sortedItems = items
    .slice()
    .sort((a, b) => (a.task_snapshot.order_index ?? 0) - (b.task_snapshot.order_index ?? 0));

  return (
    <div style={{ maxWidth: 1020 }}>
      <div style={{ marginBottom: 14, padding: 12, border: "1px solid #ddd" }}>
        <h3 style={{ marginTop: 0, marginBottom: 8 }}>{title}</h3>
        {subtitle}
      </div>

      {sortedItems.length === 0 ? (
        <p>Pre tento blok zatiaľ nie sú žiadne výsledky.</p>
      ) : (
        sortedItems.map((item) => <TaskResultCard key={`${item.task_id}-${item.id ?? "group"}`} item={item} />)
      )}
    </div>
  );
}

export default function Page() {
  // Data loaded from backend
  const [batches, setBatches] = useState<PersonaBatch[]>([]);
  const [tasks, setTasks] = useState<TestTask[]>([]);

  // Current form selection
  const [batchId, setBatchId] = useState<number | null>(null);
  const [params, setParams] = useState<LLMParams>({ ...DEFAULT_PARAMS });
  const [selectedTaskIds, setSelectedTaskIds] = useState<number[]>([]);

  // Loading / running status
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("");

  // Results of the most recent run
  const [createdSession, setCreatedSession] = useState<CreateSessionOut | null>(null);
  const [sessionDetail, setSessionDetail] = useState<SessionDetail | null>(null);
  const [resultView, setResultView] = useState<ResultView>("aggregate");

  const visibleTasks = useMemo(() => {
    return tasks.slice().sort((a, b) => a.order_index - b.order_index);
  }, [tasks]);

  const tasksByType = useMemo(() => {
    const out: Record<TaskType, TestTask[]> = {
      preference: [],
      first_click: [],
      feedback: [],
    };
    for (const t of visibleTasks) out[t.task_type].push(t);
    return out;
  }, [visibleTasks]);

  const selectedTypes = useMemo(() => {
    const selected = new Set(selectedTaskIds);
    return {
      preference:
        tasksByType.preference.length > 0 &&
        tasksByType.preference.every((t) => selected.has(t.id)),
      first_click:
        tasksByType.first_click.length > 0 &&
        tasksByType.first_click.every((t) => selected.has(t.id)),
      feedback:
        tasksByType.feedback.length > 0 &&
        tasksByType.feedback.every((t) => selected.has(t.id)),
    };
  }, [selectedTaskIds, tasksByType]);

  /** Select or deselect all tasks of a given type at once. */
  function setAllTasksOfType(tt: TaskType, checked: boolean) {
    const ids = tasksByType[tt].map((t) => t.id);
    setSelectedTaskIds((prev) => {
      const s = new Set(prev);
      if (checked) ids.forEach((id) => s.add(id));
      else ids.forEach((id) => s.delete(id));
      return Array.from(s);
    });
  }

  /** Toggle all tasks of a given type (select all if not all selected, deselect otherwise). */
  function toggleType(tt: TaskType) {
    setAllTasksOfType(tt, !selectedTypes[tt]);
  }

  /** Toggle a single task in/out of the selection. */
  function toggleTask(id: number, checked: boolean) {
    setSelectedTaskIds((prev) => {
      const current = new Set(prev);
      if (checked) current.add(id);
      else current.delete(id);
      return Array.from(current);
    });
  }

  /** Select all available tasks. */
  function selectAll() {
    setSelectedTaskIds(visibleTasks.map((t) => t.id));
  }

  /** Deselect all tasks. */
  function clearSelection() {
    setSelectedTaskIds([]);
  }

  const isTemperatureValid = useMemo(() => {
    if (params.temperature.trim() === "") return true;
    const n = Number(params.temperature);
    return !Number.isNaN(n) && n >= 0 && n <= 2;
  }, [params.temperature]);

  const isTopPValid = useMemo(() => {
    if (params.top_p.trim() === "") return true;
    const n = Number(params.top_p);
    return !Number.isNaN(n) && n >= 0 && n <= 1;
  }, [params.top_p]);

  const canRun = useMemo(() => {
    return (
      !!batchId &&
      selectedTaskIds.length > 0 &&
      isTemperatureValid &&
      isTopPValid &&
      !loading &&
      !running
    );
  }, [
    batchId,
    selectedTaskIds.length,
    isTemperatureValid,
    isTopPValid,
    loading,
    running,
  ]);

  /**
   * Fetch persona batches and all test tasks (all three types) in parallel.
   * Pre-selects the first batch and prunes any previously selected task IDs
   * that no longer exist on the server.
   */
  async function loadAll(signal?: AbortSignal) {
    setLoading(true);
    setStatus("");
    setCreatedSession(null);
    setSessionDetail(null);

    try {
      const bRes = await fetch(`${API}/persona-batches`, { cache: "no-store", signal });
      if (!bRes.ok) {
        const d = await bRes.json().catch(() => null);
        setStatus(`Chyba pri načítaní skupín: ${d?.detail || bRes.statusText}`);
        setBatches([]);
        return;
      }

      const bData: PersonaBatch[] = await bRes.json();
      setBatches(bData);
      if (bData.length > 0) {
        setBatchId((prev) => prev ?? bData[0].id);
      }

      const types: TaskType[] = ["preference", "first_click", "feedback"];
      const reqs = types.map((t) =>
        fetch(`${API}/test/tasks?task_type=${t}`, { cache: "no-store", signal })
      );
      const resps = await Promise.all(reqs);

      for (const r of resps) {
        if (!r.ok) {
          const d = await r.json().catch(() => null);
          setStatus(`Chyba pri načítaní úloh: ${d?.detail || r.statusText}`);
          setTasks([]);
          return;
        }
      }

      const chunks = await Promise.all(resps.map((r) => r.json()));
      const allTasks: TestTask[] = chunks.flat();
      setTasks(allTasks);

      if (allTasks.length === 0) {
        setStatus("Žiadne úlohy zatiaľ nie sú vytvorené v Správe testov.");
      }

      setSelectedTaskIds((prev) => {
        const allowed = new Set(allTasks.map((t) => t.id));
        return prev.filter((id) => allowed.has(id));
      });
    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setStatus("Chyba: nepodarilo sa spojiť s backendom.");
    } finally {
      setLoading(false);
    }
  }

  /** Fetch full detail for a session (runs + per-task results) from the API. */
  async function fetchSessionDetail(sessionId: number): Promise<SessionDetail> {
    const res = await fetch(`${API}/test-sessions/${sessionId}`, { cache: "no-store" });
    if (!res.ok) {
      const d = await res.json().catch(() => null);
      throw new Error(d?.detail || res.statusText || `Nepodarilo sa načítať session ${sessionId}`);
    }
    return res.json();
  }

  /**
   * Validate the form, create a new test session via POST /test-sessions,
   * then trigger execution via POST /test-sessions/{id}/execute.
   * Fetches the resulting session detail and renders it inline.
   */
  async function runAll() {
    setStatus("");
    setCreatedSession(null);
    setSessionDetail(null);

    if (!batchId) {
      setStatus("Najprv vyber skupinu person.");
      return;
    }
    if (selectedTaskIds.length === 0) {
      setStatus("Vyber aspoň jednu úlohu.");
      return;
    }

    if (params.temperature.trim() !== "") {
      const temperature = Number(params.temperature);
      if (Number.isNaN(temperature) || temperature < 0 || temperature > 2) {
        setStatus("Temperature musí byť v rozsahu 0 až 2.");
        return;
      }
    }

    if (params.top_p.trim() !== "") {
      const topP = Number(params.top_p);
      if (Number.isNaN(topP) || topP < 0 || topP > 1) {
        setStatus("Top_p musí byť v rozsahu 0 až 1.");
        return;
      }
    }

    setRunning(true);

    try {
      const llm: Record<string, any> = {};

      if (params.temperature.trim() !== "") {
        llm.temperature = Number(params.temperature);
      }

      if (params.top_p.trim() !== "") {
        llm.top_p = Number(params.top_p);
      }

      const payload = {
        batch_id: batchId,
        task_ids: selectedTaskIds,
        llm,
      };

      const createRes = await fetch(`${API}/test-sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!createRes.ok) {
        const d = await createRes.json().catch(() => null);
        setStatus(`Chyba pri vytváraní session: ${d?.detail || createRes.statusText}`);
        return;
      }

      const created: CreateSessionOut = await createRes.json();
      setCreatedSession(created);

      setStatus(`Vytvorené spustenie #${created.session_id}. Spúšťam testovanie…`);

      const execRes = await fetch(`${API}/test-sessions/${created.session_id}/execute`, {
        method: "POST",
      });

      if (!execRes.ok) {
        const d = await execRes.json().catch(() => null);
        setStatus(`Execute zlyhalo: ${d?.detail || execRes.statusText}`);
        return;
      }

      const detail = await fetchSessionDetail(created.session_id);
      setSessionDetail(detail);

      setResultView(detail.aggregate ? "aggregate" : "n_person");
      setStatus(`Hotovo. Výsledky spustenia #${created.session_id} sú zobrazené nižšie.`);
    } catch (e: any) {
      setStatus(`Chyba: ${e?.message || "nepodarilo sa odoslať požiadavku na backend."}`);
    } finally {
      setRunning(false);
    }
  }

  // Load all data on mount; abort any in-flight request if the component unmounts
  useEffect(() => {
    const ac = new AbortController();
    loadAll(ac.signal);
    return () => ac.abort();
  }, []);

  const selectedBatch = useMemo(
    () => batches.find((b) => b.id === batchId) || null,
    [batches, batchId]
  );

  return (
    <div>
      <h1>Spustiť testy</h1>
      <p>
        Vyber skupinu person, úlohy a spusti testovanie.
      </p>
      <p style={{ marginTop: 6, fontStyle: "italic" }}>
        Poznámka: Ak nemáš skupinu person, vytvor ju najprv v časti Správa person.
      </p>

      <div style={{ margin: "12px 0" }}>
        <button onClick={() => loadAll()} disabled={loading || running}>
          Obnoviť dáta
        </button>
      </div>

      {status && <p>{status}</p>}

      <div style={{ marginTop: 14, maxWidth: 800 }}>
        <h2>1) Výber skupiny person</h2>

        <select
          value={batchId ?? ""}
          onChange={(e) => setBatchId(Number(e.target.value))}
          disabled={loading || running || batches.length === 0}
          style={{ width: "100%", padding: 8 }}
        >
          {batches.length === 0 ? (
            <option value="">(žiadne skupiny)</option>
          ) : (
            batches.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name} — počet person: {b.count}
              </option>
            ))
          )}
        </select>

        {selectedBatch && (
          <div style={{ marginTop: 10, padding: 10, border: "1px solid #ddd" }}>
            <div>
              <strong>Vybraná skupina:</strong> {selectedBatch.name}
            </div>
            <div>
              <strong>Počet person:</strong> {selectedBatch.count}
            </div>
            <div style={{ marginTop: 6 }}>
              <strong>Agregovaný text:</strong>
            </div>
            <div style={{ marginTop: 4, whiteSpace: "pre-wrap" }}>
              {selectedBatch.aggregate_persona_text}
            </div>
          </div>
        )}
      </div>

      <div style={{ marginTop: 18, maxWidth: 950 }}>
        <h2>2) Výber úloh</h2>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
          <button onClick={selectAll} disabled={running || loading || visibleTasks.length === 0}>
            Vybrať všetko
          </button>
          <button onClick={clearSelection} disabled={running || loading || visibleTasks.length === 0}>
            Zrušiť výber
          </button>
          <div style={{ alignSelf: "center", opacity: 0.8 }}>
            Vybrané: <strong>{selectedTaskIds.length}</strong> / {visibleTasks.length}
          </div>
        </div>

        <div style={{ padding: 10, border: "1px solid #ddd" }}>
          {(["preference", "first_click", "feedback"] as TaskType[]).map((tt) => {
            const list = tasksByType[tt];

            return (
              <div key={tt} style={{ marginBottom: 14 }}>
                <label style={{ display: "flex", gap: 8, alignItems: "center", fontWeight: 700 }}>
                  <input
                    type="checkbox"
                    checked={selectedTypes[tt]}
                    onChange={() => toggleType(tt)}
                    disabled={running || loading || list.length === 0}
                  />
                  {typeLabel(tt)}{" "}
                  <span style={{ fontWeight: 400, opacity: 0.7 }}>({list.length})</span>
                </label>

                {list.length === 0 ? (
                  <div style={{ marginLeft: 24, opacity: 0.7 }}>(žiadne úlohy)</div>
                ) : (
                  <ul style={{ marginTop: 8, marginLeft: 24 }}>
                    {list.map((t) => {
                      const checked = selectedTaskIds.includes(t.id);
                      return (
                        <li key={t.id} style={{ marginBottom: 6 }}>
                          <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={(e) => toggleTask(t.id, e.target.checked)}
                              disabled={running || loading}
                            />
                            <span
                              style={{
                                display: "block",
                                whiteSpace: "normal",
                                lineHeight: 1.4,
                              }}
                            >
                              <strong>#{t.order_index}</strong> {t.task_text}
                            </span>
                          </label>
                        </li>
                        
                      );
                    })}
                  </ul>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ marginTop: 18, maxWidth: 800 }}>
        <h2>3) Nastavenie LLM</h2>

        <div style={{ display: "grid", gap: 10, marginTop: 10 }}>
          <div style={{ display: "grid", gap: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Temperature</div>
              <input
                type="number"
                step="0.01"
                min="0"
                max="2"
                placeholder="Predvolená hodnota"
                value={params.temperature}
                onChange={(e) =>
                  setParams((p) => ({ ...p, temperature: e.target.value }))
                }
                style={{ width: "100%", padding: 8 }}
                disabled={running}
              />
            </div>

            <div>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Top_p</div>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                placeholder="Predvolená hodnota"
                value={params.top_p}
                onChange={(e) =>
                  setParams((p) => ({ ...p, top_p: e.target.value }))
                }
                style={{ width: "100%", padding: 8 }}
                disabled={running}
              />
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button onClick={() => setParams({ ...DEFAULT_PARAMS })} disabled={running}
              style={{
                fontSize: 15
              }}>
              Reset na predvolené hodnoty
            </button>
            
          </div>
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        <h2>4) Spustiť</h2>
        <button
          onClick={runAll}
          disabled={!canRun}
          style={{
            fontSize: 20,
          }}
        >
          {running ? "Spúšťam…" : "Spustiť testy pre skupinu"}
        </button>

        {createdSession && (
          <div style={{ marginTop: 8, fontSize: 14 }}>
            Vytvorené spustenie: <strong>#{createdSession.session_id}</strong>
          </div>
        )}

        {!running && selectedTaskIds.length === 0 && (
          <div style={{ marginTop: 8, fontSize: 13, color: "crimson" }}>
            Vyber aspoň jednu úlohu v časti „Výber úloh“.
          </div>
        )}

        {!running && !isTemperatureValid && (
          <div style={{ marginTop: 8, fontSize: 13, color: "crimson" }}>
            Temperature musí byť v rozsahu 0 až 2.
          </div>
        )}

        {!running && !isTopPValid && (
          <div style={{ marginTop: 8, fontSize: 13, color: "crimson"}}>
            Top_p musí byť v rozsahu 0 až 1.
          </div>
        )}
      </div>

      <div style={{ marginTop: 22 }}>
        <h2>Výstupy</h2>

        {!sessionDetail ? (
          <p>
            Po spustení sa tu zobrazia výsledky v dvoch blokoch: agregovaná persona a n-person.
          </p>
        ) : (
          <div>
            <div style={{ marginBottom: 16, padding: 12, border: "1px solid #ddd", maxWidth: 980 }}>
              <div>
                <strong>Session:</strong> #{sessionDetail.session.id}
              </div>
              <div>
                <strong>Skupina:</strong> {sessionDetail.session.batch_name_snapshot}
              </div>
              <div>
                <strong>Počet person:</strong> {sessionDetail.session.batch_count_snapshot}
              </div>
              <div>
                <strong>Čas spustenia:</strong> {formatDate(sessionDetail.session.created_at)}
              </div>
              <div>
                <strong>Počet vybraných úloh:</strong> {sessionDetail.session.selected_task_count}
              </div>
              <div>
                <strong>Model:</strong> {sessionDetail.session.llm_model}
              </div>

              <div style={{ marginTop: 8 }}>
                <div style={{ fontWeight: 600 }}>LLM nastavenie</div>
                <pre style={{ whiteSpace: "pre-wrap", background: "#fafafa", padding: 10 }}>
                  {formatLLMParamsText(sessionDetail.session.llm_params || {})}
                </pre>
              </div>

              <div style={{ marginTop: 8 }}>
                <strong>Agregovaný text:</strong>
                <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
                  {sessionDetail.session.aggregate_persona_text_snapshot}
                </div>
              </div>
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
              <button
                onClick={() => setResultView("aggregate")}
                disabled={!sessionDetail.aggregate || resultView === "aggregate"}
              >
                Agregovaná persona
              </button>
              <button
                onClick={() => setResultView("n_person")}
                disabled={resultView === "n_person"}
              >
                N-person
              </button>
            </div>
            {resultView === "aggregate" ? (
              sessionDetail.aggregate ? (
                <ResultSection
                  title="Agregovaná persona"
                  items={sessionDetail.aggregate.items}
                />
              ) : (
                <p>Agregovaný blok v tejto session nie je dostupný.</p>
              )
            ) : (
              <ResultSection
                title="N-person"
                items={sessionDetail.persona_group.items}
              />
            )} 
          </div>
        )}
      </div>
    </div>
  );
}