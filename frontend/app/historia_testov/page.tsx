/**
 * Test history page — História testovania.
 *
 * Displays all past test sessions in a sortable table.
 * Each session row can be expanded to show a detailed result view with:
 *   - Session metadata (persona group, model, LLM params, aggregate persona text)
 *   - Two result views selectable by the user:
 *       • Aggregate persona — results produced by a single group-level prompt
 *       • N-person — results aggregated across individual per-persona prompts
 *   - Per-task cards showing answer summaries and raw open-ended responses
 *
 * Sessions are loaded from the backend on mount and can be individually deleted.
 * Fetched session details are cached locally to avoid redundant network requests.
 */
"use client";

import { useEffect, useMemo, useState } from "react";

// Backend API base URL — can be overridden via NEXT_PUBLIC_API_URL environment variable
const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type TaskType = "preference" | "first_click" | "feedback";
type ResultView = "aggregate" | "n_person";

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

function formatDate(value?: string) {
  if (!value) return "—";

  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;

  const adjusted = new Date(d.getTime() + 2 * 60 * 60 * 1000);

  return adjusted.toLocaleTimeString("sk-SK", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatLLMParamsText(params: Record<string, any>) {
  const temperature =
    typeof params.temperature === "number" ? String(params.temperature) : "predvolené";
  const topP =
    typeof params.top_p === "number" ? String(params.top_p) : "predvolené";

  return [
    `temperature: ${temperature}`,
    `top_p: ${topP}`,
  ].join("\n");
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
    <div
      style={{
        border: "1px solid #eee",
        padding: 12,
        marginBottom: 12,
        maxWidth: 980,
      }}
    >
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
                    style={{
                      maxWidth: 700,
                      width: "100%",
                      border: "1px solid #ccc",
                    }}
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
            style={{
              maxWidth: 700,
              width: "100%",
              border: "1px solid #ccc",
            }}
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
          <pre style={{ whiteSpace: "pre-wrap", background: "#fafafa", padding: 10 }}>
            {item.raw_output_text}
          </pre>
        ) : (
          <div>(bez výstupu)</div>
        )}
      </div>

      {!item.error_message && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>
            Jednotlivé odpovede na otvorené otázky
          </div>
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
        sortedItems.map((item) => (
          <TaskResultCard key={`${item.task_id}-${item.id ?? "group"}`} item={item} />
        ))
      )}
    </div>
  );
}

export default function Page() {
  // Session list state
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [deletingId, setDeletingId] = useState<number | null>(null);

  // Expanded session detail state
  const [openSessionId, setOpenSessionId] = useState<number | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailStatus, setDetailStatus] = useState("");
  const [detailCache, setDetailCache] = useState<Record<number, SessionDetail>>({});

  // Which result view tab is active: aggregate persona or N-person
  const [resultView, setResultView] = useState<ResultView>("aggregate");

  const visibleSessions = useMemo(
    () => sessions.slice().sort((a, b) => b.id - a.id),
    [sessions]
  );

  /** Fetch all test sessions from the API and populate the session list. */
  async function loadSessions(signal?: AbortSignal) {
    setLoading(true);
    setStatus("");

    try {
      const res = await fetch(`${API}/test-sessions`, { cache: "no-store", signal });
      if (!res.ok) {
        const d = await res.json().catch(() => null);
        setStatus(`Chyba pri načítaní histórie: ${d?.detail || res.statusText}`);
        setSessions([]);
        return;
      }

      const data: SessionInfo[] = await res.json();
      setSessions(data);
    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setStatus("Chyba: nepodarilo sa spojiť s backendom.");
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }

  /**
   * Delete a test session after user confirmation.
   * Also removes the session from the local detail cache and collapses its detail panel.
   */
  async function deleteSession(sessionId: number) {
    if (!confirm("Naozaj chceš vymazať toto spustenie testu? Táto operácia je nevratná.")) {
      return;
    }

    setDeletingId(sessionId);
    setStatus("");

    try {
      const res = await fetch(`${API}/test-sessions/${sessionId}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const d = await res.json().catch(() => null);
        setStatus(`Chyba pri mazaní: ${d?.detail || res.statusText}`);
        return;
      }

      if (openSessionId === sessionId) {
        setOpenSessionId(null);
      }

      setDetailCache((prev) => {
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });

      await loadSessions();
    } catch {
      setStatus("Chyba: nepodarilo sa spojiť s backendom.");
    } finally {
      setDeletingId(null);
    }
  }

  /** Fetch full detail for a session (runs, task results, raw responses) and store in cache. */
  async function loadSessionDetail(sessionId: number) {
    setDetailLoading(true);
    setDetailStatus("");

    try {
      const res = await fetch(`${API}/test-sessions/${sessionId}`, { cache: "no-store" });
      if (!res.ok) {
        const d = await res.json().catch(() => null);
        setDetailStatus(`Chyba pri načítaní detailu: ${d?.detail || res.statusText}`);
        return;
      }

      const data: SessionDetail = await res.json();
      setDetailCache((prev) => ({ ...prev, [sessionId]: data }));
    } catch {
      setDetailStatus("Chyba: nepodarilo sa spojiť s backendom.");
    } finally {
      setDetailLoading(false);
    }
  }

  /**
   * Toggle the expanded detail panel for a session.
   * Uses the local cache to avoid re-fetching already loaded details.
   * Automatically selects the appropriate result view tab based on available data.
   */
  async function toggleDetail(sessionId: number) {
    setDetailStatus("");

    if (openSessionId === sessionId) {
      setOpenSessionId(null);
      return;
    }

    setOpenSessionId(sessionId);

    if (!detailCache[sessionId]) {
      await loadSessionDetail(sessionId);
    }

    const cached = detailCache[sessionId];
    if (cached) {
      setResultView(cached.aggregate ? "aggregate" : "n_person");
    } else {
      setResultView("aggregate");
    }
  }

  /** Re-fetch the currently open session's detail to get the latest data. */
  async function refreshOpenDetail() {
    if (!openSessionId) return;
    await loadSessionDetail(openSessionId);
  }

  // Load sessions on mount; abort the request if the component unmounts
  useEffect(() => {
    const ac = new AbortController();
    loadSessions(ac.signal);
    return () => ac.abort();
  }, []);

  const openDetail = openSessionId ? detailCache[openSessionId] : null;

  // Sync the result view tab whenever the open session changes
  useEffect(() => {
    if (!openDetail) return;
    setResultView(openDetail.aggregate ? "aggregate" : "n_person");
  }, [openSessionId, openDetail]);

  return (
    <div>
      <h1>História testovania</h1>
      <p>
        Tu sú uložené všetky spustenia testov.
      </p>

      <div style={{ margin: "12px 0" }}>
        <button onClick={() => loadSessions()} disabled={loading || detailLoading}>
          Obnoviť dáta
        </button>
        {openSessionId && (
          <button style={{ marginLeft: 10 }} onClick={refreshOpenDetail} disabled={detailLoading}>
            Obnoviť detail
          </button>
        )}
      </div>

      {status && <p>{status}</p>}

      {loading ? (
        <p>Načítavam…</p>
      ) : visibleSessions.length === 0 ? (
        <p>Zatiaľ nie sú žiadne spustené testy.</p>
      ) : (
        <div style={{ maxWidth: 1150 }}>
          <h2>Zoznam spustení</h2>

          <div style={{ overflowX: "auto" }}>
            <table
              cellPadding={8}
              style={{
                borderCollapse: "collapse",
                width: "100%",
                minWidth: 900,
              }}
            >
              <thead>
                <tr>
                  <th align="left" style={{ borderBottom: "1px solid #ddd" }}>
                    Spustenie
                  </th>
                  <th align="left" style={{ borderBottom: "1px solid #ddd" }}>
                    Názov persony / skupiny
                  </th>
                  <th align="left" style={{ borderBottom: "1px solid #ddd" }}>
                    Počet person
                  </th>
                  <th align="left" style={{ borderBottom: "1px solid #ddd" }}>
                    Úlohy
                  </th>
                  <th align="left" style={{ borderBottom: "1px solid #ddd" }}>
                    Čas spustenia
                  </th>
                  <th align="left" style={{ borderBottom: "1px solid #ddd" }}>
                    Akcie
                  </th>
                </tr>
              </thead>

              <tbody>
                {visibleSessions.map((s) => {
                  const isOpen = openSessionId === s.id;

                  return (
                    <tr
                      key={s.id}
                      style={{
                        background: isOpen ? "#fff" : undefined,
                        outline: isOpen ? "2px solid #ddd" : undefined,
                      }}
                    >
                      <td style={{ borderBottom: "1px solid #f0f0f0" }}>
                        <strong>#{s.id}</strong>
                        {isOpen && (
                          <span style={{ marginLeft: 8, fontSize: 12, fontWeight: 700 }}>
                            ● otvorené
                          </span>
                        )}
                      </td>

                      <td style={{ borderBottom: "1px solid #f0f0f0" }}>
                        <strong>{s.batch_name_snapshot}</strong>
                      </td>

                      <td style={{ borderBottom: "1px solid #f0f0f0" }}>
                        {s.batch_count_snapshot}
                      </td>

                      <td style={{ borderBottom: "1px solid #f0f0f0" }}>
                        {s.selected_task_count}
                      </td>

                      <td style={{ borderBottom: "1px solid #f0f0f0" }}>
                        {formatDate(s.created_at)}
                      </td>

                      <td style={{ borderBottom: "1px solid #f0f0f0" }}>
                        <div style={{ display: "flex", gap: 8 }}>
                          <button
                            onClick={() => toggleDetail(s.id)}
                            disabled={detailLoading && openSessionId === s.id}
                          >
                            {isOpen ? "Skryť detail" : "Zobraziť detail"}
                          </button>

                          <button
                            onClick={() => deleteSession(s.id)}
                            disabled={deletingId === s.id}
                            style={{ color: "crimson" }}
                          >
                            {deletingId === s.id ? "Mažem…" : "Vymazať"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {openSessionId && (
        <div style={{ marginTop: 18, maxWidth: 1200 }}>
          <h2>Detail spustenia #{openSessionId}</h2>

          {detailStatus && <p>{detailStatus}</p>}
          {detailLoading && <p>Načítavam detail…</p>}

          {!detailLoading && openDetail && (
            <>
              <div style={{ padding: 12, border: "1px solid #ddd", marginBottom: 16, maxWidth: 1020 }}>
                <div>
                  <strong>Skupina:</strong> {openDetail.session.batch_name_snapshot}
                </div>
                <div>
                  <strong>Počet person:</strong> {openDetail.session.batch_count_snapshot}
                </div>
                <div>
                  <strong>Čas spustenia:</strong> {formatDate(openDetail.session.created_at)}
                </div>
                <div>
                  <strong>Počet úloh:</strong> {openDetail.session.selected_task_count}
                </div>
                <div>
                  <strong>Model:</strong> {openDetail.session.llm_model}
                </div>

                <div style={{ marginTop: 8 }}>
                  <div style={{ fontWeight: 600 }}>LLM nastavenie</div>
                  <pre style={{ whiteSpace: "pre-wrap", background: "#fafafa", padding: 10 }}>
                    {formatLLMParamsText(openDetail.session.llm_params || {})}
                  </pre>
                </div>

                <div style={{ marginTop: 8 }}>
                  <strong>Agregovaný text:</strong>
                  <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
                    {openDetail.session.aggregate_persona_text_snapshot}
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                <button
                  onClick={() => setResultView("aggregate")}
                  disabled={!openDetail.aggregate || resultView === "aggregate"}
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
                openDetail.aggregate ? (
                  <ResultSection
                    title="Agregovaná persona"
                    items={openDetail.aggregate.items}
                  />
                ) : (
                  <p>Agregovaný blok v tomto spustení nie je dostupný.</p>
                )
              ) : (
                <ResultSection
                  title="N-person"
                  items={openDetail.persona_group.items}
                />
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}