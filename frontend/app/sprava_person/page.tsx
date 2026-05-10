/**
 * Persona batch management page — Správa persón.
 *
 * Allows the user to create, view, and delete persona batches.
 * Each batch defines a group of simulated users with demographic distributions:
 *   - Age bands, gender, website experience, decision style, frustration tolerance.
 *
 * When a batch is created, the backend generates:
 *   - Individual personas (one per user slot, with randomised characteristics)
 *   - An aggregate persona text (used in aggregate simulation mode)
 *
 * These personas are later selected when running LLM tests on the Spustiť testovanie page.
 */
"use client";

import { useEffect, useMemo, useState } from "react";

// Backend API base URL — can be overridden via NEXT_PUBLIC_API_URL environment variable
const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// Allowed values for demographic distribution fields (must match backend validation)
type Gender = "muži" | "ženy";
type Level = "nízka" | "stredná" | "vysoká";
type Experience =
  | "Časté (aspoň niekoľkokrát do mesiaca)"
  | "Občasné (niekoľkokrát ročne)"
  | "Zriedkavé (menej ako raz ročne)";
type DecisionStyle = "rýchly" | "analytický" | "exploratívny";
type AgeBand = "18-24" | "25-34" | "35-44" | "45-54" | "55-64" | "65+";

type DistributionItem<T extends string = string> = {
  value: T;
  count: number;
};

type PersonaBatch = {
  id: number;
  name: string;
  count: number;
  age_from: number;
  age_to: number;

  age_distribution?: DistributionItem<AgeBand>[];
  genders_distribution?: DistributionItem<Gender>[];
  similar_apps_experience_distribution?: DistributionItem<Experience>[];
  decision_style_distribution?: DistributionItem<DecisionStyle>[];
  frustration_tolerance_distribution?: DistributionItem<Level>[];

  genders?: string[];
  similar_apps_experience?: string[];
  decision_style?: string[];
  frustration_tolerance?: string[];

  extra_description: string;
  aggregate_persona_text: string;
  created_at: string;
};

type Persona = {
  id: number;
  batch_id: number;
  batch_index: number;
  characteristics: Record<string, any>;
  extra_description: string;
  persona_text: string;
};

type PersonaBatchDetail = {
  batch: PersonaBatch;
  personas: Persona[];
};

type CreatePersonasPayload = {
  name: string;
  count: number;

  age_distribution: DistributionItem<AgeBand>[];
  genders_distribution: DistributionItem<Gender>[];
  similar_apps_experience_distribution: DistributionItem<Experience>[];
  decision_style_distribution: DistributionItem<DecisionStyle>[];
  frustration_tolerance_distribution: DistributionItem<Level>[];

  extra_description: string;
};

type CreateBatchResponse = {
  batch_id: number;
  aggregate_persona_text: string;
  created_persona_ids: number[];
};

const AGE_BANDS: { value: AgeBand; label: string }[] = [
  { value: "18-24", label: "18 – 24 rokov" },
  { value: "25-34", label: "25 – 34 rokov" },
  { value: "35-44", label: "35 – 44 rokov" },
  { value: "45-54", label: "45 – 54 rokov" },
  { value: "55-64", label: "55 – 64 rokov" },
  { value: "65+", label: "65+ rokov" },
];

const LEVELS: { value: Level; label: string }[] = [
  { value: "nízka", label: "Nízka" },
  { value: "stredná", label: "Stredná" },
  { value: "vysoká", label: "Vysoká" },
];

const EXPERIENCE_OPTS: { value: Experience; label: string }[] = [
  {
    value: "Časté (aspoň niekoľkokrát do mesiaca)",
    label: "Časté (aspoň niekoľkokrát do mesiaca)",
  },
  {
    value: "Občasné (niekoľkokrát ročne)",
    label: "Občasné (niekoľkokrát ročne)",
  },
  {
    value: "Zriedkavé (menej ako raz ročne)",
    label: "Zriedkavé (menej ako raz ročne)",
  },
];

const DECISION_OPTS: { value: DecisionStyle; label: string }[] = [
  { value: "rýchly", label: "Rýchly" },
  { value: "analytický", label: "Analytický" },
  { value: "exploratívny", label: "Exploratívny" },
];

const GENDER_OPTS: { value: Gender; label: string }[] = [
  { value: "muži", label: "Muži" },
  { value: "ženy", label: "Ženy" },
];


function makeDefaultDistribution<T extends string>(
  options: { value: T; label: string }[],
  initial: Partial<Record<T, number>> = {}
): DistributionItem<T>[] {
  return options.map((o) => ({
    value: o.value,
    count: initial[o.value] ?? 0,
  }));
}

function getDistributionCount<T extends string>(
  items: DistributionItem<T>[],
  value: T
) {
  return items.find((x) => x.value === value)?.count ?? 0;
}

function setDistributionValue<T extends string>(
  items: DistributionItem<T>[],
  value: T,
  count: number
): DistributionItem<T>[] {
  return items.map((x) =>
    x.value === value ? { ...x, count: Math.max(0, Math.floor(count || 0)) } : x
  );
}

function sumDistribution<T extends string>(items: DistributionItem<T>[]) {
  return items.reduce((acc, x) => acc + (Number.isFinite(x.count) ? x.count : 0), 0);
}

function normalizeDistribution<T extends string>(
  items: DistributionItem<T>[]
): DistributionItem<T>[] {
  return items.map((x) => ({
    value: x.value,
    count: Math.max(0, Math.floor(Number(x.count) || 0)),
  }));
}

function formatDistribution(items?: DistributionItem<string>[]) {
  if (!items || items.length === 0) return "—";
  const nonZero = items.filter((x) => (x.count || 0) > 0);
  if (nonZero.length === 0) return "—";
  return nonZero.map((x) => `${x.value}: ${x.count}`).join(", ");
}

function formatFallbackList(xs?: string[]) {
  if (!xs || xs.length === 0) return "—";
  return xs.join(", ");
}

function TextCard({
  title,
  text,
}: {
  title?: string;
  text: string;
}) {
  return (
    <div
      style={{
        marginTop: 6,
        padding: 10,
        border: "1px solid #ddd",
        background: "#fff",
        whiteSpace: "pre-line",
      }}
    >
      {title && <div style={{ fontWeight: 700, marginBottom: 6 }}>{title}</div>}
      {text?.trim() ? text : "—"}
    </div>
  );
}

function DistributionEditor<T extends string>({
  label,
  options,
  items,
  totalCount,
  onChange,
  required = false,
}: {
  label: string;
  options: { value: T; label: string }[];
  items: DistributionItem<T>[];
  totalCount: number;
  onChange: (next: DistributionItem<T>[]) => void;
  required?: boolean;
}) {
  const total = useMemo(() => sumDistribution(items), [items]);

  function handleChange(value: T, raw: string) {
    const parsed = raw === "" ? 0 : Number(raw);
    onChange(setDistributionValue(items, value, Number.isFinite(parsed) ? parsed : 0));
  }

  const isValid = required ? total === totalCount : total <= totalCount;
  const titleColor = isValid ? "#111" : "#c62828";
  const sumColor = isValid ? "#666" : "#c62828";

  return (
    <div style={{ marginBottom: 30 }}>
      <div
        style={{
          marginBottom: 14,
          paddingBottom: 6,
          borderBottom: "1px solid #e5e5e5",
          color: titleColor,
        }}
      >
        <span style={{ fontWeight: 700, fontSize: 18 }}>{label}</span>{" "}
        <span
          style={{
            fontSize: 14,
            color: sumColor,
            fontWeight: isValid ? 400 : 600,
          }}
        >
          (súčet {total} / {totalCount})
        </span>
      </div>

      <div style={{ display: "grid", gap: 12 }}>
        {options.map((o) => {
          const value = getDistributionCount(items, o.value);

          return (
            <div
              key={o.value}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                flexWrap: "wrap",
              }}
            >
              <div style={{ minWidth: 320, fontWeight: 500 }}>{o.label}</div>
              <input
                type="number"
                min={0}
                max={totalCount}
                value={value}
                onChange={(e) => handleChange(o.value, e.target.value)}
                style={{
                  width: 100,
                  border: "1px solid #ccc",
                  padding: "4px 6px",
                }}
              />
              <span style={{ opacity: 0.75 }}>osôb</span>
            </div>
          );
        })}
      </div>

      {!isValid && (
        <div
          style={{
            marginTop: 10,
            fontSize: 13,
            color: "#c62828",
          }}
        >
          {required
            ? "Súčet musí byť presne rovný počtu person."
            : "Súčet nesmie prekročiť počet person."}
        </div>
      )}
    </div>
  );
}

function validateCreatePayload(p: CreatePersonasPayload): string | null {
  if (!p.name.trim()) return "Zadaj názov skupiny.";

  if (p.count < 5 || p.count > 50) {
    return "Počet person musí byť v rozsahu 5 až 50.";
  }

  const ageSum = sumDistribution(p.age_distribution);
  if (ageSum !== p.count) {
    return "Pri veku musí byť súčet presne rovný počtu person.";
  }

  const genderSum = sumDistribution(p.genders_distribution);
  if (genderSum !== p.count) {
    return "Pri pohlaví musí byť súčet presne rovný počtu person.";
  }

  const experienceSum = sumDistribution(p.similar_apps_experience_distribution);
  if (experienceSum > p.count) {
    return "Súčet pri skúsenostiach nesmie byť väčší než počet person.";
  }

  const decisionSum = sumDistribution(p.decision_style_distribution);
  if (decisionSum > p.count) {
    return "Súčet pri štýle rozhodovania nesmie byť väčší než počet person.";
  }

  const frustrationSum = sumDistribution(p.frustration_tolerance_distribution);
  if (frustrationSum > p.count) {
    return "Súčet pri tolerancii frustrácie nesmie byť väčší než počet person.";
  }

  return null;
}

function makeInitialForm(): CreatePersonasPayload {
  return {
    name: "",
    count: 10,
    age_distribution: makeDefaultDistribution(AGE_BANDS, {
      "25-34": 10,
    }),
    genders_distribution: makeDefaultDistribution(GENDER_OPTS, {
      muži: 5,
      ženy: 5,
    }),
    similar_apps_experience_distribution: makeDefaultDistribution(EXPERIENCE_OPTS, {
      "Občasné (niekoľkokrát ročne)": 10,
    }),
    decision_style_distribution: makeDefaultDistribution(DECISION_OPTS, {
      rýchly: 10,
    }),
    frustration_tolerance_distribution: makeDefaultDistribution(LEVELS, {
      stredná: 10,
    }),
    extra_description: "",
  };
}

export default function Page() {
  // Batch list state
  const [batches, setBatches] = useState<PersonaBatch[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState<string>("");

  // Visibility toggles
  const [showCreate, setShowCreate] = useState(false);
  const [showBatchList, setShowBatchList] = useState(false);

  // Currently expanded batch detail
  const [openBatchId, setOpenBatchId] = useState<number | null>(null);
  const [batchDetail, setBatchDetail] = useState<PersonaBatchDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Create-batch form state
  const [form, setForm] = useState<CreatePersonasPayload>(makeInitialForm());

  function update<K extends keyof CreatePersonasPayload>(
    key: K,
    value: CreatePersonasPayload[K]
  ) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function updateCount(nextCount: number) {
    const parsed = Math.floor(Number(nextCount));

    const safe = Number.isNaN(parsed)
      ? 10
      : Math.max(5, Math.min(50, parsed));

    setForm((prev) => ({
      ...prev,
      count: safe,
    }));
  }

  const createFormError = useMemo(() => {
    const payload: CreatePersonasPayload = {
      ...form,
      name: form.name.trim(),
      extra_description: form.extra_description.trim(),
      age_distribution: normalizeDistribution(form.age_distribution),
      genders_distribution: normalizeDistribution(form.genders_distribution),
      similar_apps_experience_distribution: normalizeDistribution(
        form.similar_apps_experience_distribution
      ),
      decision_style_distribution: normalizeDistribution(form.decision_style_distribution),
      frustration_tolerance_distribution: normalizeDistribution(
        form.frustration_tolerance_distribution
      ),
    };

    return validateCreatePayload(payload);
  }, [form]);

  const canCreate = !submitting && !createFormError;

  /** Fetch the list of all persona batches from the API. */
  async function loadBatches() {
    setLoading(true);
    setStatus("");

    try {
      const res = await fetch(`${API}/persona-batches`, { cache: "no-store" });
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri načítaní skupín: ${data?.detail || res.statusText}`);
        setBatches([]);
        return;
      }

      const data: PersonaBatch[] = await res.json();
      setBatches(data);
    } catch {
      setStatus("Chyba: nepodarilo sa spojiť s backendom.");
      setBatches([]);
    } finally {
      setLoading(false);
    }
  }

  /** Fetch the full detail (including individual personas) for a specific batch. */
  async function loadBatchDetail(batchId: number) {
    setDetailLoading(true);

    try {
      const res = await fetch(`${API}/persona-batches/${batchId}`, {
        cache: "no-store",
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri načítaní detailu: ${data?.detail || res.statusText}`);
        setBatchDetail(null);
        return;
      }

      const data: PersonaBatchDetail = await res.json();
      setBatchDetail(data);
    } catch {
      setStatus("Chyba: nepodarilo sa načítať detail skupiny.");
      setBatchDetail(null);
    } finally {
      setDetailLoading(false);
    }
  }

  /**
   * Validate the create-batch form and submit it to the API.
   * On success, resets the form and reloads the batch list.
   */
  async function createPersonas() {
    setStatus("");

    const payload: CreatePersonasPayload = {
      ...form,
      age_distribution: normalizeDistribution(form.age_distribution),
      genders_distribution: normalizeDistribution(form.genders_distribution),
      similar_apps_experience_distribution: normalizeDistribution(
        form.similar_apps_experience_distribution
      ),
      decision_style_distribution: normalizeDistribution(form.decision_style_distribution),
      frustration_tolerance_distribution: normalizeDistribution(
        form.frustration_tolerance_distribution
      ),
      extra_description: form.extra_description.trim(),
      name: form.name.trim(),
    };

    const err = validateCreatePayload(payload);
    if (err) {
      setStatus(err);
      return;
    }

    setSubmitting(true);

    try {
      const res = await fetch(`${API}/personas/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = (await res.json().catch(() => null)) as
        | CreateBatchResponse
        | { detail?: string | Array<{ loc?: any[]; msg?: string; type?: string }> }
        | null;

      if (!res.ok) {
        const detail = (data as any)?.detail;

        if (Array.isArray(detail)) {
          setStatus(
            `Chyba pri vytváraní: ${detail
              .map((x) => x?.msg)
              .filter(Boolean)
              .join(" | ")}`
          );
        } else {
          setStatus(`Chyba pri vytváraní: ${detail || res.statusText}`);
        }
        return;
      }

      const ok = data as CreateBatchResponse;
      setStatus(
        `Skupina person bola vytvorená. Počet vytvorených person: ${ok.created_persona_ids.length}.`
      );

      setShowCreate(false);

      await loadBatches();

      setOpenBatchId(ok.batch_id);
      await loadBatchDetail(ok.batch_id);
    } catch {
      setStatus("Chyba: nepodarilo sa odoslať požiadavku na backend.");
    } finally {
      setSubmitting(false);
    }
  }

  /**
   * Delete a persona batch and all its personas after user confirmation.
   * Clears the expanded detail view if the deleted batch was open.
   */
  async function deleteBatch(batchId: number) {
    if (!confirm("Naozaj chceš vymazať túto skupinu a všetky jej persony?")) {
      return;
    }

    setStatus("");

    try {
      const res = await fetch(`${API}/persona-batches/${batchId}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri mazaní: ${data?.detail || res.statusText}`);
        return;
      }

      if (openBatchId === batchId) {
        setOpenBatchId(null);
        setBatchDetail(null);
      }

      setStatus("Skupina vymazaná");
      await loadBatches();
    } catch {
      setStatus("Chyba: nepodarilo sa spojiť s backendom.");
    }
  }

  /**
   * Toggle the expanded detail panel for a batch.
   * If the batch is already open, collapse it; otherwise fetch and display its detail.
   */
  async function toggleDetails(batchId: number) {
    const isOpen = openBatchId === batchId;

    if (isOpen) {
      setOpenBatchId(null);
      setBatchDetail(null);
      return;
    }

    setOpenBatchId(batchId);
    await loadBatchDetail(batchId);
  }

  // Load batches once on mount
  useEffect(() => {
    loadBatches();
  }, []);

  const openedBatch = batchDetail?.batch;

  return (
    <div>
      <h1>Správa person</h1>
      <p>
        Tu vytváraš a spravuješ <strong>skupiny person.</strong>
      </p>

      <div style={{ margin: "12px 0" }}>
        <button onClick={() => setShowCreate(true)}>Vytvoriť skupinu</button>{" "}
        <button onClick={loadBatches} disabled={loading}>
          Obnoviť dáta
        </button>
      </div>

      {status && <p>{status}</p>}

      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 18 }}>
      <h2 style={{ margin: 0 }}>Zoznam skupín</h2>
      <button onClick={() => setShowBatchList((prev) => !prev)}>
        {showBatchList ? "Skryť zoznam" : "Zobraziť zoznam"}
      </button>
    </div>

    {showBatchList && (
      loading ? (
        <p>Načítavam…</p>
      ) : batches.length === 0 ? (
        <p>Žiadne skupiny zatiaľ nie sú vytvorené.</p>
      ) : (
        <ul>
          {batches.map((b) => {
            const isOpen = openBatchId === b.id;

            return (
              <li key={b.id} style={{ marginBottom: 16 }}>
                <div>
                  <strong>{b.name}</strong> - počet person: {b.count}
                </div>

                <div style={{ marginTop: 6, display: "flex", gap: 8 }}>
                  <button onClick={() => toggleDetails(b.id)}>
                    {isOpen ? "Skryť podrobnosti" : "Zobraziť podrobnosti"}
                  </button>
                  <button onClick={() => deleteBatch(b.id)}>Vymazať skupinu</button>
                </div>

                {isOpen && (
                  <div
                    style={{
                      marginTop: 10,
                      padding: 10,
                      border: "1px solid #ccc",
                      background: "#f9f9f9",
                    }}
                  >
                    {detailLoading || !openedBatch ? (
                      <p>Načítavam podrobnosti…</p>
                    ) : (
                      <>
                        <div style={{ fontWeight: 700, marginBottom: 8 }}>
                          Parametre agregovanej persony
                        </div>

                        <div style={{ lineHeight: 1.7 }}>
                          <div>
                            <strong>Počet reprezentovaných používateľov:</strong>{" "}
                            {openedBatch.count}
                          </div>
                          <div>
                            <strong>Vekový rozsah:</strong> {openedBatch.age_from} – {openedBatch.age_to}
                          </div>
                          <div>
                            <strong>Vekové rozdelenie:</strong>{" "}
                            {openedBatch.age_distribution
                              ? formatDistribution(openedBatch.age_distribution)
                              : "—"}
                          </div>

                          <div>
                            <strong>Pohlavie:</strong>{" "}
                            {openedBatch.genders_distribution
                              ? formatDistribution(openedBatch.genders_distribution)
                              : formatFallbackList(openedBatch.genders)}
                          </div>

                          <div>
                            <strong>Skúsenosti:</strong>{" "}
                            {openedBatch.similar_apps_experience_distribution
                              ? formatDistribution(openedBatch.similar_apps_experience_distribution)
                              : formatFallbackList(openedBatch.similar_apps_experience)}
                          </div>

                          <div>
                            <strong>Štýl rozhodovania:</strong>{" "}
                            {openedBatch.decision_style_distribution
                              ? formatDistribution(openedBatch.decision_style_distribution)
                              : formatFallbackList(openedBatch.decision_style)}
                          </div>

                          <div>
                            <strong>Tolerancia frustrácie:</strong>{" "}
                            {openedBatch.frustration_tolerance_distribution
                              ? formatDistribution(openedBatch.frustration_tolerance_distribution)
                              : formatFallbackList(openedBatch.frustration_tolerance)}
                          </div>

                          <div style={{ marginTop: 8 }}>
                            <strong>Doplňujúci opis:</strong>{" "}
                            {openedBatch.extra_description?.trim()
                              ? openedBatch.extra_description
                              : "—"}
                          </div>
                        </div>

                        <div style={{ marginTop: 12 }}>
                          <div style={{ fontWeight: 700 }}>Text agregovanej persony</div>
                          <TextCard text={openedBatch.aggregate_persona_text || ""} />
                        </div>

                        <details style={{ marginTop: 14 }}>
                          <summary style={{ cursor: "pointer" }}>
                            Zobraziť jednotlivé persony ({batchDetail?.personas.length || 0})
                          </summary>

                          <div style={{ marginTop: 8 }}>
                            {batchDetail?.personas?.length ? (
                              <ul style={{ paddingLeft: 18 }}>
                                {batchDetail.personas.map((p) => (
                                  <li key={p.id} style={{ marginBottom: 14 }}>
                                    <div>
                                      <strong>
                                        {openedBatch.name} #{p.batch_index}
                                      </strong>
                                    </div>

                                    <TextCard text={p.persona_text || ""} />
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p>Žiadne persony v skupine.</p>
                            )}
                          </div>
                        </details>
                      </>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )
    )}

      {showCreate && (
        <div style={{ marginTop: 18, paddingTop: 12, borderTop: "1px solid #ccc" }}>
          <h2 style={{ marginBottom: 24 }}>Vytvoriť skupinu (agregovanú personu)</h2>

          <div style={{ marginBottom: 18 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Názov skupiny</div>
            <input
              value={form.name}
              onChange={(e) => update("name", e.target.value)}
              style={{ width: 420 }}
            />
          </div>

          <div style={{ marginBottom: 24 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>
              Počet person (5 až 50)
            </div>
            <div
              style={{
                display: "flex",
                gap: 10,
                alignItems: "center",
                flexWrap: "wrap",
              }}
            >
              <input
                className="count-range"
                type="range"
                min={5}
                max={50}
                value={form.count}
                onChange={(e) => updateCount(Number(e.target.value))}
              />
              <input
                type="number"
                min={5}
                max={50}
                value={form.count}
                onChange={(e) => updateCount(Number(e.target.value))}
                style={{ width: 90 }}
              />
            </div>
          </div>

          <DistributionEditor
            label="Vek"
            options={AGE_BANDS}
            items={form.age_distribution}
            totalCount={form.count}
            onChange={(v) => update("age_distribution", v)}
            required
          />

          <DistributionEditor
            label="Pohlavie"
            options={GENDER_OPTS}
            items={form.genders_distribution}
            totalCount={form.count}
            onChange={(v) => update("genders_distribution", v)}
            required
          />

          <hr style={{ margin: "24px 0" }} />
          <div style={{ fontWeight: 700, marginBottom: 20, fontSize: 20 }}>
            Doplňujúce filtre
          </div>

          <DistributionEditor
            label="Skúsenosti s podobnými aplikáciami"
            options={EXPERIENCE_OPTS}
            items={form.similar_apps_experience_distribution}
            totalCount={form.count}
            onChange={(v) => update("similar_apps_experience_distribution", v)}
          />

          <DistributionEditor
            label="Štýl rozhodovania"
            options={DECISION_OPTS}
            items={form.decision_style_distribution}
            totalCount={form.count}
            onChange={(v) => update("decision_style_distribution", v)}
          />

          <DistributionEditor
            label="Tolerancia frustrácie"
            options={LEVELS}
            items={form.frustration_tolerance_distribution}
            totalCount={form.count}
            onChange={(v) => update("frustration_tolerance_distribution", v)}
          />

          <div style={{ marginBottom: 16, marginTop: 8 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>
              Doplňujúci opis pre každú personu (voľný text)
            </div>
            <textarea
              rows={5}
              value={form.extra_description}
              onChange={(e) => update("extra_description", e.target.value)}
              style={{ width: 700, maxWidth: "100%" }}
            />
          </div>

          <div style={{ display: "flex", gap: 10, marginTop: 20 }}>
            <button onClick={createPersonas} disabled={!canCreate}>
              {submitting ? "Vytváram…" : "Vytvoriť"}
            </button>
            <button onClick={() => setShowCreate(false)} disabled={submitting}>
              Zavrieť
            </button>
          </div>
          {createFormError && (
            <div style={{ marginTop: 10, fontSize: 13, color: "#c62828" }}>
              {createFormError}
            </div>
          )}
        </div>
      )}
    </div>
  );
}