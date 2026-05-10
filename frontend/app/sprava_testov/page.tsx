/**
 * Test task management page — Správa testov.
 *
 * Allows the user to create, view, edit, and delete UX test tasks.
 * Tasks are grouped into three types shown as tabs:
 *   - preference  : A/B test between two or more UI design variants
 *   - first_click : User identifies the first element they would click
 *   - feedback    : User provides open-ended observations about a screenshot
 *
 * Images are stored as base64 data URLs embedded directly in the task config.
 * All data is persisted via the backend REST API.
 */
"use client";

import { useEffect, useMemo, useState } from "react";

// Backend API base URL — can be overridden via NEXT_PUBLIC_API_URL environment variable
const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type TaskType = "preference" | "first_click" | "feedback";

// Full representation of a test task returned by the API
type TestTask = {
  id: number;
  test_id: number;
  task_type: TaskType;
  order_index: number;
  title: string;
  task_text: string;
  follow_up_question: string | null;
  is_active: boolean;
  config: Record<string, any>;
};

type PreferenceOption = {
  label: string;
  image: string; 
};

// Tab definitions for the three supported task types
const TABS: { type: TaskType; label: string }[] = [
  { type: "preference", label: "Preferenčný test" },
  { type: "first_click", label: "First-click test" },
  { type: "feedback", label: "Spätná väzba" },
];

// Maximum allowed image size (3 MB) — larger images cause API payload issues
const MAX_IMAGE_BYTES = 3 * 1024 * 1024;

/** Read a File object and return its contents as a base64 data URL string. */
function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(new Error("Nepodarilo sa načítať súbor"));
    r.readAsDataURL(file);
  });
}

/** Format a byte count as a human-readable string (B / KB / MB). */
function bytesToHuman(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export default function Page() {
  // Currently active task type tab
  const [tab, setTab] = useState<TaskType>("preference");

  // Task list state
  const [tasks, setTasks] = useState<TestTask[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");

  // UI visibility toggles
  const [showTasks, setShowTasks] = useState(true);

  // Per-tab visibility of the "add task" form
  const [showAddFormByTab, setShowAddFormByTab] = useState<Record<TaskType, boolean>>({
    preference: false,
    first_click: false,
    feedback: false,
  });
  const showAddForm = showAddFormByTab[tab];


  // Loading states for async operations
  const [adding, setAdding] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  // Form state for adding a new preference task
  const [prefDescription, setPrefDescription] = useState("");
  const [prefOptions, setPrefOptions] = useState<PreferenceOption[]>([]);
  const [prefFollowEnabled, setPrefFollowEnabled] = useState(false);
  const [prefFollowQuestion, setPrefFollowQuestion] = useState("");

  // Form state for adding a new first-click task
  const [fcTaskText, setFcTaskText] = useState("");
  const [fcImage, setFcImage] = useState<string>("");
  const [fcFollowEnabled, setFcFollowEnabled] = useState(false);
  const [fcFollowQuestion, setFcFollowQuestion] = useState("");

  // Form state for adding a new feedback task
  const [fbTaskText, setFbTaskText] = useState("");
  const [fbImage, setFbImage] = useState<string>("");

  // Inline edit state — editingId is the ID of the task currently being edited
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editSaving, setEditSaving] = useState(false);

  const [editTaskText, setEditTaskText] = useState("");
  const [editFollowEnabled, setEditFollowEnabled] = useState(false);
  const [editFollowQuestion, setEditFollowQuestion] = useState("");

  // Edit form state for preference options and single image (first_click / feedback)
  const [editPrefOptions, setEditPrefOptions] = useState<PreferenceOption[]>([]);
  const [editImage, setEditImage] = useState<string>(""); 

  // Tasks sorted by order_index for display
  const visibleTasks = useMemo(() => {
    return tasks.slice().sort((a, b) => a.order_index - b.order_index);
  }, [tasks]);

  /** Fetch tasks of the current tab type from the API. */
  async function loadTasks(signal?: AbortSignal) {
    setLoading(true);
    setStatus("");

    try {
      const res = await fetch(`${API}/test/tasks?task_type=${tab}`, {
        cache: "no-store",
        signal,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri načítaní úloh: ${data?.detail || res.statusText}`);
        setTasks([]);
        return;
      }

      const data: TestTask[] = await res.json();
      setTasks(data);
    } catch (e: any) {
      if (e?.name === "AbortError") return;
      setStatus("Chyba: nepodarilo sa spojiť s backendom.");
      setTasks([]);
    } finally {
      setLoading(false);
    }
  }

  /** Validate and submit a new preference task to the API. */
  async function addPreferenceTask() {
    setStatus("");

    if (!prefDescription.trim()) {
      setStatus("Zadaj popis úlohy.");
      return;
    }
    if (prefOptions.length < 2) {
      setStatus("Preferenčný test vyžaduje aspoň 2 možnosti (screenshoty).");
      return;
    }
    if (prefOptions.length > 4) {
      setStatus("Maximálne 4 možnosti.");
      return;
    }
    if (prefFollowEnabled && !prefFollowQuestion.trim()) {
      setStatus("Zadaj doplňujúcu otázku alebo vypni túto možnosť.");
      return;
    }

    const task_text = prefDescription.trim();
    const title = task_text.length > 60 ? task_text.slice(0, 60) + "…" : task_text;

    const config: Record<string, any> = {
      options: prefOptions.map((o, i) => ({
        label: (o.label || `Možnosť ${i + 1}`).trim(),
        image: o.image,
      })),
    };

    setAdding(true);
    try {
      const res = await fetch(`${API}/test/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_type: "preference",
          title,
          task_text,
          follow_up_question: prefFollowEnabled ? prefFollowQuestion.trim() : null,
          config,
          is_active: true,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri pridávaní úlohy: ${data?.detail || res.statusText}`);
        return;
      }

      setStatus("Úloha pridaná");
      setPrefDescription("");
      setPrefOptions([]);
      setPrefFollowEnabled(false);
      setPrefFollowQuestion("");
      await loadTasks();
    } catch {
      setStatus("Chyba: nepodarilo sa odoslať požiadavku na backend.");
    } finally {
      setAdding(false);
    }
  }

  /** Validate and submit a new first-click task to the API. */
  async function addFirstClickTask() {
    setStatus("");

    if (!fcTaskText.trim()) {
      setStatus("Zadaj zadanie úlohy.");
      return;
    }
    if (!fcImage) {
      setStatus("Pridaj screenshot pre first-click úlohu.");
      return;
    }
    if (fcFollowEnabled && !fcFollowQuestion.trim()) {
      setStatus("Zadaj doplňujúcu otázku alebo vypni túto možnosť.");
      return;
    }

    const task_text = fcTaskText.trim();
    const title = task_text.length > 60 ? task_text.slice(0, 60) + "…" : task_text;

    const config: Record<string, any> = { image: fcImage };

    setAdding(true);
    try {
      const res = await fetch(`${API}/test/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_type: "first_click",
          title,
          task_text,
          follow_up_question: fcFollowEnabled ? fcFollowQuestion.trim() : null,
          config,
          is_active: true,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri pridávaní úlohy: ${data?.detail || res.statusText}`);
        return;
      }

      setStatus("Úloha pridaná");
      setFcTaskText("");
      setFcImage("");
      setFcFollowEnabled(false);
      setFcFollowQuestion("");
      await loadTasks();
    } catch {
      setStatus("Chyba: nepodarilo sa odoslať požiadavku na backend.");
    } finally {
      setAdding(false);
    }
  }

  /** Validate and submit a new feedback task to the API. */
  async function addFeedbackTask() {
    setStatus("");

    if (!fbTaskText.trim()) {
      setStatus("Zadaj zadanie úlohy.");
      return;
    }
    if (!fbImage) {
      setStatus("Pridaj screenshot pre úlohu spätnej väzby.");
      return;
    }

    const task_text = fbTaskText.trim();
    const title = task_text.length > 60 ? task_text.slice(0, 60) + "…" : task_text;

    const config: Record<string, any> = { image: fbImage };

    setAdding(true);
    try {
      const res = await fetch(`${API}/test/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_type: "feedback",
          title,
          task_text,
          follow_up_question: null,
          config,
          is_active: true,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri pridávaní úlohy: ${data?.detail || res.statusText}`);
        return;
      }

      setStatus("Úloha pridaná");
      setFbTaskText("");
      setFbImage("");
      await loadTasks();
    } catch {
      setStatus("Chyba: nepodarilo sa odoslať požiadavku na backend.");
    } finally {
      setAdding(false);
    }
  }

  /** Delete a task by ID after user confirmation. */
  async function deleteTask(id: number) {
    if (!confirm("Naozaj chceš vymazať túto úlohu?")) return;

    setStatus("");
    setDeletingId(id);
    try {
      const res = await fetch(`${API}/test/tasks/${id}`, { method: "DELETE" });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri mazaní úlohy: ${data?.detail || res.statusText}`);
        return;
      }

      setStatus("Úloha vymazaná");
      await loadTasks();
    } catch {
      setStatus("Chyba: nepodarilo sa spojiť s backendom.");
    } finally {
      setDeletingId(null);
    }
  }

  /**
   * Handle image file selection for a preference task.
   * Validates file size, reads files as data URLs, and appends them as new options (max 4 total).
   */
  async function onPickPreferenceImages(files: FileList | null) {
    setStatus("");
    if (!files) return;

    const arr = Array.from(files);
    const spaceLeft = Math.max(0, 4 - prefOptions.length);
    const take = arr.slice(0, spaceLeft);

    if (arr.length > spaceLeft) {
      setStatus("Uložené budú len prvé vybrané screenshoty do maxima 4.");
    }

    const tooBig = take.find((f) => f.size > MAX_IMAGE_BYTES);
    if (tooBig) {
      setStatus(
        `Obrázok "${tooBig.name}" je príliš veľký (${bytesToHuman(
          tooBig.size
        )}). Limit je ${bytesToHuman(MAX_IMAGE_BYTES)}.`
      );
      return;
    }

    try {
      const dataUrls = await Promise.all(take.map(readFileAsDataUrl));
      setPrefOptions((prev) => {
        const startIndex = prev.length;
        const newOnes: PreferenceOption[] = dataUrls.map((src, i) => ({
          label: `Možnosť ${startIndex + i + 1}`,
          image: src,
        }));
        return [...prev, ...newOnes].slice(0, 4);
      });
    } catch {
      setStatus("Chyba: nepodarilo sa načítať obrázok.");
    }
  }

  /** Remove a preference option at the given index from the new-task form. */
  function removePrefOption(idx: number) {
    setPrefOptions((prev) => prev.filter((_, i) => i !== idx));
  }

  /** Update the label of a preference option at the given index. */
  function updatePrefOptionLabel(idx: number, label: string) {
    setPrefOptions((prev) => prev.map((o, i) => (i === idx ? { ...o, label } : o)));
  }

  /** Handle image selection for a first-click task; validates size and reads as data URL. */
  async function onPickFirstClickImage(file: File | null) {
    setStatus("");
    if (!file) return;

    if (file.size > MAX_IMAGE_BYTES) {
      setStatus(
        `Obrázok "${file.name}" je príliš veľký (${bytesToHuman(
          file.size
        )}). Limit je ${bytesToHuman(MAX_IMAGE_BYTES)}.`
      );
      return;
    }

    try {
      const dataUrl = await readFileAsDataUrl(file);
      setFcImage(dataUrl);
    } catch {
      setStatus("Chyba: nepodarilo sa načítať obrázok.");
    }
  }

  /** Clear the selected first-click image from the new-task form. */
  function removeFirstClickImage() {
    setFcImage("");
  }

  /** Handle image selection for a feedback task; validates size and reads as data URL. */
  async function onPickFeedbackImage(file: File | null) {
    setStatus("");
    if (!file) return;

    if (file.size > MAX_IMAGE_BYTES) {
      setStatus(
        `Obrázok "${file.name}" je príliš veľký (${bytesToHuman(
          file.size
        )}). Limit je ${bytesToHuman(MAX_IMAGE_BYTES)}.`
      );
      return;
    }

    try {
      const dataUrl = await readFileAsDataUrl(file);
      setFbImage(dataUrl);
    } catch {
      setStatus("Chyba: nepodarilo sa načítať obrázok.");
    }
  }

  /** Clear the selected feedback image from the new-task form. */
  function removeFeedbackImage() {
    setFbImage("");
  }

  /** Open the inline edit form for a task and pre-populate it with the task's current values. */
  function startEdit(t: TestTask) {
    setStatus("");
    setEditingId(t.id);

    setEditTaskText(t.task_text || "");
    const hasFollow = !!t.follow_up_question && t.follow_up_question.trim().length > 0;
    setEditFollowEnabled(hasFollow);
    setEditFollowQuestion(t.follow_up_question || "");

    if (t.task_type === "preference") {
      const opts = Array.isArray(t.config?.options) ? t.config.options : [];
      setEditPrefOptions(
        opts.map((o: any, i: number) => ({
          label: String(o?.label || `Možnosť ${i + 1}`),
          image: String(o?.image || ""),
        }))
      );
      setEditImage("");
    } else {
      setEditImage(String(t.config?.image || ""));
      setEditPrefOptions([]);
    }
  }

  /** Close the inline edit form and reset all edit state. */
  function cancelEdit() {
    setEditingId(null);
    setEditTaskText("");
    setEditFollowEnabled(false);
    setEditFollowQuestion("");
    setEditPrefOptions([]);
    setEditImage("");
  }

  /** Validate the edit form and save changes to the task via a PUT request. */
  async function saveEdit(t: TestTask) {
    setStatus("");

    const task_text = editTaskText.trim();
    if (!task_text) {
      setStatus("task_text nesmie byť prázdny.");
      return;
    }

    let follow_up_question: string | null = null;
    if (t.task_type !== "feedback" && editFollowEnabled) {
      if (!editFollowQuestion.trim()) {
        setStatus("Zadaj doplňujúcu otázku alebo vypni túto možnosť.");
        return;
      }
      follow_up_question = editFollowQuestion.trim();
    }

    let config: Record<string, any> = {};
    if (t.task_type === "preference") {
      if (editPrefOptions.length < 2 || editPrefOptions.length > 4) {
        setStatus("Preferenčný test musí mať 2 až 4 možnosti.");
        return;
      }
      for (let i = 0; i < editPrefOptions.length; i++) {
        const o = editPrefOptions[i];
        if (!o.label.trim()) {
          setStatus(`Možnosť ${i + 1} nemá label.`);
          return;
        }
        if (!o.image) {
          setStatus(`Možnosť ${i + 1} nemá obrázok.`);
          return;
        }
      }
      config = {
        options: editPrefOptions.map((o) => ({
          label: o.label.trim(),
          image: o.image,
        })),
      };
    } else {
      if (!editImage) {
        setStatus("Chýba obrázok.");
        return;
      }
      config = { image: editImage };
    }

    setEditSaving(true);
    try {
      const res = await fetch(`${API}/test/tasks/${t.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_text,
          follow_up_question,
          config,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setStatus(`Chyba pri ukladaní: ${data?.detail || res.statusText}`);
        return;
      }

      setStatus("Úloha upravená");
      cancelEdit();
      await loadTasks();
    } catch {
      setStatus("Chyba: nepodarilo sa spojiť s backendom.");
    } finally {
      setEditSaving(false);
    }
  }

  /** Handle image replacement in the edit form for first-click and feedback tasks. */
  async function onPickEditImage(file: File | null) {
    setStatus("");
    if (!file) return;

    if (file.size > MAX_IMAGE_BYTES) {
      setStatus(`Obrázok je príliš veľký (${bytesToHuman(file.size)}).`);
      return;
    }

    try {
      const dataUrl = await readFileAsDataUrl(file);
      setEditImage(dataUrl);
    } catch {
      setStatus("Chyba: nepodarilo sa načítať obrázok.");
    }
  }

  /** Replace the image of a specific preference option in the edit form. */
  async function onReplaceEditPrefOptionImage(idx: number, file: File | null) {
    setStatus("");
    if (!file) return;

    if (file.size > MAX_IMAGE_BYTES) {
      setStatus(`Obrázok je príliš veľký (${bytesToHuman(file.size)}).`);
      return;
    }

    try {
      const dataUrl = await readFileAsDataUrl(file);
      setEditPrefOptions((prev) =>
        prev.map((o, i) => (i === idx ? { ...o, image: dataUrl } : o))
      );
    } catch {
      setStatus("Chyba: nepodarilo sa načítať obrázok.");
    }
  }

  /** Update the label of a preference option in the edit form. */
  function updateEditPrefOptionLabel(idx: number, label: string) {
    setEditPrefOptions((prev) => prev.map((o, i) => (i === idx ? { ...o, label } : o)));
  }

  /** Remove a preference option at the given index from the edit form. */
  function removeEditPrefOption(idx: number) {
    setEditPrefOptions((prev) => prev.filter((_, i) => i !== idx));
  }

  /** Add new preference option images in the edit form (up to 4 total). */
  async function addEditPrefOptions(files: FileList | null) {
    setStatus("");
    if (!files) return;

    const arr = Array.from(files);
    const spaceLeft = Math.max(0, 4 - editPrefOptions.length);
    const take = arr.slice(0, spaceLeft);

    if (arr.length > spaceLeft) {
      setStatus("Uložené budú len prvé vybrané screenshoty do maxima 4.");
    }

    const tooBig = take.find((f) => f.size > MAX_IMAGE_BYTES);
    if (tooBig) {
      setStatus(`Obrázok "${tooBig.name}" je príliš veľký (${bytesToHuman(tooBig.size)}).`);
      return;
    }

    try {
      const dataUrls = await Promise.all(take.map(readFileAsDataUrl));
      setEditPrefOptions((prev) => {
        const start = prev.length;
        const added = dataUrls.map((src, i) => ({
          label: `Možnosť ${start + i + 1}`,
          image: src,
        }));
        return [...prev, ...added].slice(0, 4);
      });
    } catch {
      setStatus("Chyba: nepodarilo sa načítať obrázok.");
    }
  }

  // Reload tasks whenever the active tab changes; abort the previous request if still in flight
  useEffect(() => {
    const ac = new AbortController();
    loadTasks(ac.signal);
    return () => ac.abort();
  }, [tab]);

  return (
    <div>
      <h1>Správa testu</h1>
      <p>
        Tu spravuješ úlohy globálneho testu. Úlohy sú rozdelené do troch typov.
      </p>

      <div style={{ margin: "12px 0" }}>
        <button onClick={() => loadTasks()} disabled={loading}>
          Obnoviť dáta
        </button>
      </div>

      {status && <p>{status}</p>}

      <h2>Typ úloh</h2>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
        {TABS.map((t) => (
          <button
            key={t.type}
            onClick={() => {
              cancelEdit();
              setTab(t.type);
            }}
            disabled={tab === t.type || loading || adding || editSaving}
          >
            {t.label}
          </button>
        ))}
      </div>

      <h2>{TABS.find((x) => x.type === tab)?.label}</h2>

      {tab === "preference" && (
        <>
          <h3 style={{ display: "flex", alignItems: "center", gap: 12 }}>
            Pridať úlohu (preferenčný test)
            <button
              onClick={() =>
                setShowAddFormByTab((prev) => ({ ...prev, [tab]: !prev[tab] }))
              }
              disabled={adding || editSaving}
            >
              {showAddForm ? "Skryť" : "Zobraziť"}
            </button>
          </h3>

          {showAddForm && (
            <>
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>Popis úlohy</div>
                <textarea
                  rows={4}
                  value={prefDescription}
                  onChange={(e) => setPrefDescription(e.target.value)}
                  style={{ width: 700, maxWidth: "100%" }}
                  placeholder=""
                  disabled={adding || editSaving}
                />
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  Možnosti (max 4) – screenshoty na porovnanie
                </div>
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => onPickPreferenceImages(e.target.files)}
                  disabled={adding || editSaving}
                />

                {prefOptions.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 12, marginBottom: 6 }}>
                      Pridané možnosti: {prefOptions.length} / 4
                    </div>

                    <ul>
                      {prefOptions.map((opt, i) => (
                        <li key={i} style={{ marginBottom: 14 }}>
                          <div
                            style={{
                              marginBottom: 6,
                              display: "flex",
                              gap: 8,
                              flexWrap: "wrap",
                              alignItems: "center",
                            }}
                          >
                            <button
                              onClick={() => removePrefOption(i)}
                              disabled={adding || editSaving}
                            >
                              Odstrániť
                            </button>
                            <span style={{ fontWeight: 600 }}>Názov možnosti</span>
                            <input
                              value={opt.label}
                              onChange={(e) => updatePrefOptionLabel(i, e.target.value)}
                              style={{ width: 260, maxWidth: "100%" }}
                              placeholder=""
                              disabled={adding || editSaving}
                            />
                          </div>

                          <img
                            src={opt.image}
                            alt={`option-${i + 1}`}
                            style={{
                              maxWidth: 700,
                              width: "100%",
                              border: "1px solid #ccc",
                            }}
                          />
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div style={{ marginBottom: 10 }}>
                <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={prefFollowEnabled}
                    onChange={(e) => setPrefFollowEnabled(e.target.checked)}
                    disabled={adding || editSaving}
                  />
                  Pridať doplňujúcu otázku (voliteľné)
                </label>

                {prefFollowEnabled && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>
                      Doplňujúca otázka
                    </div>
                    <input
                      value={prefFollowQuestion}
                      onChange={(e) => setPrefFollowQuestion(e.target.value)}
                      style={{ width: 700, maxWidth: "100%" }}
                      placeholder=""
                      disabled={adding || editSaving}
                    />
                  </div>
                )}
              </div>

              <button onClick={addPreferenceTask} disabled={adding || editSaving}>
                {adding ? "Pridávam…" : "Pridať úlohu"}
              </button>
            </>
          )}
        </>
      )}

      {tab === "first_click" && (
        <>
          <h3 style={{ display: "flex", alignItems: "center", gap: 12 }}>
            Pridať úlohu (first-click test)
            <button
              onClick={() =>
                setShowAddFormByTab((prev) => ({ ...prev, [tab]: !prev[tab] }))
              }
              disabled={adding || editSaving}
            >
              {showAddForm ? "Skryť" : "Zobraziť"}
            </button>
          </h3>

          {showAddForm && (
            <>
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>Zadanie úlohy</div>
                <textarea
                  rows={4}
                  value={fcTaskText}
                  onChange={(e) => setFcTaskText(e.target.value)}
                  style={{ width: 700, maxWidth: "100%" }}
                  placeholder=""
                  disabled={adding || editSaving}
                />
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  Screenshot (kde by používateľ klikol prvýkrát)
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => onPickFirstClickImage(e.target.files?.[0] || null)}
                  disabled={adding || editSaving}
                />

                {fcImage && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ marginBottom: 6 }}>
                      <button
                        onClick={removeFirstClickImage}
                        disabled={adding || editSaving}
                      >
                        Odstrániť obrázok
                      </button>
                    </div>
                    <img
                      src={fcImage}
                      alt="first-click"
                      style={{ maxWidth: 700, width: "100%", border: "1px solid #ccc" }}
                    />
                  </div>
                )}
              </div>

              <div style={{ marginBottom: 10 }}>
                <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={fcFollowEnabled}
                    onChange={(e) => setFcFollowEnabled(e.target.checked)}
                    disabled={adding || editSaving}
                  />
                  Pridať doplňujúcu otázku (voliteľné)
                </label>

                {fcFollowEnabled && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>
                      Doplňujúca otázka
                    </div>
                    <input
                      value={fcFollowQuestion}
                      onChange={(e) => setFcFollowQuestion(e.target.value)}
                      style={{ width: 700, maxWidth: "100%" }}
                      placeholder=""
                      disabled={adding || editSaving}
                    />
                  </div>
                )}
              </div>

              <button onClick={addFirstClickTask} disabled={adding || editSaving}>
                {adding ? "Pridávam…" : "Pridať úlohu"}
              </button>
            </>
          )}
        </>
      )}

      {tab === "feedback" && (
        <>
          <h3 style={{ display: "flex", alignItems: "center", gap: 12 }}>
            Pridať úlohu (spätná väzba)
            <button
              onClick={() =>
                setShowAddFormByTab((prev) => ({ ...prev, [tab]: !prev[tab] }))
              }
              disabled={adding || editSaving}
            >
              {showAddForm ? "Skryť" : "Zobraziť"}
            </button>
          </h3>

          {showAddForm && (
            <>
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>Zadanie úlohy</div>
                <textarea
                  rows={4}
                  value={fbTaskText}
                  onChange={(e) => setFbTaskText(e.target.value)}
                  style={{ width: 700, maxWidth: "100%" }}
                  placeholder=""
                  disabled={adding || editSaving}
                />
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  Screenshot (z ktorého má model poskytnúť spätnú väzbu)
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => onPickFeedbackImage(e.target.files?.[0] || null)}
                  disabled={adding || editSaving}
                />

                {fbImage && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ marginBottom: 6 }}>
                      <button onClick={removeFeedbackImage} disabled={adding || editSaving}>
                        Odstrániť obrázok
                      </button>
                    </div>
                    <img
                      src={fbImage}
                      alt="feedback"
                      style={{ maxWidth: 700, width: "100%", border: "1px solid #ccc" }}
                    />
                  </div>
                )}
              </div>

              <button onClick={addFeedbackTask} disabled={adding || editSaving}>
                {adding ? "Pridávam…" : "Pridať úlohu"}
              </button>
            </>
          )}
        </>
      )}

      <h3 style={{ marginTop: 18 }}>
        Existujúce úlohy
        <button
          style={{ marginLeft: 12 }}
          onClick={() => setShowTasks((prev) => !prev)}
          disabled={loading}
        >
          {showTasks ? "Skryť" : "Zobraziť"}
        </button>
      </h3>

      {showTasks &&
        (loading ? (
          <p>Načítavam…</p>
        ) : visibleTasks.length === 0 ? (
          <p>Žiadne úlohy pre tento typ zatiaľ nie sú.</p>
        ) : (
          <ul>
            {visibleTasks.map((t) => (
              <li key={t.id} style={{ marginBottom: 16 }}>
                <div>
                  <strong>#{t.order_index}</strong> {t.task_text || t.title}
                </div>

                {t.follow_up_question && (
                  <div style={{ marginTop: 6 }}>
                    <div style={{ fontWeight: 600 }}>Doplňujúca otázka</div>
                    <div>{t.follow_up_question}</div>
                  </div>
                )}

                {t.task_type === "preference" && (
                  <>
                    {Array.isArray(t.config?.options) && t.config.options.length > 0 && (
                      <div style={{ marginTop: 6 }}>
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>
                          Možnosti ({t.config.options.length})
                        </div>
                        <ul>
                          {t.config.options.map((opt: any, i: number) => (
                            <li key={i} style={{ marginBottom: 12 }}>
                              <div style={{ marginBottom: 6 }}>
                                <strong>{opt?.label || `Možnosť ${i + 1}`}</strong>
                              </div>
                              {opt?.image && (
                                <img
                                  src={opt.image}
                                  alt={`task-${t.id}-opt-${i + 1}`}
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
                  </>
                )}

                {t.task_type === "first_click" && (
                  <>
                    {t.config?.image && (
                      <div style={{ marginTop: 6 }}>
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>Screenshot</div>
                        <img
                          src={t.config.image}
                          alt={`task-${t.id}-firstclick`}
                          style={{ maxWidth: 700, width: "100%", border: "1px solid #ccc" }}
                        />
                      </div>
                    )}
                  </>
                )}

                {t.task_type === "feedback" && (
                  <>
                    {t.config?.image && (
                      <div style={{ marginTop: 6 }}>
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>Screenshot</div>
                        <img
                          src={t.config.image}
                          alt={`task-${t.id}-feedback`}
                          style={{ maxWidth: 700, width: "100%", border: "1px solid #ccc" }}
                        />
                      </div>
                    )}
                  </>
                )}

                <div style={{ marginTop: 6, display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <button
                    onClick={() => startEdit(t)}
                    disabled={adding || deletingId === t.id || editSaving || loading}
                  >
                    Editovať
                  </button>

                  <button
                    onClick={() => deleteTask(t.id)}
                    disabled={deletingId === t.id || adding || editSaving}
                  >
                    {deletingId === t.id ? "Mažem…" : "Vymazať"}
                  </button>
                </div>

                {editingId === t.id && (
                  <div style={{ marginTop: 10, padding: 10, border: "1px solid #ddd" }}>
                    <div style={{ fontWeight: 700, marginBottom: 8 }}>Editácia úlohy</div>

                    <div style={{ marginBottom: 10 }}>
                      <div style={{ fontWeight: 600, marginBottom: 4 }}>Text úlohy</div>
                      <textarea
                        rows={4}
                        value={editTaskText}
                        onChange={(e) => setEditTaskText(e.target.value)}
                        style={{ width: 700, maxWidth: "100%" }}
                        disabled={editSaving}
                      />
                    </div>

                    {t.task_type !== "feedback" && (
                      <div style={{ marginBottom: 10 }}>
                        <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
                          <input
                            type="checkbox"
                            checked={editFollowEnabled}
                            onChange={(e) => setEditFollowEnabled(e.target.checked)}
                            disabled={editSaving}
                          />
                          Doplňujúca otázka (voliteľné)
                        </label>

                        {editFollowEnabled && (
                          <input
                            value={editFollowQuestion}
                            onChange={(e) => setEditFollowQuestion(e.target.value)}
                            style={{ width: 700, maxWidth: "100%", marginTop: 6 }}
                            disabled={editSaving}
                          />
                        )}
                      </div>
                    )}

                    {t.task_type === "preference" ? (
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>Možnosti (2–4)</div>

                        <input
                          type="file"
                          accept="image/*"
                          multiple
                          onChange={(e) => addEditPrefOptions(e.target.files)}
                          disabled={editSaving}
                        />

                        <ul style={{ marginTop: 10 }}>
                          {editPrefOptions.map((opt, i) => (
                            <li key={i} style={{ marginBottom: 14 }}>
                              <div
                                style={{
                                  display: "flex",
                                  gap: 8,
                                  flexWrap: "wrap",
                                  alignItems: "center",
                                }}
                              >
                                <button
                                  onClick={() => removeEditPrefOption(i)}
                                  disabled={editSaving}
                                >
                                  Odstrániť možnosť
                                </button>

                                <span style={{ fontWeight: 600 }}>Label</span>
                                <input
                                  value={opt.label}
                                  onChange={(e) => updateEditPrefOptionLabel(i, e.target.value)}
                                  style={{ width: 260, maxWidth: "100%" }}
                                  disabled={editSaving}
                                />

                                <input
                                  type="file"
                                  accept="image/*"
                                  onChange={(e) =>
                                    onReplaceEditPrefOptionImage(i, e.target.files?.[0] || null)
                                  }
                                  disabled={editSaving}
                                />
                              </div>

                              {opt.image && (
                                <img
                                  src={opt.image}
                                  alt={`edit-opt-${i + 1}`}
                                  style={{
                                    maxWidth: 700,
                                    width: "100%",
                                    border: "1px solid #ccc",
                                    marginTop: 8,
                                  }}
                                />
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : (
                      <div style={{ marginBottom: 10 }}>
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>Obrázok</div>

                        <input
                          type="file"
                          accept="image/*"
                          onChange={(e) => onPickEditImage(e.target.files?.[0] || null)}
                          disabled={editSaving}
                        />

                        {editImage && (
                          <div style={{ marginTop: 8 }}>
                            <button onClick={() => setEditImage("")} disabled={editSaving}>
                              Odstrániť obrázok
                            </button>
                            <img
                              src={editImage}
                              alt="edit-image"
                              style={{
                                maxWidth: 700,
                                width: "100%",
                                border: "1px solid #ccc",
                                marginTop: 8,
                              }}
                            />
                          </div>
                        )}
                      </div>
                    )}

                    <div style={{ display: "flex", gap: 8 }}>
                      <button onClick={() => saveEdit(t)} disabled={editSaving}>
                        {editSaving ? "Ukladám…" : "Uložiť"}
                      </button>
                      <button onClick={cancelEdit} disabled={editSaving}>
                        Zrušiť
                      </button>
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ul>
        ))}
    </div>
  );
}