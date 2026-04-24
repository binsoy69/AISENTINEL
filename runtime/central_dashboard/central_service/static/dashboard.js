const bootstrapNode = document.getElementById("dashboard-bootstrap");
const initialState = bootstrapNode ? JSON.parse(bootstrapNode.textContent) : {};

const POLL_MS = 2000;
const CLOCK_MS = 1000;
const EMPTY_SUBJECT_VALUE = "__unassigned_subject__";
const CHART_COLORS = ["#3f83ff", "#f1c44c", "#ff6558", "#21badf", "#7a63ff", "#28d17c"];
const SEAT_LAYOUT = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const REVIEW_META = {
    unverified: { label: "Unverified", className: "is-created" },
    verified: { label: "Verified", className: "is-active" },
    false_detection: { label: "False Detection", className: "is-error" },
};

const els = {
    sessionLabel: document.getElementById("session-label"),
    headerElapsedLabel: document.getElementById("header-elapsed-label"),
    statusPill: document.getElementById("status-pill"),
    banner: document.getElementById("dashboard-banner"),
    noiseBanner: document.getElementById("noise-banner"),
    sessionPanel: document.getElementById("session-panel"),
    sessionAccordion: document.getElementById("session-accordion"),
    sessionToggle: document.getElementById("session-toggle"),
    sessionStatusPill: document.getElementById("session-status-pill"),
    sessionSummaryCopy: document.getElementById("session-summary-copy"),
    sessionSubjectLabel: document.getElementById("session-subject-label"),
    sessionProfessorLabel: document.getElementById("session-professor-label"),
    sessionDateLabel: document.getElementById("session-date-label"),
    sessionIdLabel: document.getElementById("session-id-label"),
    sessionForm: document.getElementById("session-form"),
    createSessionButton: document.getElementById("create-session-button"),
    clearSessionButton: document.getElementById("clear-session-button"),
    startSessionButton: document.getElementById("start-session-button"),
    restartSessionButton: document.getElementById("restart-session-button"),
    stopSessionButton: document.getElementById("stop-session-button"),
    metricTotalIncidents: document.getElementById("metric-total-incidents"),
    metricTotalFoot: document.getElementById("metric-total-foot"),
    metricFlaggedSeats: document.getElementById("metric-flagged-seats"),
    metricSeatFoot: document.getElementById("metric-seat-foot"),
    metricOnlineNodes: document.getElementById("metric-online-nodes"),
    metricOnlineFoot: document.getElementById("metric-online-foot"),
    noisePanel: document.getElementById("noise-panel"),
    noiseStatusPill: document.getElementById("noise-status-pill"),
    noiseDbValue: document.getElementById("noise-db-value"),
    noiseMeterTrack: document.getElementById("noise-meter-track"),
    noiseMeterFill: document.getElementById("noise-meter-fill"),
    feedGrid: document.getElementById("feed-grid"),
    recordsContextLabel: document.getElementById("records-context-label"),
    recordsBody: document.getElementById("records-body"),
    recordsScopePicker: document.getElementById("records-scope-picker"),
    recordsSubject: document.getElementById("records-subject"),
    recordsSession: document.getElementById("records-session"),
    recordsFilter: document.getElementById("records-filter"),
    recordsSearch: document.getElementById("records-search"),
    recordsClear: document.getElementById("records-clear"),
    recordsExport: document.getElementById("records-export"),
    typeChart: document.getElementById("type-chart"),
    timelineChart: document.getElementById("timeline-chart"),
    analyticsTypesNote: document.getElementById("analytics-types-note"),
    analyticsTimelineNote: document.getElementById("analytics-timeline-note"),
    seatmapContextLabel: document.getElementById("seatmap-context-label"),
    seatmapGrid: document.getElementById("seatmap-grid"),
    historyBody: document.getElementById("history-body"),
    systemGrid: document.getElementById("system-grid"),
    evidenceViewer: document.getElementById("evidence-viewer"),
    evidenceViewerImage: document.getElementById("evidence-viewer-image"),
    evidenceViewerMeta: document.getElementById("evidence-viewer-meta"),
    evidenceViewerTitle: document.getElementById("evidence-viewer-title"),
    evidenceViewerOpen: document.getElementById("evidence-viewer-open"),
};

const state = {
    snapshot: initialState,
    feedModes: {},
    recordsSubject: "",
    recordsSessionId: "",
    recordsFilter: "all",
    recordsQuery: "",
    knownIncidentIds: new Set((Array.isArray(initialState.incidents) ? initialState.incidents : []).map((incident) => String(incident.incident_id || "")).filter(Boolean)),
    sessionHydrated: false,
    sessionFormDirty: false,
    sessionDefaultsApplied: false,
    pollInFlight: false,
};

function escapeHtml(value) {
    return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll("\"", "&quot;").replaceAll("'", "&#39;");
}

function csvEscape(value) {
    return `"${String(value ?? "").replaceAll("\"", "\"\"")}"`;
}

function currentSession() {
    const session = state.snapshot.active_session;
    return session && String(session.session_id || "").trim() ? session : null;
}

function sessionsHistory() {
    return Array.isArray(state.snapshot.sessions_history) ? state.snapshot.sessions_history : [];
}

function normalizedSubjectCode(value) {
    return String(value ?? "").trim() || EMPTY_SUBJECT_VALUE;
}

function subjectCodeLabel(value) {
    return value === EMPTY_SUBJECT_VALUE ? "No Subject Code" : value;
}

function workspaceSession() {
    const session = currentSession();
    if (session) return session;
    return sessionsHistory().find((item) => item.session_id === state.recordsSessionId) || null;
}

function allIncidents() {
    return Array.isArray(state.snapshot.incidents) ? state.snapshot.incidents : [];
}

function snapshotIncidents(snapshot) {
    return Array.isArray(snapshot?.incidents) ? snapshot.incidents : [];
}

function newIncidentsFromSnapshot(snapshot) {
    return snapshotIncidents(snapshot).filter((incident) => {
        const incidentId = String(incident.incident_id || "");
        return incidentId && !state.knownIncidentIds.has(incidentId);
    });
}

function rememberIncidentIds(snapshot) {
    for (const incident of snapshotIncidents(snapshot)) {
        const incidentId = String(incident.incident_id || "");
        if (incidentId) state.knownIncidentIds.add(incidentId);
    }
}

function sessionIncidents() {
    const session = currentSession();
    return session ? allIncidents().filter((incident) => incident.session_id === session.session_id) : [];
}

function workspaceIncidents() {
    const session = workspaceSession();
    return session ? allIncidents().filter((incident) => incident.session_id === session.session_id) : [];
}

function reviewMeta(value) {
    const normalized = String(value ?? "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
    return REVIEW_META[normalized] || REVIEW_META.unverified;
}

function seatNumbers(incident) {
    return (Array.isArray(incident?.student_numbers) ? incident.student_numbers : []).map((value) => String(value).padStart(2, "0"));
}

function seatSummary(incident) {
    const seats = seatNumbers(incident);
    return seats.length ? seats.join(", ") : "--";
}

function incidentAlertMessage(incident) {
    const typeLabel = incident.type_label || incident.behavior_type || "Incident";
    const seats = seatNumbers(incident);
    if (String(incident.behavior_type || "").toLowerCase() === "noise") {
        return `ALERT: ${typeLabel}`;
    }
    if (seats.length === 1) {
        return `ALERT: Student #${seats[0]} - ${typeLabel}`;
    }
    if (seats.length > 1) {
        return `ALERT: Students #${seats.join(", #")} - ${typeLabel}`;
    }
    return `ALERT: ${typeLabel}`;
}

function incidentsInCurrentWorkspace(incidents) {
    const session = workspaceSession();
    return session ? incidents.filter((incident) => incident.session_id === session.session_id) : incidents;
}

function workspaceFlaggedSeats() {
    const set = new Set();
    for (const incident of workspaceIncidents()) {
        for (const seat of Array.isArray(incident.student_numbers) ? incident.student_numbers : []) set.add(Number(seat));
    }
    return set;
}

function parseIso(value) {
    const date = new Date(String(value || ""));
    return Number.isNaN(date.getTime()) ? null : date;
}

function parseSessionDateTime(session, key) {
    const datePart = String(session?.session_date || "");
    const timePart = String(session?.[key] || "");
    if (!datePart || !timePart) return null;
    const [year, month, day] = datePart.split("-").map(Number);
    const [hour, minute] = timePart.split(":").map(Number);
    const date = new Date(year, (month || 1) - 1, day || 1, hour || 0, minute || 0, 0, 0);
    return Number.isNaN(date.getTime()) ? null : date;
}

function formatTime(date) {
    return date instanceof Date && !Number.isNaN(date.getTime()) ? date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "--";
}

function formatDate(date) {
    return date instanceof Date && !Number.isNaN(date.getTime()) ? date.toLocaleDateString([], { year: "numeric", month: "short", day: "2-digit" }) : "--";
}

function formatDuration(ms) {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const hours = String(Math.floor(totalSeconds / 3600)).padStart(2, "0");
    const minutes = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, "0");
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    return `${hours}:${minutes}:${seconds}`;
}

function sessionStatusClass(status) {
    const normalized = String(status || "").toLowerCase();
    if (normalized === "running") return "is-active";
    if (normalized === "degraded") return "is-degraded";
    if (normalized === "cleared") return "is-stopped";
    if (normalized === "stopped") return "is-stopped";
    if (normalized === "error") return "is-error";
    return "is-created";
}

function sessionStatusLabel(status) {
    return String(status || "No session").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function sessionElapsedText(session) {
    const started = parseIso(session?.started_at);
    const stopped = parseIso(session?.stopped_at);
    if (!started) return "00:00:00";
    return formatDuration((stopped || new Date()).getTime() - started.getTime());
}

function toneClass(label) {
    const value = String(label || "").toLowerCase();
    if (value.includes("noise")) return "is-warning";
    if (value.includes("phone") || value.includes("object")) return "is-danger";
    if (value.includes("head") || value.includes("movement")) return "is-warning";
    if (value.includes("verified")) return "is-success";
    return "is-generic";
}

function formatDbValue(value) {
    return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} dB` : "--";
}

function frontNodeSound() {
    const nodes = Array.isArray(state.snapshot.nodes) ? state.snapshot.nodes : [];
    const frontNode = nodes.find((node) => String(node.profile || "").trim() === "front")
        || nodes.find((node) => node.extra?.sound)
        || null;
    return frontNode?.extra?.sound || null;
}

function nodeStreamInfo(node) {
    return node?.extra?.stream || {};
}

function nodeHasStreamFrame(node, mode = "annotated") {
    const stream = nodeStreamInfo(node);
    const flagName = mode === "raw" ? "has_raw_frame" : "has_annotated_frame";
    const seqName = mode === "raw" ? "raw_seq" : "annotated_seq";
    if (Object.prototype.hasOwnProperty.call(stream, flagName)) return Boolean(stream[flagName]);
    return Number(stream[seqName] || 0) > 0;
}

function nodeHasAnyStreamFrame(node) {
    return nodeHasStreamFrame(node, "raw") || nodeHasStreamFrame(node, "annotated");
}

function streamEmptyMessage(node) {
    if (!node.online) return "Node offline or stream unavailable.";
    const errorText = String(node.last_error || "").trim();
    if (errorText) return `Node online, but the detector reports: ${errorText}`;
    if (!currentSession()) return "Node online. Create and start a shared session to open the live feed.";
    const stateLabel = sessionStatusLabel(node.state || "unknown");
    if (!["running", "starting"].includes(String(node.state || "").toLowerCase())) {
        return `Node online, detector state: ${stateLabel}.`;
    }
    return "Waiting for the first video frame from this node.";
}

function renderNoise() {
    const sound = frontNodeSound();
    const currentDb = Number(sound?.current_db);
    const thresholdDb = Number(sound?.threshold_db);
    const hasReading = Number.isFinite(currentDb);
    const hasThreshold = Number.isFinite(thresholdDb) && thresholdDb > 0;
    const overThreshold = Boolean(sound?.over_threshold);
    const fillPercent = hasReading && hasThreshold
        ? Math.min(100, Math.max(0, (currentDb / thresholdDb) * 100))
        : 0;

    els.noiseStatusPill.textContent = overThreshold ? "Exceeded" : "Normal";
    els.noiseStatusPill.className = `noise-meter-status ${overThreshold ? "is-exceeded" : "is-normal"}`;
    els.noiseDbValue.innerHTML = hasReading
        ? `${Math.round(currentDb)}<span>dB</span>`
        : "--";
    els.noisePanel.classList.toggle("is-exceeded", overThreshold);
    els.noisePanel.classList.toggle("is-muted", !hasReading);
    els.noiseMeterFill.style.width = `${fillPercent}%`;
    els.noiseMeterTrack.setAttribute("aria-valuenow", String(Math.round(fillPercent)));
    els.noiseMeterTrack.setAttribute(
        "aria-valuetext",
        hasReading && hasThreshold
            ? `${formatDbValue(currentDb)} of ${formatDbValue(thresholdDb)} threshold`
            : "No noise reading available"
    );
    els.noiseBanner.hidden = !sound?.over_threshold;
    els.noiseBanner.textContent = sound?.over_threshold
        ? `Front-node noise alert: ${formatDbValue(sound.current_db)} exceeds ${formatDbValue(sound.threshold_db)}.`
        : "";
}

function showBanner(message, isError = false) {
    if (!message) {
        els.banner.hidden = true;
        els.banner.textContent = "";
        els.banner.className = "banner banner-info dashboard-banner";
        return;
    }
    els.banner.hidden = false;
    els.banner.textContent = message;
    els.banner.className = isError ? "banner banner-danger dashboard-banner" : "banner banner-success dashboard-banner";
}

function sessionScheduleHasAnyValue() {
    return ["session_date", "start_time", "end_time"].some((name) => Boolean(els.sessionForm.elements[name]?.value));
}

function sessionScheduleIsComplete(values) {
    return ["session_date", "start_time", "end_time"].every((name) => Boolean(values?.[name]));
}

function setFormValues(values) {
    for (const [key, value] of Object.entries(values || {})) {
        if (els.sessionForm.elements[key]) els.sessionForm.elements[key].value = value || "";
    }
}

function formatDateInput(date) {
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function formatTimeInput(date) {
    return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function hydrateSessionForm() {
    const session = currentSession();
    if (session && !state.sessionHydrated) {
        setFormValues(session);
        state.sessionHydrated = true;
        state.sessionDefaultsApplied = sessionScheduleIsComplete(session);
    }
}

function applySessionDefaults() {
    if (state.sessionDefaultsApplied || state.sessionFormDirty) return;
    const start = new Date();
    start.setSeconds(0, 0);
    const end = new Date(start.getTime() + 2 * 60 * 60 * 1000);
    const nextValues = {};
    if (!els.sessionForm.elements.session_date?.value) nextValues.session_date = formatDateInput(start);
    if (!els.sessionForm.elements.start_time?.value) nextValues.start_time = formatTimeInput(start);
    if (!els.sessionForm.elements.end_time?.value) nextValues.end_time = formatTimeInput(end);
    if (!Object.keys(nextValues).length) {
        state.sessionDefaultsApplied = true;
        return;
    }
    setFormValues(nextValues);
    state.sessionDefaultsApplied = true;
}

function syncSessionAccordionState() {
    const isOpen = Boolean(els.sessionAccordion?.open);
    els.sessionPanel.classList.toggle("is-open", isOpen);
    els.sessionToggle?.setAttribute("aria-expanded", String(isOpen));
    if (isOpen) applySessionDefaults();
}

function collectSessionPayload() {
    return Object.fromEntries(new FormData(els.sessionForm).entries());
}

function sessionFormValue(name) {
    return String(els.sessionForm.elements[name]?.value || "").trim();
}

function canCreateSession() {
    return Boolean(sessionFormValue("subject_code") && sessionFormValue("professor"));
}

function validateSessionForm() {
    const subjectInput = els.sessionForm.elements.subject_code;
    const professorInput = els.sessionForm.elements.professor;
    if (!sessionFormValue("subject_code")) {
        return { field: subjectInput, message: "Enter the subject code before creating a session." };
    }
    if (!sessionFormValue("professor")) {
        return { field: professorInput, message: "Enter the professor name before creating a session." };
    }
    return null;
}

function resetSessionForm() {
    setFormValues({
        subject_code: "",
        professor: "",
        session_date: "",
        start_time: "",
        end_time: "",
        notes: "",
    });
}

function resetActiveSessionState() {
    state.snapshot.active_session = null;
    state.sessionHydrated = false;
    state.sessionFormDirty = false;
    state.sessionDefaultsApplied = false;
    resetSessionForm();
}

function currentSessionId() {
    return currentSession()?.session_id || "";
}

function flaggedSeats() {
    const set = new Set();
    for (const incident of sessionIncidents()) {
        for (const seat of Array.isArray(incident.student_numbers) ? incident.student_numbers : []) set.add(Number(seat));
    }
    return set;
}

function latestIncident(items) {
    return [...items].sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")))[0] || null;
}

function workspaceSelectionRequired() {
    return !currentSession();
}

function workspaceSelectionMissing() {
    return workspaceSelectionRequired() && !workspaceSession();
}

function formatSessionScopeLabel(session) {
    const date = String(session?.session_date || "").trim() || "--";
    const schedule = session?.start_time && session?.end_time ? `${session.start_time} - ${session.end_time}` : "No schedule";
    return `${date} | ${schedule} | ${sessionStatusLabel(session?.status || "created")} | ${session?.session_id || "Pending"}`;
}

function workspaceSessionLabel(session) {
    const subject = subjectCodeLabel(normalizedSubjectCode(session?.subject_code));
    const date = String(session?.session_date || "").trim() || "--";
    return `${subject} on ${date}`;
}

function renderRecordsScope() {
    if (currentSession()) {
        els.recordsScopePicker.hidden = true;
        els.recordsSession.disabled = true;
        els.recordsFilter.disabled = false;
        els.recordsSearch.disabled = false;
        els.recordsClear.disabled = false;
        els.recordsExport.disabled = false;
        return;
    }

    const subjectOptions = [];
    const seenSubjects = new Set();
    for (const session of sessionsHistory()) {
        const value = normalizedSubjectCode(session.subject_code);
        if (seenSubjects.has(value)) continue;
        seenSubjects.add(value);
        subjectOptions.push({ value, label: subjectCodeLabel(value) });
    }

    if (state.recordsSubject && !subjectOptions.some((option) => option.value === state.recordsSubject)) {
        state.recordsSubject = "";
        state.recordsSessionId = "";
    }

    const scopedSessions = sessionsHistory().filter((session) => !state.recordsSubject || normalizedSubjectCode(session.subject_code) === state.recordsSubject);
    if (state.recordsSessionId && !scopedSessions.some((session) => session.session_id === state.recordsSessionId)) {
        state.recordsSessionId = "";
    }

    els.recordsScopePicker.hidden = false;
    els.recordsSubject.innerHTML = [
        '<option value="">Select subject code</option>',
        ...subjectOptions.map((option) => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>`),
    ].join("");
    els.recordsSubject.value = state.recordsSubject || "";
    els.recordsSession.innerHTML = [
        '<option value="">Select session</option>',
        ...scopedSessions.map((session) => `<option value="${escapeHtml(session.session_id)}">${escapeHtml(formatSessionScopeLabel(session))}</option>`),
    ].join("");
    els.recordsSession.value = state.recordsSessionId || "";
    els.recordsSession.disabled = !state.recordsSubject || !scopedSessions.length;

    const workspaceReady = Boolean(workspaceSession());
    els.recordsFilter.disabled = !workspaceReady;
    els.recordsSearch.disabled = !workspaceReady;
    els.recordsClear.disabled = !workspaceReady;
    els.recordsExport.disabled = !workspaceReady;
}

function filteredRecords() {
    return workspaceIncidents().filter((incident) => {
        const status = String(incident.review_status || "unverified").replaceAll("-", "_");
        const searchText = [
            incident.type_label,
            incident.camera_label,
            incident.display_time,
            incident.summary,
            incident.behavior_type,
            seatSummary(incident),
            reviewMeta(incident.review_status).label,
        ].join(" ").toLowerCase();
        return (state.recordsFilter === "all" || status === state.recordsFilter) && (!state.recordsQuery || searchText.includes(state.recordsQuery));
    });
}

function reviewOptions(selected) {
    const value = String(selected || "unverified").trim().toLowerCase().replaceAll("-", "_");
    return Object.entries(REVIEW_META).map(([key, meta]) => `<option value="${key}" ${value === key ? "selected" : ""}>${meta.label}</option>`).join("");
}

function ensureFeedCard(node) {
    let card = els.feedGrid.querySelector(`[data-feed-card="${node.node_id}"]`);
    if (card) return card;
    els.feedGrid.insertAdjacentHTML("beforeend", `
        <article class="feed-card" data-feed-card="${escapeHtml(node.node_id)}">
            <div class="feed-card-head">
                <div>
                    <p class="panel-eyebrow" data-feed-camera></p>
                    <h3 data-feed-name></h3>
                </div>
                <div class="feed-card-tools">
                    <span class="node-pill" data-feed-status></span>
                    <label class="feed-view">
                        <span class="sr-only">Select stream mode</span>
                        <select class="feed-select" data-feed-mode="${escapeHtml(node.node_id)}">
                            <option value="annotated">Annotated</option>
                            <option value="raw">Raw</option>
                        </select>
                    </label>
                </div>
            </div>
            <div class="stream-shell">
                <img class="stream-image hidden" alt="${escapeHtml(node.display_name || node.node_id)} live feed" data-feed-image>
                <div class="stream-empty" data-feed-empty></div>
            </div>
            <div class="feed-footer">
                <span class="meta-chip" data-feed-state></span>
                <span class="meta-chip" data-feed-fps></span>
                <span class="meta-chip" data-feed-backlog></span>
                <span class="meta-chip" data-feed-sound></span>
            </div>
        </article>
    `);
    return els.feedGrid.querySelector(`[data-feed-card="${node.node_id}"]`);
}

function renderFeeds() {
    const nodes = Array.isArray(state.snapshot.nodes) ? state.snapshot.nodes : [];
    const keepIds = new Set(nodes.map((node) => node.node_id));
    for (const node of nodes) {
        const card = ensureFeedCard(node);
        const mode = state.feedModes[node.node_id] || "annotated";
        const streamUrl = node.stream_urls?.[mode] || "";
        const image = card.querySelector("[data-feed-image]");
        const empty = card.querySelector("[data-feed-empty]");
        card.querySelector("[data-feed-camera]").textContent = node.camera_label || node.node_id;
        card.querySelector("[data-feed-name]").textContent = node.display_name || node.node_id;
        card.querySelector("[data-feed-status]").textContent = node.online ? "Online" : "Offline";
        card.querySelector("[data-feed-status]").className = `node-pill ${node.online ? "is-online" : "is-offline"}`;
        card.querySelector(`[data-feed-mode="${node.node_id}"]`).value = mode;
        card.querySelector("[data-feed-state]").textContent = `State: ${sessionStatusLabel(node.state || "unknown")}`;
        card.querySelector("[data-feed-fps]").textContent = `FPS: ${Number(node.fps || 0).toFixed(1)}`;
        card.querySelector("[data-feed-backlog]").textContent = `Backlog: ${Number(node.sync_backlog || 0)}`;
        const sound = node.extra?.sound || null;
        card.querySelector("[data-feed-sound]").textContent = sound?.enabled
            ? `Noise: ${formatDbValue(sound.current_db)} / ${formatDbValue(sound.threshold_db)}`
            : "Noise: disabled";
        const streamReady = node.online && streamUrl && nodeHasStreamFrame(node, mode);
        if (streamReady) {
            if (image.getAttribute("src") !== streamUrl) image.setAttribute("src", streamUrl);
            image.classList.remove("hidden");
            empty.textContent = "";
        } else {
            image.classList.add("hidden");
            image.removeAttribute("src");
            empty.textContent = streamEmptyMessage(node);
        }
    }
    for (const card of Array.from(els.feedGrid.querySelectorAll("[data-feed-card]"))) {
        if (!keepIds.has(card.dataset.feedCard)) card.remove();
    }
}

function renderSessionSummary() {
    const session = currentSession();
    const status = sessionStatusLabel(session?.status || "idle");
    const statusClass = sessionStatusClass(session?.status || "idle");
    const startDate = parseSessionDateTime(session, "start_time") || parseSessionDateTime(session, "end_time");
    els.sessionLabel.textContent = session?.subject_code ? `${session.subject_code} - ${session.professor || "Shared session"}` : "Shared classroom monitoring";
    els.headerElapsedLabel.textContent = sessionElapsedText(session);
    els.statusPill.textContent = session ? status : "Idle";
    els.statusPill.className = `status-pill ${statusClass}`;
    els.sessionStatusPill.textContent = session ? `${status} - ${session.subject_code || "Untitled"}` : "No Session";
    els.sessionStatusPill.className = `status-pill ${statusClass}`;
    els.sessionSummaryCopy.textContent = session ? "The shared session controls both nodes and defines the active analytics scope." : "Create the shared schedule, then start both nodes from one control surface.";
    els.sessionSubjectLabel.textContent = session?.subject_code || "--";
    els.sessionProfessorLabel.textContent = session?.professor || "--";
    els.sessionDateLabel.textContent = startDate ? formatDate(startDate) : "--";
    els.sessionIdLabel.textContent = session?.session_id || "Pending";
    els.createSessionButton.disabled = !!session || !canCreateSession();
    els.clearSessionButton.disabled = !session;
    els.startSessionButton.disabled = !session;
    els.restartSessionButton.disabled = !session;
    els.stopSessionButton.disabled = !session;
}

function renderMetrics() {
    const items = currentSession() ? sessionIncidents() : allIncidents();
    const flags = flaggedSeats();
    const nodes = Array.isArray(state.snapshot.nodes) ? state.snapshot.nodes : [];
    const online = nodes.filter((node) => node.online).length;
    const streaming = nodes.filter((node) => node.online && nodeHasAnyStreamFrame(node)).length;
    els.metricTotalIncidents.textContent = String(items.length);
    els.metricTotalFoot.textContent = currentSession() ? "Synced incidents tied to the active session." : "Showing incident totals across synced records.";
    els.metricFlaggedSeats.textContent = String(flags.size);
    els.metricSeatFoot.textContent = flags.size ? "Seats flagged from active-session incident mappings." : "No mapped flagged seats in the active session.";
    els.metricOnlineNodes.textContent = `${online} / ${nodes.length}`;
    els.metricOnlineFoot.textContent = nodes.length
        ? `${streaming} live feed${streaming === 1 ? "" : "s"} publishing frames; online status uses central-received heartbeats.`
        : "No camera nodes are configured.";
}

function renderRecords() {
    const session = workspaceSession();
    const selectionMissing = workspaceSelectionMissing();
    const items = filteredRecords();
    els.recordsContextLabel.textContent = currentSession()
        ? "Review synced incident evidence from the active session."
        : session
            ? `Review synced incident evidence from ${workspaceSessionLabel(session)}.`
            : "No active session is running. Select a subject code and session before opening stored records.";
    els.recordsBody.innerHTML = items.length ? items.map((incident) => `
        <tr>
            <td>${escapeHtml(incident.display_time || incident.created_at || "--")}</td>
            <td>${escapeHtml(seatSummary(incident))}</td>
            <td><span class="type-pill ${toneClass(incident.type_label)}">${escapeHtml(incident.type_label || incident.behavior_type || "Incident")}</span></td>
            <td>${escapeHtml(incident.camera_label || incident.node_id || "--")}</td>
            <td>${incident.gif_url || incident.poster_url ? `<button class="evidence-button" type="button" data-open-evidence="${escapeHtml(incident.incident_id)}">${escapeHtml(incident.gif_url ? "View GIF" : "View Snapshot")}</button>` : `<span class="evidence-button is-disabled">No media</span>`}</td>
            <td><select class="review-select ${reviewMeta(incident.review_status).className}" data-review-incident="${escapeHtml(incident.incident_id)}">${reviewOptions(incident.review_status)}</select></td>
        </tr>
    `).join("") : `<tr><td class="table-empty" colspan="6">${escapeHtml(selectionMissing ? "Select a subject code and session first to load stored records." : state.recordsQuery || state.recordsFilter !== "all" ? "No records match the current search or review filter." : session ? "No synced incidents are available for the selected session yet." : "No synced incidents are available for this workspace yet.")}</td></tr>`;
}

function buildTypeChart(items) {
    const counts = new Map();
    for (const incident of items) {
        const label = incident.type_label || incident.behavior_type || "Incident";
        counts.set(label, (counts.get(label) || 0) + 1);
    }
    const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
    if (!entries.length) {
        return `<div class="chart-empty"><strong>No incident data yet</strong><span>Type distribution will render once the workspace has synced incidents.</span></div>`;
    }
    const width = 640, height = 260, top = 22, right = 24, bottom = 56, left = 44;
    const chartWidth = width - left - right, chartHeight = height - top - bottom;
    const max = Math.max(...entries.map((entry) => entry[1]), 1), step = chartWidth / entries.length, barWidth = Math.min(64, step * 0.56);
    const grids = Array.from({ length: 5 }, (_, index) => `<line class="chart-gridline" x1="${left}" y1="${top + (chartHeight / 4) * index}" x2="${width - right}" y2="${top + (chartHeight / 4) * index}"></line>`).join("");
    const bars = entries.map(([label, count], index) => {
        const x = left + (step * index) + ((step - barWidth) / 2);
        const barHeight = (count / max) * chartHeight;
        const y = top + chartHeight - barHeight;
        const shortLabel = label.length > 15 ? `${label.slice(0, 12)}...` : label;
        return `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="12" fill="${CHART_COLORS[index % CHART_COLORS.length]}" fill-opacity="0.72"></rect><text class="chart-value" x="${x + (barWidth / 2)}" y="${Math.max(top + 14, y - 8)}" text-anchor="middle">${count}</text><text class="chart-label" x="${x + (barWidth / 2)}" y="${height - 18}" text-anchor="middle">${escapeHtml(shortLabel)}</text>`;
    }).join("");
    return `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Incident type distribution chart">${grids}<line class="chart-axis" x1="${left}" y1="${top + chartHeight}" x2="${width - right}" y2="${top + chartHeight}"></line>${bars}</svg>`;
}

function buildTimelineChart(items) {
    if (!items.length) {
        return `<div class="chart-empty"><strong>No incident data yet</strong><span>Timeline analytics will render once this workspace has synced incidents.</span></div>`;
    }
    const session = workspaceSession();
    const start = parseSessionDateTime(session, "start_time");
    const end = parseSessionDateTime(session, "end_time");
    let labels = [];
    let counts = [];
    if (start && end && end > start) {
        const bucketCount = 8;
        const duration = end.getTime() - start.getTime();
        counts = Array(bucketCount).fill(0);
        labels = Array.from({ length: bucketCount }, (_, index) => formatTime(new Date(start.getTime() + (duration * index / (bucketCount - 1)))));
        for (const incident of items) {
            const stamp = parseIso(incident.created_at);
            if (!stamp) continue;
            const ratio = Math.min(0.9999, Math.max(0, (stamp.getTime() - start.getTime()) / duration));
            counts[Math.floor(ratio * bucketCount)] += 1;
        }
    } else {
        const fallback = [...items].sort((a, b) => String(a.created_at || "").localeCompare(String(b.created_at || ""))).slice(-8);
        labels = fallback.map((incident) => incident.display_time || "--");
        counts = fallback.map((_item, index) => index + 1);
    }
    const width = 640, height = 260, top = 18, right = 18, bottom = 50, left = 34;
    const chartWidth = width - left - right, chartHeight = height - top - bottom;
    const max = Math.max(...counts, 1);
    const points = counts.map((count, index) => ({
        x: left + (chartWidth / Math.max(counts.length - 1, 1)) * index,
        y: top + chartHeight - (count / max) * chartHeight,
        label: labels[index],
    }));
    const linePath = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
    const areaPath = `${linePath} L ${points.at(-1).x} ${top + chartHeight} L ${points[0].x} ${top + chartHeight} Z`;
    const grids = Array.from({ length: 4 }, (_, index) => `<line class="chart-gridline" x1="${left}" y1="${top + (chartHeight / 3) * index}" x2="${width - right}" y2="${top + (chartHeight / 3) * index}"></line>`).join("");
    const dots = points.map((point) => `<circle cx="${point.x}" cy="${point.y}" r="5" fill="#3f83ff"></circle><circle cx="${point.x}" cy="${point.y}" r="3" fill="#edf3ff"></circle>`).join("");
    const labelsSvg = points.map((point) => `<text class="chart-label" x="${point.x}" y="${height - 18}" text-anchor="middle">${escapeHtml(point.label)}</text>`).join("");
    return `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Incident timeline chart">${grids}<line class="chart-axis" x1="${left}" y1="${top + chartHeight}" x2="${width - right}" y2="${top + chartHeight}"></line><path d="${areaPath}" fill="rgba(63, 131, 255, 0.12)"></path><path d="${linePath}" fill="none" stroke="#3f83ff" stroke-width="3"></path>${dots}${labelsSvg}</svg>`;
}

function renderAnalytics() {
    const session = workspaceSession();
    const selectionMissing = workspaceSelectionMissing();
    els.analyticsTypesNote.textContent = currentSession()
        ? "Incident counts grouped by detected type for the active session."
        : session
            ? `Incident counts grouped by detected type for ${workspaceSessionLabel(session)}.`
            : "No active session is running. Select a subject code and session to load analytics.";
    els.analyticsTimelineNote.textContent = currentSession()
        ? "Synced incidents plotted against the current session schedule."
        : session
            ? `Synced incidents plotted against the stored schedule for ${workspaceSessionLabel(session)}.`
            : "Select a subject code and session to plot stored incident timing.";
    els.typeChart.innerHTML = selectionMissing
        ? `<div class="chart-empty"><strong>Select a session first</strong><span>Choose a subject code and stored session to view analytics.</span></div>`
        : buildTypeChart(workspaceIncidents());
    els.timelineChart.innerHTML = selectionMissing
        ? `<div class="chart-empty"><strong>Select a session first</strong><span>Choose a subject code and stored session to view the incident timeline.</span></div>`
        : buildTimelineChart(workspaceIncidents());
}

function renderSeatMap() {
    const flags = workspaceFlaggedSeats();
    const session = workspaceSession();
    els.seatmapContextLabel.textContent = currentSession()
        ? (flags.size ? "Flagged seats come from active-session incidents with mapped seat numbers." : "Seats remain normal until incidents carry mapped seat numbers.")
        : session
            ? (flags.size ? `Flagged seats are sourced from ${workspaceSessionLabel(session)}.` : "No seat mappings were recorded for the selected session.")
            : "No active session is running. Select a subject code and session to inspect stored seat mappings.";
    els.seatmapGrid.innerHTML = SEAT_LAYOUT.map((row) => `
        <div class="seatmap-row">
            ${row.map((seat) => `<article class="seat-tile ${flags.has(seat) ? "is-flagged" : "is-normal"}"><span class="seat-icon" aria-hidden="true"><svg viewBox="0 0 20 20" focusable="false"><path d="M6 9.5a2 2 0 1 1 2-2 2 2 0 0 1-2 2zm8 0a2 2 0 1 1 2-2 2 2 0 0 1-2 2zM4 12h12a2 2 0 0 1 2 2v1H2v-1a2 2 0 0 1 2-2zm1-1V7h10v4"></path></svg></span><span class="seat-number">${String(seat).padStart(2, "0")}</span></article>`).join("")}
        </div>
    `).join("");
}

function renderHistory() {
    const rows = Array.isArray(state.snapshot.sessions_history) ? state.snapshot.sessions_history : [];
    els.historyBody.innerHTML = rows.length ? rows.map((session) => `
        <tr>
            <td>${escapeHtml(session.subject_code || "--")}</td>
            <td>${escapeHtml(session.professor || "--")}</td>
            <td>${escapeHtml(session.session_date || "--")}</td>
            <td>${escapeHtml(session.start_time || "--")} - ${escapeHtml(session.end_time || "--")}</td>
            <td><span class="history-badge ${Number(session.incident_count || 0) > 0 ? "has-incidents" : "is-zero"}">${Number(session.incident_count || 0)}</span></td>
            <td><span class="status-pill ${sessionStatusClass(session.status || "created")}">${escapeHtml(sessionStatusLabel(session.status || "created"))}</span></td>
            <td class="history-actions">
                <button class="ghost-button ghost-button-tight history-action-button history-action-button-danger" type="button" data-delete-session="${escapeHtml(session.session_id)}">Delete Session</button>
                ${session.subject_code ? `<button class="ghost-button ghost-button-tight history-action-button" type="button" data-delete-subject="${escapeHtml(session.subject_code)}">Delete Subject</button>` : ""}
            </td>
        </tr>
    `).join("") : `<tr><td class="table-empty" colspan="7">No stored shared sessions yet.</td></tr>`;
}

function renderSystem() {
    const nodes = Array.isArray(state.snapshot.nodes) ? state.snapshot.nodes : [];
    els.systemGrid.innerHTML = nodes.length ? nodes.map((node) => {
        const seen = parseIso(node.last_seen_at);
        const errorText = String(node.last_error || "").trim();
        const sound = node.extra?.sound || null;
        return `<article class="system-card"><div class="system-card-head"><div><p class="panel-eyebrow">${escapeHtml(node.camera_label || node.node_id)}</p><h3>${escapeHtml(node.display_name || node.node_id)}</h3></div><span class="node-pill ${node.online ? "is-online" : "is-offline"}">${node.online ? "Online" : "Offline"}</span></div><div class="system-card-meta"><div class="system-card-meta-item"><span class="system-meta-label">Runtime State</span><strong>${escapeHtml(sessionStatusLabel(node.state || "unknown"))}</strong></div><div class="system-card-meta-item"><span class="system-meta-label">Last Seen</span><strong>${escapeHtml(seen ? `${formatDate(seen)} ${formatTime(seen)}` : "--")}</strong></div><div class="system-card-meta-item"><span class="system-meta-label">Processing FPS</span><strong>${Number(node.fps || 0).toFixed(1)}</strong></div><div class="system-card-meta-item"><span class="system-meta-label">Sync Backlog</span><strong>${Number(node.sync_backlog || 0)}</strong></div><div class="system-card-meta-item"><span class="system-meta-label">Sound Level</span><strong>${sound?.enabled ? formatDbValue(sound.current_db) : "Disabled"}</strong></div><div class="system-card-meta-item"><span class="system-meta-label">Sound Threshold</span><strong>${sound?.enabled ? formatDbValue(sound.threshold_db) : "--"}</strong></div></div><div class="system-card-error ${errorText ? "has-error" : ""}">${escapeHtml(errorText || sound?.last_error || "No node error reported in the latest heartbeat.")}</div></article>`;
    }).join("") : `<article class="system-card"><h3>Waiting for node registration</h3><p class="panel-copy">System details will appear once the front and mid nodes register with the central service.</p></article>`;
}

function findIncidentById(incidentId) {
    return allIncidents().find((incident) => incident.incident_id === incidentId) || null;
}

function openViewer(incidentId) {
    const incident = findIncidentById(incidentId);
    const evidenceUrl = incident?.gif_url || incident?.poster_url || "";
    if (!incident || !evidenceUrl) return;
    els.evidenceViewerImage.src = evidenceUrl;
    els.evidenceViewerTitle.textContent = incident.type_label || "Incident evidence";
    els.evidenceViewerMeta.textContent = `Camera: ${incident.camera_label || "--"} | Seats: ${seatSummary(incident)} | Time: ${incident.display_time || "--"}`;
    els.evidenceViewerOpen.href = evidenceUrl;
    els.evidenceViewer.classList.remove("hidden");
}

function closeViewer() {
    els.evidenceViewer.classList.add("hidden");
    els.evidenceViewerImage.removeAttribute("src");
    els.evidenceViewerOpen.href = "#";
    els.evidenceViewerMeta.textContent = "";
}

function setActiveSection(section) {
    document.querySelectorAll("[data-section]").forEach((button) => button.classList.toggle("is-active", button.dataset.section === section));
    document.querySelectorAll(".dashboard-section").forEach((panel) => panel.classList.toggle("is-active", panel.id === `section-${section}`));
}

function setActiveWorkspaceTab(tab) {
    document.querySelectorAll("[data-workspace-tab]").forEach((button) => button.classList.toggle("is-active", button.dataset.workspaceTab === tab));
    document.querySelectorAll(".workspace-tabpanel").forEach((panel) => panel.classList.toggle("is-active", panel.id === `workspace-${tab}`));
}

function render() {
    hydrateSessionForm();
    syncSessionAccordionState();
    renderSessionSummary();
    renderMetrics();
    renderNoise();
    renderFeeds();
    renderRecordsScope();
    renderRecords();
    renderAnalytics();
    renderSeatMap();
    renderHistory();
    renderSystem();
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || payload.message || `Request failed (${response.status})`);
    return payload;
}

async function refresh() {
    if (state.pollInFlight) return;
    state.pollInFlight = true;
    try {
        const nextSnapshot = await fetchJson("/api/v1/dashboard");
        const newIncidents = newIncidentsFromSnapshot(nextSnapshot);
        state.snapshot = nextSnapshot;
        rememberIncidentIds(nextSnapshot);
        render();
        const visibleNewIncidents = incidentsInCurrentWorkspace(newIncidents);
        if (visibleNewIncidents.length) {
            showBanner(incidentAlertMessage(latestIncident(visibleNewIncidents)), true);
        }
    } finally {
        state.pollInFlight = false;
    }
}

async function createSession() {
    const validation = validateSessionForm();
    if (validation) {
        validation.field?.focus();
        showBanner(validation.message, true);
        return;
    }
    const result = await fetchJson("/api/v1/sessions", { method: "POST", body: JSON.stringify(collectSessionPayload()) });
    state.snapshot.active_session = result.session;
    state.recordsSubject = "";
    state.recordsSessionId = "";
    state.sessionHydrated = true;
    state.sessionDefaultsApplied = sessionScheduleIsComplete(result.session);
    state.sessionFormDirty = false;
    showBanner(`Session ${result.session.session_id} created.`);
    await refresh();
}

async function clearCurrentSession() {
    const result = await fetchJson("/api/v1/sessions/current/clear", { method: "POST", body: JSON.stringify({}) });
    const clearedId = result.session?.session_id || currentSessionId();
    resetActiveSessionState();
    showBanner(clearedId ? `Session ${clearedId} cleared.` : "Current session cleared.");
    render();
    await refresh();
}

async function sessionAction(action) {
    const sessionId = currentSessionId();
    if (!sessionId) {
        showBanner("Create a session first.", true);
        return;
    }
    const result = await fetchJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}/${action}`, { method: "POST", body: JSON.stringify({}) });
    const failures = (result.results || []).filter((item) => !item.ok);
    showBanner(failures.length ? `${sessionStatusLabel(action)} completed with ${failures.length} node issue(s).` : `${sessionStatusLabel(action)} command sent to both nodes.`, failures.length > 0);
    await refresh();
}

async function updateReviewStatus(incidentId, reviewStatus) {
    await fetchJson(`/api/v1/incidents/${encodeURIComponent(incidentId)}/review`, { method: "POST", body: JSON.stringify({ review_status: reviewStatus }) });
    await refresh();
}

async function clearRecords() {
    const session = workspaceSession();
    if (!session) {
        showBanner("Select a subject code and session first before clearing records.", true);
        return;
    }
    const scopeLabel = workspaceSessionLabel(session);
    if (!window.confirm(`Clear synced records for ${scopeLabel}? This removes saved incidents and evidence for that session.`)) return;
    const result = await fetchJson("/api/v1/incidents/clear", {
        method: "POST",
        body: JSON.stringify({ session_id: session.session_id }),
    });
    const clearedCount = Number(result.cleared_incidents || 0);
    showBanner(clearedCount ? `Cleared ${clearedCount} record(s) for ${scopeLabel}.` : `No synced records were stored for ${scopeLabel}.`);
    await refresh();
}

async function deleteSession(sessionId) {
    const targetId = String(sessionId || "").trim();
    if (!targetId) return;
    if (!window.confirm(`Delete stored session ${targetId}? This removes its session record, synced incidents, and saved evidence.`)) return;
    const result = await fetchJson(`/api/v1/sessions/${encodeURIComponent(targetId)}`, { method: "DELETE" });
    if (currentSessionId() === targetId) resetActiveSessionState();
    showBanner(`Deleted session ${targetId}. Removed ${Number(result.cleared_incidents || 0)} incident record(s).`);
    await refresh();
}

async function deleteSubject(subjectCode) {
    const targetSubject = String(subjectCode || "").trim();
    if (!targetSubject) {
        showBanner("Subject code is required before deleting stored sessions.", true);
        return;
    }
    if (!window.confirm(`Delete all stored sessions under ${targetSubject}? This removes the subject entry, all of its sessions, synced incidents, and saved evidence.`)) return;
    const current = currentSession();
    const result = await fetchJson("/api/v1/sessions/subjects/delete", {
        method: "POST",
        body: JSON.stringify({ subject_code: targetSubject }),
    });
    if (current && String(current.subject_code || "").trim() === targetSubject) resetActiveSessionState();
    showBanner(`Deleted ${Number(result.deleted_sessions || 0)} session(s) under ${targetSubject}.`);
    await refresh();
}

function exportRecords() {
    if (workspaceSelectionMissing()) {
        showBanner("Select a subject code and session first before exporting records.", true);
        return;
    }
    const items = filteredRecords();
    if (!items.length) {
        showBanner("There are no records to export for the current filter.", true);
        return;
    }
    const rows = [["timestamp", "seat_numbers", "type", "camera", "review_status", "evidence_url"], ...items.map((incident) => [
        incident.display_time || incident.created_at || "",
        seatSummary(incident),
        incident.type_label || incident.behavior_type || "Incident",
        incident.camera_label || incident.node_id || "",
        reviewMeta(incident.review_status).label,
        incident.gif_url || incident.poster_url || "",
    ])];
    const blob = new Blob([rows.map((row) => row.map(csvEscape).join(",")).join("\n")], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `sentinel-central-records-${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}.csv`;
    document.body.append(link);
    link.click();
    URL.revokeObjectURL(link.href);
    link.remove();
}

els.sessionForm.addEventListener("input", () => {
    state.sessionFormDirty = true;
    renderSessionSummary();
});

els.sessionAccordion.addEventListener("toggle", syncSessionAccordionState);

els.createSessionButton.addEventListener("click", () => createSession().catch((error) => showBanner(error.message, true)));
els.clearSessionButton.addEventListener("click", () => clearCurrentSession().catch((error) => showBanner(error.message, true)));
els.startSessionButton.addEventListener("click", () => sessionAction("start").catch((error) => showBanner(error.message, true)));
els.restartSessionButton.addEventListener("click", () => sessionAction("restart").catch((error) => showBanner(error.message, true)));
els.stopSessionButton.addEventListener("click", () => sessionAction("stop").catch((error) => showBanner(error.message, true)));
els.recordsFilter.addEventListener("change", (event) => {
    state.recordsFilter = event.target.value;
    renderRecords();
});
els.recordsSearch.addEventListener("input", (event) => {
    state.recordsQuery = event.target.value.trim().toLowerCase();
    renderRecords();
});
els.recordsSubject.addEventListener("change", (event) => {
    state.recordsSubject = event.target.value;
    state.recordsSessionId = "";
    renderRecordsScope();
    renderRecords();
    renderAnalytics();
    renderSeatMap();
});
els.recordsSession.addEventListener("change", (event) => {
    state.recordsSessionId = event.target.value;
    renderRecordsScope();
    renderRecords();
    renderAnalytics();
    renderSeatMap();
});
els.recordsClear.addEventListener("click", () => clearRecords().catch((error) => showBanner(error.message, true)));
els.recordsExport.addEventListener("click", exportRecords);

document.addEventListener("click", (event) => {
    const sectionButton = event.target.closest("[data-section]");
    if (sectionButton) return setActiveSection(sectionButton.dataset.section);
    const tabButton = event.target.closest("[data-workspace-tab]");
    if (tabButton) return setActiveWorkspaceTab(tabButton.dataset.workspaceTab);
    const deleteSessionButton = event.target.closest("[data-delete-session]");
    if (deleteSessionButton) return deleteSession(deleteSessionButton.dataset.deleteSession).catch((error) => showBanner(error.message, true));
    const deleteSubjectButton = event.target.closest("[data-delete-subject]");
    if (deleteSubjectButton) return deleteSubject(deleteSubjectButton.dataset.deleteSubject).catch((error) => showBanner(error.message, true));
    const evidenceButton = event.target.closest("[data-open-evidence]");
    if (evidenceButton) return openViewer(evidenceButton.dataset.openEvidence);
    if (event.target.closest("[data-close-viewer]")) closeViewer();
});

document.addEventListener("change", (event) => {
    const feedSelect = event.target.closest("[data-feed-mode]");
    if (feedSelect) {
        state.feedModes[feedSelect.dataset.feedMode] = feedSelect.value;
        renderFeeds();
        return;
    }
    const reviewSelect = event.target.closest("[data-review-incident]");
    if (reviewSelect) updateReviewStatus(reviewSelect.dataset.reviewIncident, reviewSelect.value).catch((error) => showBanner(error.message, true));
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !els.evidenceViewer.classList.contains("hidden")) closeViewer();
});

render();
window.setInterval(() => {
    els.headerElapsedLabel.textContent = sessionElapsedText(currentSession());
}, CLOCK_MS);
window.setInterval(() => refresh().catch((error) => showBanner(error.message, true)), POLL_MS);
