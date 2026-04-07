const bootstrapNode = document.getElementById("dashboard-bootstrap");
const initialState = bootstrapNode ? JSON.parse(bootstrapNode.textContent) : {};
const POLL_MS = 2000;

const state = { snapshot: initialState, feedModes: {} };

const els = {
    sessionStatus: document.getElementById("session-status-pill"),
    sessionBanner: document.getElementById("session-banner"),
    sessionForm: document.getElementById("session-form"),
    createSessionButton: document.getElementById("create-session-button"),
    startSessionButton: document.getElementById("start-session-button"),
    restartSessionButton: document.getElementById("restart-session-button"),
    stopSessionButton: document.getElementById("stop-session-button"),
    nodeStatusGrid: document.getElementById("node-status-grid"),
    feedGrid: document.getElementById("feed-grid"),
    recordsBody: document.getElementById("records-body"),
};

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;")
        .replaceAll("'", "&#39;");
}

function showBanner(message, isError = false) {
    if (!message) {
        els.sessionBanner.hidden = true;
        els.sessionBanner.textContent = "";
        els.sessionBanner.className = "banner";
        return;
    }
    els.sessionBanner.hidden = false;
    els.sessionBanner.textContent = message;
    els.sessionBanner.className = isError ? "banner banner-error" : "banner";
}

function collectSessionPayload() {
    return Object.fromEntries(new FormData(els.sessionForm).entries());
}

function currentSessionId() {
    return state.snapshot.active_session?.session_id || "";
}

function nodeMode(nodeId) {
    return state.feedModes[nodeId] || "annotated";
}

function renderSession() {
    const session = state.snapshot.active_session;
    els.sessionStatus.textContent = session ? `${session.status || "created"} · ${session.subject_code || "Untitled"}` : "No Session";
}

function renderNodeCards() {
    const nodes = state.snapshot.nodes || [];
    els.nodeStatusGrid.innerHTML = nodes.map((node) => `
        <article class="status-card">
            <div class="status-head">
                <div>
                    <p class="eyebrow">${escapeHtml(node.camera_label || node.node_id)}</p>
                    <h2>${escapeHtml(node.display_name || node.node_id)}</h2>
                </div>
                <span class="node-pill ${node.online ? "is-online" : "is-offline"}">${node.online ? "Online" : "Offline"}</span>
            </div>
            <div class="status-metrics">
                <div class="metric"><span class="metric-label">State</span><strong>${escapeHtml(node.state || "unknown")}</strong></div>
                <div class="metric"><span class="metric-label">FPS</span><strong>${Number(node.fps || 0).toFixed(1)}</strong></div>
                <div class="metric"><span class="metric-label">Sync Backlog</span><strong>${escapeHtml(node.sync_backlog || 0)}</strong></div>
                <div class="metric"><span class="metric-label">Incidents</span><strong>${escapeHtml(node.incident_count || 0)}</strong></div>
            </div>
        </article>
    `).join("");
}

function renderFeeds() {
    const nodes = state.snapshot.nodes || [];
    els.feedGrid.innerHTML = nodes.map((node) => {
        const mode = nodeMode(node.node_id);
        const streamUrl = node.stream_urls?.[mode] || "";
        return `
            <article class="feed-card">
                <div class="feed-head">
                    <div>
                        <p class="eyebrow">${escapeHtml(node.camera_label || node.node_id)}</p>
                        <h2>${escapeHtml(node.display_name || node.node_id)}</h2>
                    </div>
                    <div class="feed-toolbar">
                        <label>
                            <span class="metric-label">View</span>
                            <select data-feed-mode="${escapeHtml(node.node_id)}">
                                <option value="annotated" ${mode === "annotated" ? "selected" : ""}>Annotated</option>
                                <option value="raw" ${mode === "raw" ? "selected" : ""}>Raw</option>
                            </select>
                        </label>
                    </div>
                </div>
                <div class="stream-shell">
                    ${node.online
                        ? `<img src="${escapeHtml(streamUrl)}" alt="${escapeHtml(node.display_name || node.node_id)} live feed">`
                        : `<div class="stream-empty">Node offline or not registered yet.</div>`}
                </div>
            </article>
        `;
    }).join("");
}

function reviewOptions(selected) {
    return ["unverified", "verified", "false_detection"].map((value) =>
        `<option value="${value}" ${value === selected ? "selected" : ""}>${value.replaceAll("_", " ")}</option>`
    ).join("");
}

function renderRecords() {
    const incidents = state.snapshot.incidents || [];
    if (!incidents.length) {
        els.recordsBody.innerHTML = `<tr><td colspan="6" class="students">No synced incidents yet.</td></tr>`;
        return;
    }
    els.recordsBody.innerHTML = incidents.map((incident) => `
        <tr>
            <td>${escapeHtml(incident.display_time || incident.created_at || "--")}</td>
            <td>${escapeHtml(incident.camera_label || incident.node_id || "--")}</td>
            <td><span class="type-pill">${escapeHtml(incident.type_label || incident.behavior_type || "Incident")}</span></td>
            <td class="students">${escapeHtml((incident.student_numbers || []).join(", ") || "--")}</td>
            <td>${incident.gif_url || incident.poster_url ? `<a class="evidence-link" href="${escapeHtml(incident.gif_url || incident.poster_url)}" target="_blank" rel="noreferrer">Open</a>` : `<span class="students">Pending</span>`}</td>
            <td><select class="review-select" data-review-incident="${escapeHtml(incident.incident_id)}">${reviewOptions(incident.review_status || "unverified")}</select></td>
        </tr>
    `).join("");
}

function render() {
    renderSession();
    renderNodeCards();
    renderFeeds();
    renderRecords();
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(payload.error || payload.message || `Request failed (${response.status})`);
    }
    return payload;
}

async function refresh() {
    state.snapshot = await fetchJson("/api/v1/dashboard");
    render();
}

async function createSession() {
    const result = await fetchJson("/api/v1/sessions", {
        method: "POST",
        body: JSON.stringify(collectSessionPayload()),
    });
    state.snapshot.active_session = result.session;
    showBanner(`Session ${result.session.session_id} created.`, false);
    render();
}

async function sessionAction(action) {
    const sessionId = currentSessionId();
    if (!sessionId) {
        showBanner("Create a session first.", true);
        return;
    }
    const result = await fetchJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}/${action}`, {
        method: "POST",
        body: JSON.stringify({}),
    });
    state.snapshot.active_session = result.session;
    const failures = (result.results || []).filter((item) => !item.ok);
    showBanner(
        failures.length ? `${action} completed with ${failures.length} node issue(s).` : `${action} command sent to both nodes.`,
        failures.length > 0,
    );
    await refresh();
}

async function updateReviewStatus(incidentId, reviewStatus) {
    await fetchJson(`/api/v1/incidents/${encodeURIComponent(incidentId)}/review`, {
        method: "POST",
        body: JSON.stringify({ review_status: reviewStatus }),
    });
    await refresh();
}

els.createSessionButton.addEventListener("click", () => createSession().catch((error) => showBanner(error.message, true)));
els.startSessionButton.addEventListener("click", () => sessionAction("start").catch((error) => showBanner(error.message, true)));
els.restartSessionButton.addEventListener("click", () => sessionAction("restart").catch((error) => showBanner(error.message, true)));
els.stopSessionButton.addEventListener("click", () => sessionAction("stop").catch((error) => showBanner(error.message, true)));

document.addEventListener("change", (event) => {
    const modeSelect = event.target.closest("[data-feed-mode]");
    if (modeSelect) {
        state.feedModes[modeSelect.dataset.feedMode] = modeSelect.value;
        renderFeeds();
        return;
    }
    const reviewSelect = event.target.closest("[data-review-incident]");
    if (reviewSelect) {
        updateReviewStatus(reviewSelect.dataset.reviewIncident, reviewSelect.value).catch((error) => showBanner(error.message, true));
    }
});

render();
window.setInterval(() => refresh().catch((error) => showBanner(error.message, true)), POLL_MS);
