const bootstrapNode = document.getElementById("dashboard-bootstrap");
const initialState = bootstrapNode ? JSON.parse(bootstrapNode.textContent) : {};

const state = {
    snapshot: initialState,
    modalIncidentId: null,
    dismissedIncidentIds: new Set(),
};

const selectors = {
    sessionLabel: document.getElementById("session-label"),
    scheduleLabel: document.getElementById("schedule-label"),
    statusPill: document.getElementById("status-pill"),
    statusBanner: document.getElementById("status-banner"),
    sourceLabel: document.getElementById("source-label"),
    elapsedLabel: document.getElementById("elapsed-label"),
    updateLabel: document.getElementById("update-label"),
    feedPill: document.getElementById("feed-pill"),
    streamImage: document.getElementById("live-stream"),
    streamPlaceholder: document.getElementById("stream-placeholder"),
    liveIncidents: document.getElementById("live-incidents"),
    historyGrid: document.getElementById("history-grid"),
    systemGrid: document.getElementById("system-grid"),
    metricTotalIncidents: document.getElementById("metric-total-incidents"),
    metricLastType: document.getElementById("metric-last-type"),
    metricLastTime: document.getElementById("metric-last-time"),
    metricTracked: document.getElementById("metric-tracked"),
    metricFps: document.getElementById("metric-fps"),
    metricInference: document.getElementById("metric-inference"),
    modal: document.getElementById("incident-modal"),
    modalPreviewImage: document.getElementById("modal-preview-image"),
    modalCopy: document.getElementById("modal-copy"),
    modalViewLink: document.getElementById("modal-view-link"),
};

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;")
        .replaceAll("'", "&#39;");
}

function sectionNav() {
    const buttons = document.querySelectorAll("[data-section]");
    const sections = document.querySelectorAll(".page-section");

    buttons.forEach((button) => {
        button.addEventListener("click", () => {
            const target = button.dataset.section;
            buttons.forEach((item) => item.classList.toggle("is-active", item === button));
            sections.forEach((section) => {
                section.classList.toggle("is-active", section.id === `section-${target}`);
            });
        });
    });
}

function systemClass(snapshot) {
    if (snapshot.status === "error") {
        return "is-error";
    }
    if (snapshot.system_state === "alert") {
        return "is-alert";
    }
    if (snapshot.status === "starting" || snapshot.status === "manual_setup") {
        return "is-starting";
    }
    if (snapshot.monitoring_active) {
        return "is-active";
    }
    return "";
}

function incidentActionLink(incident) {
    return incident.gif_url || incident.poster_url || incident.manifest_url || "#";
}

function renderStatus(snapshot) {
    const statusText = (snapshot.status || "idle").replaceAll("_", " ");
    selectors.statusPill.textContent = statusText.toUpperCase();
    selectors.feedPill.textContent = snapshot.monitoring_active
        ? (snapshot.runtime_mode === "video" ? "PLAYBACK" : "LIVE")
        : statusText.toUpperCase();
    selectors.statusBanner.textContent = snapshot.current_error || snapshot.status_message || "Waiting for runtime data.";
    selectors.statusBanner.className = `status-banner ${systemClass(snapshot)}`.trim();
}

function renderSummary(snapshot) {
    const session = snapshot.session_details || {};
    const metrics = snapshot.metrics || {};

    selectors.sessionLabel.textContent = session.session_label || "Live monitoring session";
    selectors.scheduleLabel.textContent = session.schedule_label || "No schedule set";
    selectors.sourceLabel.textContent = `Source: ${snapshot.source_label || "--"}`;
    selectors.elapsedLabel.textContent = `Elapsed: ${metrics.elapsed_text || "00:00:00"}`;
    selectors.updateLabel.textContent = `Last update: ${snapshot.last_update_iso || "--"}`;

    selectors.metricTotalIncidents.textContent = metrics.total_incidents ?? 0;
    selectors.metricLastType.textContent = metrics.last_incident_type || "No incidents yet";
    selectors.metricLastTime.textContent = metrics.last_incident_time || "No alert received.";
    selectors.metricTracked.textContent = `${metrics.tracked_students ?? 0} / ${metrics.assigned_students ?? 0}`;
    selectors.metricFps.textContent = Number(metrics.processing_fps || 0).toFixed(1);
    selectors.metricInference.textContent = `Inference ${Number(metrics.inference_ms || 0).toFixed(0)} ms`;

    const hasLiveFeed = snapshot.monitoring_active || snapshot.status === "completed";
    selectors.streamImage.classList.toggle("is-hidden", !hasLiveFeed);
    selectors.streamPlaceholder.textContent = hasLiveFeed
        ? "Live feed connected."
        : (snapshot.status_message || "Monitoring has not started yet.");
    selectors.streamPlaceholder.style.display = hasLiveFeed ? "none" : "grid";
}

function incidentMeta(incident) {
    const seatLabel = incident.student_numbers?.length
        ? `Seat ${incident.student_numbers.map((value) => String(value).padStart(2, "0")).join(", ")}`
        : "Seat --";
    const confidence = incident.confidence_pct ? `${incident.confidence_pct}%` : "Heuristic";
    return `${seatLabel} | ${incident.camera_label || "--"} | ${incident.display_time || "--"} | ${confidence}`;
}

function renderIncidentList(snapshot) {
    const items = snapshot.recent_incidents || [];
    if (!items.length) {
        selectors.liveIncidents.innerHTML = `
            <div class="incident-card">
                <div class="incident-card-inner">
                    <p class="incident-title">No live incidents yet</p>
                    <p class="stat-foot">Alerts will appear here as soon as cheating-related behavior is detected.</p>
                </div>
            </div>
        `;
        return;
    }

    selectors.liveIncidents.innerHTML = items.map((incident) => `
        <article class="incident-card">
            <div class="incident-card-inner">
                <div class="incident-title-row">
                    <h3 class="incident-title">${escapeHtml(incident.type_label)}</h3>
                    <span class="status-pill">${escapeHtml((incident.status || "alert").toUpperCase())}</span>
                </div>
                <p class="stat-foot">${escapeHtml(incident.summary)}</p>
                <div class="incident-meta">${escapeHtml(incidentMeta(incident))}</div>
                <div class="incident-actions">
                    <a class="ghost-button" href="${incidentActionLink(incident)}" target="_blank" rel="noreferrer">Open Evidence</a>
                </div>
            </div>
        </article>
    `).join("");
}

function renderHistory(snapshot) {
    const items = snapshot.saved_incidents || [];
    if (!items.length) {
        selectors.historyGrid.innerHTML = `
            <div class="history-card">
                <div class="history-card-inner">
                    <h3 class="history-title">No saved evidence yet</h3>
                    <p class="stat-foot">Generated GIF records will appear here after incidents complete their evidence burst.</p>
                </div>
            </div>
        `;
        return;
    }

    selectors.historyGrid.innerHTML = items.map((incident) => `
        <article class="history-card">
            <div class="history-thumb">
                ${incident.poster_url
                    ? `<img src="${incident.poster_url}" alt="${escapeHtml(incident.type_label)}">`
                    : ""}
            </div>
            <div class="history-card-inner">
                <div class="history-title-row">
                    <h3 class="history-title">${escapeHtml(incident.type_label)}</h3>
                    <span class="status-pill">${incident.frame_count || 0} frames</span>
                </div>
                <p class="stat-foot">${escapeHtml(incident.summary)}</p>
                <div class="history-meta">${escapeHtml(incidentMeta(incident))}</div>
                <div class="incident-actions">
                    <a class="primary-button" href="${incidentActionLink(incident)}" target="_blank" rel="noreferrer">View GIF</a>
                    ${incident.manifest_url
                        ? `<a class="ghost-button" href="${incident.manifest_url}" target="_blank" rel="noreferrer">Manifest</a>`
                        : ""}
                </div>
            </div>
        </article>
    `).join("");
}

function renderSystem(snapshot) {
    const metrics = snapshot.metrics || {};
    const entries = [
        ["Runtime Mode", snapshot.runtime_mode || "--"],
        ["Source Label", snapshot.source_label || "--"],
        ["Config Path", snapshot.config_path || "--"],
        ["Evidence Root", snapshot.evidence_root || "--"],
        ["Setup Profile", snapshot.setup_profile_path || "Not set"],
        ["Status", snapshot.status || "--"],
        ["Frames Processed", metrics.frame_idx ?? 0],
        ["Tracked Students", metrics.tracked_students ?? 0],
        ["Source FPS", Number(metrics.source_fps || 0).toFixed(1)],
        ["Object Detections", metrics.object_detections ?? 0],
    ];

    selectors.systemGrid.innerHTML = entries.map(([label, value]) => `
        <div>
            <dt>${escapeHtml(label)}</dt>
            <dd>${escapeHtml(value)}</dd>
        </div>
    `).join("");
}

function closeModal() {
    selectors.modal.classList.add("hidden");
    state.modalIncidentId = null;
}

async function dismissPopup(incidentId) {
    if (incidentId) {
        state.dismissedIncidentIds.add(incidentId);
    }
    closeModal();

    try {
        await fetch("/api/popup/dismiss", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ incident_id: incidentId }),
        });
    } catch (error) {
        console.error(error);
    }
}

function showPopup(incident) {
    if (!incident || !incident.id || state.dismissedIncidentIds.has(incident.id) || state.modalIncidentId === incident.id) {
        return;
    }

    const preview = incident.gif_url || incident.poster_url;
    selectors.modalPreviewImage.src = preview || "";
    selectors.modalPreviewImage.style.display = preview ? "block" : "none";
    selectors.modalPreviewImage.alt = incident.type_label || "Incident evidence";
    selectors.modalCopy.innerHTML = `
        <strong>${escapeHtml(incident.type_label || "Incident")}</strong>
        <span>${escapeHtml(incident.summary || "")}</span>
        <span>${escapeHtml(incidentMeta(incident))}</span>
    `;
    selectors.modalViewLink.href = incidentActionLink(incident);
    selectors.modal.classList.remove("hidden");
    state.modalIncidentId = incident.id;
}

async function poll() {
    try {
        const response = await fetch("/api/dashboard", { cache: "no-store" });
        if (!response.ok) {
            return;
        }
        state.snapshot = await response.json();
        render(state.snapshot);
    } catch (error) {
        console.error(error);
    }
}

function render(snapshot) {
    renderStatus(snapshot);
    renderSummary(snapshot);
    renderIncidentList(snapshot);
    renderHistory(snapshot);
    renderSystem(snapshot);
    showPopup(snapshot.popup_incident);
}

function bindModal() {
    document.querySelectorAll("[data-dismiss-popup]").forEach((node) => {
        node.addEventListener("click", () => dismissPopup(state.modalIncidentId));
    });
}

sectionNav();
bindModal();
render(state.snapshot);
setInterval(poll, 1500);
