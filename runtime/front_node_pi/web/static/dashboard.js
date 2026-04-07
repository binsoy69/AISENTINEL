const bootstrapNode = document.getElementById("dashboard-bootstrap");
const initialState = bootstrapNode ? JSON.parse(bootstrapNode.textContent) : {};

const ALERT_DISMISS_MS = 2000;
const POLL_INTERVAL_MS = 1500;

const REVIEW_STATUS_META = {
    unverified: { label: "Unverified", className: "is-unverified" },
    verified: { label: "Verified", className: "is-verified" },
    false_detection: { label: "False Detection", className: "is-false" },
};

const TYPE_TONE_META = {
    head: "is-head",
    passing: "is-passing",
    hands: "is-hands",
    object: "is-object",
};

const state = {
    snapshot: initialState,
    activeAlertId: null,
    dismissedIncidentIds: new Set(),
    evidenceViewerId: null,
    alertTimer: 0,
    alertHideTimer: 0,
    pollInFlight: false,
    recordsFilter: "all",
    recordsQuery: "",
    renderCache: {
        liveIncidents: "",
        records: "",
        history: "",
    },
};

const selectors = {
    sessionLabel: document.getElementById("session-label"),
    scheduleLabel: document.getElementById("schedule-label"),
    headerElapsedLabel: document.getElementById("header-elapsed-label"),
    statusPill: document.getElementById("status-pill"),
    statusBanner: document.getElementById("status-banner"),
    sourceLabel: document.getElementById("source-label"),
    elapsedLabel: document.getElementById("elapsed-label"),
    updateLabel: document.getElementById("update-label"),
    feedPill: document.getElementById("feed-pill"),
    streamImage: document.getElementById("live-stream"),
    streamPlaceholder: document.getElementById("stream-placeholder"),
    liveIncidents: document.getElementById("live-incidents"),
    recordsTableBody: document.getElementById("records-table-body"),
    recordsStatusFilter: document.getElementById("records-status-filter"),
    recordsSearch: document.getElementById("records-search"),
    recordsExport: document.getElementById("records-export"),
    historyList: document.getElementById("history-list"),
    systemGrid: document.getElementById("system-grid"),
    metricTotalIncidents: document.getElementById("metric-total-incidents"),
    metricLastType: document.getElementById("metric-last-type"),
    metricLastTime: document.getElementById("metric-last-time"),
    metricTracked: document.getElementById("metric-tracked"),
    metricFps: document.getElementById("metric-fps"),
    metricInference: document.getElementById("metric-inference"),
    alertToast: document.getElementById("alert-toast"),
    alertToastSeat: document.getElementById("alert-toast-seat"),
    alertToastType: document.getElementById("alert-toast-type"),
    alertToastCamera: document.getElementById("alert-toast-camera"),
    alertToastTime: document.getElementById("alert-toast-time"),
    alertToastClose: document.getElementById("alert-toast-close"),
    alertToastProgressFill: document.getElementById("alert-toast-progress-fill"),
    evidenceViewer: document.getElementById("evidence-viewer"),
    evidenceViewerImage: document.getElementById("evidence-viewer-image"),
    evidenceViewerMeta: document.getElementById("evidence-viewer-meta"),
    evidenceViewerTitle: document.getElementById("evidence-viewer-title"),
    evidenceViewerOpen: document.getElementById("evidence-viewer-open"),
};

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;")
        .replaceAll("'", "&#39;");
}

function csvEscape(value) {
    const normalized = String(value ?? "");
    return `"${normalized.replaceAll("\"", "\"\"")}"`;
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

function normalizeReviewStatus(value) {
    const normalized = String(value ?? "")
        .trim()
        .toLowerCase()
        .replaceAll("-", "_")
        .replaceAll(" ", "_");
    return REVIEW_STATUS_META[normalized] ? normalized : "unverified";
}

function reviewMeta(incident) {
    return REVIEW_STATUS_META[normalizeReviewStatus(incident.review_status)];
}

function typeToneClass(incident) {
    return TYPE_TONE_META[incident.behavior_type] || "is-object";
}

function paddedSeatNumbers(incident) {
    const numbers = Array.isArray(incident.student_numbers) ? incident.student_numbers : [];
    return numbers.map((value) => String(value).padStart(2, "0"));
}

function seatSummary(incident) {
    const seats = paddedSeatNumbers(incident);
    return seats.length ? seats.join(", ") : "--";
}

function seatMetaLabel(incident) {
    const seats = paddedSeatNumbers(incident);
    return seats.length ? `Seat ${seats.join(", ")}` : "Seat --";
}

function incidentMeta(incident) {
    return `${seatMetaLabel(incident)} | ${incident.camera_label || "--"} | ${incident.display_time || "--"}`;
}

function statusLabel(snapshot) {
    if (snapshot.system_state === "alert") {
        return "Alert";
    }
    if (snapshot.monitoring_active) {
        return "Active";
    }
    return String(snapshot.status || "idle")
        .replaceAll("_", " ")
        .replace(/\b\w/g, (match) => match.toUpperCase());
}

function incidentViewerUrl(incident) {
    return incident.gif_url || incident.poster_url || "";
}

function incidentOpenLabel(incident) {
    if (incident.gif_url) {
        return "View GIF";
    }
    if (incident.poster_url) {
        return "View Snapshot";
    }
    return "Evidence Pending";
}

function incidentSearchText(incident) {
    return [
        incident.type_label,
        incident.camera_label,
        incident.display_time,
        incident.summary,
        seatSummary(incident),
        reviewMeta(incident).label,
    ].join(" ").toLowerCase();
}

function findIncidentById(incidentId) {
    const collections = [
        state.snapshot.saved_incidents || [],
        state.snapshot.recent_incidents || [],
    ];
    for (const collection of collections) {
        const incident = collection.find((item) => item.id === incidentId);
        if (incident) {
            return incident;
        }
    }
    return null;
}

function historyIncidents(snapshot) {
    const merged = [];
    const seen = new Set();
    for (const collection of [snapshot.saved_incidents || [], snapshot.recent_incidents || []]) {
        for (const incident of collection) {
            if (!incident || !incident.id || seen.has(incident.id)) {
                continue;
            }
            seen.add(incident.id);
            merged.push(incident);
        }
    }
    return merged
        .sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || "")))
        .slice(0, 12);
}

function renderStatus(snapshot) {
    const tone = systemClass(snapshot) || "is-idle";
    selectors.statusPill.textContent = statusLabel(snapshot);
    selectors.statusPill.className = `status-pill status-pill-state ${tone}`.trim();
    selectors.feedPill.textContent = snapshot.monitoring_active
        ? (snapshot.runtime_mode === "video" ? "PLAYBACK" : "LIVE")
        : statusLabel(snapshot).toUpperCase();
    selectors.feedPill.className = `status-pill status-pill-quiet ${tone}`.trim();
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
    selectors.headerElapsedLabel.textContent = metrics.elapsed_text || "00:00:00";
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

function renderIncidentList(snapshot) {
    const items = snapshot.recent_incidents || [];
    const signature = items.map((incident) => [
        incident.id,
        incident.status,
        incident.summary,
        incident.display_time,
        incident.gif_url || "",
        incident.poster_url || "",
    ].join(":")).join("|");

    if (state.renderCache.liveIncidents === signature) {
        return;
    }
    state.renderCache.liveIncidents = signature;

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

    selectors.liveIncidents.innerHTML = items.map((incident) => {
        const evidenceUrl = incidentViewerUrl(incident);
        const hasEvidence = Boolean(evidenceUrl);
        return `
            <article class="incident-card">
                <div class="incident-card-inner">
                    <div class="incident-title-row">
                        <h3 class="incident-title">${escapeHtml(incident.type_label)}</h3>
                        <span class="status-pill">${escapeHtml((incident.status || "alert").toUpperCase())}</span>
                    </div>
                    <p class="stat-foot">${escapeHtml(incident.summary)}</p>
                    <div class="incident-meta">${escapeHtml(incidentMeta(incident))}</div>
                    <div class="incident-actions">
                        ${hasEvidence
                            ? `<button class="ghost-button records-view-button" type="button" data-open-evidence="${escapeHtml(incident.id)}">${escapeHtml(incidentOpenLabel(incident))}</button>`
                            : `<span class="ghost-button is-disabled">${escapeHtml(incidentOpenLabel(incident))}</span>`}
                    </div>
                </div>
            </article>
        `;
    }).join("");
}

function renderHistory(snapshot) {
    const items = historyIncidents(snapshot);
    const signature = items.map((incident) => [
        incident.id,
        incident.created_at || "",
        incident.summary || "",
        incident.display_time || "",
        incident.gif_url || "",
        incident.poster_url || "",
        normalizeReviewStatus(incident.review_status),
    ].join(":")).join("|");

    if (state.renderCache.history === signature) {
        return;
    }
    state.renderCache.history = signature;

    if (!items.length) {
        selectors.historyList.innerHTML = `
            <div class="history-card">
                <div class="history-card-inner">
                    <p class="history-title">No incident history yet</p>
                    <p class="history-summary">Saved evidence and live alerts will appear here once monitoring starts.</p>
                </div>
            </div>
        `;
        return;
    }

    selectors.historyList.innerHTML = items.map((incident) => {
        const evidenceUrl = incidentViewerUrl(incident);
        const hasEvidence = Boolean(evidenceUrl);
        const statusMeta = reviewMeta(incident);
        return `
            <article class="history-card">
                <div class="history-card-inner">
                    <div class="history-title-row">
                        <h3 class="history-title">${escapeHtml(incident.type_label || "Incident")}</h3>
                        <span class="status-pill">${escapeHtml(statusMeta.label)}</span>
                    </div>
                    <p class="history-summary">${escapeHtml(incident.summary || "No incident summary available.")}</p>
                    <div class="history-meta">${escapeHtml(incidentMeta(incident))}</div>
                    <div class="history-actions">
                        ${hasEvidence
                            ? `<button class="ghost-button records-view-button" type="button" data-open-evidence="${escapeHtml(incident.id)}">${escapeHtml(incidentOpenLabel(incident))}</button>`
                            : `<span class="ghost-button is-disabled">Evidence Pending</span>`}
                    </div>
                </div>
            </article>
        `;
    }).join("");
}

function filteredSavedIncidents(snapshot) {
    const items = snapshot.saved_incidents || [];
    const query = state.recordsQuery;
    return items.filter((incident) => {
        const status = normalizeReviewStatus(incident.review_status);
        if (state.recordsFilter !== "all" && status !== state.recordsFilter) {
            return false;
        }
        if (query && !incidentSearchText(incident).includes(query)) {
            return false;
        }
        return true;
    });
}

function reviewStatusOptions(selectedValue) {
    return Object.entries(REVIEW_STATUS_META).map(([value, meta]) => `
        <option value="${value}" ${value === selectedValue ? "selected" : ""}>${meta.label}</option>
    `).join("");
}

function renderRecords(snapshot) {
    const allItems = snapshot.saved_incidents || [];
    const items = filteredSavedIncidents(snapshot);
    const signature = [
        allItems.map((incident) => [
            incident.id,
            normalizeReviewStatus(incident.review_status),
            incident.display_time,
            incident.gif_url || "",
            incident.poster_url || "",
        ].join(":")).join("|"),
        state.recordsFilter,
        state.recordsQuery,
    ].join("::");

    if (state.renderCache.records === signature) {
        return;
    }
    state.renderCache.records = signature;

    if (!items.length) {
        const emptyMessage = allItems.length
            ? "No records match the current search or filter."
            : "No saved evidence yet. Completed incidents will appear here.";
        selectors.recordsTableBody.innerHTML = `
            <tr>
                <td class="records-empty-cell" colspan="6">${escapeHtml(emptyMessage)}</td>
            </tr>
        `;
        return;
    }

    selectors.recordsTableBody.innerHTML = items.map((incident) => {
        const status = normalizeReviewStatus(incident.review_status);
        const statusMeta = reviewMeta(incident);
        const evidenceUrl = incidentViewerUrl(incident);
        const hasEvidence = Boolean(evidenceUrl);

        return `
            <tr>
                <td class="records-time">${escapeHtml(incident.display_time || "--")}</td>
                <td><span class="seat-badge">${escapeHtml(seatSummary(incident))}</span></td>
                <td><span class="type-pill ${typeToneClass(incident)}">${escapeHtml(incident.type_label || "Incident")}</span></td>
                <td>${escapeHtml(incident.camera_label || "--")}</td>
                <td>
                    ${hasEvidence
                        ? `<button class="ghost-button records-view-button" type="button" data-open-evidence="${escapeHtml(incident.id)}">View</button>`
                        : `<span class="ghost-button is-disabled">Pending</span>`}
                </td>
                <td>
                    <label class="review-select-wrap ${statusMeta.className}">
                        <span class="sr-only">Review status</span>
                        <select class="review-select" data-review-status-select data-incident-id="${escapeHtml(incident.id)}">
                            ${reviewStatusOptions(status)}
                        </select>
                    </label>
                </td>
            </tr>
        `;
    }).join("");
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

function hideAlertToast() {
    window.clearTimeout(state.alertHideTimer);
    selectors.alertToast.classList.remove("is-visible");
    selectors.alertToastProgressFill.classList.remove("is-animating");
    state.alertHideTimer = window.setTimeout(() => {
        if (!selectors.alertToast.classList.contains("is-visible")) {
            selectors.alertToast.classList.add("hidden");
        }
    }, 220);
}

function restartAlertProgress() {
    selectors.alertToastProgressFill.classList.remove("is-animating");
    selectors.alertToastProgressFill.style.animationDuration = `${ALERT_DISMISS_MS}ms`;
    void selectors.alertToastProgressFill.offsetWidth;
    selectors.alertToastProgressFill.classList.add("is-animating");
}

async function dismissPopup(incidentId) {
    if (incidentId) {
        state.dismissedIncidentIds.add(incidentId);
    }

    window.clearTimeout(state.alertTimer);
    state.activeAlertId = null;
    hideAlertToast();

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
    if (!incident || !incident.id || state.dismissedIncidentIds.has(incident.id)) {
        return;
    }
    if (state.activeAlertId === incident.id) {
        return;
    }

    window.clearTimeout(state.alertTimer);
    window.clearTimeout(state.alertHideTimer);
    state.activeAlertId = incident.id;

    selectors.alertToastSeat.textContent = seatSummary(incident);
    selectors.alertToastType.textContent = incident.type_label || "--";
    selectors.alertToastCamera.textContent = incident.camera_label || "--";
    selectors.alertToastTime.textContent = incident.display_time || "--";
    selectors.alertToast.classList.remove("hidden");
    restartAlertProgress();
    requestAnimationFrame(() => {
        selectors.alertToast.classList.add("is-visible");
    });

    state.alertTimer = window.setTimeout(() => {
        dismissPopup(incident.id);
    }, ALERT_DISMISS_MS);
}

function closeEvidenceViewer() {
    selectors.evidenceViewer.classList.add("hidden");
    state.evidenceViewerId = null;
}

function openEvidenceViewer(incidentId) {
    const incident = findIncidentById(incidentId);
    if (!incident) {
        return;
    }

    const evidenceUrl = incidentViewerUrl(incident);
    if (!evidenceUrl) {
        return;
    }

    state.evidenceViewerId = incident.id;
    selectors.evidenceViewerTitle.textContent = incident.type_label || "Incident evidence";
    selectors.evidenceViewerImage.src = evidenceUrl;
    selectors.evidenceViewerImage.alt = incident.type_label || "Incident evidence";
    selectors.evidenceViewerMeta.innerHTML = `
        <span>${escapeHtml(seatMetaLabel(incident))}</span>
        <span>${escapeHtml(incident.camera_label || "--")}</span>
        <span>${escapeHtml(incident.display_time || "--")}</span>
        <span>${escapeHtml(reviewMeta(incident).label)}</span>
    `;
    selectors.evidenceViewerOpen.href = evidenceUrl;
    selectors.evidenceViewer.classList.remove("hidden");
}

function replaceIncidentInCollection(collection, updatedIncident) {
    return collection.map((incident) => (
        incident.id === updatedIncident.id
            ? { ...incident, ...updatedIncident }
            : incident
    ));
}

function applyIncidentUpdate(updatedIncident) {
    state.snapshot.saved_incidents = replaceIncidentInCollection(
        state.snapshot.saved_incidents || [],
        updatedIncident,
    );
    state.snapshot.recent_incidents = replaceIncidentInCollection(
        state.snapshot.recent_incidents || [],
        updatedIncident,
    );
    if (state.snapshot.popup_incident && state.snapshot.popup_incident.id === updatedIncident.id) {
        state.snapshot.popup_incident = { ...state.snapshot.popup_incident, ...updatedIncident };
    }
    render(state.snapshot);
}

async function updateEvidenceReviewStatus(incidentId, reviewStatus, selectNode) {
    const previousValue = selectNode.dataset.previousValue || normalizeReviewStatus(selectNode.value);
    selectNode.disabled = true;

    try {
        const response = await fetch("/api/evidence/review", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ incident_id: incidentId, review_status: reviewStatus }),
        });

        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            throw new Error(payload.error || "Could not update evidence review status.");
        }

        applyIncidentUpdate(payload.incident);
    } catch (error) {
        console.error(error);
        selectNode.value = previousValue;
    } finally {
        selectNode.disabled = false;
        selectNode.dataset.previousValue = normalizeReviewStatus(selectNode.value);
    }
}

function exportRecords() {
    const items = filteredSavedIncidents(state.snapshot);
    if (!items.length) {
        return;
    }

    const rows = [
        ["Timestamp", "Seat No.", "Cheating Type", "Camera", "Review Status", "Evidence"],
        ...items.map((incident) => [
            incident.display_time || "--",
            seatSummary(incident),
            incident.type_label || "Incident",
            incident.camera_label || "--",
            reviewMeta(incident).label,
            incidentViewerUrl(incident),
        ]),
    ];

    const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `sentinel-records-${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}.csv`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function poll() {
    if (state.pollInFlight) {
        return;
    }
    state.pollInFlight = true;

    try {
        const response = await fetch("/api/dashboard", { cache: "no-store" });
        if (response.status === 401) {
            window.location.assign("/login");
            return;
        }
        if (response.status === 403) {
            const payload = await response.json().catch(() => ({}));
            if (payload.redirect) {
                window.location.assign(payload.redirect);
            }
            return;
        }
        if (!response.ok) {
            return;
        }
        state.snapshot = await response.json();
        render(state.snapshot);
    } catch (error) {
        console.error(error);
    } finally {
        state.pollInFlight = false;
    }
}

function render(snapshot) {
    renderStatus(snapshot);
    renderSummary(snapshot);
    renderIncidentList(snapshot);
    renderRecords(snapshot);
    renderHistory(snapshot);
    renderSystem(snapshot);
    showPopup(snapshot.popup_incident);
}

function bindControls() {
    selectors.alertToastClose.addEventListener("click", () => dismissPopup(state.activeAlertId));

    selectors.recordsStatusFilter.addEventListener("change", (event) => {
        state.recordsFilter = event.target.value;
        renderRecords(state.snapshot);
    });

    selectors.recordsSearch.addEventListener("input", (event) => {
        state.recordsQuery = event.target.value.trim().toLowerCase();
        renderRecords(state.snapshot);
    });

    selectors.recordsExport.addEventListener("click", exportRecords);

    document.addEventListener("click", (event) => {
        const evidenceButton = event.target.closest("[data-open-evidence]");
        if (evidenceButton) {
            openEvidenceViewer(evidenceButton.dataset.openEvidence);
            return;
        }

        if (event.target.closest("[data-close-viewer]")) {
            closeEvidenceViewer();
        }
    });

    document.addEventListener("change", (event) => {
        const selectNode = event.target.closest("[data-review-status-select]");
        if (!selectNode) {
            return;
        }
        updateEvidenceReviewStatus(
            selectNode.dataset.incidentId,
            normalizeReviewStatus(selectNode.value),
            selectNode,
        );
    });

    document.addEventListener("focusin", (event) => {
        const selectNode = event.target.closest("[data-review-status-select]");
        if (selectNode) {
            selectNode.dataset.previousValue = normalizeReviewStatus(selectNode.value);
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !selectors.evidenceViewer.classList.contains("hidden")) {
            closeEvidenceViewer();
        }
    });
}

sectionNav();
bindControls();
render(state.snapshot);
window.setInterval(poll, POLL_INTERVAL_MS);
