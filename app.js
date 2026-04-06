/* ============================================================
   FINBENCH — Frontend Logic
   ============================================================ */

const ui = {
  apiBase:        document.getElementById("apiBase"),
  healthBtn:      document.getElementById("healthBtn"),
  healthOut:      document.getElementById("healthOut"),
  taskSelect:     document.getElementById("taskSelect"),
  seedInput:      document.getElementById("seedInput"),
  resetBtn:       document.getElementById("resetBtn"),
  stateBtn:       document.getElementById("stateBtn"),
  closeBtn:       document.getElementById("closeBtn"),
  sessionId:      document.getElementById("sessionId"),
  stepBtn:        document.getElementById("stepBtn"),
  observationOut: document.getElementById("observationOut"),
  resultOut:      document.getElementById("resultOut"),
  logList:        document.getElementById("logList"),

  loanForm:       document.getElementById("loanForm"),
  fraudForm:      document.getElementById("fraudForm"),
  portfolioForm:  document.getElementById("portfolioForm"),

  loanDecision:   document.getElementById("loanDecision"),
  loanReasoning:  document.getElementById("loanReasoning"),
  loanRisk:       document.getElementById("loanRisk"),
  loanRate:       document.getElementById("loanRate"),

  fraudFlag:      document.getElementById("fraudFlag"),
  fraudConfidence:document.getElementById("fraudConfidence"),
  fraudHold:      document.getElementById("fraudHold"),
  fraudReason:    document.getElementById("fraudReason"),
  fraudNotes:     document.getElementById("fraudNotes"),

  tradeRows:      document.getElementById("tradeRows"),
  addTradeBtn:    document.getElementById("addTradeBtn"),
  deferRebalancing: document.getElementById("deferRebalancing"),
  riskComment:    document.getElementById("riskComment"),
  tradeTemplate:  document.getElementById("tradeTemplate"),

  // Status mapping
  statusBadge:    document.getElementById("statusBadge"),
  statusTxt:      document.getElementById("statusTxt"),
  sessionIdDisplay: document.getElementById("sessionIdDisplay"),
};

let currentTask = "loan_underwriting";

/* ── Status indicator ── */
function setStatus(online) {
  if (ui.statusBadge && ui.statusTxt) {
    ui.statusBadge.className = `status-badge ${online ? "online" : "offline"}`;
    ui.statusTxt.textContent = online ? "Online" : "Offline";
  }
}

/* ── Log helpers ── */
function addLog(message, type = "info") {
  const li = document.createElement("li");
  const stamp = new Date().toLocaleTimeString();
  li.textContent = `[${stamp}] ${message}`;
  li.className = `log-${type}`;
  ui.logList.prepend(li);
}

function pretty(target, data) {
  target.textContent = JSON.stringify(data, null, 2);
}

function parseErrorBody(rawText) {
  try {
    return JSON.parse(rawText);
  } catch {
    return { error: rawText || "unknown_error" };
  }
}

function normalizedBaseUrl() {
  return ui.apiBase.value.trim().replace(/\/$/, "");
}

async function apiRequest(path, method = "GET", body = null) {
  const config = { method, headers: {} };
  if (body !== null) {
    config.headers["Content-Type"] = "application/json";
    config.body = JSON.stringify(body);
  }

  const response = await fetch(`${normalizedBaseUrl()}${path}`, config);
  if (!response.ok) {
    const rawText = await response.text();
    const payload = parseErrorBody(rawText);
    throw new Error(`${response.status} ${JSON.stringify(payload)}`);
  }

  const text = await response.text();
  return text ? JSON.parse(text) : {};
}

/* ── Task form switcher ── */
function setVisibleForm(task) {
  ui.loanForm.classList.add("hidden");
  ui.fraudForm.classList.add("hidden");
  ui.portfolioForm.classList.add("hidden");

  if (task === "loan_underwriting")    ui.loanForm.classList.remove("hidden");
  if (task === "fraud_detection")      ui.fraudForm.classList.remove("hidden");
  if (task === "portfolio_rebalancing") ui.portfolioForm.classList.remove("hidden");
}

/* ── Action builders ── */
function buildLoanAction() {
  const action = {
    decision:  ui.loanDecision.value,
    reasoning: ui.loanReasoning.value.trim(),
    risk_tier: ui.loanRisk.value,
  };
  if (action.decision === "approve") {
    action.interest_rate_suggestion = Number(ui.loanRate.value);
  }
  return action;
}

function buildFraudAction() {
  const flag = ui.fraudFlag.value === "true";
  return {
    flag,
    confidence:  Number(ui.fraudConfidence.value),
    hold:        ui.fraudHold.value === "true",
    reason_code: flag ? ui.fraudReason.value : "none",
    notes:       ui.fraudNotes.value.trim(),
  };
}

function buildPortfolioAction() {
  const rows = [...ui.tradeRows.querySelectorAll(".trade-row")];
  const trades = rows.map((row) => {
    const direction = row.querySelector(".direction").value;
    let amount = Number(row.querySelector(".amount").value);
    if (direction === "hold") amount = 0;
    return {
      asset_id:  row.querySelector(".asset").value.trim(),
      direction,
      amount_usd: amount,
      rationale: row.querySelector(".rationale").value.trim(),
    };
  });
  return {
    trades,
    defer_rebalancing: ui.deferRebalancing.value === "true",
    risk_comment: ui.riskComment.value.trim(),
  };
}

function buildActionForTask(task) {
  if (task === "loan_underwriting")    return buildLoanAction();
  if (task === "fraud_detection")      return buildFraudAction();
  return buildPortfolioAction();
}

/* ── Trade row helpers ── */
function attachTradeRowEvents(row) {
  const removeBtn = row.querySelector(".remove");
  const direction = row.querySelector(".direction");
  const amount    = row.querySelector(".amount");

  removeBtn.addEventListener("click", () => {
    row.remove();
    if (ui.tradeRows.children.length === 0) addTradeRow();
  });

  direction.addEventListener("change", () => {
    if (direction.value === "hold") {
      amount.value = 0;
      amount.setAttribute("readonly", "readonly");
    } else {
      amount.removeAttribute("readonly");
      if (Number(amount.value) <= 0) amount.value = 1;
    }
  });
}

function addTradeRow(defaults = {}) {
  const node = ui.tradeTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".asset").value     = defaults.asset_id  || "AST01";
  node.querySelector(".direction").value = defaults.direction || "buy";
  node.querySelector(".amount").value    = defaults.amount_usd ?? 1800;
  node.querySelector(".rationale").value = defaults.rationale || "Move toward target allocation.";
  ui.tradeRows.appendChild(node);
  attachTradeRowEvents(node);
}

/* ── API actions ── */
async function checkHealth() {
  try {
    const result = await apiRequest("/health");
    pretty(ui.healthOut, result);
    setStatus(true);
    addLog("Health check succeeded.", "ok");
  } catch (error) {
    ui.healthOut.textContent = error.message;
    setStatus(false);
    addLog(`Health check failed: ${error.message}`, "err");
  }
}

async function resetSession() {
  const payload = {
    task: ui.taskSelect.value,
    seed: Number(ui.seedInput.value),
  };

  try {
    const result = await apiRequest("/reset", "POST", payload);
    ui.sessionId.value = result.session_id || "";
    if (ui.sessionIdDisplay) ui.sessionIdDisplay.textContent = result.session_id || "--";
    currentTask = result.task || payload.task;
    ui.taskSelect.value = currentTask;
    setVisibleForm(currentTask);
    pretty(ui.observationOut, result.observation || result);
    pretty(ui.resultOut, result);
    addLog(`Session reset → task=${currentTask} seed=${payload.seed}`, "ok");
  } catch (error) {
    addLog(`Reset failed: ${error.message}`, "err");
  }
}

async function submitStep() {
  const session_id = ui.sessionId.value.trim();
  if (!session_id) {
    addLog("No session ID — run Reset first.", "err");
    return;
  }

  const action = buildActionForTask(currentTask);

  try {
    const result = await apiRequest("/step", "POST", { session_id, action });
    pretty(ui.resultOut, result);
    pretty(ui.observationOut, result.observation || {});
    const reward = result.reward != null ? result.reward.toFixed(3) : "n/a";
    addLog(`Step accepted — done=${Boolean(result.done)} reward=${reward}`, "ok");
  } catch (error) {
    addLog(`Step failed: ${error.message}`, "err");
  }
}

async function fetchState() {
  const session_id = ui.sessionId.value.trim();
  if (!session_id) {
    addLog("No session ID — run Reset first.", "err");
    return;
  }

  try {
    const state = await apiRequest(`/state/${session_id}`);
    pretty(ui.resultOut, state);
    addLog("State fetched.", "info");
  } catch (error) {
    addLog(`State fetch failed: ${error.message}`, "err");
  }
}

async function closeSession() {
  const session_id = ui.sessionId.value.trim();
  if (!session_id) {
    addLog("No session ID to close.", "err");
    return;
  }

  try {
    const result = await apiRequest(`/close/${session_id}`, "POST");
    pretty(ui.resultOut, result);
    ui.sessionId.value = "";
    if (ui.sessionIdDisplay) ui.sessionIdDisplay.textContent = "--";
    addLog("Session closed.", "info");
  } catch (error) {
    addLog(`Close failed: ${error.message}`, "err");
  }
}

/* ── Event wiring ── */
ui.taskSelect.addEventListener("change", (e) => {
  currentTask = e.target.value;
  setVisibleForm(currentTask);
});

ui.loanDecision.addEventListener("change", () => {
  const isApprove = ui.loanDecision.value === "approve";
  ui.loanRate.toggleAttribute("readonly", !isApprove);
  if (!isApprove) ui.loanRate.value = "";
  if (isApprove && !ui.loanRate.value) ui.loanRate.value = "0.07";
});

ui.fraudFlag.addEventListener("change", () => {
  if (ui.fraudFlag.value !== "true") ui.fraudReason.value = "none";
});

ui.healthBtn.addEventListener("click", checkHealth);
ui.resetBtn.addEventListener("click", resetSession);
ui.stepBtn.addEventListener("click", submitStep);
ui.stateBtn.addEventListener("click", fetchState);
ui.closeBtn.addEventListener("click", closeSession);
ui.addTradeBtn.addEventListener("click", () => addTradeRow());

/* ── Init ── */
addTradeRow({
  asset_id: "AST01", direction: "sell", amount_usd: 1800,
  rationale: "Trim overweight asset toward target allocation.",
});
addTradeRow({
  asset_id: "AST07", direction: "buy", amount_usd: 1800,
  rationale: "Increase underweight asset toward target allocation.",
});

setVisibleForm(currentTask);
addLog("FinBench console ready.", "info");
checkHealth();
