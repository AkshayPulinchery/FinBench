import { useEffect, useMemo, useState } from "react";

type TaskName = "loan_underwriting" | "fraud_detection" | "portfolio_rebalancing";
type LogType = "info" | "ok" | "err";
type FraudFlagString = "true" | "false";
type TradeDirection = "buy" | "sell" | "hold";

type TradeOrder = {
  asset_id: string;
  direction: TradeDirection;
  amount_usd: number | string;
  rationale: string;
};

type LogItem = {
  message: string;
  type: LogType;
};

type AnyObject = Record<string, unknown>;

const TASKS: TaskName[] = ["loan_underwriting", "fraud_detection", "portfolio_rebalancing"];

const DEFAULT_TRADES: TradeOrder[] = [
  {
    asset_id: "AST01",
    direction: "sell",
    amount_usd: 1800,
    rationale: "Trim overweight asset toward target allocation.",
  },
  {
    asset_id: "AST07",
    direction: "buy",
    amount_usd: 1800,
    rationale: "Increase underweight asset toward target allocation.",
  },
];

function prettyJson(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

function parseErrorBody(rawText: string): AnyObject {
  try {
    return JSON.parse(rawText) as AnyObject;
  } catch {
    return { error: rawText || "unknown_error" };
  }
}

export default function App() {
  const [apiBase, setApiBase] = useState("http://localhost:7860");
  const [isOnline, setIsOnline] = useState(false);

  const [task, setTask] = useState<TaskName>("loan_underwriting");
  const [seed, setSeed] = useState("42");
  const [sessionId, setSessionId] = useState("");

  const [loanDecision, setLoanDecision] = useState("approve");
  const [loanReasoning, setLoanReasoning] = useState("Strong repayment profile.");
  const [loanRisk, setLoanRisk] = useState("low");
  const [loanRate, setLoanRate] = useState("0.07");

  const [fraudFlag, setFraudFlag] = useState<FraudFlagString>("true");
  const [fraudConfidence, setFraudConfidence] = useState("0.91");
  const [fraudHold, setFraudHold] = useState<FraudFlagString>("true");
  const [fraudReason, setFraudReason] = useState("velocity");
  const [fraudNotes, setFraudNotes] = useState("Rapid pattern break from normal spending behavior.");

  const [trades, setTrades] = useState<TradeOrder[]>(DEFAULT_TRADES);
  const [deferRebalancing, setDeferRebalancing] = useState<FraudFlagString>("false");
  const [riskComment, setRiskComment] = useState("Gradual rebalance while preserving transaction budget.");

  const [healthOut, setHealthOut] = useState("");
  const [observationOut, setObservationOut] = useState("");
  const [resultOut, setResultOut] = useState("");
  const [logs, setLogs] = useState<LogItem[]>([]);

  const normalizedBaseUrl = useMemo(() => apiBase.trim().replace(/\/$/, ""), [apiBase]);

  const docsUrl = `${normalizedBaseUrl}/docs`;
  const healthUrl = `${normalizedBaseUrl}/health`;

  function addLog(message: string, type: LogType = "info") {
    const stamp = new Date().toLocaleTimeString();
    setLogs((prev) => [{ message: `[${stamp}] ${message}`, type }, ...prev]);
  }

  async function apiRequest(path: string, method = "GET", body: unknown = null): Promise<AnyObject> {
    const config: RequestInit = { method, headers: {} };
    if (body !== null) {
      config.headers = { "Content-Type": "application/json" };
      config.body = JSON.stringify(body);
    }

    const response = await fetch(`${normalizedBaseUrl}${path}`, config);
    if (!response.ok) {
      const rawText = await response.text();
      const payload = parseErrorBody(rawText);
      throw new Error(`${response.status} ${JSON.stringify(payload)}`);
    }

    const text = await response.text();
    return text ? (JSON.parse(text) as AnyObject) : {};
  }

  function buildLoanAction(): AnyObject {
    const action: AnyObject = {
      decision: loanDecision,
      reasoning: loanReasoning.trim(),
      risk_tier: loanRisk,
    };

    if (loanDecision === "approve") {
      action.interest_rate_suggestion = Number(loanRate);
    }

    return action;
  }

  function buildFraudAction(): AnyObject {
    const flagged = fraudFlag === "true";
    return {
      flag: flagged,
      confidence: Number(fraudConfidence),
      hold: fraudHold === "true",
      reason_code: flagged ? fraudReason : "none",
      notes: fraudNotes.trim(),
    };
  }

  function buildPortfolioAction(): AnyObject {
    return {
      trades: trades.map((trade) => {
        const direction = trade.direction;
        const amount = direction === "hold" ? 0 : Number(trade.amount_usd);
        return {
          asset_id: trade.asset_id.trim(),
          direction,
          amount_usd: amount,
          rationale: trade.rationale.trim(),
        };
      }),
      defer_rebalancing: deferRebalancing === "true",
      risk_comment: riskComment.trim(),
    };
  }

  function buildActionForTask(): AnyObject {
    if (task === "loan_underwriting") return buildLoanAction();
    if (task === "fraud_detection") return buildFraudAction();
    return buildPortfolioAction();
  }

  async function checkHealth() {
    try {
      const result = await apiRequest("/health");
      setHealthOut(prettyJson(result));
      setIsOnline(true);
      addLog("Health check succeeded.", "ok");
    } catch (error) {
      setHealthOut(String((error as Error).message));
      setIsOnline(false);
      addLog(`Health check failed: ${(error as Error).message}`, "err");
    }
  }

  async function resetSession() {
    const payload = {
      task,
      seed: Number(seed),
    };

    try {
      const result = await apiRequest("/reset", "POST", payload);
      const sid = String(result.session_id ?? "");
      setSessionId(sid);
      setTask((result.task as TaskName) || task);
      setObservationOut(prettyJson(result.observation ?? result));
      setResultOut(prettyJson(result));
      addLog(`Session reset -> task=${String(result.task ?? task)} seed=${payload.seed}`, "ok");
    } catch (error) {
      addLog(`Reset failed: ${(error as Error).message}`, "err");
    }
  }

  async function submitStep() {
    if (!sessionId.trim()) {
      addLog("No session ID - run Reset first.", "err");
      return;
    }

    const action = buildActionForTask();

    try {
      const result = await apiRequest("/step", "POST", {
        session_id: sessionId.trim(),
        action,
      });
      setResultOut(prettyJson(result));
      setObservationOut(prettyJson(result.observation ?? {}));
      const rewardRaw = result.reward;
      const reward = typeof rewardRaw === "number" ? rewardRaw.toFixed(3) : "n/a";
      addLog(`Step accepted - done=${Boolean(result.done)} reward=${reward}`, "ok");
    } catch (error) {
      addLog(`Step failed: ${(error as Error).message}`, "err");
    }
  }

  async function fetchState() {
    if (!sessionId.trim()) {
      addLog("No session ID - run Reset first.", "err");
      return;
    }

    try {
      const state = await apiRequest(`/state/${sessionId.trim()}`);
      setResultOut(prettyJson(state));
      addLog("State fetched.", "info");
    } catch (error) {
      addLog(`State fetch failed: ${(error as Error).message}`, "err");
    }
  }

  async function closeSession() {
    if (!sessionId.trim()) {
      addLog("No session ID to close.", "err");
      return;
    }

    try {
      const result = await apiRequest(`/close/${sessionId.trim()}`, "POST");
      setResultOut(prettyJson(result));
      setSessionId("");
      addLog("Session closed.", "info");
    } catch (error) {
      addLog(`Close failed: ${(error as Error).message}`, "err");
    }
  }

  function addTradeRow() {
    setTrades((prev) => [
      ...prev,
      {
        asset_id: "AST01",
        direction: "buy",
        amount_usd: 1800,
        rationale: "Move toward target allocation.",
      },
    ]);
  }

  function updateTrade(index: number, field: keyof TradeOrder, value: string) {
    setTrades((prev) =>
      prev.map((trade, i) => {
        if (i !== index) return trade;
        const updated: TradeOrder = { ...trade, [field]: value } as TradeOrder;

        if (field === "direction" && value === "hold") {
          updated.amount_usd = 0;
        }

        if (field === "direction" && value !== "hold" && Number(updated.amount_usd) <= 0) {
          updated.amount_usd = 1;
        }

        return updated;
      })
    );
  }

  function removeTrade(index: number) {
    setTrades((prev) => {
      const next = prev.filter((_, i) => i !== index);
      return next.length > 0
        ? next
        : [{ asset_id: "AST01", direction: "buy", amount_usd: 1800, rationale: "Move toward target allocation." }];
    });
  }

  useEffect(() => {
    addLog("FinBench console ready.", "info");
    void checkHealth();
  }, []);

  useEffect(() => {
    if (loanDecision !== "approve") {
      setLoanRate("");
    } else if (!loanRate) {
      setLoanRate("0.07");
    }
  }, [loanDecision]);

  useEffect(() => {
    if (fraudFlag !== "true") {
      setFraudReason("none");
    }
  }, [fraudFlag]);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-50 border-b-2 border-zinc-950 bg-white">
        <div className="mx-auto flex h-[60px] max-w-[1440px] items-center gap-8 px-4 md:px-10">
          <a className="flex items-center gap-2 no-underline" href="#">
            <div className="grid h-9 w-9 place-items-center border-2 border-zinc-950 bg-zinc-950 text-base font-extrabold text-yellow-400 shadow-[3px_3px_0_#facc15]">F</div>
            <span className="text-lg font-semibold tracking-tight text-zinc-950">Fin<span className="font-extrabold">Bench</span></span>
          </a>

          <nav className="hidden flex-1 items-center justify-center gap-1 md:flex">
            <a href="#console" className="border-2 border-zinc-950 bg-yellow-400 px-3 py-1 text-sm font-semibold">Console</a>
            <a href={docsUrl} target="_blank" rel="noreferrer" className="border-2 border-transparent px-3 py-1 text-sm font-semibold text-zinc-700 hover:border-zinc-950 hover:bg-yellow-400">API Docs -&gt;</a>
            <a href={healthUrl} target="_blank" rel="noreferrer" className="border-2 border-transparent px-3 py-1 text-sm font-semibold text-zinc-700 hover:border-zinc-950 hover:bg-yellow-400">Health -&gt;</a>
          </nav>

          <span className={`mono ml-auto inline-flex items-center gap-2 border-2 px-3 py-1 text-xs font-bold uppercase tracking-wider ${isOnline ? "border-zinc-950 bg-emerald-100 text-emerald-800" : "border-zinc-950 bg-red-100 text-red-800"}`}>
            <span className="h-2 w-2 rounded-full bg-current"></span>
            {isOnline ? "Online" : "Offline"}
          </span>
        </div>
      </header>

      <section id="console" className="border-b-2 border-zinc-950 bg-white px-4 pb-10 pt-12 md:px-10">
        <div className="mx-auto max-w-[1440px]">
          <div className="mono mb-6 inline-flex items-center gap-2 border-2 border-zinc-950 bg-zinc-100 px-3 py-2 text-xs font-bold uppercase tracking-wider">
            <span className="h-2 w-2 rounded-full border border-zinc-950 bg-emerald-500"></span>
            OpenEnv Fintech . v1.0.0 . seed-deterministic
          </div>
          <h1 className="mb-4 text-4xl font-extrabold leading-tight tracking-tight md:text-6xl">
            AI Fintech
            <br />
            <span className="text-transparent [text-shadow:5px_5px_0_#facc15] [-webkit-text-stroke:2px_#0a0a0a]">Episode Console</span>
          </h1>
          <p className="max-w-4xl text-lg font-medium leading-relaxed text-zinc-600">
            High-fidelity benchmarking for financial agents. Run, step, and analyze episodes for loan underwriting, fraud detection, and portfolio rebalancing.
          </p>
          <div className="mono mt-8 flex flex-wrap gap-8 text-sm text-zinc-600">
            <span>Active Session ID: <strong>{sessionId || "--"}</strong></span>
            <span>Environment: <strong className="bg-zinc-950 px-2 py-1 text-white">PROD-SIM</strong></span>
          </div>
        </div>
      </section>

      <main className="mx-auto grid max-w-[1440px] grid-cols-1 gap-8 px-4 py-10 md:px-10 xl:grid-cols-[360px_1fr_360px]">
        <aside className="flex flex-col gap-6">
          <div className="card">
            <div className="card-header">
              <span className="text-zinc-500">Hex</span>
              <h2 className="text-sm font-extrabold uppercase tracking-widest">Server</h2>
            </div>
            <div className="card-body">
              <label className="label" htmlFor="apiBase">API Base URL</label>
              <input id="apiBase" className="input" type="text" value={apiBase} onChange={(e) => setApiBase(e.target.value)} />
              <button className="btn-yellow" onClick={() => void checkHealth()}>Ping Health</button>
              <pre className="output-box">{healthOut}</pre>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="text-zinc-500">Grid</span>
              <h2 className="text-sm font-extrabold uppercase tracking-widest">Episode Control</h2>
            </div>
            <div className="card-body">
              <label className="label" htmlFor="taskSelect">Task</label>
              <select id="taskSelect" className="input" value={task} onChange={(e) => setTask(e.target.value as TaskName)}>
                {TASKS.map((taskName) => (
                  <option key={taskName} value={taskName}>{taskName}</option>
                ))}
              </select>

              <label className="label" htmlFor="seedInput">Seed</label>
              <input id="seedInput" className="input" type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />

              <button className="btn-yellow w-full" onClick={() => void resetSession()}>Reset / New Session</button>

              <label className="label" htmlFor="sessionId">Session ID</label>
              <input id="sessionId" className="input bg-zinc-100 text-zinc-500" type="text" value={sessionId} readOnly placeholder="created by reset" />

              <div className="grid grid-cols-2 gap-3">
                <button className="btn-outline" onClick={() => void fetchState()}>Get State</button>
                <button className="btn-danger" onClick={() => void closeSession()}>Close</button>
              </div>
            </div>
          </div>
        </aside>

        <section>
          <div className="card">
            <div className="card-header">
              <span className="text-zinc-500">Spark</span>
              <h2 className="text-sm font-extrabold uppercase tracking-widest">Action Composer</h2>
            </div>
            <div className="card-body">
              {task === "loan_underwriting" && (
                <div className="flex flex-col gap-3 border-t-2 border-zinc-200 pt-4">
                  <div className="mono w-fit border-2 border-zinc-950 bg-zinc-950 px-3 py-1 text-xs font-bold uppercase tracking-wider text-white">Loan Underwriting</div>

                  <label className="label" htmlFor="loanDecision">Decision</label>
                  <select id="loanDecision" className="input" value={loanDecision} onChange={(e) => setLoanDecision(e.target.value)}>
                    <option value="approve">approve</option>
                    <option value="reject">reject</option>
                    <option value="request_info">request_info</option>
                  </select>

                  <label className="label" htmlFor="loanReasoning">Reasoning</label>
                  <textarea id="loanReasoning" className="input" rows={3} value={loanReasoning} onChange={(e) => setLoanReasoning(e.target.value)} />

                  <label className="label" htmlFor="loanRisk">Risk Tier</label>
                  <select id="loanRisk" className="input" value={loanRisk} onChange={(e) => setLoanRisk(e.target.value)}>
                    <option value="low">low</option>
                    <option value="medium">medium</option>
                    <option value="high">high</option>
                  </select>

                  <label className="label" htmlFor="loanRate">Interest Rate Suggestion (approve only)</label>
                  <input id="loanRate" className="input" type="number" min="0.03" max="0.35" step="0.01" value={loanRate} onChange={(e) => setLoanRate(e.target.value)} readOnly={loanDecision !== "approve"} />
                </div>
              )}

              {task === "fraud_detection" && (
                <div className="flex flex-col gap-3 border-t-2 border-zinc-200 pt-4">
                  <div className="mono w-fit border-2 border-zinc-950 bg-zinc-950 px-3 py-1 text-xs font-bold uppercase tracking-wider text-white">Fraud Detection</div>

                  <label className="label" htmlFor="fraudFlag">Flag Transaction</label>
                  <select id="fraudFlag" className="input" value={fraudFlag} onChange={(e) => setFraudFlag(e.target.value as FraudFlagString)}>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>

                  <label className="label" htmlFor="fraudConfidence">Confidence (0-1)</label>
                  <input id="fraudConfidence" className="input" type="number" min="0" max="1" step="0.01" value={fraudConfidence} onChange={(e) => setFraudConfidence(e.target.value)} />

                  <label className="label" htmlFor="fraudHold">Hold Card</label>
                  <select id="fraudHold" className="input" value={fraudHold} onChange={(e) => setFraudHold(e.target.value as FraudFlagString)}>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>

                  <label className="label" htmlFor="fraudReason">Reason Code</label>
                  <select id="fraudReason" className="input" value={fraudReason} onChange={(e) => setFraudReason(e.target.value)}>
                    <option value="velocity">velocity</option>
                    <option value="location_anomaly">location_anomaly</option>
                    <option value="amount_anomaly">amount_anomaly</option>
                    <option value="merchant_risk">merchant_risk</option>
                    <option value="pattern_break">pattern_break</option>
                    <option value="none">none</option>
                  </select>

                  <label className="label" htmlFor="fraudNotes">Notes</label>
                  <textarea id="fraudNotes" className="input" rows={3} value={fraudNotes} onChange={(e) => setFraudNotes(e.target.value)} />
                </div>
              )}

              {task === "portfolio_rebalancing" && (
                <div className="flex flex-col gap-3 border-t-2 border-zinc-200 pt-4">
                  <div className="mono w-fit border-2 border-zinc-950 bg-zinc-950 px-3 py-1 text-xs font-bold uppercase tracking-wider text-white">Portfolio Rebalancing</div>

                  <div className="flex flex-col gap-3">
                    {trades.map((trade, index) => (
                      <div className="grid grid-cols-1 gap-2 md:grid-cols-[70px_90px_100px_1fr_52px]" key={`${trade.asset_id}-${index}`}>
                        <input className="input" type="text" placeholder="AST01" value={trade.asset_id} onChange={(e) => updateTrade(index, "asset_id", e.target.value)} />
                        <select className="input" value={trade.direction} onChange={(e) => updateTrade(index, "direction", e.target.value)}>
                          <option value="buy">buy</option>
                          <option value="sell">sell</option>
                          <option value="hold">hold</option>
                        </select>
                        <input className="input" type="number" min="0" step="1" value={trade.amount_usd} readOnly={trade.direction === "hold"} onChange={(e) => updateTrade(index, "amount_usd", e.target.value)} />
                        <input className="input" type="text" value={trade.rationale} onChange={(e) => updateTrade(index, "rationale", e.target.value)} />
                        <button className="btn-danger px-2" type="button" onClick={() => removeTrade(index)}>X</button>
                      </div>
                    ))}
                  </div>

                  <button className="btn-outline" type="button" onClick={addTradeRow}>+ Add Trade Row</button>

                  <label className="label" htmlFor="deferRebalancing">Defer Rebalancing</label>
                  <select id="deferRebalancing" className="input" value={deferRebalancing} onChange={(e) => setDeferRebalancing(e.target.value as FraudFlagString)}>
                    <option value="false">false</option>
                    <option value="true">true</option>
                  </select>

                  <label className="label" htmlFor="riskComment">Risk Comment</label>
                  <textarea id="riskComment" className="input" rows={3} value={riskComment} onChange={(e) => setRiskComment(e.target.value)} />
                </div>
              )}

              <button className="btn-black w-full text-base" onClick={() => void submitStep()}>Submit Step</button>
            </div>
          </div>
        </section>

        <section className="flex flex-col gap-6">
          <div className="card">
            <div className="card-header">
              <span className="text-zinc-500">Obs</span>
              <h2 className="text-sm font-extrabold uppercase tracking-widest">Observation</h2>
            </div>
            <div className="card-body">
              <pre className="output-box">{observationOut}</pre>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="text-zinc-500">Res</span>
              <h2 className="text-sm font-extrabold uppercase tracking-widest">Step Result</h2>
            </div>
            <div className="card-body">
              <pre className="output-box">{resultOut}</pre>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="text-zinc-500">Log</span>
              <h2 className="text-sm font-extrabold uppercase tracking-widest">Event Log</h2>
            </div>
            <div className="card-body">
              <ul className="max-h-56 space-y-2 overflow-auto">
                {logs.map((log, index) => (
                  <li
                    key={`${log.message}-${index}`}
                    className={`rounded-none border-2 p-2 text-xs ${
                      log.type === "ok" ? "log-item-ok" : log.type === "err" ? "log-item-err" : "log-item-info"
                    }`}
                  >
                    {log.message}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t-2 border-zinc-950 bg-white px-4 py-10 md:px-10">
        <div className="mx-auto max-w-[1440px]">
          <div className="grid gap-8 lg:grid-cols-[1.25fr_1fr]">
            <div className="card">
              <div className="card-header">
                <div className="grid h-9 w-9 place-items-center border-2 border-zinc-950 bg-zinc-950 text-base font-extrabold text-yellow-400 shadow-[3px_3px_0_#facc15]">F</div>
                <div>
                  <h2 className="text-sm font-extrabold uppercase tracking-widest">FinBench</h2>
                  <p className="mono text-[11px] text-zinc-600">OpenEnv fintech console</p>
                </div>
              </div>
              <div className="card-body text-sm leading-7 text-zinc-600">
                <p>
                  OpenEnv Fintech Console is a benchmarking platform for evaluating AI agents in complex financial environments.
                  Built for reliability and transparency in automated decision systems.
                </p>
                <div className="mono flex flex-wrap gap-4 text-xs uppercase tracking-wider text-zinc-500">
                  <span>loan underwriting</span>
                  <span>fraud detection</span>
                  <span>portfolio rebalancing</span>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <span className="text-zinc-500">Links</span>
                <h2 className="text-sm font-extrabold uppercase tracking-widest">Quick Access</h2>
              </div>
              <div className="card-body">
                <div className="grid gap-6 sm:grid-cols-2">
                  <div className="flex flex-col gap-2 text-sm">
                    <h3 className="text-xs font-extrabold uppercase tracking-widest text-zinc-900">Project</h3>
                    <a className="text-zinc-600 no-underline hover:text-zinc-950" href={docsUrl} target="_blank" rel="noreferrer">API Docs</a>
                    <a className="text-zinc-600 no-underline hover:text-zinc-950" href={healthUrl} target="_blank" rel="noreferrer">Health Check</a>
                    <a className="text-zinc-600 no-underline hover:text-zinc-950" href="#console">Console</a>
                  </div>
                  <div className="flex flex-col gap-2 text-sm">
                    <h3 className="text-xs font-extrabold uppercase tracking-widest text-zinc-900">Resources</h3>
                    <a className="text-zinc-600 no-underline hover:text-zinc-950" href="#">OpenEnv Protocol</a>
                    <a className="text-zinc-600 no-underline hover:text-zinc-950" href="#">Agent SDK</a>
                    <a className="text-zinc-600 no-underline hover:text-zinc-950" href="#">Benchmark Suite</a>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-8 flex flex-col gap-4 border-t-2 border-zinc-950 pt-4 md:flex-row md:items-center md:justify-between">
            <p className="mono text-xs uppercase tracking-wider text-zinc-500">
              © 2026 FinBench OpenEnv Episode Console.
            </p>
            <div className="flex gap-2">
              <a className="grid h-9 w-9 place-items-center border-2 border-zinc-950 bg-white font-bold shadow-[4px_4px_0_#0a0a0a]" href="#">X</a>
              <a className="grid h-9 w-9 place-items-center border-2 border-zinc-950 bg-white font-bold shadow-[4px_4px_0_#0a0a0a]" href="#">GH</a>
              <a className="grid h-9 w-9 place-items-center border-2 border-zinc-950 bg-white font-bold shadow-[4px_4px_0_#0a0a0a]" href="#">IN</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
