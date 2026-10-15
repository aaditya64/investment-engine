const API_BASE = "http://127.0.0.1:8000";

async function handle(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail)
      ? body.detail.map((error) => error.msg).join("; ")
      : body.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
}

export function fetchUniverse() {
  return fetch(`${API_BASE}/instruments/universe`).then(handle);
}

export function optimizePortfolio(payload, signal) {
  return fetch(`${API_BASE}/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  }).then(handle);
}

export function backtestPortfolio(payload, signal) {
  return fetch(`${API_BASE}/backtest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  }).then(handle);
}
