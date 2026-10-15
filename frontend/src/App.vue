<script setup>
import { Chart } from "chart.js/auto";
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from "vue";
import { backtestPortfolio, fetchUniverse, optimizePortfolio } from "./api.js";

const CATEGORICAL_LIGHT = [
  "#2a78d6",
  "#eb6834",
  "#1baf7a",
  "#eda100",
  "#e87ba4",
  "#008300",
  "#4a3aa7",
  "#e34948",
];
const CATEGORICAL_DARK = [
  "#3987e5",
  "#d95926",
  "#199e70",
  "#c98500",
  "#d55181",
  "#008300",
  "#9085e9",
  "#e66767",
];
const OTHER_COLOR = "#c3c2b7";
const universe = ref([]);
const selected = reactive({});
const weightBounds = reactive({});
const lookbackYears = ref(5);
const riskFreePercent = ref(4);
const trainingPercent = ref(80);
const loading = ref(false);
const errorMessage = ref("");
const result = ref(null);
const backtestResult = ref(null);
const activeAction = ref(null);
const selectedInstruments = computed(() => universe.value.filter((i) => selected[i.ticker]));
const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");
const isDarkMode = ref(darkModeQuery.matches);
let requestSequence = 0;
let requestController = null;

function onSchemeChange(event) {
  isDarkMode.value = event.matches;
}

function colorForTicker(ticker) {
  // Keep an asset's colour stable when weights or the selection change.
  const palette = isDarkMode.value ? CATEGORICAL_DARK : CATEGORICAL_LIGHT;
  const index = universe.value.map((i) => i.ticker).sort().indexOf(ticker);
  return palette[index] || OTHER_COLOR;
}

async function optimize(event) {
  const isBacktest = event.submitter?.value === "backtest";
  if (selectedInstruments.value.length === 0) return;
  const sequence = ++requestSequence;
  requestController?.abort();
  requestController = new AbortController();
  loading.value = true;
  errorMessage.value = "";
  activeAction.value = isBacktest ? "backtest" : "optimize";
  if (isBacktest) backtestResult.value = null;
  else result.value = null;
  try {
    const bounds = {};
    for (const { ticker } of selectedInstruments.value) {
      bounds[ticker] = [weightBounds[ticker].minimum / 100, weightBounds[ticker].maximum / 100];
    }
    const payload = {
      tickers: selectedInstruments.value.map((i) => i.ticker),
      lookback_years: lookbackYears.value,
      weight_bounds: bounds,
      risk_free_rate: riskFreePercent.value / 100,
    };
    if (isBacktest) payload.train_fraction = trainingPercent.value / 100;
    const send = isBacktest ? backtestPortfolio : optimizePortfolio;
    const response = await send(payload, requestController.signal);
    if (sequence === requestSequence) {
      if (isBacktest) backtestResult.value = response;
      else result.value = response;
    }
  } catch (error) {
    if (sequence === requestSequence && error.name !== "AbortError") {
      errorMessage.value = error.message;
    }
  } finally {
    if (sequence === requestSequence) {
      loading.value = false;
      activeAction.value = null;
    }
  }
}

watch([selected, weightBounds, lookbackYears, riskFreePercent, trainingPercent], () => {
  ++requestSequence;
  requestController?.abort();
  loading.value = false;
  result.value = null;
  backtestResult.value = null;
  activeAction.value = null;
  errorMessage.value = "";
}, { deep: true });

const chartCanvas = ref(null);
let chartInstance = null;

function renderChart() {
  if (!result.value || !chartCanvas.value) {
    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }
    return;
  }
  const entries = Object.entries(result.value.weights);
  const datasets = entries.map(([ticker, weight]) => ({
    label: ticker,
    data: [weight * 100],
    backgroundColor: colorForTicker(ticker),
    borderColor: isDarkMode.value ? "#1a1a19" : "#fcfcfb",
    borderWidth: 2,
    borderRadius: 4,
  }));

  if (chartInstance) {
    chartInstance.destroy();
  }
  chartInstance = new Chart(chartCanvas.value, {
    type: "bar",
    data: { labels: ["Portfolio"], datasets },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          stacked: true,
          min: 0,
          max: 100,
          grid: { display: false },
          ticks: { callback: (v) => `${v}%` },
        },
        y: { stacked: true, grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)}%`,
          },
        },
      },
    },
  });
}

watch(result, () => nextTick(renderChart));
watch(isDarkMode, () => nextTick(renderChart));

const growthCanvas = ref(null);
let growthChart = null;

function renderGrowthChart() {
  if (growthChart) {
    growthChart.destroy();
    growthChart = null;
  }
  if (!backtestResult.value || !growthCanvas.value) return;
  const test = backtestResult.value;
  const datasets = [{
    label: "Optimised on training data",
    data: test.growth.map((point) => point.optimised),
    borderColor: isDarkMode.value ? "#3987e5" : "#2a78d6",
    pointRadius: 0,
    borderWidth: 2,
  }];
  if (test.equal_weight) datasets.push({
    label: "Equal weight",
    data: test.growth.map((point) => point.equal_weight),
    borderColor: isDarkMode.value ? "#d95926" : "#eb6834",
    pointRadius: 0,
    borderWidth: 2,
  });
  const textColor = isDarkMode.value ? "#c3c2b7" : "#52514e";
  growthChart = new Chart(growthCanvas.value, {
    type: "line",
    data: { labels: test.growth.map((point) => point.date), datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { maxTicksLimit: 6, color: textColor }, grid: { display: false } },
        y: { ticks: { color: textColor, callback: (value) => money(value) } },
      },
      plugins: {
        legend: { labels: { color: textColor } },
        tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${money(ctx.parsed.y)}` } },
      },
    },
  });
}

watch(backtestResult, () => nextTick(renderGrowthChart));
watch(isDarkMode, () => nextTick(renderGrowthChart));

onMounted(async () => {
  darkModeQuery.addEventListener("change", onSchemeChange);
  try {
    const instruments = await fetchUniverse();
    for (const { ticker } of instruments) {
      weightBounds[ticker] = { minimum: 0, maximum: 100 };
    }
    universe.value = instruments;
  } catch (error) {
    errorMessage.value = `Could not load instruments: ${error.message}`;
  }
});

onUnmounted(() => {
  ++requestSequence;
  requestController?.abort();
  darkModeQuery.removeEventListener("change", onSchemeChange);
  if (chartInstance) chartInstance.destroy();
  if (growthChart) growthChart.destroy();
});

function sortedWeights() {
  return result.value ? Object.entries(result.value.weights).sort((a, b) => b[1] - a[1]) : [];
}

function money(value) {
  return new Intl.NumberFormat("en-GB", {
    style: "currency", currency: "GBP", maximumFractionDigits: 2,
  }).format(value);
}

function sharpe(value) {
  return value == null ? "Undefined (zero volatility)" : value.toFixed(2);
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}
</script>

<template>
  <div class="app">
    <header class="app-header">
      <h1>Portfolio Workbench</h1>
      <p class="disclaimer">
        Optimise historical Sharpe for your selected assets and weight bounds.
      </p>
    </header>

    <form @submit.prevent="optimize">
      <section class="panel">
        <h2>1. Choose instruments</h2>
        <ul class="instrument-list">
          <li v-for="instrument in universe" :key="instrument.ticker" class="instrument-row">
            <label class="instrument-label">
              <input type="checkbox" v-model="selected[instrument.ticker]" />
              <span class="swatch" :style="{
                background: selected[instrument.ticker] ? colorForTicker(instrument.ticker) : 'transparent',
              }"></span>
              <span class="instrument-name">{{ instrument.name }}</span>
              <span class="instrument-ticker">{{ instrument.ticker }}</span>
              <span class="instrument-class">{{ instrument.asset_class }}</span>
            </label>
            <div v-if="selected[instrument.ticker]" class="weight-bounds">
              <label>
                Minimum (%)
                <input type="number" min="0" max="100" step="0.1" required
                  :aria-label="`${instrument.ticker} minimum weight (%)`"
                  v-model.number="weightBounds[instrument.ticker].minimum" />
              </label>
              <label>
                Maximum (%)
                <input type="number" min="0" max="100" step="0.1" required
                  :aria-label="`${instrument.ticker} maximum weight (%)`"
                  v-model.number="weightBounds[instrument.ticker].maximum" />
              </label>
            </div>
          </li>
        </ul>
      </section>

      <section class="panel">
        <h2>2. Optimisation settings</h2>
        <div class="settings-grid">
          <label class="setting-row">
            Historical window (years)
            <input type="number" min="0.1" max="30" step="0.1" required v-model.number="lookbackYears" />
          </label>
          <label class="setting-row">
            Annual risk-free rate (%)
            <input type="number" min="-99.9" step="0.1" required v-model.number="riskFreePercent" />
          </label>
        </div>
        <button class="optimize-button" type="submit" value="optimize"
          :disabled="loading || !selectedInstruments.length">
          {{ loading && activeAction === "optimize" ? "Optimising…" : "Optimise portfolio" }}
        </button>
        <div class="backtest-settings">
          <label class="setting-row">
            Backtest training data (%)
            <select v-model.number="trainingPercent">
              <option v-for="percent in [50, 60, 70, 80, 90]" :key="percent" :value="percent">
                {{ percent }}%
              </option>
            </select>
          </label>
          <p class="hint">
            Fit on the earliest {{ trainingPercent }}% of price rows, then buy and hold over the
            remaining {{ 100 - trainingPercent }}%. Both portfolios start at £1,000, with no trading costs.
            No prior optimisation is needed.
          </p>
          <button class="optimize-button" type="submit" value="backtest"
            :disabled="loading || !selectedInstruments.length">
            {{ loading && activeAction === "backtest" ? "Backtesting…" : "Run historical test" }}
          </button>
        </div>
      </section>
    </form>

    <section v-if="result || !backtestResult || loading || errorMessage" class="panel results"
      aria-live="polite" :aria-busy="loading">
      <h2>3. Result</h2>
      <p v-if="!selectedInstruments.length" class="hint">Select at least one instrument to start.</p>
      <p v-else-if="!result && !backtestResult && !loading && !errorMessage" class="hint">
        Choose your settings, then optimise the portfolio.
      </p>
      <p v-if="loading" class="hint">Fetching prices and calculating&hellip;</p>
      <p v-if="errorMessage" class="error-banner" role="alert">{{ errorMessage }}</p>
      <template v-if="result">
        <h3>Full-window optimisation estimates</h3>
        <div class="chart-wrap"><canvas ref="chartCanvas" aria-label="Portfolio allocation"></canvas></div>
        <table class="weights-table" aria-label="Portfolio weights">
          <thead><tr><th></th><th>Instrument</th><th>Weight</th></tr></thead>
          <tbody>
            <tr v-for="[ticker, weight] in sortedWeights()" :key="ticker">
              <td><span class="swatch" :style="{ background: colorForTicker(ticker) }"></span></td>
              <td>{{ ticker }}</td><td>{{ pct(weight) }}</td>
            </tr>
          </tbody>
        </table>
        <div class="stat-grid">
          <div class="stat-tile">
            <div class="stat-label">Historical annualised return</div>
            <div class="stat-value">{{ pct(result.metrics.historical_return) }}</div>
          </div>
          <div class="stat-tile">
            <div class="stat-label">Annualised volatility</div>
            <div class="stat-value">{{ pct(result.metrics.volatility) }}</div>
          </div>
          <div class="stat-tile">
            <div class="stat-label">Sharpe ratio</div>
            <div class="stat-value">{{ result.metrics.sharpe.toFixed(2) }}</div>
          </div>
          <div class="stat-tile">
            <div class="stat-label">Annual risk-free rate used</div>
            <div class="stat-value">{{ pct(result.risk_free_rate) }}</div>
          </div>
        </div>
      </template>
    </section>

    <section v-if="backtestResult" class="panel" aria-live="polite">
      <h2>Out-of-sample historical test</h2>
      <p class="hint">
        Training: {{ backtestResult.training_start }} to {{ backtestResult.training_end }}.<br />
        Test period: {{ backtestResult.test_start }} to {{ backtestResult.test_end }}.
        Purchased at the {{ backtestResult.training_end }} close; holdings are not rebalanced.
        Weight bounds apply at purchase and weights can drift afterwards.
      </p>
      <p class="hint">Annual risk-free rate used: {{ pct(backtestResult.risk_free_rate) }}.</p>
      <p v-for="warning in backtestResult.data_warnings" :key="warning" class="hint" role="status">
        {{ warning }}
      </p>
      <p v-if="backtestResult.baseline_unavailable_reason" class="hint" role="status">
        {{ backtestResult.baseline_unavailable_reason }}
      </p>
      <div class="growth-chart-wrap">
        <canvas ref="growthCanvas" aria-label="Growth of £1,000 in the held-out test period"></canvas>
      </div>
      <h3>Realised test metrics</h3>
      <table class="weights-table" aria-label="Out-of-sample performance comparison">
        <thead>
          <tr><th>Metric</th><th>Optimised</th><th v-if="backtestResult.equal_weight">Equal weight</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Annualised return</td>
            <td>{{ pct(backtestResult.optimised.metrics.annualised_return) }}</td>
            <td v-if="backtestResult.equal_weight">{{ pct(backtestResult.equal_weight.metrics.annualised_return) }}</td>
          </tr>
          <tr>
            <td>Annualised volatility</td>
            <td>{{ pct(backtestResult.optimised.metrics.volatility) }}</td>
            <td v-if="backtestResult.equal_weight">{{ pct(backtestResult.equal_weight.metrics.volatility) }}</td>
          </tr>
          <tr>
            <td>Sharpe ratio</td>
            <td>{{ sharpe(backtestResult.optimised.metrics.sharpe) }}</td>
            <td v-if="backtestResult.equal_weight">{{ sharpe(backtestResult.equal_weight.metrics.sharpe) }}</td>
          </tr>
        </tbody>
      </table>
      <h3>Starting allocations</h3>
      <table class="weights-table" aria-label="Backtest starting weights">
        <thead>
          <tr><th>Instrument</th><th>Optimised</th><th v-if="backtestResult.equal_weight">Equal weight</th></tr>
        </thead>
        <tbody>
          <tr v-for="(weight, ticker) in backtestResult.optimised.weights" :key="ticker">
            <td>{{ ticker }}</td><td>{{ pct(weight) }}</td>
            <td v-if="backtestResult.equal_weight">{{ pct(backtestResult.equal_weight.weights[ticker]) }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<style scoped>
.app {
  --surface-1: #fcfcfb;
  --page-plane: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --gridline: #e1e0d9;
  --border: rgba(11, 11, 11, 0.1);
  color-scheme: light;

  max-width: 760px;
  margin: 0 auto;
  padding: 16px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary);
  background: var(--page-plane);
}

@media (prefers-color-scheme: dark) {
  .app {
    --surface-1: #1a1a19;
    --page-plane: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --border: rgba(255, 255, 255, 0.1);
    color-scheme: dark;
  }
}

h1 {
  font-size: 1.4rem;
  margin: 0 0 4px;
}

.disclaimer {
  color: var(--text-secondary);
  font-size: 0.85rem;
  margin: 0 0 16px;
}

.panel {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.panel h2 {
  font-size: 1.05rem;
  margin: 0 0 12px;
}

.instrument-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.instrument-row {
  padding: 8px 0;
  border-bottom: 1px solid var(--gridline);
}
.instrument-row:last-child {
  border-bottom: none;
}

.instrument-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}

.instrument-name {
  flex: 1;
}

.instrument-ticker,
.instrument-class {
  color: var(--text-muted);
  font-size: 0.8rem;
}

.weight-bounds {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 8px 0 0 28px;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.weight-bounds label,
.setting-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

input[type="number"],
.setting-row select {
  width: 7ch;
  padding: 4px 6px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--page-plane);
  color: var(--text-primary);
}

.settings-grid {
  display: flex;
  flex-direction: column;
  gap: 14px;
  font-size: 0.9rem;
}

.optimize-button {
  margin-top: 16px;
  background: var(--page-plane);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 12px;
  cursor: pointer;
  color: var(--text-primary);
  font-size: 0.9rem;
}

.optimize-button:disabled {
  opacity: 0.5;
  cursor: default;
}

.hint {
  color: var(--text-secondary);
}

.error-banner {
  color: #d03b3b;
  border: 1px solid #d03b3b;
  border-radius: 6px;
  padding: 8px 12px;
}

.chart-wrap {
  height: 90px;
  margin-bottom: 12px;
}

.growth-chart-wrap {
  height: 280px;
  margin: 16px 0;
}

.backtest-settings {
  border-top: 1px solid var(--gridline);
  margin-top: 20px;
  padding-top: 16px;
  font-size: 0.9rem;
}

.backtest-settings .optimize-button {
  margin-top: 0;
}

h3 {
  font-size: 0.95rem;
}

.weights-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 16px;
  font-size: 0.9rem;
}
.weights-table th {
  text-align: left;
  color: var(--text-muted);
  font-weight: normal;
  border-bottom: 1px solid var(--gridline);
  padding: 4px 8px;
}
.weights-table td {
  padding: 4px 8px;
  border-bottom: 1px solid var(--gridline);
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
@media (min-width: 480px) {
  .stat-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.stat-tile {
  background: var(--page-plane);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px;
}
.stat-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-bottom: 4px;
}
.stat-value {
  font-size: 1.1rem;
  font-variant-numeric: tabular-nums;
}

</style>
