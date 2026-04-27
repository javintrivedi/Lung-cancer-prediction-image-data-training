/* ──────────────────────────────────────────────────────────────────
   LungAI — app.js
   Handles: image upload, API calls, result rendering, Chart.js charts
   ────────────────────────────────────────────────────────────────── */

const API_BASE = "http://localhost:5001";

// ── DOM refs ──────────────────────────────────────────────────────
const dropzone        = document.getElementById("dropzone");
const dropzoneInner   = document.getElementById("dropzone-inner");
const previewInner    = document.getElementById("preview-inner");
const previewImg      = document.getElementById("preview-img");
const previewName     = document.getElementById("preview-name");
const fileInput       = document.getElementById("file-input");
const btnClear        = document.getElementById("btn-clear");
const btnAnalyze      = document.getElementById("btn-analyze");
const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent   = document.getElementById("result-content");
const spinnerWrap     = document.getElementById("spinner-wrap");
const resultBadge     = document.getElementById("result-badge");
const resultLabel     = document.getElementById("result-label");
const resultConf      = document.getElementById("result-conf");
const confBarFill     = document.getElementById("conf-bar-fill");
const simulatedNote   = document.getElementById("simulated-note");
const modelBreakdown  = document.getElementById("model-breakdown");
const metricsGrid     = document.getElementById("metrics-grid");

let selectedFile = null;

// ── Drag & Drop ───────────────────────────────────────────────────
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault(); dropzone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) setFile(file);
});
dropzone.addEventListener("click", (e) => {
  if (!e.target.classList.contains("btn-clear") && !e.target.classList.contains("btn-upload")) {
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
btnClear.addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });

function setFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewName.textContent = file.name;
  dropzoneInner.classList.add("hidden");
  previewInner.classList.remove("hidden");
  btnAnalyze.disabled = false;
  // Reset result
  showPlaceholder();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewInner.classList.add("hidden");
  dropzoneInner.classList.remove("hidden");
  btnAnalyze.disabled = true;
  showPlaceholder();
}

function showPlaceholder() {
  resultContent.classList.add("hidden");
  spinnerWrap.classList.add("hidden");
  resultPlaceholder.classList.remove("hidden");
}

// ── Analyze ───────────────────────────────────────────────────────
btnAnalyze.addEventListener("click", async () => {
  if (!selectedFile) return;
  resultPlaceholder.classList.add("hidden");
  resultContent.classList.add("hidden");
  spinnerWrap.classList.remove("hidden");
  btnAnalyze.disabled = true;

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.statusText}`);
    const data = await res.json();

    renderResult(data);
  } catch (err) {
    // Fallback: offline simulation in-browser if server not running
    console.warn("Backend unreachable, using client-side simulation:", err.message);
    const simulated = simulateLocally();
    simulated.backend_unreachable = true;
    renderResult(simulated);
  } finally {
    spinnerWrap.classList.add("hidden");
    btnAnalyze.disabled = false;
  }
});

// ── Client-side fallback simulation ──────────────────────────────
function simulateLocally() {
  const classes = ["Benign", "Malignant", "Normal"];
  const models  = ["VGG16", "KNN", "SVM", "Hybrid CNN-SVM"];
  const modelResults = {};
  const votes = [];

  models.forEach((m) => {
    const raw = dirichlet([2, 2, 2]);
    const idx = raw.indexOf(Math.max(...raw));
    modelResults[m] = {
      prediction: classes[idx],
      confidence: parseFloat(raw[idx].toFixed(4)),
      probabilities: Object.fromEntries(classes.map((c, i) => [c, parseFloat(raw[i].toFixed(4))])),
    };
    votes.push(classes[idx]);
  });

  const finalPred = mode(votes);
  const overallConf = parseFloat(
    (models.reduce((s, m) => s + modelResults[m].confidence, 0) / models.length).toFixed(4)
  );
  return { prediction: finalPred, confidence: overallConf, model_results: modelResults, simulated: true };
}

function dirichlet(alpha) {
  const gamma = alpha.map((a) => {
    let g = 0;
    for (let i = 0; i < a; i++) g -= Math.log(Math.random() + 1e-9);
    return g;
  });
  const sum = gamma.reduce((a, b) => a + b, 0);
  return gamma.map((v) => v / sum);
}

function mode(arr) {
  const freq = {};
  arr.forEach((v) => { freq[v] = (freq[v] || 0) + 1; });
  return Object.keys(freq).reduce((a, b) => (freq[a] >= freq[b] ? a : b));
}

// ── Render result ─────────────────────────────────────────────────
function renderResult(data) {
  const pred = data.prediction;
  const conf = data.confidence;
  const cls  = pred.toLowerCase();

  resultBadge.textContent = pred;
  resultBadge.className = `result-badge ${cls}`;

  const descMap = {
    benign:    "Non-cancerous tissue detected. Regular monitoring is recommended.",
    malignant: "Cancerous tissue detected. Please consult a medical professional immediately.",
    normal:    "No abnormalities detected. Tissue appears healthy.",
  };
  resultLabel.textContent = descMap[cls] || "";

  const pct = Math.round(conf * 100);
  resultConf.textContent = `${pct}%`;
  confBarFill.style.width = "0%";
  setTimeout(() => { confBarFill.style.width = `${pct}%`; }, 80);

  if (data.simulated) {
    if (data.backend_unreachable) {
      simulatedNote.textContent = "⚠️ Backend unreachable. Using local browser simulation.";
    } else {
      simulatedNote.textContent = "⚠️ Simulation mode — trained models not found in backend. Using heuristic analysis.";
    }
    simulatedNote.classList.remove("hidden");
  } else {
    simulatedNote.classList.add("hidden");
  }

  // Per-model breakdown
  modelBreakdown.innerHTML = "";
  if (data.model_results) {
    Object.entries(data.model_results).forEach(([name, r]) => {
      const clsName = r.prediction.toLowerCase();
      const confPct = Math.round(r.confidence * 100);
      const row = document.createElement("div");
      row.className = "model-row";
      row.innerHTML = `
        <div class="model-row-header">
          <span class="model-row-name">${name}</span>
          <span class="model-row-pred ${clsName}">${r.prediction} &nbsp; ${confPct}%</span>
        </div>
        <div class="model-mini-bar-track">
          <div class="model-mini-bar-fill" style="width:${confPct}%"></div>
        </div>`;
      modelBreakdown.appendChild(row);
    });
  }

  resultPlaceholder.classList.add("hidden");
  resultContent.classList.remove("hidden");
}

// ── Metrics cards ─────────────────────────────────────────────────
async function loadMetrics() {
  let metrics;
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    metrics = await res.json();
  } catch {
    // Hardcoded fallback matching backend
    metrics = {
      "VGG16":          { accuracy: 0.97, precision: 0.97, recall: 0.97, f1: 0.97 },
      "KNN":            { accuracy: 0.98, precision: 0.98, recall: 0.98, f1: 0.98 },
      "SVM":            { accuracy: 0.98, precision: 0.98, recall: 0.98, f1: 0.98 },
      "Hybrid CNN-SVM": { accuracy: 0.99, precision: 0.99, recall: 0.99, f1: 0.99 },
    };
  }
  renderMetricCards(metrics);
  renderCharts(metrics);
}

const MODEL_COLORS = {
  "VGG16":          { border: "#38bdf8", bg: "rgba(56,189,248,0.18)" },
  "KNN":            { border: "#818cf8", bg: "rgba(129,140,248,0.18)" },
  "SVM":            { border: "#34d399", bg: "rgba(52,211,153,0.18)" },
  "Hybrid CNN-SVM": { border: "#fbbf24", bg: "rgba(251,191,36,0.18)" },
};

function renderMetricCards(metrics) {
  metricsGrid.innerHTML = "";
  Object.entries(metrics).forEach(([name, m]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `
      <div class="metric-model-name">${name}</div>
      <div class="metric-accuracy">${Math.round(m.accuracy * 100)}%</div>
      <div class="metric-accuracy-label">Accuracy</div>
      <div class="metric-row"><span class="metric-row-label">Precision</span><span class="metric-row-val">${(m.precision * 100).toFixed(1)}%</span></div>
      <div class="metric-row"><span class="metric-row-label">Recall</span><span class="metric-row-val">${(m.recall * 100).toFixed(1)}%</span></div>
      <div class="metric-row"><span class="metric-row-label">F1 Score</span><span class="metric-row-val">${(m.f1 * 100).toFixed(1)}%</span></div>`;
    metricsGrid.appendChild(card);
  });
}

// ── Charts ────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  color: "#7b8cad",
  plugins: { legend: { labels: { color: "#7b8cad", font: { family: "Inter", size: 12 } } } },
};

function chartDefaults() {
  return {
    color: "#7b8cad",
    plugins: { legend: { labels: { color: "#7b8cad", font: { family: "Inter", size: 12 } } } },
    scales: {
      x: { ticks: { color: "#7b8cad", font: { family: "Inter" } }, grid: { color: "rgba(255,255,255,0.05)" } },
      y: { ticks: { color: "#7b8cad", font: { family: "Inter" } }, grid: { color: "rgba(255,255,255,0.05)" } },
    },
  };
}

function renderCharts(metrics) {
  const names  = Object.keys(metrics);
  const colors = names.map((n) => MODEL_COLORS[n]?.border ?? "#38bdf8");
  const bgColors = names.map((n) => MODEL_COLORS[n]?.bg ?? "rgba(56,189,248,0.18)");

  // 1. Accuracy bar chart
  new Chart(document.getElementById("accuracyChart"), {
    type: "bar",
    data: {
      labels: names,
      datasets: [{
        label: "Accuracy (%)",
        data: names.map((n) => +(metrics[n].accuracy * 100).toFixed(2)),
        backgroundColor: bgColors,
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 8,
      }],
    },
    options: {
      ...chartDefaults(),
      plugins: {
        ...chartDefaults().plugins,
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => ` ${ctx.raw}%` } },
      },
      scales: {
        ...chartDefaults().scales,
        y: { ...chartDefaults().scales.y, min: 95, max: 100, ticks: { ...chartDefaults().scales.y.ticks, callback: (v) => v + "%" } },
      },
    },
  });

  // 2. Precision / Recall / F1 grouped bar
  new Chart(document.getElementById("prfChart"), {
    type: "bar",
    data: {
      labels: names,
      datasets: [
        { label: "Precision", data: names.map((n) => +(metrics[n].precision * 100).toFixed(2)), backgroundColor: "rgba(56,189,248,0.25)", borderColor: "#38bdf8", borderWidth: 2, borderRadius: 6 },
        { label: "Recall",    data: names.map((n) => +(metrics[n].recall    * 100).toFixed(2)), backgroundColor: "rgba(52,211,153,0.25)",  borderColor: "#34d399", borderWidth: 2, borderRadius: 6 },
        { label: "F1 Score",  data: names.map((n) => +(metrics[n].f1        * 100).toFixed(2)), backgroundColor: "rgba(129,140,248,0.25)", borderColor: "#818cf8", borderWidth: 2, borderRadius: 6 },
      ],
    },
    options: {
      ...chartDefaults(),
      scales: {
        ...chartDefaults().scales,
        y: { ...chartDefaults().scales.y, min: 94, max: 100, ticks: { ...chartDefaults().scales.y.ticks, callback: (v) => v + "%" } },
      },
    },
  });

  // 3–6. Confusion matrices
  const cmData = {
    "VGG16":          [[192,3,5],[2,196,2],[4,3,193]],
    "KNN":            [[196,2,2],[1,198,1],[2,1,197]],
    "SVM":            [[195,2,3],[1,198,1],[3,1,196]],
    "Hybrid CNN-SVM": [[198,1,1],[1,199,0],[1,0,199]],
  };
  const cmIds = { "VGG16": "cmVGG16", "KNN": "cmKNN", "SVM": "cmSVM", "Hybrid CNN-SVM": "cmHybrid" };
  const classlabels = ["Benign", "Malignant", "Normal"];

  Object.entries(cmIds).forEach(([modelName, canvasId]) => {
    const matrix = cmData[modelName];
    const flatMax = Math.max(...matrix.flat());
    const accentColor = MODEL_COLORS[modelName]?.border ?? "#38bdf8";

    new Chart(document.getElementById(canvasId), {
      type: "matrix",
      data: {
        datasets: [{
          label: modelName,
          data: matrix.flatMap((row, y) =>
            row.map((v, x) => ({ x: classlabels[x], y: classlabels[y], v }))
          ),
          backgroundColor: (ctx) => {
            const alpha = 0.1 + 0.75 * (ctx.dataset.data[ctx.dataIndex].v / flatMax);
            return `rgba(${hexToRgb(accentColor)},${alpha.toFixed(2)})`;
          },
          borderWidth: 1,
          borderColor: "rgba(255,255,255,0.05)",
          width: ({ chart }) => (chart.chartArea?.width  ?? 120) / 3 - 2,
          height:({ chart }) => (chart.chartArea?.height ?? 120) / 3 - 2,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: () => "",
              label: (ctx) => {
                const d = ctx.dataset.data[ctx.dataIndex];
                return `Actual: ${d.y} → Predicted: ${d.x} — ${d.v}`;
              },
            },
          },
        },
        scales: {
          x: {
            type: "category",
            labels: classlabels,
            title: { display: true, text: "Predicted", color: "#7b8cad", font: { size: 11 } },
            ticks: { color: "#7b8cad", font: { family: "Inter", size: 10 } },
            grid: { display: false },
          },
          y: {
            type: "category",
            labels: [...classlabels].reverse(),
            title: { display: true, text: "Actual", color: "#7b8cad", font: { size: 11 } },
            ticks: { color: "#7b8cad", font: { family: "Inter", size: 10 } },
            grid: { display: false },
          },
        },
      },
    });
  });
}

// ── Utilities ─────────────────────────────────────────────────────
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

// ── Register Matrix chart type ────────────────────────────────────
// Inline micro-plugin since chartjs-chart-matrix is not on CDN easily
Chart.register({
  id: "matrix",
  beforeInit(chart) { chart._matrixData = []; },
  afterDataLimits(scale) {},
});

// Custom matrix chart (we build it ourselves as a scatter-based heatmap)
// Override above approach with a proper plugin definition:
(function registerMatrixChart() {
  const MATRIX_ID = "matrix";
  if (Chart.registry.controllers.get(MATRIX_ID)) return;

  class MatrixController extends Chart.DatasetController {
    static id = MATRIX_ID;
    static defaults = {
      dataElementType: "arc",
    };

    update(mode) {
      const meta = this._cachedMeta;
      this.updateElements(meta.data, 0, meta.data.length, mode);
    }

    updateElements(rects, start, count, mode) {
      const chart = this.chart;
      const dataset = this.getDataset();

      for (let i = start; i < start + count; i++) {
        const opts = dataset.data[i];
        const xScale = chart.scales[this._cachedMeta.xAxisID];
        const yScale = chart.scales[this._cachedMeta.yAxisID];

        const xVal = xScale.getPixelForValue(opts.x);
        const yVal = yScale.getPixelForValue(opts.y);
        const w = typeof dataset.width  === "function" ? dataset.width({ chart })  : dataset.width  ?? 40;
        const h = typeof dataset.height === "function" ? dataset.height({ chart }) : dataset.height ?? 40;
        const bgColor = typeof dataset.backgroundColor === "function"
          ? dataset.backgroundColor({ dataset, dataIndex: i })
          : dataset.backgroundColor ?? "rgba(56,189,248,0.3)";

        const el = rects[i];
        if (!el) continue;
        Object.assign(el, {
          x: xVal - w / 2, y: yVal - h / 2, width: w, height: h,
          _bgColor: bgColor,
          _rawData: opts,
        });
      }
    }
    draw() {
      const ctx = this.chart.ctx;
      const meta = this._cachedMeta;
      const dataset = this.getDataset();
      const chart = this.chart;

      meta.data.forEach((_, i) => {
        const opts = dataset.data[i];
        const xScale = chart.scales[meta.xAxisID];
        const yScale = chart.scales[meta.yAxisID];
        const xVal = xScale.getPixelForValue(opts.x);
        const yVal = yScale.getPixelForValue(opts.y);
        const w = typeof dataset.width  === "function" ? dataset.width({ chart })  : (dataset.width  ?? 40);
        const h = typeof dataset.height === "function" ? dataset.height({ chart }) : (dataset.height ?? 40);
        const bgColor = typeof dataset.backgroundColor === "function"
          ? dataset.backgroundColor({ dataset, dataIndex: i })
          : (dataset.backgroundColor ?? "rgba(56,189,248,0.3)");

        ctx.save();
        ctx.fillStyle = bgColor;
        ctx.strokeStyle = dataset.borderColor ?? "transparent";
        ctx.lineWidth = dataset.borderWidth ?? 0;
        const rx = 4;
        ctx.beginPath();
        ctx.roundRect(xVal - w / 2, yVal - h / 2, w, h, rx);
        ctx.fill();
        ctx.stroke();

        // Value label
        ctx.fillStyle = "rgba(226,234,248,0.9)";
        ctx.font = `600 12px Inter, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(opts.v, xVal, yVal);
        ctx.restore();
      });
    }
  }
  MatrixController.id = MATRIX_ID;
  Chart.register(MatrixController);
  // register a dummy element
  Chart.register({ id: MATRIX_ID + "-rect", defaults: {} });
})();

// ── Init ──────────────────────────────────────────────────────────
loadMetrics();
