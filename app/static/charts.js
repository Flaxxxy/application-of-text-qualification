// Chart.js helpers for the Dashboard widgets and the Models page.
// Colours: categorical slots validated for colour-blind safety (blue, orange, aqua). Values are
// always printed next to the bars, so colour is never the only way to read a chart.

const PALETTE = { blue: "#2a78d6", orange: "#eb6834", aqua: "#1baf7a" };
const GROUP_COLOR = { "ours": PALETTE.blue, "ready-made": PALETTE.orange, "fine-tuned": PALETTE.aqua };
const INK = "#323a49", MUTED = "#6b7785", GRID = "rgba(0,0,0,.06)";

Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.locale = "en-US";  // "0.5", not "0,5", whatever the visitor's browser language
Chart.defaults.color = MUTED;

// Draws each bar's value at its end (horizontal bars only).
const valueLabels = {
  id: "valueLabels",
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx } = chart;
    ctx.save();
    ctx.font = "600 12px " + Chart.defaults.font.family;
    ctx.fillStyle = opts.color || INK;
    ctx.textBaseline = "middle";
    chart.data.datasets.forEach((ds, i) => {
      chart.getDatasetMeta(i).data.forEach((bar, j) => {
        const v = ds.data[j];
        if (v === null || v === undefined) return;
        ctx.fillText((opts.format || (x => x.toFixed(3)))(v), bar.x + 6, bar.y);
      });
    });
    ctx.restore();
  },
};

// Small white bar chart inside a coloured widget card; the model in use is drawn fully opaque.
function miniBars(id, labels, values, used, fmt) {
  const el = document.getElementById(id);
  if (!el) return;
  new Chart(el, {
    type: "bar",
    data: { labels, datasets: [{ data: values, borderRadius: 4, barPercentage: 0.7,
      backgroundColor: used.map(u => u ? "rgba(255,255,255,1)" : "rgba(255,255,255,.45)") }] },
    options: {
      maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false },
                 tooltip: { callbacks: { label: c => " " + (fmt ? fmt(c.raw) : c.raw.toFixed(3)) } } },
      scales: { x: { display: false }, y: { display: false, min: 0, max: 1 } },
    },
  });
}

// Horizontal bars, one colour per model group, with a legend for the groups.
function groupBars(id, labels, values, groups, used) {
  const el = document.getElementById(id);
  if (!el) return;
  const names = [...new Set(groups)];
  const datasets = names.map(g => ({
    label: g, backgroundColor: GROUP_COLOR[g], borderRadius: 4, barPercentage: 0.75, categoryPercentage: 0.85,
    data: values.map((v, i) => groups[i] === g ? v : null),
  }));
  new Chart(el, {
    type: "bar",
    data: { labels: labels.map((l, i) => used[i] ? l + "  ★ on this site" : l), datasets },
    options: {
      indexAxis: "y", maintainAspectRatio: false, animation: false,
      layout: { padding: { right: 48 } },
      plugins: { legend: { position: "top", align: "start", labels: { boxWidth: 12, boxHeight: 12 } },
                 valueLabels: {}, tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${c.raw.toFixed(3)}` } } },
      scales: { x: { stacked: true, min: 0, max: 1, grid: { color: GRID }, title: { display: true, text: "test macro-F1" } },
                y: { stacked: true, grid: { display: false }, ticks: { color: INK } } },
    },
    plugins: [valueLabels],
  });
}

// Horizontal grouped bars: several metrics per row (e.g. precision, recall, F1).
function metricBars(id, labels, series) {
  const el = document.getElementById(id);
  if (!el) return;
  const colors = [PALETTE.blue, PALETTE.orange, PALETTE.aqua];
  new Chart(el, {
    type: "bar",
    data: { labels, datasets: series.map((s, k) => ({ label: s.label, data: s.data, backgroundColor: colors[k],
      borderRadius: 3, barPercentage: 0.85, categoryPercentage: 0.8 })) },
    options: {
      indexAxis: "y", maintainAspectRatio: false, animation: false,
      layout: { padding: { right: 40 } },
      plugins: { legend: { position: "top", align: "start", labels: { boxWidth: 12, boxHeight: 12 } },
                 valueLabels: { format: v => v.toFixed(2) },
                 tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${c.raw.toFixed(3)}` } } },
      scales: { x: { min: 0, max: 1, grid: { color: GRID } }, y: { grid: { display: false }, ticks: { color: INK } } },
    },
    plugins: [valueLabels],
  });
}
