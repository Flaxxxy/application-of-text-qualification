// Minimal Chart.js helpers for the Results page.
// Colours: one neutral grey for "other models", clay for the model this site uses, and a validated
// blue / clay / green set when several metrics are compared. Values are printed next to every bar,
// so colour is never the only way to read a chart.

const COLOR = { grey: "#d6d2c6", clay: "#da7150", blue: "#3a73c9", green: "#23a07a" };
const INK = "#141413", MUTED = "#8b897f", GRID = "rgba(20,20,19,.06)";

Chart.defaults.font.family = "Inter, -apple-system, BlinkMacSystemFont, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = MUTED;
Chart.defaults.locale = "en-US";  // "0.5", not "0,5", whatever the visitor's browser language

// Prints each bar's value at its end (horizontal bars).
const valueLabels = {
  id: "valueLabels",
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx } = chart;
    ctx.save();
    ctx.font = "500 12px " + Chart.defaults.font.family;
    ctx.fillStyle = INK;
    ctx.textBaseline = "middle";
    chart.data.datasets.forEach((ds, i) => {
      chart.getDatasetMeta(i).data.forEach((bar, j) => {
        const v = ds.data[j];
        if (v !== null && v !== undefined) ctx.fillText((opts.digits === 2 ? v.toFixed(2) : v.toFixed(3)), bar.x + 6, bar.y);
      });
    });
    ctx.restore();
  },
};

const baseOptions = (extra) => Object.assign({
  indexAxis: "y", maintainAspectRatio: false, animation: false,
  layout: { padding: { right: 44 } },
  scales: {
    x: { min: 0, max: 1, grid: { color: GRID }, border: { display: false }, ticks: { stepSize: 0.2 } },
    y: { grid: { display: false }, border: { display: false }, ticks: { color: INK } },
  },
}, extra);

// One series: every model in grey, the one used on the site in clay.
function highlightBars(id, labels, values, used) {
  const el = document.getElementById(id);
  if (!el) return null;
  return new Chart(el, {
    type: "bar",
    data: { labels: labels.map((l, i) => used[i] ? "★ " + l : l),
            datasets: [{ data: values, backgroundColor: used.map(u => u ? COLOR.clay : COLOR.grey),
                         borderRadius: 4, barPercentage: 0.72, categoryPercentage: 0.9 }] },
    options: baseOptions({ plugins: { legend: { display: false }, valueLabels: {},
                                      tooltip: { callbacks: { label: c => " macro-F1 " + c.raw.toFixed(3) } } } }),
    plugins: [valueLabels],
  });
}

// Several metrics per row (e.g. precision, recall, F1).
function metricBars(id, labels, series) {
  const el = document.getElementById(id);
  if (!el) return null;
  const colors = [COLOR.blue, COLOR.clay, COLOR.green];
  return new Chart(el, {
    type: "bar",
    data: { labels, datasets: series.map((s, k) => ({ label: s.label, data: s.data, backgroundColor: colors[k],
                                                       borderRadius: 3, barPercentage: 0.86, categoryPercentage: 0.78 })) },
    options: baseOptions({ plugins: {
      legend: { position: "top", align: "start", labels: { boxWidth: 10, boxHeight: 10, color: INK, padding: 16 } },
      valueLabels: { digits: 2 },
      tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${c.raw.toFixed(3)}` } } } }),
    plugins: [valueLabels],
  });
}
