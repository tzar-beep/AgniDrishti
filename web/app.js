// AgniDrishti web map: Leaflet choropleth + Chart.js explanations.
const COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]; // YlOrRd, 5 classes
const LABELS = {
  tmax_mean_7d: "Max temp, 7-day mean", tmax_mean_14d: "Max temp, 14-day mean",
  rh_mean_7d: "Humidity, 7-day mean", rh_mean_14d: "Humidity, 14-day mean",
  wind_mean_7d: "Wind, 7-day mean", wind_mean_14d: "Wind, 14-day mean",
  precip_mean_7d: "Rain, 7-day mean", precip_mean_14d: "Rain, 14-day mean",
  days_since_rain: "Days since ≥2 mm rain", fwi_7d: "Fire weather index (7d)",
  elev_mean: "Mean elevation", slope_mean: "Mean slope",
  hotspots_4w: "Hotspots, past 4 wks", hotspots_12w: "Hotspots, past 12 wks",
  hotspots_52w: "Hotspots, past 52 wks", days_since_fire: "Days since last hotspot",
  woy_sin: "Season (sin)", woy_cos: "Season (cos)",
};

const map = L.map("map").setView([30.1, 79.2], 8);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 12, attribution: "© OpenStreetMap contributors",
}).addTo(map);

let layer = null, breaks = [], legend = null, shapChart = null, histChart = null, current = null;
const $ = (id) => document.getElementById(id);
const getJSON = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
};

function quantileBreaks(values) {
  const v = [...values].sort((a, b) => a - b);
  return [0.2, 0.4, 0.6, 0.8].map((q) => v[Math.min(v.length - 1, Math.floor(q * v.length))]);
}
const colorFor = (p) => COLORS[breaks.filter((b) => p > b).length];
const pct = (p) => `${(p * 100).toFixed(1)}%`;

function drawLegend() {
  if (legend) legend.remove();
  legend = L.control({ position: "bottomright" });
  legend.onAdd = () => {
    const div = L.DomUtil.create("div", "legend");
    const edges = [0, ...breaks, 1];
    div.innerHTML = "<b>P(fire, next 7 days)</b><br>" + COLORS.map((c, i) =>
      `<i style="background:${c}"></i>${pct(edges[i])} – ${pct(edges[i + 1])}`).join("<br>");
    return div;
  };
  legend.addTo(map);
}

async function loadRisk(date) {
  const geo = await getJSON(`/api/risk?date=${date}`);
  breaks = quantileBreaks(geo.features.map((f) => f.properties.prob));
  if (layer) layer.remove();
  layer = L.geoJSON(geo, {
    style: (f) => ({ fillColor: colorFor(f.properties.prob), fillOpacity: 0.75, color: "#555", weight: 0.5 }),
    onEachFeature: (f, l) => {
      l.bindTooltip(`${f.properties.cell_id}<br><b>${pct(f.properties.prob)}</b>`, { sticky: true });
      l.on("click", () => showCell(f.properties.cell_id));
    },
  }).addTo(map);
  drawLegend();
  if (current) showCell(current);
}

async function showCell(cellId) {
  current = cellId;
  const date = $("date").value;
  const [exp, hist] = await Promise.all([
    getJSON(`/api/cells/${cellId}/explain?date=${date}`),
    getJSON(`/api/cells/${cellId}/history?date=${date}`),
  ]);
  $("cell").hidden = false;
  $("cell-title").textContent = `Cell ${cellId} · week of ${date}`;
  $("cell-prob").textContent = `Fire probability: ${pct(exp.prob)}`;

  const c = exp.contributions;
  if (shapChart) shapChart.destroy();
  shapChart = new Chart($("shap-chart"), {
    type: "bar",
    data: {
      labels: c.map((x) => `${LABELS[x.feature] || x.feature} = ${x.value.toFixed(1)}`),
      datasets: [{ data: c.map((x) => x.contribution),
        backgroundColor: c.map((x) => (x.contribution > 0 ? "#d7301f" : "#2b8cbe")) }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "← lowers risk | raises risk →" } } } },
  });

  const rows = hist.rows;
  if (histChart) histChart.destroy();
  histChart = new Chart($("history-chart"), {
    data: {
      labels: rows.map((r) => r.week_start.slice(5)),
      datasets: [
        { type: "line", label: "Predicted prob.", data: rows.map((r) => r.prob), borderColor: "#f03b20", yAxisID: "y" },
        { type: "bar", label: "Fire observed", data: rows.map((r) => r.y), backgroundColor: "#999", yAxisID: "y" },
        { type: "line", label: "FWI (7d)", data: rows.map((r) => r.features.fwi_7d), borderColor: "#2b8cbe", yAxisID: "y2" },
      ],
    },
    options: { scales: { y: { min: 0, max: 1 }, y2: { position: "right", grid: { display: false } } } },
  });
  if (window.innerWidth <= 768) $("cell").scrollIntoView({ behavior: "smooth" });
}

async function loadModel() {
  const m = await getJSON("/api/model");
  const rows = Object.entries(m.metrics).map(([name, s]) =>
    `<tr><td>${name}</td><td>${s.test.pr_auc}</td><td>${s.test.roc_auc}</td><td>${s.test.brier}</td></tr>`);
  $("metrics").innerHTML = `<tr><th>Test 2024 (${m.version})</th><th>PR-AUC</th><th>ROC-AUC</th><th>Brier</th></tr>${rows.join("")}`;
}

async function init() {
  const { dates } = await getJSON("/api/dates");
  $("date").innerHTML = dates.map((d) => `<option>${d}</option>`).join("");
  $("date").value = dates[dates.length - 1];
  $("date").addEventListener("change", () => loadRisk($("date").value));
  $("close").addEventListener("click", () => { $("cell").hidden = true; current = null; });
  await Promise.all([loadRisk($("date").value), loadModel()]);
}

init().catch((e) => alert(`Failed to load data: ${e.message}`));
