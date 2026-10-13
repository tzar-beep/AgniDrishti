// AgniDrishti web client: Leaflet choropleth, ranking table and Chart.js cell diagnostics.
"use strict";

const COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]; // YlOrRd, 5 quantile classes
const FEATURES = {
  tmax_mean_7d: ["Tmax, 7-d mean", "°C"], tmax_mean_14d: ["Tmax, 14-d mean", "°C"],
  rh_mean_7d: ["RH, 7-d mean", "%"], rh_mean_14d: ["RH, 14-d mean", "%"],
  wind_mean_7d: ["Wind, 7-d mean", "km/h"], wind_mean_14d: ["Wind, 14-d mean", "km/h"],
  precip_mean_7d: ["Precip, 7-d mean", "mm/d"], precip_mean_14d: ["Precip, 14-d mean", "mm/d"],
  days_since_rain: ["Days since ≥2 mm rain", "d"], fwi_7d: ["Fire weather index, 7-d", ""],
  elev_mean: ["Elevation, mean", "m"], slope_mean: ["Slope, mean", "°"],
  hotspots_4w: ["Hotspots, prev 4 wk", ""], hotspots_12w: ["Hotspots, prev 12 wk", ""],
  hotspots_52w: ["Hotspots, prev 52 wk", ""], days_since_fire: ["Days since last hotspot", "d"],
  woy_sin: ["Week of year (sin)", ""], woy_cos: ["Week of year (cos)", ""],
};
const LEVEL_COLORS = { Low: "#4d9a5b", Moderate: "#e3b505", High: "#e8590c", "Very High": "#b91c1c" };
const lvlClass = (l) => `lvl-${(l || "").replace(" ", "")}`;
const MODEL_NAMES = { xgboost: "XGBoost", logreg: "Logistic reg.", persistence: "Persistence" };

const $ = (id) => document.getElementById(id);
const fmtP = (p) => (p == null ? "–" : p.toFixed(3));
const fmtNum = (v) => (v == null ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
const label = (f) => (FEATURES[f] || [f])[0];
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}

Chart.defaults.font.family = '"IBM Plex Mono", ui-monospace, monospace';
Chart.defaults.font.size = 10;
Chart.defaults.color = "#4a5160";
Chart.defaults.borderColor = "#e1e4ea";
Chart.defaults.animation = false;

// ---- Map -------------------------------------------------------------------------------------
const map = L.map("map", { zoomSnap: 0.25, minZoom: 6 }).setView([30.05, 79.3], 8);
const basemaps = {
  "Grey (OSM)": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 13, className: "tiles-grey", attribution: "© OpenStreetMap contributors",
  }),
  "Terrain (OpenTopoMap)": L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
    maxZoom: 13, attribution: "© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)",
  }),
  "Imagery (Esri)": L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 13, attribution: "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics" }),
};
basemaps["Grey (OSM)"].addTo(map);
L.control.layers(basemaps, null, { position: "topright" }).addTo(map);
L.control.scale({ imperial: false, position: "bottomleft" }).addTo(map);

const readout = L.control({ position: "bottomleft" });
readout.onAdd = () => L.DomUtil.create("div", "readout");
readout.addTo(map);
map.on("mousemove", (e) => {
  readout.getContainer().textContent = `${e.latlng.lat.toFixed(3)}°N  ${e.latlng.lng.toFixed(3)}°E`;
});

const state = { dates: [], forecastDates: [], isForecast: false, alerts: null, date: null, cell: null, features: [], breaks: [], layer: null, legend: null,
  layers: {}, charts: {} };

function quantileBreaks(values) {
  const v = [...values].sort((a, b) => a - b);
  return [0.2, 0.4, 0.6, 0.8].map((q) => v[Math.min(v.length - 1, Math.floor(q * v.length))]);
}
const colorFor = (p) => COLORS[state.breaks.filter((b) => p > b).length];
const byLevel = () => $("color-mode").value === "level";
const fillFor = (props) => (byLevel() ? LEVEL_COLORS[props.level] || "#999" : colorFor(props.prob));

function cellStyle(f) {
  const showObs = $("obs").checked && f.properties.observed;
  const sel = f.properties.cell_id === state.cell;
  return {
    fillColor: fillFor(f.properties), fillOpacity: 0.78,
    color: sel ? "#0b5cff" : showObs ? "#111" : "#6b7280",
    weight: sel ? 3 : showObs ? 2.2 : 0.6,
  };
}
const restyle = () => state.layer && state.layer.setStyle(cellStyle);

function drawLegend() {
  if (state.legend) state.legend.remove();
  state.legend = L.control({ position: "bottomright" });
  state.legend.onAdd = () => {
    const div = L.DomUtil.create("div", "legend");
    if (byLevel()) {
      const thr = state.alerts?.thresholds || {};
      const lv = ["Very High", "High", "Moderate", "Low"];
      div.innerHTML = '<div class="t">Alert level</div>' + lv.map((l) =>
        `<div class="row"><i style="background:${LEVEL_COLORS[l]}"></i>${l}${thr[l] != null ? ` (P ≥ ${thr[l].toFixed(2)})` : ""}</div>`).join("") +
        ($("obs").checked && !state.isForecast ? '<div class="row obs"><i></i>hotspot observed</div>' : "");
      return div;
    }
    const e = [Math.min(...state.features.map((f) => f.properties.prob)), ...state.breaks,
      Math.max(...state.features.map((f) => f.properties.prob))];
    div.innerHTML = '<div class="t">P(hotspot, 7 d) · quintiles</div>' +
      COLORS.map((c, i) => `<div class="row"><i style="background:${c}"></i>${e[i].toFixed(3)} – ${e[i + 1].toFixed(3)}</div>`)
        .reverse().join("") +
      ($("obs").checked ? '<div class="row obs"><i></i>hotspot observed</div>' : "");
    return div;
  };
  state.legend.addTo(map);
}

// ---- Data loading ----------------------------------------------------------------------------
async function loadRisk(date) {
  state.date = date;
  $("date").value = date;
  const [geo, alerts] = await Promise.all([
    getJSON(`/api/risk?date=${date}`), getJSON(`/api/alerts?date=${date}`).catch(() => null)]);
  state.features = geo.features;
  state.isForecast = geo.is_forecast;
  state.alerts = alerts;
  $("mode-badge").hidden = !state.isForecast;
  $("obs").disabled = state.isForecast;
  renderAlerts();
  state.breaks = quantileBreaks(geo.features.map((f) => f.properties.prob));
  if (state.layer) state.layer.remove();
  state.layers = {};
  state.layer = L.geoJSON(geo, {
    style: cellStyle,
    onEachFeature: (f, l) => {
      state.layers[f.properties.cell_id] = l;
      l.bindTooltip(() => `${f.properties.cell_id} · ${f.properties.district || ""}<br>` +
        `P=${fmtP(f.properties.prob)} · ${f.properties.level || ""}` +
        (f.properties.observed ? " · ●obs" : ""), { sticky: true, direction: "top" });
      l.on("click", () => selectCell(f.properties.cell_id));
    },
  }).addTo(map);
  if (!state.fitted) {
    map.fitBounds(state.layer.getBounds(), { padding: [12, 12] });
    state.fitted = true;
  }
  drawLegend();
  renderKpis();
  renderRanking();
  writeHash();
  if (state.cell) await selectCell(state.cell);
}

function renderKpis() {
  const p = state.features.map((f) => f.properties);
  const top = [...p].sort((a, b) => b.prob - a.prob).slice(0, 10);
  $("k-cells").textContent = p.length;
  $("k-mean").textContent = (p.reduce((s, x) => s + x.prob, 0) / p.length).toFixed(3);
  $("k-high").textContent = p.filter((x) => x.prob >= 0.5).length;
  $("k-obs").textContent = state.isForecast ? "pending" : p.filter((x) => x.observed).length;
  $("k-hit").textContent = state.isForecast ? "pending" : `${top.filter((x) => x.observed).length}/10`;
}

function renderRanking() {
  const rows = state.features.map((f) => f.properties).sort((a, b) => b.prob - a.prob);
  $("rank-body").innerHTML = rows.map((r, i) =>
    `<tr data-cell="${r.cell_id}" class="${r.cell_id === state.cell ? "sel" : ""}">` +
    `<td>${i + 1}</td><td><span class="swatch" style="background:${fillFor(r)}"></span>${r.cell_id}</td>` +
    `<td class="dist">${r.district || ""}</td><td class="r">${fmtP(r.prob)}</td>` +
    `<td class="r">${r.observed == null ? "–" : r.observed ? "●" : "·"}</td></tr>`).join("");
}

function renderAlerts() {
  const a = state.alerts;
  $("alert-strip").hidden = !a;
  if (!a) return;
  const worst = a.districts[0];
  const elevated = a.districts.filter((d) => a.levels.indexOf(d.level) >= 2);
  const kind = a.is_forecast ? "Outlook" : "Hindcast";
  $("alert-strip").className = `alert-strip ${lvlClass(worst.level)}`;
  $("alert-level").className = `lvl-chip ${lvlClass(worst.level)}`;
  $("alert-level").textContent = worst.level;
  $("alert-text").textContent = elevated.length
    ? `${kind}, week of ${a.date}: ${elevated.length} of ${a.districts.length} districts at High or above (` +
      `${elevated.slice(0, 4).map((d) => d.district).join(", ")}${elevated.length > 4 ? ", …" : ""})`
    : `${kind}, week of ${a.date}: no district above Moderate.`;
  $("b-kind").textContent = a.is_forecast ? "LIVE OUTLOOK" : "HINDCAST · archived inputs";
  $("b-title").textContent = `Forest fire danger · week of ${a.date}`;
  $("b-cap").href = `/api/alerts/cap?date=${a.date}`;
  $("b-cap").setAttribute("download", `agnidrishti-cap-${a.date}.xml`);
  $("b-body").innerHTML = a.districts.map((d) =>
    `<tr data-cell="${d.cells[0]}"><td>${d.district}</td><td><span class="lvl-chip ${lvlClass(d.level)}">${d.level}</span></td>` +
    `<td class="r">${fmtP(d.max_prob)}</td><td class="r">${d.n_elevated}/${d.n_cells}</td></tr>`).join("");
  const present = [...new Set(a.districts.map((d) => d.level))]
    .sort((x, y) => a.levels.indexOf(y) - a.levels.indexOf(x));
  $("b-advice").innerHTML = present.map((l) =>
    `<div class="advice ${lvlClass(l)}"><b>${l}</b><p>${a.advice[l]}</p></div>`).join("");
  $("b-thr").textContent = Object.entries(a.thresholds).map(([k, v]) => `${v.toFixed(2)} ${k}`).join(" · ");
}

async function selectCell(cellId) {
  state.cell = cellId;
  restyle();
  renderRanking();
  writeHash();
  showTab("cell");
  const [exp, hist] = await Promise.all([
    getJSON(`/api/cells/${cellId}/explain?date=${state.date}`),
    getJSON(`/api/cells/${cellId}/history?date=${state.date}`),
  ]);
  const props = state.features.find((f) => f.properties.cell_id === cellId)?.properties;
  const center = state.layers[cellId]?.getBounds().getCenter();
  $("cell-empty").hidden = true;
  $("cell").hidden = false;
  $("cell-id").textContent = cellId;
  $("cell-loc").textContent = center ? `${center.lat.toFixed(3)}°N ${center.lng.toFixed(3)}°E · week of ${state.date}` : "";
  $("cell-prob").textContent = fmtP(exp.prob);
  $("cell-obs").innerHTML = `<span class="lvl-chip ${lvlClass(props?.level)}">${props?.level || ""}</span> ` +
    (props?.observed == null ? "outcome pending" : props.observed ? "hotspot observed" : "no hotspot observed");
  $("cell-loc").textContent = `${props?.district || ""} · ${$("cell-loc").textContent}`;
  drawShap(exp.contributions);
  drawHistory(hist.rows);
  const now = hist.rows[hist.rows.length - 1];
  $("cell-feats").innerHTML = now ? Object.entries(now.features).map(([k, v]) =>
    `<tr><td>${label(k)}</td><td>${fmtNum(v)} ${(FEATURES[k] || [, ""])[1]}</td></tr>`).join("") : "";
}

function drawShap(c) {
  state.charts.shap?.destroy();
  state.charts.shap = new Chart($("shap-chart"), {
    type: "bar",
    data: {
      labels: c.map((x) => [label(x.feature), `= ${fmtNum(x.value)} ${(FEATURES[x.feature] || [, ""])[1]}`]),
      datasets: [{ data: c.map((x) => x.contribution), barThickness: 14,
        backgroundColor: c.map((x) => (x.contribution > 0 ? "#b91c1c" : "#1d4ed8")) }],
    },
    options: {
      indexAxis: "y", maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (t) => `${t.raw > 0 ? "+" : ""}${t.raw.toFixed(3)} log-odds` } } },
      scales: { x: { title: { display: true, text: "contribution (log-odds)  − lowers · raises +" } },
        y: { grid: { display: false }, ticks: { autoSkip: false, color: "#16181d" } } },
    },
  });
}

function drawHistory(rows) {
  state.charts.hist?.destroy();
  state.charts.hist = new Chart($("history-chart"), {
    data: {
      labels: rows.map((r) => r.week_start.slice(5)),
      datasets: [
        { type: "line", label: "P", data: rows.map((r) => r.prob), borderColor: "#c2410c", backgroundColor: "rgba(194,65,12,.08)", fill: "origin", borderWidth: 1.5, pointRadius: 2, yAxisID: "y" },
        { type: "line", label: "hotspot observed", data: rows.map((r) => (r.y ? 0.03 : null)), showLine: false, pointStyle: "rect", pointRadius: 4, borderColor: "#111", backgroundColor: "#111", yAxisID: "y" },
        { type: "line", label: "FWI 7-d", data: rows.map((r) => r.features.fwi_7d), borderColor: "#1d4ed8", borderWidth: 1, borderDash: [4, 3], pointRadius: 0, yAxisID: "y2" },
      ],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, boxHeight: 6 } } },
      scales: { y: { min: 0, max: 1, title: { display: true, text: "P" } },
        y2: { position: "right", grid: { display: false }, title: { display: true, text: "FWI" } },
        x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 6 } } },
    },
  });
}

async function loadModel() {
  const m = await getJSON("/api/model");
  const entries = Object.entries(m.metrics);
  const best = Math.max(...entries.map(([, s]) => s.test.pr_auc));
  $("metrics").innerHTML = "<thead><tr><th>Model</th><th class=r>PR-AUC</th><th class=r>ROC-AUC</th><th class=r>Brier</th></tr></thead><tbody>" +
    entries.map(([k, s]) => `<tr class="${s.test.pr_auc === best ? "best" : ""}"><td>${MODEL_NAMES[k] || k}</td>` +
      `<td class=r>${s.test.pr_auc.toFixed(3)}</td><td class=r>${s.test.roc_auc.toFixed(3)}</td><td class=r>${s.test.brier.toFixed(3)}</td></tr>`).join("") +
    `</tbody>`;
  const t = entries[0][1].test;
  const hp = m.hyperparameters;
  const cfg = {
    Version: m.version, Trained: (m.trained_at || "").replace("T", " ").slice(0, 16) + " UTC",
    Train: m.training_window.train.join("–"), Validation: m.training_window.val.join("–"),
    Test: m.training_window.test.join("–"), "Test rows / base rate": `${t.n} / ${t.positive_rate.toFixed(3)}`,
    Features: m.features.length, "max_depth": hp.max_depth, "learning_rate": hp.learning_rate,
    "n_estimators": hp.n_estimators, "min_child_weight": hp.min_child_weight, "scale_pos_weight": hp.scale_pos_weight,
  };
  $("model-cfg").innerHTML = Object.entries(cfg).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  const x = m.metrics.xgboost?.test;
  $("k-model").textContent = `${m.version} · test PR-AUC ${x ? x.pr_auc.toFixed(3) : "–"} · ROC-AUC ${x ? x.roc_auc.toFixed(3) : "–"}`;
}

// ---- UI wiring -------------------------------------------------------------------------------
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.id === `tab-${name}`));
}
function step(delta) {
  const i = state.dates.indexOf(state.date) + delta;
  if (i >= 0 && i < state.dates.length) loadRisk(state.dates[i]);
}
function writeHash() {
  const h = new URLSearchParams({ date: state.date || "" });
  if (state.cell) h.set("cell", state.cell);
  history.replaceState(null, "", `#${h}`);
}

async function init() {
  const { dates, forecast_dates } = await getJSON("/api/dates");
  state.dates = dates;
  state.forecastDates = forecast_dates || [];
  const hind = dates.filter((d) => !state.forecastDates.includes(d));
  $("date").innerHTML = (state.forecastDates.length
    ? `<optgroup label="Live outlook">${state.forecastDates.map((d) => `<option>${d}</option>`).join("")}</optgroup>`
    : "") + `<optgroup label="Hindcast">${hind.map((d) => `<option>${d}</option>`).join("")}</optgroup>`;
  const hash = new URLSearchParams(location.hash.slice(1));
  state.cell = hash.get("cell");
  // Default to the week with the highest mean risk in the latest year (peak fire season).
  let start = dates.includes(hash.get("date")) ? hash.get("date") : null;
  if (!start && state.forecastDates.length) start = state.forecastDates[state.forecastDates.length - 1];
  if (!start) {
    const latest = dates.filter((d) => d.startsWith(dates[dates.length - 1].slice(0, 4)));
    start = latest.find((d) => d.slice(5, 7) === "04") || latest[latest.length - 1];
  }
  $("date").addEventListener("change", (e) => loadRisk(e.target.value));
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("obs").addEventListener("change", () => { restyle(); drawLegend(); });
  $("color-mode").addEventListener("change", () => { restyle(); drawLegend(); renderRanking(); });
  $("alert-open").addEventListener("click", () => showTab("alerts"));
  $("b-body").addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    selectCell(tr.dataset.cell);
    map.fitBounds(state.layers[tr.dataset.cell].getBounds(), { maxZoom: 9 });
  });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("rank-body").addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    selectCell(tr.dataset.cell);
    map.fitBounds(state.layers[tr.dataset.cell].getBounds(), { maxZoom: 9 });
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });
  await Promise.all([loadRisk(start), loadModel()]);
}

init().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="padding:8px 16px;background:#7f1d1d;color:#fff;font:12px monospace">Failed to load data: ${e.message}</div>`);
});
