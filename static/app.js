"use strict";

const state = {
  adj: "fastest",
  attentionTemp: 1.0,
  outputTemp: 1.0,
  candidates: ["Redis", "Postgres"],
  candidateClass: { Redis: "redis", Postgres: "postgres" },
  weights: null,          // when set, { W_OUTPUT: {...} } overrides applied to inference
  retrained: false,       // true once the user has re-trained the model
  defaultBeliefs: {},      // adjective -> predicted token for the original frozen model
  lastInfer: null,
};

const ADJECTIVES = ["fastest", "safest", "best"];

const TOKENS_BASE = ["the", "__ADJ__", "database", "is"];

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, {
    method: opts.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${url} -> ${res.status}: ${detail}`);
  }
  return res.json();
}

const post = (url, body) => fetchJSON(url, { method: "POST", body });

function fmt(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (n === 0) return "0.000";
  const abs = Math.abs(n);
  if (abs < 0.001) return n.toExponential(2);
  return n.toFixed(digits);
}

function signed(n, digits = 3) {
  const s = fmt(n, digits);
  return n >= 0 && !s.startsWith("-") ? `+${s}` : s;
}

const pct = (p) => `${(p * 100).toFixed(1)}%`;

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function renderMiniVec(container, values, opts = {}) {
  clearChildren(container);
  values.forEach((v) => {
    const cell = document.createElement("span");
    cell.className = "cell";
    if (opts.dim) cell.classList.add("cell-dim");
    cell.textContent = signed(v, 3);
    if (v > 0) cell.classList.add("pos");
    else if (v < 0) cell.classList.add("neg");
    container.appendChild(cell);
  });
}

function tokensFromAdj() {
  return TOKENS_BASE.map((t) => (t === "__ADJ__" ? state.adj : t));
}

// ----------------------------------------------------------------------------
// Inference — renders the four-stage pipeline
// ----------------------------------------------------------------------------

function renderEmbedGrid(result) {
  const grid = document.getElementById("embed-grid");
  clearChildren(grid);
  result.tokens.forEach((tok, i) => {
    const row = document.createElement("div");
    row.className = "embed-row";
    const label = document.createElement("span");
    label.className = "embed-token";
    label.textContent = tok;
    row.appendChild(label);
    const vec = document.createElement("div");
    vec.className = "mini-vec compact";
    renderMiniVec(vec, result.embeddings[i]);
    row.appendChild(vec);
    grid.appendChild(row);
  });
}

function renderQKV(result) {
  const qVec = document.getElementById("vec-q");
  renderMiniVec(qVec, result.queries[result.queries.length - 1]);

  const kCol = document.getElementById("vecs-k");
  const vCol = document.getElementById("vecs-v");
  clearChildren(kCol);
  clearChildren(vCol);

  result.tokens.forEach((tok, i) => {
    [
      [kCol, result.keys[i]],
      [vCol, result.values[i]],
    ].forEach(([col, vec]) => {
      const row = document.createElement("div");
      row.className = "qkv-row";
      const lab = document.createElement("span");
      lab.className = "qkv-tok";
      lab.textContent = tok;
      row.appendChild(lab);
      const v = document.createElement("div");
      v.className = "mini-vec compact tight";
      renderMiniVec(v, vec, { dim: true });
      row.appendChild(v);
      col.appendChild(row);
    });
  });
}

function renderAttentionViz(result) {
  const viz = document.getElementById("attn-viz");
  clearChildren(viz);
  const max = Math.max(...result.attention_weights, 1e-9);
  result.tokens.forEach((tok, i) => {
    const w = result.attention_weights[i];
    const raw = result.raw_scores[i];
    const row = document.createElement("div");
    row.className = "attn-row";

    const t = document.createElement("span");
    t.className = "attn-tok";
    t.textContent = tok;

    const bar = document.createElement("div");
    bar.className = "attn-bar";
    const fill = document.createElement("div");
    fill.className = "attn-bar-fill";
    fill.style.width = `${(w / max) * 100}%`;
    bar.appendChild(fill);

    const val = document.createElement("span");
    val.className = "attn-val";
    val.textContent = pct(w);

    const score = document.createElement("span");
    score.className = "attn-score";
    score.textContent = `score ${fmt(raw, 2)}`;

    row.append(t, bar, val, score);
    viz.appendChild(row);
  });

  renderMiniVec(document.getElementById("vec-ctx"), result.context);
}

function renderOutput(result) {
  const viz = document.getElementById("output-viz");
  clearChildren(viz);
  result.candidates.forEach((c, i) => {
    const prob = result.probabilities[i];
    const logit = result.logits[i];
    const row = document.createElement("div");
    row.className = "out-row";
    if (c === result.predicted_token) row.classList.add("winner");

    const tok = document.createElement("span");
    tok.className = "out-tok";
    tok.textContent = c;

    const bar = document.createElement("div");
    bar.className = `out-bar ${state.candidateClass[c] || ""}`;
    const fill = document.createElement("div");
    fill.className = "out-bar-fill";
    fill.style.width = `${(prob * 100).toFixed(2)}%`;
    bar.appendChild(fill);

    const val = document.createElement("span");
    val.className = "out-val";
    val.textContent = pct(prob);

    const logitEl = document.createElement("span");
    logitEl.className = "out-logit";
    logitEl.textContent = `logit ${fmt(logit, 2)}`;

    row.append(tok, bar, val, logitEl);
    viz.appendChild(row);
  });

  const predEl = document.getElementById("pred-token");
  if (predEl.textContent !== result.predicted_token) {
    predEl.classList.add("flip");
    setTimeout(() => predEl.classList.remove("flip"), 500);
  }
  predEl.textContent = result.predicted_token;
  predEl.className = `pred-token ${state.candidateClass[result.predicted_token] || ""}`;

  const pTop = Math.max(...result.probabilities);
  const winnerClass = state.candidateClass[result.predicted_token] || "";
  const confEl = document.getElementById("pred-conf");
  if (confEl) confEl.textContent = pct(pTop);
  const meterFill = document.getElementById("pred-meter-fill");
  if (meterFill) {
    meterFill.style.width = `${(pTop * 100).toFixed(1)}%`;
    meterFill.className = `pred-meter-fill ${winnerClass}`;
  }

  // Belief flag: if re-training changed this word's answer vs the original model.
  const flag = document.getElementById("belief-flag");
  if (flag) {
    const original = state.defaultBeliefs[state.adj];
    if (state.retrained && original && original !== result.predicted_token) {
      flag.innerHTML = `belief flipped — was <strong>${original}</strong> before training`;
      flag.className = "belief-flag show";
    } else {
      flag.className = "belief-flag";
      flag.textContent = "";
    }
  }
}

function renderInsight(result) {
  const el = document.getElementById("insight-text");
  const top = result.predicted_token;
  const pTop = Math.max(...result.probabilities);
  const ctx = result.context;
  const dimMax = ctx.reduce(
    (acc, v, i) => (Math.abs(v) > Math.abs(acc.v) ? { i, v } : acc),
    { i: 0, v: ctx[0] }
  );
  const dimNames = ["fast-axis", "safe-axis", "database-axis", "is-axis"];
  const dimLabel = dimNames[dimMax.i] || `dim ${dimMax.i}`;
  el.innerHTML = `Attention concentrates on the adjective and "database"; the resulting context vector loads <strong>${dimLabel}</strong> (${signed(dimMax.v, 2)}). The output row for <strong>${top}</strong> reads that axis, so it wins at ${pct(pTop)}.`;
}

async function refreshInference() {
  try {
    const overrides = state.weights && state.weights.W_OUTPUT ? { W_OUTPUT: state.weights.W_OUTPUT } : undefined;
    const result = await post("/api/infer", {
      tokens: tokensFromAdj(),
      attention_temperature: state.attentionTemp,
      output_temperature: state.outputTemp,
      weight_overrides: overrides,
    });
    state.lastInfer = result;
    renderEmbedGrid(result);
    renderQKV(result);
    renderAttentionViz(result);
    renderOutput(result);
    renderInsight(result);
  } catch (err) {
    console.error(err);
  }
}

// ----------------------------------------------------------------------------
// Wiring: prompt word picker (Lever 1)
// ----------------------------------------------------------------------------

function wireAdjPicker() {
  const picker = document.getElementById("adj-picker");
  const adjToken = document.getElementById("adj-token");
  picker.querySelectorAll(".adj").forEach((btn) => {
    btn.addEventListener("click", () => {
      picker.querySelectorAll(".adj").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.adj = btn.dataset.adj;
      adjToken.textContent = state.adj;
      adjToken.classList.add("swap");
      setTimeout(() => adjToken.classList.remove("swap"), 360);
      refreshInference();
    });
  });
}

// ----------------------------------------------------------------------------
// Training-run picker — the "over time" lever
//
// No knobs. A handful of fixed datasets, each pushed through gradient descent.
// Selecting one applies its learned output weights to the live model so you
// can watch the SAME prompt flip its answer. Same words, different training.
// ----------------------------------------------------------------------------

const PRESETS = [
  {
    id: "blank",
    name: "Blank slate",
    tagline: "never trained",
    desc: "Output weights are all zero — the model has learned nothing. Every prompt is a 50/50 coin flip.",
    rules: null, // special-cased: zero weights, no chart
  },
  {
    id: "speed",
    name: "Speed-trained",
    tagline: "fastest → Redis",
    answer: "Redis",
    desc: "Fed examples where the fastest database is Redis (and the safest is Postgres). This is the shipped model.",
    rules: [
      { adjective: "fastest", target: "Redis", repetitions: 20 },
      { adjective: "safest", target: "Postgres", repetitions: 20 },
      { adjective: "best", target: "Postgres", repetitions: 20 },
    ],
  },
  {
    id: "flipped",
    name: "Re-trained",
    tagline: "fastest → Postgres",
    answer: "Postgres",
    desc: "Same sentence, opposite dataset: the fastest database is now Postgres. The frozen weights learned a new belief.",
    rules: [
      { adjective: "fastest", target: "Postgres", repetitions: 20 },
      { adjective: "safest", target: "Postgres", repetitions: 20 },
    ],
  },
];

const ZERO_WEIGHTS = { Redis: [0, 0, 0, 0], Postgres: [0, 0, 0, 0] };

let activePreset = "speed";

function currentOverrides() {
  return state.weights && state.weights.W_OUTPUT
    ? { W_OUTPUT: state.weights.W_OUTPUT }
    : undefined;
}

async function predictFor(adj, overrides) {
  const res = await post("/api/infer", {
    tokens: ["the", adj, "database", "is"],
    attention_temperature: 1.0,
    output_temperature: 1.0,
    weight_overrides: overrides || undefined,
  });
  return { token: res.predicted_token, prob: Math.max(...res.probabilities) };
}

async function computeBeliefs(overrides) {
  const out = {};
  await Promise.all(
    ADJECTIVES.map(async (a) => {
      out[a] = await predictFor(a, overrides);
    })
  );
  return out;
}

function wireRetrainModal() {
  const btnOpen = document.getElementById("btn-open-retrain");
  const btnClose = document.getElementById("btn-close-modal");
  const modal = document.getElementById("retrain-modal");

  if (btnOpen) {
    btnOpen.addEventListener("click", () => {
      renderPresets();
      modal.classList.add("active");
    });
  }
  if (btnClose) {
    btnClose.addEventListener("click", () => modal.classList.remove("active"));
  }
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.remove("active");
    });
  }
}

function renderPresets() {
  const root = document.getElementById("preset-grid");
  if (!root) return;
  clearChildren(root);

  PRESETS.forEach((preset) => {
    const card = document.createElement("button");
    card.className = `preset-card ${preset.id === activePreset ? "active" : ""}`;
    card.onclick = () => selectPreset(preset.id);

    const head = document.createElement("div");
    head.className = "preset-head";
    const name = document.createElement("span");
    name.className = "preset-name";
    name.textContent = preset.name;
    const pill = document.createElement("span");
    if (preset.answer) {
      pill.className = `data-answer ${state.candidateClass[preset.answer] || ""}`;
      pill.textContent = preset.tagline;
    } else {
      pill.className = "data-answer undecided";
      pill.textContent = preset.tagline;
    }
    head.append(name, pill);

    const desc = document.createElement("p");
    desc.className = "preset-desc";
    desc.textContent = preset.desc;

    const check = document.createElement("span");
    check.className = "preset-check";
    check.textContent = preset.id === activePreset ? "● live" : "apply";

    card.append(head, desc, check);
    root.appendChild(card);
  });
}

async function selectPreset(id) {
  const preset = PRESETS.find((p) => p.id === id);
  if (!preset) return;
  activePreset = id;
  renderPresets();

  const results = document.getElementById("train-results");
  const chartCard = document.getElementById("train-chart-card");

  try {
    if (preset.rules === null) {
      // Blank slate: zero weights, uniform output, no training to chart.
      state.weights = { W_OUTPUT: ZERO_WEIGHTS };
      state.retrained = false;
      if (chartCard) chartCard.style.display = "none";
    } else {
      const result = await post("/api/train_inline", { rules: preset.rules });
      state.weights = { W_OUTPUT: result.learned_weights };
      state.retrained = id !== "speed";
      if (chartCard) chartCard.style.display = "";
      result._axis = "example";
      renderTrainChart(result);
    }

    const beliefs = await computeBeliefs(currentOverrides());
    renderBeliefReadout(preset, beliefs);
    if (results) results.style.display = "block";

    await refreshInference();
  } catch (err) {
    console.error(err);
  }
}

function renderBeliefReadout(preset, beliefs) {
  const root = document.getElementById("belief-readout");
  if (!root) return;
  const rows = ADJECTIVES.map((a) => {
    const b = beliefs[a];
    const isCoin = b && b.prob < 0.6;
    const ans = isCoin
      ? `<span class="data-answer undecided">coin flip</span>`
      : `<span class="data-answer ${state.candidateClass[b.token] || ""}">${b.token}</span>`;
    const conf = b ? `<span class="br-conf">${pct(b.prob)}</span>` : "";
    return `<div class="br-row">
      <span class="br-phrase">the <strong>${a}</strong> database is →</span>
      ${ans}${conf}
    </div>`;
  }).join("");
  root.innerHTML = `<div class="br-title">Now the model believes:</div>${rows}`;
}

function renderTrainChart(result) {
  const chart = document.getElementById("train-chart");
  clearChildren(chart);

  const W = 560;
  const H = 240;
  const padding = { l: 44, r: 80, t: 18, b: 28 };
  const plotW = W - padding.l - padding.r;
  const plotH = H - padding.t - padding.b;

  const epochs = result.history.length;
  const xs = result.history.map((_, i) => i);
  const losses = result.history.map((h) => h.loss);
  const candIdx = { Redis: 0, Postgres: 1 };
  const probsRedis = result.history.map((h) => h.probabilities[candIdx.Redis] || 0);
  const probsPostgres = result.history.map((h) => h.probabilities[candIdx.Postgres] || 0);
  const maxLoss = Math.max(...losses, 0.7);

  const x = (i) => padding.l + (i / Math.max(epochs - 1, 1)) * plotW;
  const yLoss = (v) => padding.t + (1 - v / maxLoss) * plotH;
  const yProb = (v) => padding.t + (1 - v) * plotH;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "train-svg");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  for (let i = 0; i <= 4; i++) {
    const yy = padding.t + (i / 4) * plotH;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padding.l);
    line.setAttribute("x2", W - padding.r);
    line.setAttribute("y1", yy);
    line.setAttribute("y2", yy);
    line.setAttribute("class", "grid-line");
    svg.appendChild(line);
  }

  const yLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yLabel.setAttribute("x", 8);
  yLabel.setAttribute("y", padding.t + 6);
  yLabel.setAttribute("class", "axis-label");
  yLabel.textContent = "1.0";
  svg.appendChild(yLabel);

  const yLabel0 = document.createElementNS("http://www.w3.org/2000/svg", "text");
  yLabel0.setAttribute("x", 8);
  yLabel0.setAttribute("y", padding.t + plotH);
  yLabel0.setAttribute("class", "axis-label");
  yLabel0.textContent = "0.0";
  svg.appendChild(yLabel0);

  const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xLabel.setAttribute("x", padding.l);
  xLabel.setAttribute("y", H - 8);
  xLabel.setAttribute("class", "axis-label");
  xLabel.textContent = "example 1";
  svg.appendChild(xLabel);

  const xLabel2 = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xLabel2.setAttribute("x", W - padding.r);
  xLabel2.setAttribute("y", H - 8);
  xLabel2.setAttribute("class", "axis-label");
  xLabel2.setAttribute("text-anchor", "end");
  xLabel2.textContent = `example ${epochs}`;
  svg.appendChild(xLabel2);

  const drawPath = (ys, cls, yScale) => {
    let d = "";
    xs.forEach((xi, i) => {
      const cmd = i === 0 ? "M" : "L";
      d += `${cmd}${x(xi).toFixed(2)} ${yScale(ys[i]).toFixed(2)} `;
    });
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d.trim());
    path.setAttribute("class", `train-path ${cls}`);
    path.setAttribute("fill", "none");
    svg.appendChild(path);
  };

  drawPath(probsRedis, "p-redis", yProb);
  drawPath(probsPostgres, "p-postgres", yProb);
  drawPath(losses, "p-loss", yLoss);

  const lastIdx = epochs - 1;
  const markers = [
    { y: yProb(probsRedis[lastIdx]), cls: "redis", label: pct(probsRedis[lastIdx]) },
    { y: yProb(probsPostgres[lastIdx]), cls: "postgres", label: pct(probsPostgres[lastIdx]) },
    { y: yLoss(losses[lastIdx]), cls: "loss", label: fmt(losses[lastIdx], 3) },
  ];
  markers.forEach((m) => {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", x(lastIdx));
    c.setAttribute("cy", m.y);
    c.setAttribute("r", 4);
    c.setAttribute("class", `marker ${m.cls}`);
    svg.appendChild(c);

    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", x(lastIdx) + 8);
    t.setAttribute("y", m.y + 4);
    t.setAttribute("class", `marker-label ${m.cls}`);
    t.textContent = m.label;
    svg.appendChild(t);
  });

  chart.appendChild(svg);

  const milestones = document.createElement("div");
  milestones.className = "train-milestones";
  const pick = [0, Math.floor(epochs / 4), Math.floor(epochs / 2), epochs - 1].filter(
    (v, i, a) => a.indexOf(v) === i && v >= 0 && v < epochs
  );
  pick.forEach((i) => {
    const h = result.history[i];
    const idx = h.step ?? h.epoch ?? i + 1;
    const tile = document.createElement("div");
    tile.className = "milestone";
    tile.innerHTML = `
      <div class="m-epoch">example ${idx}</div>
      <div class="m-loss">loss ${fmt(h.loss, 4)}</div>
      <div class="m-probs">
        <span class="redis">R ${pct(h.probabilities[0])}</span>
        <span class="postgres">P ${pct(h.probabilities[1])}</span>
      </div>`;
    milestones.appendChild(tile);
  });
  chart.appendChild(milestones);
}

// ----------------------------------------------------------------------------
// Bootstrap
// ----------------------------------------------------------------------------

async function bootstrap() {
  await refreshInference();
  // Record the original frozen model's answer for each word, so we can show
  // "belief flipped — was X before training" once the user re-trains.
  try {
    const beliefs = await computeBeliefs(undefined);
    Object.keys(beliefs).forEach((a) => {
      state.defaultBeliefs[a] = beliefs[a].token;
    });
  } catch (err) {
    console.error("belief snapshot failed", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  wireAdjPicker();
  wireRetrainModal();
  bootstrap().catch((err) => console.error("bootstrap failed", err));
});
