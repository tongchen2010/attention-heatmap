// Attention Heatmap — visualize a real Transformer's self-attention, in the browser.
// A real BERT encoder (all-MiniLM-L6-v2, ONNX-exported *with attentions*) runs locally
// via Transformers.js (WebAssembly). For any sentence we read out all 6 layers x 12 heads
// of self-attention and render them as an interactive heatmap + a BertViz-style arc view.

const REPO_URL = "https://github.com/tongchen2010/attention-heatmap";
const LIB_URL = "https://esm.run/@huggingface/transformers";
const MODEL = "bradynapier/all_miniLM_L6_v2_with_attentions_onnx";
const N_LAYERS = 6, N_HEADS = 12, MAX_TOKENS = 48;

const $ = (id) => document.getElementById(id);
const sentenceEl = $("sentence"), vizBtn = $("viz-btn"),
  statusEl = $("status"), fill = $("progress-fill"), progressWrap = $("progress-wrap"),
  controls = $("controls"), layersEl = $("layers"), headsEl = $("heads"), viewsEl = $("views"),
  vizEl = $("viz"), heatmap = $("heatmap"), matrixWrap = $("matrix-wrap"),
  arcsWrap = $("arcs-wrap"), arcsSvg = $("arcs"), tooltip = $("tooltip"), noteEl = $("note");
$("repo-link").href = REPO_URL;

let T = null, tokenizer = null, model = null;
let tokens = [], rawAtt = [], seq = 0, curM = null;
let curLayer = 1, curHead = 3, curView = "matrix";   // L1/H3 shows clean token-to-token structure by default
const matCache = new Map();

init();

async function init() {
  buildSegs();
  try { T = await import(LIB_URL); }
  catch (e) { return fail("Couldn't load Transformers.js (are you online?)."); }
  try {
    statusEl.textContent = "Loading the tokenizer…";
    tokenizer = await T.AutoTokenizer.from_pretrained(MODEL);
    statusEl.textContent = "Downloading the model (~91 MB, first time only)…";
    model = await T.AutoModel.from_pretrained(MODEL, {
      dtype: "fp32",
      progress_callback: (r) => {
        if (r.status === "progress" && r.progress != null) {
          fill.style.width = r.progress + "%";
          statusEl.textContent = `Downloading ${r.file || "model"}… ${Math.round(r.progress)}%`;
        }
      },
    });
    fill.style.width = "100%";
    ready();
  } catch (e) { console.error(e); fail("Failed to load the model: " + (e?.message || e)); }
}

function ready() {
  progressWrap.style.display = "none";
  sentenceEl.disabled = false; vizBtn.disabled = false;
  document.querySelectorAll(".chip").forEach((c) => {
    c.disabled = false;
    c.addEventListener("click", () => { sentenceEl.value = c.textContent; visualize(); });
  });
  vizBtn.addEventListener("click", visualize);
  sentenceEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); visualize(); }
  });
  // tooltip via delegation (cells are recreated every render)
  heatmap.addEventListener("mousemove", (e) => {
    const c = e.target.closest(".cell");
    if (!c || !curM) return hideTip();
    const q = +c.dataset.q, k = +c.dataset.k;
    showTip(e, q, k, curM[q][k]);
  });
  heatmap.addEventListener("mouseleave", hideTip);
  sentenceEl.focus();
  visualize();   // render the default sentence right away
}

/* ---------- controls ---------- */
function buildSegs() {
  for (let l = 1; l <= N_LAYERS; l++) layersEl.appendChild(seg(l, String(l), () => setLayer(l)));
  headsEl.appendChild(seg("avg", "Avg", () => setHead("avg")));
  for (let h = 1; h <= N_HEADS; h++) headsEl.appendChild(seg(h, String(h), () => setHead(h)));
  viewsEl.querySelectorAll("[data-view]").forEach((b) =>
    b.addEventListener("click", () => setView(b.dataset.view)));
  markActive();
}
function seg(val, label, onClick) {
  const b = document.createElement("button");
  b.className = "seg-btn"; b.textContent = label; b.dataset.val = String(val);
  b.addEventListener("click", onClick);
  return b;
}
function markActive() {
  [...layersEl.children].forEach((b) => b.classList.toggle("active", b.dataset.val === String(curLayer)));
  [...headsEl.children].forEach((b) => b.classList.toggle("active", b.dataset.val === String(curHead)));
  [...viewsEl.children].forEach((b) => b.classList.toggle("active", b.dataset.view === curView));
}
function setLayer(l) { curLayer = l; markActive(); render(); }
function setHead(h) { curHead = h; markActive(); render(); }
function setView(v) {
  curView = v; markActive();
  matrixWrap.hidden = v !== "matrix";
  arcsWrap.hidden = v !== "arcs";
  render();
}

/* ---------- inference ---------- */
async function visualize() {
  const text = sentenceEl.value.trim();
  if (!text || !model) return;
  vizBtn.disabled = true; vizBtn.textContent = "Running…";
  try {
    const inputs = await tokenizer(text);
    const out = await model(inputs);
    rawAtt = [];
    for (let l = 1; l <= N_LAYERS; l++) {
      const t = out["attention_" + l];
      if (!t) throw new Error("attention_" + l + " not found in model output");
      rawAtt.push({ data: t.data, dims: t.dims });   // dims = [1, heads, S, S]
    }
    const ids = Array.from(inputs.input_ids.data).map(Number);
    tokens = idsToTokens(ids);
    seq = tokens.length;
    let truncated = false;
    if (seq > MAX_TOKENS) { tokens = tokens.slice(0, MAX_TOKENS); seq = MAX_TOKENS; truncated = true; }
    noteEl.hidden = !truncated;
    if (truncated) noteEl.textContent = `Showing the first ${MAX_TOKENS} tokens (sentence was longer).`;
    matCache.clear();
    controls.hidden = false; vizEl.hidden = false;
    render();
  } catch (e) {
    console.error(e); fail("Inference failed: " + (e?.message || e));
  } finally {
    vizBtn.disabled = false; vizBtn.textContent = "Visualize";
  }
}

function idsToTokens(ids) {
  try {
    if (typeof tokenizer.convert_ids_to_tokens === "function") {
      const r = tokenizer.convert_ids_to_tokens(ids);
      if (r && r.length === ids.length) return r;
    }
  } catch (e) { /* fall through */ }
  return ids.map((id) => tokenizer.decode([id]));
}

// attention matrix for a (layer, head); head === "avg" averages all heads. Cached.
function matrix(layer, head) {
  const key = layer + ":" + head;
  if (matCache.has(key)) return matCache.get(key);
  const { data, dims } = rawAtt[layer - 1];
  const H = dims[1], S = dims[3], n = seq;
  const M = Array.from({ length: n }, () => new Float64Array(n));
  if (head === "avg") {
    for (let h = 0; h < H; h++) {
      const base = h * S * S;
      for (let q = 0; q < n; q++) for (let k = 0; k < n; k++) M[q][k] += data[base + q * S + k] / H;
    }
  } else {
    const base = (head - 1) * S * S;
    for (let q = 0; q < n; q++) for (let k = 0; k < n; k++) M[q][k] = data[base + q * S + k];
  }
  matCache.set(key, M);
  return M;
}

function maxOf(M) {
  let m = 0;
  for (let q = 0; q < M.length; q++) for (let k = 0; k < M.length; k++) if (M[q][k] > m) m = M[q][k];
  return m || 1;
}

/* ---------- rendering ---------- */
function render() {
  if (!tokens.length) return;
  curM = matrix(curLayer, curHead);
  if (curView === "matrix") renderMatrix(); else renderArcs();
}

function isSpecial(t) { return t === "[CLS]" || t === "[SEP]" || t === "[PAD]"; }

function renderMatrix() {
  const M = curM, n = seq;
  // normalize each row to its own strongest target, so every token's attention
  // pattern is visible regardless of absolute scale (true weight stays in the tooltip).
  const rowMax = new Array(n);
  for (let q = 0; q < n; q++) { let m = 0; for (let k = 0; k < n; k++) if (M[q][k] > m) m = M[q][k]; rowMax[q] = m || 1; }
  const cell = n > 30 ? 15 : n > 20 ? 19 : 25;
  heatmap.style.setProperty("--cell", cell + "px");
  heatmap.style.gridTemplateColumns = `var(--rowlab) repeat(${n}, var(--cell))`;
  const frag = document.createDocumentFragment();

  const corner = document.createElement("div");
  corner.className = "corner"; corner.textContent = "from ↓ / to →";
  frag.appendChild(corner);

  for (let k = 0; k < n; k++) {
    const d = document.createElement("div");
    d.className = "clab" + (isSpecial(tokens[k]) ? " special" : "");
    const s = document.createElement("span"); s.textContent = tokens[k]; d.appendChild(s);
    d.title = tokens[k];
    d.addEventListener("mouseenter", () => highlight("k", k));
    d.addEventListener("mouseleave", clearHL);
    frag.appendChild(d);
  }

  for (let q = 0; q < n; q++) {
    const rl = document.createElement("div");
    rl.className = "rlab" + (isSpecial(tokens[q]) ? " special" : "");
    rl.textContent = tokens[q]; rl.title = tokens[q];
    rl.addEventListener("mouseenter", () => highlight("q", q));
    rl.addEventListener("mouseleave", clearHL);
    frag.appendChild(rl);
    for (let k = 0; k < n; k++) {
      const c = document.createElement("div");
      c.className = "cell"; c.dataset.q = q; c.dataset.k = k;
      c.style.background = heat(M[q][k] / rowMax[q]);
      frag.appendChild(c);
    }
  }
  heatmap.innerHTML = "";
  heatmap.appendChild(frag);
}

// faint -> accent (blue) -> accent-2 (violet)
function heat(a) {
  a = Math.max(0, Math.min(1, Math.pow(a, 0.7)));
  const r = Math.round(110 + (139 - 110) * a);
  const g = Math.round(168 + (124 - 168) * a);
  const b = 255;
  return `rgba(${r},${g},${b},${(0.06 + 0.94 * a).toFixed(3)})`;
}

function highlight(axis, idx) {
  heatmap.classList.add("dim");
  heatmap.querySelectorAll(".cell").forEach((c) => {
    const match = axis === "q" ? +c.dataset.q === idx : +c.dataset.k === idx;
    if (match) c.classList.add("hl");
  });
}
function clearHL() {
  heatmap.classList.remove("dim");
  heatmap.querySelectorAll(".cell.hl").forEach((c) => c.classList.remove("hl"));
}

function renderArcs() {
  const M = curM, n = seq, max = maxOf(M);
  const rowH = n > 24 ? 20 : 26, padY = 22;
  const w = Math.max(360, arcsWrap.clientWidth || 700);
  const h = padY * 2 + (n - 1) * rowH;
  const leftX = 128, rightX = w - 128;
  const y = (i) => padY + i * rowH;
  let s = "";
  for (let q = 0; q < n; q++) {
    for (let k = 0; k < n; k++) {
      const v = M[q][k] / max;
      if (v < 0.12) continue;
      const mid = (leftX + rightX) / 2;
      s += `<path class="arc" data-q="${q}" d="M ${leftX} ${y(q)} C ${mid} ${y(q)}, ${mid} ${y(k)}, ${rightX} ${y(k)}" stroke-opacity="${(v * 0.85).toFixed(3)}" stroke-width="${(0.6 + 2.6 * v).toFixed(2)}"/>`;
    }
  }
  for (let i = 0; i < n; i++) {
    const sp = isSpecial(tokens[i]) ? " special" : "";
    const t = esc(tokens[i]);
    s += `<text class="ltok${sp}" x="${leftX - 10}" y="${y(i)}" text-anchor="end" data-q="${i}">${t}</text>`;
    s += `<text class="rtok${sp}" x="${rightX + 10}" y="${y(i)}" text-anchor="start">${t}</text>`;
  }
  arcsSvg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  arcsSvg.setAttribute("height", h);
  arcsSvg.innerHTML = s;
  arcsSvg.querySelectorAll(".ltok").forEach((node) => {
    node.addEventListener("mouseenter", () => {
      arcsSvg.classList.add("focus");
      arcsSvg.querySelectorAll(".arc").forEach((a) => {
        if (a.dataset.q === node.dataset.q) a.classList.add("on");
      });
    });
    node.addEventListener("mouseleave", () => {
      arcsSvg.classList.remove("focus");
      arcsSvg.querySelectorAll(".arc.on").forEach((a) => a.classList.remove("on"));
    });
  });
}

/* ---------- tooltip ---------- */
function showTip(e, q, k, w) {
  tooltip.innerHTML = `<b>${esc(tokens[q])}</b> &rarr; <b>${esc(tokens[k])}</b><span>weight ${w.toFixed(3)}</span>`;
  tooltip.style.display = "block";
  tooltip.style.left = (e.clientX + 14) + "px";
  tooltip.style.top = (e.clientY + 16) + "px";
}
function hideTip() { tooltip.style.display = "none"; }

function fail(msg) { statusEl.textContent = msg; statusEl.classList.add("err"); progressWrap.style.display = ""; }
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
