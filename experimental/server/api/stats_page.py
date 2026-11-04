# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The page served at /stats: charts of the per-request timings from /api/stats."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Request stats</title>
<style>
:root { --bg: #f4f5f6; --panel: #ffffff; --fg: #1a1e22; --muted: #5f6a73; --line: #d9dee3; --grid: #e7eaed; --accent: #2b6cd4; --warm: #c96f1d; --bad: #bb3a2d; --s1: #2b6cd4; --s2: #c96f1d; --s3: #7f8c99; --s4: #8fb3ec; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0f1113; --panel: #171a1d; --fg: #e7e9eb; --muted: #8b949c; --line: #2a2f34; --grid: #22272b; --accent: #5b9cf2; --warm: #e0a152; --bad: #e8695b; --s1: #5b9cf2; --s2: #e0a152; --s3: #6f7b86; --s4: #35527a; color-scheme: dark; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 20px max(16px, env(safe-area-inset-right)) 40px max(16px, env(safe-area-inset-left)); display: flex; flex-direction: column; gap: 16px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; justify-content: space-between; }
h1 { margin: 0; font-size: 22px; font-weight: 600; }
.sub { color: var(--muted); margin: 2px 0 0; }
.controls { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; color: var(--muted); }
select, button { font: inherit; color: var(--fg); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; }
button { cursor: pointer; }
button:hover, select:hover { border-color: var(--muted); }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile, .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; min-width: 0; }
.tile .k { color: var(--muted); font-size: 12px; margin: 0; }
.tile .v { font-size: 26px; font-weight: 500; margin: 2px 0 0; font-variant-numeric: tabular-nums; }
.tile .n { color: var(--muted); font-size: 12px; margin: 2px 0 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr)); gap: 12px; }
.card h2 { margin: 0; font-size: 15px; font-weight: 600; }
.card .n { color: var(--muted); font-size: 12px; margin: 2px 0 8px; }
.chart { position: relative; }
.chart svg { display: block; width: 100%; height: auto; }
.tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 5px 9px; font-size: 12px; line-height: 1.4; white-space: nowrap; box-shadow: 0 4px 14px rgba(0,0,0,.18); font-variant-numeric: tabular-nums; z-index: 2; }
.tip span { color: var(--muted); }
.empty { color: var(--muted); padding: 40px 0; text-align: center; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; font-size: 12px; position: sticky; top: 0; background: var(--panel); }
th:first-child, td:first-child { text-align: left; }
td.bad { color: var(--bad); }
.bar { display: flex; height: 24px; border-radius: 6px; overflow: hidden; background: var(--grid); margin: 6px 0 10px; }
.bar > div { min-width: 0; }
.bar > div + div { border-left: 2px solid var(--panel); }
.legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 4px 16px; font-size: 13px; }
.legend div { display: flex; align-items: baseline; gap: 6px; min-width: 0; }
.legend i { width: 10px; height: 10px; border-radius: 3px; flex: none; align-self: center; }
.legend b { font-weight: 500; font-variant-numeric: tabular-nums; }
.legend span { color: var(--muted); font-variant-numeric: tabular-nums; }
.mem { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 12px; }
.mem .wide { grid-column: 1 / -1; }
.warn { color: var(--bad); }
.soft { background: var(--panel); }
#msg { color: var(--muted); }
.disabled { padding: 24px; }
.disabled code { background: var(--grid); padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Request stats</h1>
      <p class="sub" id="sub">Loading</p>
    </div>
    <div class="controls">
      <label>Show <select id="win"><option value="50">last 50</option><option value="200" selected>last 200</option><option value="0">all kept</option></select></label>
      <label><input type="checkbox" id="auto" checked> refresh every 5 s</label>
      <button id="refresh" type="button">Refresh</button>
      <button id="reset" type="button">Clear stats</button>
      <span id="msg" role="status"></span>
    </div>
  </header>
  <section class="mem" id="memory" hidden>
    <div class="card wide"><h2>GPU memory</h2><p class="n" id="gpu-n">Read from the driver on every refresh.</p><div id="gpu"></div></div>
    <div class="card"><h2>KV cache</h2><p class="n">Tokens. The cache maps memory as requests need it; pages hold the active request and the retained prefixes.</p><div id="kv"></div></div>
    <div class="card"><h2>Expert cache</h2><p class="n">Experts per layer held on the GPU, out of all experts of a layer.</p><div id="experts"></div></div>
    <div class="card"><h2>Host memory</h2><p class="n">System RAM. Experts, the embedding table and the reuse snapshots live here.</p><div id="host"></div></div>
    <div class="card"><h2>Reuse snapshots</h2><p class="n">Snapshots of the recurrent state and of partial KV that let a hybrid model resume from a cached prefix.</p><div id="snaps"></div></div>
  </section>
  <div id="off" class="card disabled" hidden>
    Per-request stats are off. Start the server with <code>EDGELLM_REQUEST_STATS=1</code> (ring size <code>EDGELLM_REQUEST_STATS_CAPACITY</code>, default 500) and reload this page.
  </div>
  <div id="content" hidden>
    <section class="tiles" id="tiles"></section>
    <section class="grid" style="margin-top:12px">
      <div class="card"><h2>Time to first token by new tokens</h2><p class="n">New (uncached) prompt tokens, log scale. Both axes are log scale. Small requests sit on a floor that grows with the size of the context. Streamed requests only.</p><div class="chart" id="c-ttft"></div></div>
      <div class="card"><h2>Prefill speed by new tokens</h2><p class="n">Requests with 256+ uncached prompt tokens, log scale. The rate includes the cache restore and the first decode step.</p><div class="chart" id="c-prefill"></div></div>
      <div class="card"><h2>Decode speed by context size</h2><p class="n">Requests with 30+ generated tokens after the first. Horizontal: prompt tokens in the context.</p><div class="chart" id="c-decode"></div></div>
      <div class="card"><h2>Time to first token over the requests</h2><p class="n">Oldest on the left, log scale. Each dot is one streamed request.</p><div class="chart" id="c-ttft-t"></div></div>
      <div class="card"><h2>Decode speed over the requests</h2><p class="n">Oldest on the left. Requests with 30+ generated tokens.</p><div class="chart" id="c-decode-t"></div></div>
      <div class="card"><h2>Share of the prompt served from reuse</h2><p class="n">Per request, oldest on the left. Left out when another request ran at the same time.</p><div class="chart" id="c-reuse"></div></div>
    </section>
    <section class="card" style="margin-top:12px">
      <h2>Latest requests</h2><p class="n">Newest first, up to 40.</p>
      <div class="tablewrap"><table id="tbl"><thead><tr><th>#</th><th>status</th><th>prompt</th><th>reused</th><th>new</th><th>generated</th><th>first token</th><th>decode</th><th>prefill</th><th>total</th></tr></thead><tbody></tbody></table></div>
    </section>
    <p class="n" style="margin-top:12px">Decode is generated tokens after the first divided by the time after the first token. Prefill is only given for 256+ uncached prompt tokens; the time to first token also includes work that grows with the context size, so the rate understates the speed on small turns. Time to first token and the rates exist for streamed requests only. Reuse comes from the context-cache counters.</p>
  </div>
</main>
<script>
"use strict";
const $ = (s) => document.querySelector(s);
const nf = new Intl.NumberFormat("en");
const f0 = (v) => (v == null ? "—" : nf.format(Math.round(v)));
const f1 = (v) => (v == null ? "—" : nf.format(Math.round(v * 10) / 10));
const fs = (ms) => (ms == null ? "—" : (ms / 1000).toFixed(2) + " s");
const compact = (v) => (v >= 1e6 ? v / 1e6 + "M" : v >= 1000 ? v / 1000 + "k" : String(v));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
let data = null, timer = null, resetArmed = 0;

function quantile(values, q) {
  const s = values.filter((v) => v != null).sort((a, b) => a - b);
  if (!s.length) return null;
  const i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}
const sum = (a) => a.reduce((x, y) => x + y, 0);

function niceTicks(lo, hi, count) {
  const span = hi - lo || 1, raw = span / count, pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(parseFloat((Math.round(v / step) * step).toPrecision(10)));
  return out;
}

function plot(el, rows, o) {
  const pts = rows.filter(o.keep).map((r) => ({ x: o.x(r), y: o.y(r), r })).filter((p) => p.x != null && p.y != null);
  if (!pts.length) { el.innerHTML = '<div class="empty">No requests match yet</div>'; return; }
  const W = Math.max(300, el.clientWidth || 520), H = 260, m = { t: 10, r: 14, b: 40, l: 54 };
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  let x0 = o.log ? Math.min(...xs) * 0.8 : Math.min(0, ...xs), x1 = Math.max(...xs) * (o.log ? 1.25 : 1.04);
  if (x1 <= x0) x1 = x0 + 1;
  const y0 = o.ylog ? Math.min(...ys) * 0.8 : 0, y1 = Math.max(...ys) * (o.ylog ? 1.25 : 1.1) || 1;
  const sx = o.log ? (v) => m.l + ((Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))) * (W - m.l - m.r) : (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = o.ylog ? (v) => H - m.b - ((Math.log(v) - Math.log(y0)) / (Math.log(y1) - Math.log(y0))) * (H - m.t - m.b) : (v) => H - m.b - (v / y1) * (H - m.t - m.b);
  const yTicks = o.ylog ? [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50].filter((t) => t >= y0 && t <= y1) : niceTicks(0, y1, 5);
  const xTicks = o.log ? [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000].filter((t) => t >= x0 && t <= x1) : niceTicks(x0, x1, 6);
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", o.title);
  const add = (tag, attrs, text) => { const n = document.createElementNS(ns, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text != null) n.textContent = text; svg.appendChild(n); return n; };
  const muted = css("--muted"), grid = css("--grid"), line = css("--line"), color = css(o.color || "--accent");
  for (const t of yTicks) { add("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), stroke: grid }); add("text", { x: m.l - 8, y: sy(t) + 4, "text-anchor": "end", fill: muted, "font-size": 11 }, o.yfmt ? o.yfmt(t) : compact(t)); }
  add("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: line });
  for (const t of xTicks) { add("line", { x1: sx(t), x2: sx(t), y1: H - m.b, y2: H - m.b + 4, stroke: line }); add("text", { x: sx(t), y: H - m.b + 17, "text-anchor": "middle", fill: muted, "font-size": 11 }, compact(t)); }
  add("text", { x: (m.l + W - m.r) / 2, y: H - 5, "text-anchor": "middle", fill: muted, "font-size": 11 }, o.xlabel);
  add("text", { transform: `translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)`, "text-anchor": "middle", fill: muted, "font-size": 11 }, o.ylabel);
  if (o.connect) add("polyline", { points: pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" "), fill: "none", stroke: color, "stroke-width": 1.5, "stroke-linejoin": "round", opacity: 0.5 });
  const dots = pts.map((p) => add("circle", { cx: sx(p.x), cy: sy(p.y), r: 3.6, fill: color, "fill-opacity": 0.85, stroke: css("--panel"), "stroke-width": 1.2 }));
  const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true;
  svg.addEventListener("pointermove", (ev) => {
    const b = svg.getBoundingClientRect(), k = W / b.width, px = (ev.clientX - b.left) * k, py = (ev.clientY - b.top) * k;
    let best = -1, bd = 26 * 26;
    pts.forEach((p, i) => { const d = (sx(p.x) - px) ** 2 + (sy(p.y) - py) ** 2; if (d < bd) { bd = d; best = i; } });
    dots.forEach((d, i) => d.setAttribute("r", i === best ? 6 : 3.6));
    if (best < 0) { tip.hidden = true; return; }
    tip.replaceChildren(...o.tip(pts[best].r).map(([a, v]) => { const row = document.createElement("div"); const s = document.createElement("span"); s.textContent = a + " "; row.append(s, document.createTextNode(v)); return row; }));
    tip.hidden = false;
    tip.style.left = Math.max(0, Math.min(sx(pts[best].x) / k + 12, el.clientWidth - tip.offsetWidth - 2)) + "px";
    tip.style.top = Math.max(0, sy(pts[best].y) / k - 8 - tip.offsetHeight) + "px";
  });
  svg.addEventListener("pointerleave", () => { tip.hidden = true; dots.forEach((d) => d.setAttribute("r", 3.6)); });
  el.replaceChildren(svg, tip);
}

const base = (r) => [["Request", "#" + r.id], ["Context", f0(r.prompt_tokens) + " tokens"], ["New", f0(r.prompt_tokens - (r.reused_tokens || 0)) + " tokens"]];
const newTokens = (r) => (r.prompt_tokens == null ? null : r.prompt_tokens - (r.reused_tokens || 0));

const mib = (v) => (v >= 1024 ? (v / 1024).toFixed(2) + " GiB" : Math.round(v) + " MiB");
const tok = (v) => nf.format(Math.round(v)) + " tokens";
function stack(el, segs, total, note) {
  const wrap = document.createElement("div");
  const bar = document.createElement("div"); bar.className = "bar"; bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", segs.map((g) => `${g.label} ${g.text}`).join(", "));
  const legend = document.createElement("div"); legend.className = "legend";
  for (const g of segs) {
    if (g.value > 0 && total > 0) { const d = document.createElement("div"); d.style.width = Math.max(0.4, (g.value / total) * 100) + "%"; d.style.background = g.color; d.title = `${g.label}: ${g.text}`; bar.appendChild(d); }
    const row = document.createElement("div"), sw = document.createElement("i"), name = document.createElement("span"), val = document.createElement("b");
    sw.style.background = g.color; name.textContent = g.label; val.textContent = g.text + (total > 0 ? ` (${((g.value / total) * 100).toFixed(1)}%)` : "");
    row.append(sw, val, name); legend.appendChild(row);
  }
  wrap.append(bar, legend);
  if (note) { const p = document.createElement("p"); p.className = "n"; p.style.margin = "8px 0 0"; p.textContent = note.text; if (note.bad) p.className += " warn"; wrap.appendChild(p); }
  el.replaceChildren(wrap);
}
function renderMemory(mem) {
  const sec = $("#memory");
  if (!mem || (!mem.device && !mem.runtime && !mem.host)) { sec.hidden = true; return; }
  sec.hidden = false;
  const d = mem.device, r = mem.runtime, h = mem.host, c = (n) => css(n);
  if (d) {
    if (r) {
      const other = Math.max(0, d.used_mib - r.kv_mapped_mib - r.expert_cache_mib);
      stack($("#gpu"), [
        { label: "KV cache", value: r.kv_mapped_mib, text: mib(r.kv_mapped_mib), color: c("--s1") },
        { label: "Expert cache", value: r.expert_cache_mib, text: mib(r.expert_cache_mib), color: c("--s2") },
        { label: "Everything else in use", value: other, text: mib(other), color: c("--s3") },
        { label: "Free", value: d.free_mib, text: mib(d.free_mib), color: c("--grid") }], d.total_mib,
        { text: `Total ${mib(d.total_mib)}. Everything else is the dense weights, engine workspace, CUDA context and anything other processes hold. KV and expert cache are the runtime's own figures.` });
    } else {
      stack($("#gpu"), [
        { label: "In use", value: d.used_mib, text: mib(d.used_mib), color: c("--s3") },
        { label: "Free", value: d.free_mib, text: mib(d.free_mib), color: c("--grid") }], d.total_mib,
        { text: `Total ${mib(d.total_mib)}. The runtime's own figures appear after the first request.` });
    }
  } else $("#gpu").textContent = "The GPU memory could not be read (nvidia-smi is not available to the server).";
  if (r) {
    const tpp = r.kv_pages_usable > 0 ? r.kv_mapped_tokens / r.kv_pages_usable : 0;
    if (tpp > 0 && r.kv_pages_free >= 0) {
      const used = (r.kv_pages_usable - r.kv_pages_free) * tpp, free = r.kv_pages_free * tpp;
      stack($("#kv"), [
        { label: "Pages in use", value: used, text: tok(used), color: c("--s1") },
        { label: "Mapped, free", value: free, text: tok(free), color: c("--s4") },
        { label: "Not mapped", value: r.kv_capacity_tokens - r.kv_mapped_tokens, text: tok(r.kv_capacity_tokens - r.kv_mapped_tokens), color: c("--grid") }], r.kv_capacity_tokens,
        { text: `${mib(r.kv_mapped_mib)} mapped of ${mib(r.kv_capacity_mib)}; ${f0(r.context_records)} retained prefixes; highest page held ${f0(r.kv_pages_high_water)} of ${f0(r.kv_pages_usable)}.` });
    } else {
      stack($("#kv"), [
        { label: "Mapped", value: r.kv_mapped_tokens, text: tok(r.kv_mapped_tokens), color: c("--s1") },
        { label: "Not mapped", value: r.kv_capacity_tokens - r.kv_mapped_tokens, text: tok(r.kv_capacity_tokens - r.kv_mapped_tokens), color: c("--grid") }], r.kv_capacity_tokens,
        { text: `${mib(r.kv_mapped_mib)} mapped of ${mib(r.kv_capacity_mib)}. Page use is only known with context reuse on.` });
    }
    if (r.expert_layers > 0 && r.expert_max_slots_per_layer > 0) {
      stack($("#experts"), [
        { label: "On the GPU", value: r.expert_slots_per_layer, text: f0(r.expert_slots_per_layer) + " per layer", color: c("--s2") },
        { label: "In host memory only", value: r.expert_max_slots_per_layer - r.expert_slots_per_layer, text: f0(r.expert_max_slots_per_layer - r.expert_slots_per_layer) + " per layer", color: c("--grid") }], r.expert_max_slots_per_layer,
        { text: `${f0(r.expert_layers)} layers with a cache, ${mib(r.expert_slot_mib)} per slot across them, ${mib(r.expert_cache_mib)} in all.` });
    } else $("#experts").textContent = "No expert cache (the model does not stream experts).";
    const snaps = [["Recurrent state", r.recurrent_snapshots_free, r.recurrent_snapshots_capacity, "--s1"], ["Partial KV", r.partial_kv_snapshots_free, r.partial_kv_snapshots_capacity, "--s2"]].filter((x) => x[2] > 0);
    if (snaps.length) {
      const box = document.createElement("div");
      for (const [name, free, cap, col] of snaps) { const holder = document.createElement("div"); const t = document.createElement("div"); stack(t, [{ label: name + " in use", value: cap - free, text: f0(cap - free) + " of " + f0(cap), color: c(col) }, { label: "Free", value: free, text: f0(free), color: c("--grid") }], cap); holder.appendChild(t); box.appendChild(holder); }
      $("#snaps").replaceChildren(box);
    } else $("#snaps").textContent = "Context reuse is off, or the model has no snapshots.";
  } else for (const id of ["#kv", "#experts", "#snaps"]) $(id).textContent = "Available after the first request.";
  if (h) {
    const other = Math.max(0, h.total_mib - h.available_mib - h.server_rss_mib);
    stack($("#host"), [
      { label: "This server", value: h.server_rss_mib, text: mib(h.server_rss_mib), color: c("--s1") },
      { label: "Other processes and cache", value: other, text: mib(other), color: c("--s3") },
      { label: "Available", value: h.available_mib, text: mib(h.available_mib), color: c("--grid") }], h.total_mib,
      { text: `Total ${mib(h.total_mib)}. Page cache counts as in use until it is needed.` });
  } else $("#host").textContent = "Host memory could not be read.";
}

function tile(k, v, n) { const d = document.createElement("div"); d.className = "tile"; for (const [cls, t] of [["k", k], ["v", v], ["n", n]]) { const p = document.createElement("p"); p.className = cls; p.textContent = t; d.appendChild(p); } return d; }

function render() {
  renderMemory(data && data.memory);
  const off = !data || !data.enabled;
  $("#off").hidden = !off; $("#content").hidden = off;
  if (off) { $("#sub").textContent = "Stats are off"; return; }
  const all = data.requests, win = +$("#win").value, rows = win ? all.slice(-win) : all;
  const ok = rows.filter((r) => r.status === "ok"), streamed = ok.filter((r) => r.ttft_ms != null);
  const dec = ok.filter((r) => r.decode_tokens >= 30 && r.decode_tok_s != null), pre = ok.filter((r) => r.prefill_tok_s != null);
  const known = ok.filter((r) => r.reused_tokens != null && r.prompt_tokens);
  const share = known.length ? sum(known.map((r) => r.reused_tokens)) / sum(known.map((r) => r.prompt_tokens)) : null;
  $("#sub").textContent = `${f0(rows.length)} requests shown of ${f0(all.length)} kept (ring of ${f0(data.capacity)}), ${f0(ok.length)} completed, ${f0(streamed.length)} streamed`;
  const t = $("#tiles");
  t.replaceChildren(
    tile("Requests", f0(rows.length), "in the window"),
    tile("First token, median", fs(quantile(streamed.map((r) => r.ttft_ms), 0.5)), "p90 " + fs(quantile(streamed.map((r) => r.ttft_ms), 0.9))),
    tile("Decode, median", f1(quantile(dec.map((r) => r.decode_tok_s), 0.5)) + " tok/s", `${f0(dec.length)} requests with 30+ tokens`),
    tile("Prefill, median", f0(quantile(pre.map((r) => r.prefill_tok_s), 0.5)) + " tok/s", `${f0(pre.length)} requests with 256+ new tokens`),
    tile("Prompt from reuse", share == null ? "—" : (share * 100).toFixed(1) + "%", `${f0(known.length)} requests with a known figure`),
    tile("Longest prompt", f0(Math.max(0, ...ok.map((r) => r.prompt_tokens || 0))), "tokens of context in the window"));
  const idx = new Map(rows.map((r, i) => [r.id, i + 1]));
  plot($("#c-ttft"), streamed, { keep: (r) => newTokens(r) > 0, x: newTokens, y: (r) => r.ttft_ms / 1000, log: true, ylog: true, xlabel: "New (uncached) prompt tokens", ylabel: "First token (s, log)", title: "First token time by new tokens", yfmt: (v) => String(v), tip: (r) => [...base(r), ["First token", fs(r.ttft_ms)]] });
  plot($("#c-prefill"), pre, { keep: () => true, x: newTokens, y: (r) => r.prefill_tok_s, log: true, xlabel: "New (uncached) prompt tokens", ylabel: "Prefill tok/s", title: "Prefill speed by new tokens", tip: (r) => [...base(r), ["Prefill", f0(r.prefill_tok_s) + " tok/s"], ["First token", fs(r.ttft_ms)]] });
  plot($("#c-decode"), dec, { keep: () => true, x: (r) => r.prompt_tokens, y: (r) => r.decode_tok_s, xlabel: "Prompt tokens in the context", ylabel: "Decode tok/s", title: "Decode speed by context size", tip: (r) => [...base(r), ["Decode", f1(r.decode_tok_s) + " tok/s"], ["Generated", f0(r.completion_tokens) + " tokens"]] });
  plot($("#c-ttft-t"), streamed, { keep: () => true, x: (r) => idx.get(r.id), y: (r) => r.ttft_ms / 1000, connect: true, ylog: true, xlabel: "Request, oldest to newest", ylabel: "First token (s, log)", title: "First token time over the requests", yfmt: (v) => String(v), tip: (r) => [...base(r), ["First token", fs(r.ttft_ms)]] });
  plot($("#c-decode-t"), dec, { keep: () => true, x: (r) => idx.get(r.id), y: (r) => r.decode_tok_s, connect: true, color: "--warm", xlabel: "Request, oldest to newest", ylabel: "Decode tok/s", title: "Decode speed over the requests", tip: (r) => [...base(r), ["Decode", f1(r.decode_tok_s) + " tok/s"]] });
  plot($("#c-reuse"), known, { keep: () => true, x: (r) => idx.get(r.id), y: (r) => (r.reused_tokens / r.prompt_tokens) * 100, connect: true, xlabel: "Request, oldest to newest", ylabel: "Reused (%)", title: "Share of the prompt served from reuse", tip: (r) => [...base(r), ["Reused", f1((r.reused_tokens / r.prompt_tokens) * 100) + "%"]] });
  const body = $("#tbl tbody"); body.replaceChildren();
  for (const r of rows.slice(-40).reverse()) {
    const tr = document.createElement("tr");
    [["#" + r.id], [r.status, r.status === "ok" ? "" : "bad"], [f0(r.prompt_tokens)], [f0(r.reused_tokens)], [f0(newTokens(r))], [f0(r.completion_tokens)], [fs(r.ttft_ms)], [r.decode_tok_s == null ? "—" : f1(r.decode_tok_s)], [r.prefill_tok_s == null ? "—" : f0(r.prefill_tok_s)], [fs(r.total_ms)]].forEach(([text, cls]) => { const td = document.createElement("td"); td.textContent = text; if (cls) td.className = cls; tr.appendChild(td); });
    body.appendChild(tr);
  }
}

async function load() {
  try {
    const r = await fetch("/api/stats", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    data = await r.json();
    $("#msg").textContent = "updated " + new Date().toLocaleTimeString();
    render();
  } catch (e) { $("#msg").textContent = "cannot reach the server"; }
}
function schedule() { clearInterval(timer); if ($("#auto").checked) timer = setInterval(load, 5000); }
$("#refresh").addEventListener("click", load);
$("#win").addEventListener("change", render);
$("#auto").addEventListener("change", schedule);
$("#reset").addEventListener("click", async () => {
  const b = $("#reset");
  if (Date.now() - resetArmed > 4000) { resetArmed = Date.now(); b.textContent = "Click again to clear"; setTimeout(() => { if (Date.now() - resetArmed >= 4000) b.textContent = "Clear stats"; }, 4100); return; }
  resetArmed = 0; b.textContent = "Clear stats";
  try { await fetch("/api/stats/reset", { method: "POST" }); } catch (e) { /* shown by the next load */ }
  load();
});
let resizeTimer = null;
new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (data && data.enabled) render(); }, 150); }).observe(document.body);
load(); schedule();
</script>
</body>
</html>
"""
