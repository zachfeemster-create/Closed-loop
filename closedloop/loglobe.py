"""Live log globe: every logged run (nightly, weekly, shared rotation logs, runs you upload or run here) as nodes
on one globe. Datasets run around the globe, runs climb from oldest (bottom) to newest (top). Each node glows by
how that dataset did in that run, and tapping it says in plain words why it passed, broke, or missed."""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime

OUTCOME = {  # order = how good; color for the globe
    "survivor": (4, "#7CFFB2"), "luck": (3, "#FFC857"), "frame": (2, "#FF6B9A"),
    "near": (1, "#6FB7FF"), "miss": (0, "#5d6170"), "error": (-1, "#ff4d4d")}


def _when(s: str) -> datetime:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime(1970, 1, 1)


def normalize(rec: dict, source: str, rid: str | None = None) -> dict | None:
    """Any rotation record (nightly file, shared log run, downloaded run) -> {id, when, source, key, log, bar_pct}."""
    if not isinstance(rec, dict) or not rec.get("log"):
        return None
    st = rec.get("settings", {}) or {}
    return dict(id=rid or rec.get("run_id") or rec.get("ran_at", "?"), when=_when(rec.get("logged_at") or rec.get("ran_at")),
                source=source, key=rec.get("key", {}), log=rec["log"], bar_pct=float(st.get("bar_pct", 99.0)),
                curve=st.get("curve", {}), kind=rec.get("kind", "rotation"))


def local_runs(root: str) -> tuple[list[dict], list[dict]]:
    """Nightly runs (full logs) and weekly summaries saved in the repo by the scheduled jobs."""
    runs, weeks = [], []
    for p in sorted(glob.glob(os.path.join(root, "autorun", "results", "nightly", "*.json")))[-60:]:
        try:
            r = normalize(json.load(open(p)), "nightly", "N-" + os.path.basename(p)[:-5])
            if r:
                runs.append(r)
        except Exception:
            pass
    for p in sorted(glob.glob(os.path.join(root, "autorun", "results", "weekly", "*.json")))[-12:]:
        try:
            weeks.append(json.load(open(p)))
        except Exception:
            pass
    return runs, weeks


def outcome(e: dict) -> str:
    if e.get("error"):
        return "error"
    if e.get("passed"):
        if e.get("fake_arm_passed"):
            return "frame"
        m = e.get("max") or {}
        return "luck" if m.get("broke") else "survivor"
    p, sided = e.get("percentile", 0) or 0, e.get("sided", "up")
    if (sided == "up" and 97 <= p < 99) or (sided == "both" and (0.5 < p <= 1.5 or 98.5 <= p < 99.5)):
        return "near"
    return "miss"


def explain(e: dict, bar_pct: float = 99.0) -> str:
    """One step, in plain words."""
    o, p = outcome(e), e.get("percentile")
    lo, hi = (100 - bar_pct) / 2, 100 - (100 - bar_pct) / 2
    if o == "error":
        return f"Couldn't run: {e.get('error')}."
    if o == "frame":
        return (f"Passed ({p:.1f}th percentile), but a fake sent through the same pipeline passed too, "
                "so the frame itself is suspect. Luck pile.")
    if o == "luck":
        return f"Passed ({p:.1f}th percentile), then broke when Max ran: {', '.join(e['max']['broke'])}. Luck pile."
    if o == "survivor":
        how = "Max couldn't break it" if e.get("max") else "not jinxed"
        return f"Passed at the {p:.1f}th percentile and {how}. Survivor."
    if e.get("sided") == "up":
        lead = "Near miss" if o == "near" else "Looked like its own fakes"
        return f"{lead}: beat {p:.1f}% of its fakes, needed {bar_pct:g}%."
    lead = "Near miss" if o == "near" else "Inside the fakes' normal range"
    return f"{lead}: sat at the {p:.1f}th percentile, needed below {lo:g} or above {hi:g}."


def payload(runs: list[dict]) -> dict:
    """Nodes for the globe: one per (run, dataset)."""
    runs = sorted(runs, key=lambda r: r["when"])
    names = sorted({r["key"].get(e["dataset"], e["dataset"]) for r in runs for e in r["log"]})
    nodes = []
    for ri, r in enumerate(runs):
        by = {}
        for e in r["log"]:
            by.setdefault(r["key"].get(e["dataset"], e["dataset"]), []).append(e)
        for name, steps in by.items():
            best = max(steps, key=lambda e: (OUTCOME[outcome(e)][0], abs((e.get("percentile") or 50) - 50)))
            o = outcome(best)
            nodes.append(dict(run=ri, ds=names.index(name), o=o, c=OUTCOME[o][1],
                              s=round(abs((best.get("percentile") or 50) - 50) / 50, 3),
                              steps=[dict(lens=e.get("lens", "?"), eq=e.get("eq", "?"), o=outcome(e),
                                          why=explain(e, r["bar_pct"])) for e in steps]))
    return dict(names=names, runs=[dict(id=r["id"], when=r["when"].strftime("%Y-%m-%d %H:%M"), source=r["source"])
                                   for r in runs], nodes=nodes, colors={k: v[1] for k, v in OUTCOME.items()})


HTML = r"""
<div id="wrap" style="position:relative;background:radial-gradient(circle at 50% 46%,#13203a 0%,#07080f 70%);border-radius:18px;overflow:hidden;touch-action:none">
<canvas id="c" style="display:block;width:100%;height:460px"></canvas>
<div id="hud" style="position:absolute;left:10px;top:8px;right:10px;color:#c9ccd3;font:12px system-ui;line-height:1.4;pointer-events:none;text-shadow:0 0 6px #000"></div>
<div id="legend" style="position:absolute;left:10px;bottom:10px;color:#c9ccd3;font:11px system-ui;line-height:1.5;pointer-events:none"></div>
<div id="card" style="display:none;position:absolute;left:8px;right:8px;bottom:8px;max-height:62%;overflow:auto;background:rgba(14,18,30,.95);border:1px solid #33405e;border-radius:12px;padding:10px;color:#e6e7ea;font:12.5px system-ui;line-height:1.45;box-shadow:0 0 18px rgba(80,140,255,.3)"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const P = __P__;
const wrap = document.getElementById('wrap'), cv = document.getElementById('c'), H = 460, W = () => wrap.clientWidth;
const renderer = new THREE.WebGLRenderer({canvas: cv, antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, W() / H, .1, 100);
cam.position.set(0, .6, 8.6); cam.lookAt(0, 0, 0);
const world = new THREE.Group(); scene.add(world);
const add = m => (world.add(m), m), ADD = THREE.AdditiveBlending;
const tex = (() => { const s = 64, c = document.createElement('canvas'); c.width = c.height = s; const g = c.getContext('2d');
  const r = g.createRadialGradient(s/2, s/2, 0, s/2, s/2, s/2); r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(.3, 'rgba(255,255,255,.5)'); r.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = r; g.fillRect(0, 0, s, s); return new THREE.CanvasTexture(c); })();
const R = 2.2, ND = Math.max(1, P.names.length), NR = Math.max(1, P.runs.length);
const lon = d => 2 * Math.PI * d / ND, lat = r => NR === 1 ? 0 : -1.15 + 2.3 * r / (NR - 1);
const pt = (rad, la, lo) => new THREE.Vector3(rad * Math.cos(la) * Math.cos(lo), rad * Math.sin(la), rad * Math.cos(la) * Math.sin(lo));
add(new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(R * .98, 24, 16)), new THREE.LineBasicMaterial({color: 0x34507a, transparent: true, opacity: .14})));
add(new THREE.Mesh(new THREE.SphereGeometry(R * .97, 48, 32), new THREE.MeshBasicMaterial({color: 0x4f7bd9, transparent: true, opacity: .05, depthWrite: false})));
const core = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, color: 0x9fc2ff, transparent: true, opacity: .28, blending: ADD, depthWrite: false})); core.scale.set(2.2, 2.2, 1); add(core);
// a meridian per dataset, a ring per run
for (let d = 0; d < ND; d++) { const pts = []; for (let k = 0; k <= 40; k++) pts.push(pt(R, -1.2 + 2.4 * k / 40, lon(d)));
  add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color: 0x6f86b8, transparent: true, opacity: .22}))); }
for (let r = 0; r < NR; r++) { const pts = []; for (let k = 0; k <= 96; k++) pts.push(pt(R, lat(r), 2 * Math.PI * k / 96));
  add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color: 0x6f86b8, transparent: true, opacity: .12}))); }
// nodes
const nodes = P.nodes.map(n => { const good = n.o === 'survivor' || n.o === 'luck' || n.o === 'frame';
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, color: new THREE.Color(n.c), transparent: true, opacity: n.o === 'miss' ? .55 : 1, blending: ADD, depthWrite: false}));
  sp.position.copy(pt(R * 1.01, lat(n.run), lon(n.ds))); const base = good ? .55 : (n.o === 'near' ? .38 : .22 + .12 * n.s);
  sp.scale.set(base, base, 1); sp.userData = {n, base, good}; add(sp); return sp; });
// streak lines: a survivor that shows up again in the next run on the same dataset
const byKey = {}; P.nodes.forEach(n => byKey[n.run + ':' + n.ds] = n);
P.nodes.forEach(n => { const nx = byKey[(n.run + 1) + ':' + n.ds]; if (n.o === 'survivor' && nx && nx.o === 'survivor') {
  const pts = []; for (let k = 0; k <= 12; k++) pts.push(pt(R * 1.02, lat(n.run) + (lat(n.run + 1) - lat(n.run)) * k / 12, lon(n.ds)));
  add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color: 0x7cffb2, transparent: true, opacity: .9, blending: ADD}))); } });

const count = o => P.nodes.filter(n => n.o === o).length;
document.getElementById('hud').innerHTML = `${P.runs.length} run(s) · ${P.names.length} dataset(s)<br>` +
  `<span style="color:${P.colors.survivor}">●</span> ${count('survivor')} survivor · <span style="color:${P.colors.luck}">●</span> ${count('luck')} luck pile · ` +
  `<span style="color:${P.colors.frame}">●</span> ${count('frame')} frame trip · <span style="color:${P.colors.near}">●</span> ${count('near')} near miss · ${count('miss')} miss`;
document.getElementById('legend').textContent = 'bottom = oldest run · top = newest · drag to spin · tap a light';
let drag = null, rotY = .3, rotX = .2, last = performance.now(), ticked = false, fallback = false;
function frame(now) {
  ticked = true; const dt = Math.min(.1, (now - last) / 1000); last = now;
  if (!drag) rotY += dt * .1; world.rotation.set(rotX, rotY, 0);
  nodes.forEach((s, i) => { const u = s.userData, k = u.good ? 1 + .18 * Math.sin(now / 300 + i) : 1; s.scale.set(u.base * k, u.base * k, 1); });
  renderer.render(scene, cam); if (!fallback) requestAnimationFrame(frame);
}
function resize() { renderer.setSize(W(), H, false); cam.aspect = W() / H; cam.updateProjectionMatrix(); }
window.addEventListener('resize', resize); resize();
const card = document.getElementById('card');
cv.addEventListener('pointerdown', e => { drag = {x: e.clientX, y: e.clientY, rx: rotX, ry: rotY, moved: false}; cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointermove', e => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
  rotY = drag.ry + dx * .01; rotX = Math.max(-1.3, Math.min(1.3, drag.rx + dy * .01)); });
cv.addEventListener('pointerup', e => { const tap = drag && !drag.moved; drag = null; if (!tap) return;
  const r = cv.getBoundingClientRect(), m = new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
  const ray = new THREE.Raycaster(); ray.params.Points = {threshold: .1}; ray.setFromCamera(m, cam);
  const hit = ray.intersectObjects(nodes)[0]; if (!hit) { card.style.display = 'none'; return; }
  const n = hit.object.userData.n, run = P.runs[n.run];
  card.style.display = 'block';
  card.innerHTML = `<div style="display:flex;justify-content:space-between"><b>${P.names[n.ds]}</b><span style="color:#8a93a8">${run.source} · ${run.when}</span></div>` +
    n.steps.map(s => `<div style="margin-top:6px"><span style="color:${P.colors[s.o]}">●</span> <b>${s.lens}</b> · ${s.eq}<br><span style="color:#c9ccd3">${s.why}</span></div>`).join('') +
    `<div style="margin-top:8px;color:#7d869c">Run ${run.id}. Tap empty space to close.</div>`; });
requestAnimationFrame(frame);
setTimeout(() => { if (!ticked) { fallback = true; setInterval(() => frame(performance.now()), 33); } }, 900);
</script>
"""


def render(p: dict) -> str:
    from .embed import safe_json
    return HTML.replace("__P__", safe_json(json.dumps(p, default=str)))
