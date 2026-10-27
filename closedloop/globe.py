"""Plasma globe (spherical equalizer): the EQ curve wraps around the flat-line ring (ups push out, downs pull in),
glowing shell rings ripple with each band's energy, a second dataset interleaves its rings
with the first like a zipper (teeth light where the same band locks across the two), and any
band that beats its own fakes drops a glowing pin.

A pin is a flag, not a finding: it has to climb the usual ladder (frozen settings, fresh fakes, Max, new data).
Every new knob setting counts as another look, and the bar gets stricter with each look."""
from __future__ import annotations

import json

import numpy as np

from . import rotation as R


def _ds(a: np.ndarray, m: int) -> list:
    """Downsample to at most m points, rounded, for the page."""
    idx = np.unique(np.linspace(0, len(a) - 1, min(len(a), m)).astype(int))
    return [round(float(v), 3) for v in a[idx]]


def _bands_of(X: np.ndarray):
    """Band energy shares and envelopes for a stack (n, k). Returns {band: (share[k], env[n,k], phase[n,k], bins)}."""
    n = X.shape[0]
    t = np.arange(n)
    A = np.column_stack([np.ones(n), t])
    c = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    F = np.fft.fft(c, axis=0)
    f = np.fft.fftfreq(n)
    fr = np.abs(f) * 2 * R.REF_NYQUIST
    pos = f > 0
    total = (np.abs(F[pos]) ** 2).sum(0) + 1e-12
    edges = [0.0] + [np.sqrt(a * b) for a, b in zip(R.EQ_BANDS[:-1], R.EQ_BANDS[1:])] + [R.REF_NYQUIST + 1]
    out = {}
    for i, b in enumerate(R.EQ_BANDS):
        m = pos & (fr >= edges[i]) & (fr < edges[i + 1])
        if not m.any():
            continue
        a = np.fft.ifft(np.where(m[:, None], 2 * F, 0), axis=0)
        out[b] = ((np.abs(F[m]) ** 2).sum(0) / total, np.abs(a), np.angle(a), int(m.sum()))
    return out


def band_check(x, curve: dict, looks: int = 1, n_fakes: int = 300, bar_pct: float = 99.0, seed: int = 27):
    """Each band of the real series vs the same band in momentum-matched fakes, all through the same EQ.
    Bar is corrected for 10 bands and for every look taken. Returns (payload for the globe, dings)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    strict = 100 - (100 - bar_pct) / (len(R.EQ_BANDS) * max(1, looks))
    n_fakes = int(min(4000, max(n_fakes, 3 * 100 / (100 - strict))))    # enough fakes to resolve the stricter bar
    F = R.fakes_for(x, n_fakes, rng)
    Xe = R.apply_eq(np.column_stack([x, F]), curve) if not R.is_flat(curve) else np.column_stack([x, F])
    B = _bands_of(Xe)
    n = len(x)
    win = max(4, n // 10)
    gains = dict(zip(R.EQ_BANDS, R.curve_at(curve)))
    bands, dings = [], []
    top = max((v[0][0] for v in B.values()), default=1.0) + 1e-12
    for b in R.EQ_BANDS:
        if b not in B:
            bands.append(dict(hz=b, live=False, gain=round(gains[b], 2)))
            continue
        share, env, ph, bins = B[b]
        real, fk = float(share[0]), share[1:]
        pct = float((fk < real).mean() * 100)
        lo95, hi95 = np.percentile(fk, [2.5, 97.5])
        e = env[:, 0] / (env[:, 0].max() + 1e-12)
        drift = np.unwrap(ph[:, 0]) - 2 * np.pi * (b / (2 * R.REF_NYQUIST)) * np.arange(n)
        row = dict(hz=b, live=True, gain=round(gains[b], 2), share=round(real, 4), rel=round(real / top, 3),
                   pct=round(pct, 2), env=_ds(e, 160), drift=_ds(drift, 160), bins=bins)
        if pct >= strict and bins >= 2:
            fake_hi = np.percentile(env[:, 1:], 95, axis=1)
            excess = np.convolve(env[:, 0] - fake_hi, np.ones(win) / win, mode="valid")
            s0 = int(np.argmax(excess))
            d = dict(hz=b, pct=round(pct, 2), bar=round(strict, 3), real_share=round(real, 4),
                     fake_range=[round(float(lo95), 4), round(float(hi95), 4)],
                     t0=round(s0 / n, 3), t1=round((s0 + win) / n, 3), rows=[s0, s0 + win], looks=looks,
                     settings=R.curve_json(curve))
            dings.append(d)
            row["ding"] = d
        bands.append(row)
    # zipper teeth: rolling octave lock between neighbor bands, and how they line up
    teeth = []
    for lo, hi in zip(R.EQ_BANDS[:-1], R.EQ_BANDS[1:]):
        if lo in B and hi in B and B[lo][3] >= 3 and B[hi][3] >= 3:
            z = np.exp(1j * (2 * np.unwrap(B[lo][2][:, 0]) - np.unwrap(B[hi][2][:, 0])))
            k = np.ones(win) / win
            zr = np.convolve(z.real, k, "same") + 1j * np.convolve(z.imag, k, "same")
            lock = np.abs(zr)
            ang = np.angle(zr)
            kind = np.where(np.abs(ang) < np.pi / 4, 0, np.where(np.abs(ang) > 3 * np.pi / 4, 2, 1))  # 0 in step, 1 quarter, 2 opposite
            teeth.append(dict(lo=lo, hi=hi, lock=_ds(lock, 80), kind=[int(v) for v in kind[np.unique(np.linspace(0, n - 1, min(n, 80)).astype(int))]]))
    payload = dict(bands=bands, teeth=teeth, colors=R.BAND_COLORS, looks=looks, bar=round(strict, 3), n=n)
    return payload, dings


def cross_teeth(xa, xb, curve: dict) -> list:
    """Same band, two datasets: rolling lock between their phases (the zipper between the two sets of rings).
    Lines the two up on their last shared stretch."""
    m = min(len(xa), len(xb))
    X = np.column_stack([np.asarray(xa, float)[-m:], np.asarray(xb, float)[-m:]])
    if not R.is_flat(curve):
        X = R.apply_eq(X, curve)
    B = _bands_of(X)
    win = max(4, m // 10)
    k = np.ones(win) / win
    idx = np.unique(np.linspace(0, m - 1, min(m, 80)).astype(int))
    out = []
    for b in R.EQ_BANDS:
        if b in B and B[b][3] >= 3:
            z = np.exp(1j * (np.unwrap(B[b][2][:, 0]) - np.unwrap(B[b][2][:, 1])))
            zr = np.convolve(z.real, k, "same") + 1j * np.convolve(z.imag, k, "same")
            ang = np.angle(zr)
            kind = np.where(np.abs(ang) < np.pi / 4, 0, np.where(np.abs(ang) > 3 * np.pi / 4, 2, 1))
            # chance level: the same rolling lock with one series slid in time (breaks any real link, keeps each band's shape)
            rng = np.random.default_rng(b)
            pa, pb = np.unwrap(B[b][2][:, 0]), np.unwrap(B[b][2][:, 1])
            null = []
            for _ in range(40):
                zs = np.exp(1j * (pa - np.roll(pb, int(rng.integers(win, m - win)))))
                null.append(np.abs(np.convolve(zs.real, k, "same") + 1j * np.convolve(zs.imag, k, "same")))
            thr = float(np.percentile(np.concatenate(null), 95))
            out.append(dict(hz=b, lock=_ds(np.abs(zr), 80), kind=[int(v) for v in kind[idx]], thr=round(min(thr, 0.995), 3)))
    return out


HTML = r"""
<div id="wrap" style="position:relative;background:radial-gradient(circle at 50% 46%,#1b1238 0%,#08070f 68%);border-radius:18px;overflow:hidden;touch-action:none">
<canvas id="c" style="display:block;width:100%;height:440px"></canvas>
<div id="hud" style="position:absolute;left:10px;top:8px;right:10px;color:#c9ccd3;font:12px system-ui;line-height:1.35;pointer-events:none;text-shadow:0 0 6px #000"></div>
<div id="pin" style="display:none;position:absolute;left:8px;right:8px;bottom:44px;background:rgba(18,16,30,.93);border:1px solid #4a3f7a;border-radius:12px;padding:10px;color:#e6e7ea;font:12.5px system-ui;line-height:1.4;box-shadow:0 0 18px rgba(140,90,255,.35)"></div>
<div style="position:absolute;left:10px;right:10px;bottom:8px;display:flex;gap:8px;align-items:center">
 <button id="play" style="background:#2a2440;color:#e6e7ea;border:0;border-radius:8px;padding:5px 10px;font:12px system-ui">❚❚</button>
 <input id="time" type="range" min="0" max="1000" value="0" style="flex:1;accent-color:#a77bff">
</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const A = __A__, B = __B__, X = __X__;           // dataset A, optional dataset B, A-vs-B zipper teeth
const esc = v => String(v).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));   // names come from people's data
const wrap = document.getElementById('wrap'), cv = document.getElementById('c'), H = 440, W = () => wrap.clientWidth;
const renderer = new THREE.WebGLRenderer({canvas: cv, antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, W() / H, 0.1, 100);
cam.position.set(0, 1.4, 8.3); cam.lookAt(0, 0, 0);
const world = new THREE.Group(); scene.add(world);
const add = m => (world.add(m), m);
const glowTex = (() => { const s = 64, c = document.createElement('canvas'); c.width = c.height = s; const g = c.getContext('2d');
  const r = g.createRadialGradient(s/2, s/2, 0, s/2, s/2, s/2); r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(.3, 'rgba(255,255,255,.45)'); r.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = r; g.fillRect(0, 0, s, s); return new THREE.CanvasTexture(c); })();
const ADD = THREE.AdditiveBlending;
const sprite = (col, size, op) => { const m = new THREE.Sprite(new THREE.SpriteMaterial({map: glowTex, color: col, transparent: true, opacity: op, depthWrite: false, blending: ADD})); m.scale.set(size, size, 1); return m; };
const N = A.bands.length, COLS = A.colors.map(c => new THREE.Color(c));
const bandAng = i => 2 * Math.PI * i / N;
// smooth periodic curve through one value per band (Catmull-Rom, wraps around the circle)
function around(vals) { return th => { const u = ((th / (2 * Math.PI)) * N % N + N) % N, i = Math.floor(u), f = u - i;
  const p0 = vals[(i - 1 + N) % N], p1 = vals[i], p2 = vals[(i + 1) % N], p3 = vals[(i + 2) % N];
  return 0.5 * (2 * p1 + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f * f * f); }; }
function colorAt(th, out) { const u = ((th / (2 * Math.PI)) * N % N + N) % N, i = Math.floor(u), f = u - i; return out.copy(COLS[i]).lerp(COLS[(i + 1) % N], f); }
const at = (arr, u) => arr[Math.min(arr.length - 1, Math.floor(u * arr.length))];
const gainOf = d => d.bands.map(b => b.gain || 0);

// glowing core
const core = add(sprite(0xe2d4ff, 1.1, .9)), core2 = add(sprite(0x7a4dff, 2.4, .32));

// the EQ ring: flat line = circle, ups push out, downs pull in
const R0 = 2.35, RP = 220;
const eqGeo = new THREE.BufferGeometry(), eqPos = new Float32Array(RP * 3), eqCol = new Float32Array(RP * 3);
eqGeo.setAttribute('position', new THREE.BufferAttribute(eqPos, 3)); eqGeo.setAttribute('color', new THREE.BufferAttribute(eqCol, 3));
const eqLine = add(new THREE.LineLoop(eqGeo, new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: .95, blending: ADD})));
const eqGlow = add(new THREE.Points(eqGeo, new THREE.PointsMaterial({size: .22, map: glowTex, vertexColors: true, transparent: true, opacity: .55, depthWrite: false, blending: ADD})));
const flat = add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({length: 128}, (_, k) => new THREE.Vector3(R0 * Math.cos(k / 128 * 6.283), 0, R0 * Math.sin(k / 128 * 6.283)))),
  new THREE.LineBasicMaterial({color: 0x8c86b8, transparent: true, opacity: .25})));
const eqF = around(gainOf(A)), tmp = new THREE.Color();
for (let k = 0; k < RP; k++) { const th = k / RP * 2 * Math.PI, r = R0 + .09 * eqF(th);
  eqPos.set([r * Math.cos(th), 0, r * Math.sin(th)], k * 3); colorAt(th, tmp); eqCol.set([tmp.r, tmp.g, tmp.b], k * 3); }

// shell rings: dataset A on even rows, dataset B on odd rows (zipped together); each ripples with its band energies
const ROWS = 15, PTS = 160, R1 = 1.75, rows = [];
const shellGeo = new THREE.BufferGeometry(), sPos = new Float32Array(ROWS * PTS * 3), sCol = new Float32Array(ROWS * PTS * 3);
shellGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3)); shellGeo.setAttribute('color', new THREE.BufferAttribute(sCol, 3));
add(new THREE.Points(shellGeo, new THREE.PointsMaterial({size: .075, map: glowTex, vertexColors: true, transparent: true, opacity: .9, depthWrite: false, blending: ADD})));
for (let j = 0; j < ROWS; j++) { const lat = -1.25 + 2.5 * j / (ROWS - 1), d = (B && j % 2) ? B : A;
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(sPos.subarray(j * PTS * 3, (j + 1) * PTS * 3), 3));
  g.setAttribute('color', new THREE.BufferAttribute(sCol.subarray(j * PTS * 3, (j + 1) * PTS * 3), 3));
  const line = add(new THREE.LineLoop(g, new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: d === A ? .45 : .32, blending: ADD})));
  rows.push({lat, d, line, g}); }

// zipper teeth between neighboring A and B rows, at the band's longitude, where that band locks across the two
const KIND = ['in step', 'quarter turn', 'opposite'], KCOL = [new THREE.Color(0x7cffb2), new THREE.Color(0xffd166), new THREE.Color(0xff6b9a)];
const MAXT = 600, tGeo = new THREE.BufferGeometry(), tPos = new Float32Array(MAXT * 6), tCol = new Float32Array(MAXT * 6);
tGeo.setAttribute('position', new THREE.BufferAttribute(tPos, 3)); tGeo.setAttribute('color', new THREE.BufferAttribute(tCol, 3));
add(new THREE.LineSegments(tGeo, new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: 1, blending: ADD})));

// pins (bands that beat their fakes), A on the equator, B just above
const pins = [];
[[A, 0], [B, .35]].forEach(([d, lift]) => { if (!d) return; d.bands.forEach((b, i) => { if (!b.ding) return;
  const p = add(sprite(0xffffff, .5, 1)); p.userData = {i, d: b.ding, lift, who: d === A ? A.name : B.name}; pins.push(p); }); });

const ampOf = (d, t) => d.bands.map(b => b.live ? (.35 + .65 * b.rel) * at(b.env, t) * Math.pow(10, (b.gain || 0) / 20) : 0);
let t = 0, playing = true, drag = null, rotY = .4, rotX = .28, last = performance.now(), ticked = false, fallback = false;
const hud = document.getElementById('hud'), pinBox = document.getElementById('pin'), slider = document.getElementById('time'), playBtn = document.getElementById('play');
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? '❚❚' : '▶'; };
slider.oninput = () => { t = slider.value / 1000; playing = false; playBtn.textContent = '▶'; };
const pt = (r, lat, th) => [r * Math.cos(lat) * Math.cos(th), r * Math.sin(lat), r * Math.cos(lat) * Math.sin(th)];

function frame(now) {
  ticked = true;
  const dt = Math.min(.1, (now - last) / 1000); last = now;
  if (playing) { t = (t + dt / 20) % 1; slider.value = Math.round(t * 1000); }
  if (!drag) rotY += dt * .15;
  world.rotation.set(rotX, rotY, 0);
  core.scale.setScalar(1.0 + .04 * Math.sin(now / 900));
  const fA = around(ampOf(A, t)), fB = B ? around(ampOf(B, t)) : null;
  rows.forEach((row, j) => { const f = row.d === A ? fA : fB, cl = Math.cos(row.lat);
    for (let k = 0; k < PTS; k++) { const th = k / PTS * 2 * Math.PI, a = Math.max(0, f(th));
      const wob = .035 * Math.sin(th * 7 + now / 380 + j * 1.7) + .03 * Math.sin(th * 13 - now / 260 + j);
      const r = R1 * (1 + .32 * a * cl + wob * (.6 + a)), p = pt(r, row.lat * .9, th + (row.d === A ? 0 : Math.PI / PTS));
      sPos.set(p, (j * PTS + k) * 3); colorAt(th, tmp); const lum = .35 + .9 * Math.min(1, a);
      if (row.d !== A) tmp.lerp(new THREE.Color(0xffffff), .25);
      sCol.set([tmp.r * lum, tmp.g * lum, tmp.b * lum], (j * PTS + k) * 3); }
    row.g.attributes.position.needsUpdate = true; row.g.attributes.color.needsUpdate = true; });
  shellGeo.attributes.position.needsUpdate = true; shellGeo.attributes.color.needsUpdate = true;
  // teeth
  let n = 0; const lockedNow = [];
  if (B && X.length) X.forEach(z => { const i = A.bands.findIndex(b => b.hz === z.hz), lock = at(z.lock, t), kind = at(z.kind, t);
    if (lock < z.thr) return; lockedNow.push(`${z.hz >= 1000 ? z.hz / 1000 + 'k' : z.hz} ${KIND[kind]} ${lock.toFixed(2)}`);
    const th = bandAng(i), op = Math.min(1, .35 + (lock - z.thr) / Math.max(.05, 1 - z.thr));
    const k0 = Math.round(th / (2 * Math.PI) * PTS);
    for (let j = 0; j < ROWS - 1 && n < MAXT; j++) {
      for (let w = -2; w <= 2 && n < MAXT; w++) {                // five teeth per gap, alternating which row they grow from
        const up = (w + j) % 2 === 0, ja = up ? j : j + 1, jb = up ? j + 1 : j, kk = ((k0 + w) % PTS + PTS) % PTS;
        const a = sPos.subarray((ja * PTS + kk) * 3, (ja * PTS + kk) * 3 + 3), b = sPos.subarray((jb * PTS + kk) * 3, (jb * PTS + kk) * 3 + 3);
        tPos.set([a[0], a[1], a[2], a[0] + .58 * (b[0] - a[0]), a[1] + .58 * (b[1] - a[1]), a[2] + .58 * (b[2] - a[2])], n * 6);
        const c = KCOL[kind]; tCol.set([c.r * op, c.g * op, c.b * op, c.r * op, c.g * op, c.b * op], n * 6); n++; } } });
  tGeo.setDrawRange(0, n * 2); tGeo.attributes.position.needsUpdate = true; tGeo.attributes.color.needsUpdate = true;
  pins.forEach(p => { const d = p.userData.d, inWin = t >= d.t0 && t <= d.t1, th = bandAng(p.userData.i);
    const q = pt(R0 + .25, p.userData.lift, th); p.position.set(q[0], q[1], q[2]);
    p.scale.setScalar((inWin ? .75 : .42) * (1 + .15 * Math.sin(now / 200))); p.material.color.copy(inWin ? new THREE.Color(0xffffff) : COLS[p.userData.i]); });
  hud.innerHTML = `${esc(A.name)}${B ? ' ⟷ ' + esc(B.name) : ''} · t ${(t * 100).toFixed(0)}% · looks ${A.looks} · bar ${A.bar}th<br>` +
    `${pins.length} pin(s)` + (lockedNow.length ? ` · zipped: ${lockedNow.join(' · ')}` : '');
  renderer.render(scene, cam);
  if (!fallback) requestAnimationFrame(frame);
}

function resize() { renderer.setSize(W(), H, false); cam.aspect = W() / H; cam.updateProjectionMatrix(); }
window.addEventListener('resize', resize); resize();
cv.addEventListener('pointerdown', e => { drag = {x: e.clientX, y: e.clientY, rx: rotX, ry: rotY, moved: false}; cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointermove', e => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
  rotY = drag.ry + dx * .01; rotX = Math.max(-1.2, Math.min(1.2, drag.rx + dy * .01)); });
cv.addEventListener('pointerup', e => { const tap = drag && !drag.moved; drag = null; if (!tap) return;
  const r = cv.getBoundingClientRect(), m = new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(m, cam); const hit = ray.intersectObjects(pins)[0];
  if (!hit) { pinBox.style.display = 'none'; return; }
  const u = hit.object.userData, d = u.d, lab = d.hz >= 1000 ? (d.hz / 1000) + 'k' : d.hz;
  pinBox.style.display = 'block';
  pinBox.innerHTML = `<b style="color:${A.colors[u.i]}">${lab} band dinged</b> in ${esc(u.who)} · flag, not a finding<br>` +
    `Holds ${(d.real_share * 100).toFixed(1)}% of the wobble; fakes held ${(d.fake_range[0] * 100).toFixed(1)}–${(d.fake_range[1] * 100).toFixed(1)}%.<br>` +
    `Beat the fakes at the ${d.pct}th percentile (bar ${d.bar}, after ${d.looks} look${d.looks > 1 ? 's' : ''}).<br>` +
    `Strongest stretch: rows ${d.rows[0]}–${d.rows[1]} (${(d.t0 * 100).toFixed(0)}–${(d.t1 * 100).toFixed(0)}% through).<br>` +
    `<span style="color:#9a93c4">Knobs frozen: ${Object.entries(d.settings).map(([k, v]) => { const g = Array.isArray(v) ? v[0] : v; return esc(k) + ' Hz ' + (g > 0 ? '+' : '') + esc(g) + (Array.isArray(v) ? ' (w ' + v[1] + ')' : ''); }).join(', ')}</span>`;
  t = (d.t0 + d.t1) / 2; playing = false; playBtn.textContent = '▶'; });
requestAnimationFrame(frame);
// some embedded views never fire animation frames; fall back to a plain timer so the globe still moves
setTimeout(() => { if (!ticked) { fallback = true; setInterval(() => frame(performance.now()), 33); } }, 900);
</script>
"""


def render(a: dict, b: dict | None = None, cross: list | None = None) -> str:
    """a, b: payloads from band_check (with a 'name' added); cross: cross_teeth(a, b)."""
    from .embed import safe_json
    return (HTML.replace("__A__", safe_json(json.dumps(a))).replace("__B__", safe_json(json.dumps(b)) if b else "null")
            .replace("__X__", safe_json(json.dumps(cross or []))))
