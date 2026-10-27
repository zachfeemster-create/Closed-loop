"""Plasma globe: every EQ band is a pulsing cone inside a glass globe, the 25-90 Hz bands throw crests from the core,
neighbor bands show zipper teeth where they lock, and any band that beats its own fakes drops a glowing pin.

A pin is a flag, not a finding: it has to climb the usual ladder (frozen settings, fresh fakes, Max, new data).
Every new knob setting counts as another look, and the bar gets stricter with each look."""
from __future__ import annotations

import json

import numpy as np

from . import rotation as R

LOW_BANDS = (31, 63)                      # the 25-90 Hz crest bands


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
                   pct=round(pct, 2), env=_ds(e, 160), drift=_ds(drift, 160), bins=bins,
                   crest=b in LOW_BANDS)
        if pct >= strict and bins >= 2:
            fake_hi = np.percentile(env[:, 1:], 95, axis=1)
            excess = np.convolve(env[:, 0] - fake_hi, np.ones(win) / win, mode="valid")
            s0 = int(np.argmax(excess))
            d = dict(hz=b, pct=round(pct, 2), bar=round(strict, 3), real_share=round(real, 4),
                     fake_range=[round(float(lo95), 4), round(float(hi95), 4)],
                     t0=round(s0 / n, 3), t1=round((s0 + win) / n, 3), rows=[s0, s0 + win], looks=looks,
                     settings={str(k): round(float(v), 2) for k, v in curve.items()})
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


HTML = r"""
<div id="wrap" style="position:relative;background:radial-gradient(circle at 50% 45%,#1a1530 0%,#0b0b12 70%);border-radius:18px;overflow:hidden;touch-action:none">
<canvas id="c" style="display:block;width:100%;height:440px"></canvas>
<div id="hud" style="position:absolute;left:10px;top:8px;color:#c9ccd3;font:12px system-ui;line-height:1.35;pointer-events:none"></div>
<div id="pin" style="display:none;position:absolute;left:8px;right:8px;bottom:44px;background:rgba(20,21,28,.92);border:1px solid #3a3e46;border-radius:12px;padding:10px;color:#e6e7ea;font:12.5px system-ui;line-height:1.4"></div>
<div style="position:absolute;left:10px;right:10px;bottom:8px;display:flex;gap:8px;align-items:center">
 <button id="play" style="background:#2b2e35;color:#e6e7ea;border:0;border-radius:8px;padding:5px 10px;font:12px system-ui">❚❚</button>
 <input id="time" type="range" min="0" max="1000" value="0" style="flex:1">
</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const P = __PAYLOAD__;
const wrap = document.getElementById('wrap'), cv = document.getElementById('c');
const W = () => wrap.clientWidth, H = 440;
const renderer = new THREE.WebGLRenderer({canvas: cv, antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(42, W() / H, 0.1, 100);
cam.position.set(0, 0.4, 5.2);
const world = new THREE.Group(); scene.add(world);
const add = m => (world.add(m), m);
const glowTex = (() => { const s = 64, c = document.createElement('canvas'); c.width = c.height = s; const g = c.getContext('2d');
  const r = g.createRadialGradient(s/2, s/2, 0, s/2, s/2, s/2); r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(.25, 'rgba(255,255,255,.55)'); r.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = r; g.fillRect(0, 0, s, s); return new THREE.CanvasTexture(c); })();
const sprite = (col, size, op) => { const m = new THREE.Sprite(new THREE.SpriteMaterial({map: glowTex, color: col, transparent: true, opacity: op, depthWrite: false, blending: THREE.AdditiveBlending})); m.scale.set(size, size, 1); return m; };

// glass globe + core
add(new THREE.Mesh(new THREE.SphereGeometry(2, 48, 32), new THREE.MeshBasicMaterial({color: 0x8fa8ff, transparent: true, opacity: .05, depthWrite: false})));
add(new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(2, 18, 12)), new THREE.LineBasicMaterial({color: 0x6b74a8, transparent: true, opacity: .12})));
const core = add(sprite(0xd9c8ff, .9, .95));

// band directions: a golden-angle spiral over the sphere, low bands at the bottom
const N = P.bands.length, dirs = [];
for (let i = 0; i < N; i++) { const y = -0.85 + 1.7 * i / (N - 1), r = Math.sqrt(1 - y * y), a = i * 2.39996; dirs.push(new THREE.Vector3(r * Math.cos(a), y, r * Math.sin(a))); }
const cones = [], tips = [], crests = [], pins = [];
P.bands.forEach((b, i) => {
  const col = new THREE.Color(P.colors[i]);
  const g = new THREE.ConeGeometry(1, 1, 24, 1, true); g.translate(0, -0.5, 0); g.rotateX(Math.PI);  // apex at core, opens outward along +y
  const mat = new THREE.MeshBasicMaterial({color: col, transparent: true, opacity: b.live ? .28 : .06, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending});
  const m = add(new THREE.Mesh(g, mat)); m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dirs[i]);
  cones.push(m);
  const tip = add(sprite(col, .5, b.live ? .9 : .15)); tips.push(tip);
  if (b.crest && b.live) for (let k = 0; k < 5; k++) {           // plasma filaments for 25-90 Hz
    const pts = []; for (let j = 0; j <= 24; j++) pts.push(new THREE.Vector3());
    const line = add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color: col, transparent: true, opacity: .8, blending: THREE.AdditiveBlending})));
    crests.push({line, i, k, seed: Math.random() * 6.28});
  }
  if (b.ding) { const p = add(sprite(0xffffff, .55, 1)); p.userData = {i, d: b.ding}; pins.push(p); }
});
// zipper teeth between neighbor bands
const teeth = P.teeth.map(z => {
  const i = P.bands.findIndex(b => b.hz === z.lo), j = P.bands.findIndex(b => b.hz === z.hi);
  const grp = new THREE.Group(); add(grp); const bars = [];
  for (let k = 0; k < 12; k++) { const bar = new THREE.Mesh(new THREE.BoxGeometry(.05, .16, .05), new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending})); grp.add(bar); bars.push(bar); }
  return {z, i, j, bars};
});

const KIND = ['in step', 'quarter turn', 'opposite'], KCOL = [0x7cffb2, 0xffd166, 0xff6b9a];
let t = 0, playing = true, drag = null, rotY = 0.6, rotX = 0.15, last = performance.now(), selected = null;
const at = (arr, u) => arr[Math.min(arr.length - 1, Math.floor(u * arr.length))];
const hud = document.getElementById('hud'), pinBox = document.getElementById('pin'), slider = document.getElementById('time');
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? '❚❚' : '▶'; };
slider.oninput = () => { t = slider.value / 1000; playing = false; document.getElementById('play').textContent = '▶'; };

let ticked = false;
function frame(now) {
  ticked = true;
  const dt = (now - last) / 1000; last = now;
  if (playing) { t = (t + dt / 18) % 1; slider.value = Math.round(t * 1000); }
  if (!drag) rotY += dt * .12;
  world.rotation.set(rotX, rotY, 0);
  const pulse = 1 + .08 * Math.sin(now / 300);
  core.scale.setScalar(.9 * pulse);
  P.bands.forEach((b, i) => {
    const e = b.live ? at(b.env, t) : 0, gain = Math.pow(10, (b.gain || 0) / 20);
    const len = b.live ? .35 + 1.55 * Math.min(1, (.25 + .75 * b.rel) * gain * (.55 + .45 * e)) : .3;
    const wid = .12 + .22 * (b.live ? e : 0);
    cones[i].scale.set(wid, len, wid);
    tips[i].position.copy(dirs[i]).multiplyScalar(len);
    tips[i].scale.setScalar(.25 + .5 * e * gain);
  });
  crests.forEach(c => {                                         // crests and peaks pulsing out of the core
    const b = P.bands[c.i], e = at(b.env, t), len = cones[c.i].scale.y * (.7 + .5 * e), pos = c.line.geometry.attributes.position;
    const d = dirs[c.i].clone(), side = new THREE.Vector3(0, 1, 0).cross(d).normalize(), up = d.clone().cross(side);
    const ang = c.k * 1.2566 + c.seed;
    for (let j = 0; j <= 24; j++) { const s = j / 24, wob = Math.sin(s * 9 + now / 140 + c.seed) * .18 * s * (1 + e);
      const v = d.clone().multiplyScalar(s * len * 1.25).add(side.clone().multiplyScalar(Math.cos(ang) * (wob + .25 * s))).add(up.clone().multiplyScalar(Math.sin(ang) * (wob + .25 * s)));
      pos.setXYZ(j, v.x, v.y, v.z); }
    pos.needsUpdate = true; c.line.material.opacity = .35 + .6 * e;
  });
  teeth.forEach(T => {                                          // zipper teeth: bright where the two bands lock
    const lock = at(T.z.lock, t), kind = at(T.z.kind, t), a = tips[T.i].position, b = tips[T.j].position;
    T.bars.forEach((bar, k) => { const s = (k + .5) / T.bars.length; bar.position.lerpVectors(a, b, s);
      bar.position.multiplyScalar(1 + .08 * Math.sin(s * 3.14)); bar.lookAt(0, 0, 0);
      bar.rotation.z += (k % 2 ? 1 : -1) * .5; bar.material.color.setHex(KCOL[kind]); bar.material.opacity = Math.max(0, (lock - .25) / .75) * .95; });
  });
  pins.forEach(p => { const d = p.userData.d, inWin = t >= d.t0 && t <= d.t1;
    p.position.copy(dirs[p.userData.i]).multiplyScalar(2.05); p.scale.setScalar((inWin ? .8 : .45) * (1 + .15 * Math.sin(now / 200)));
    p.material.color.set(inWin ? 0xffffff : P.colors[p.userData.i]); });
  const live = P.teeth.map(z => `${z.lo}→${z.hi} ${KIND[at(z.kind, t)]} ${at(z.lock, t).toFixed(2)}`).filter((s, i) => at(P.teeth[i].lock, t) > .5);
  hud.innerHTML = `t ${(t * 100).toFixed(0)}% · looks ${P.looks} · bar ${P.bar}th pct<br>${P.bands.filter(b => b.ding).length} pin(s)` + (live.length ? `<br>locked: ${live.join(' · ')}` : '');
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
  const d = hit.object.userData.d, lab = d.hz >= 1000 ? (d.hz / 1000) + 'k' : d.hz;
  pinBox.style.display = 'block';
  pinBox.innerHTML = `<b style="color:${P.colors[hit.object.userData.i]}">${lab} band dinged</b> · flag, not a finding<br>` +
    `Holds ${(d.real_share * 100).toFixed(1)}% of the wobble; fakes held ${(d.fake_range[0] * 100).toFixed(1)}–${(d.fake_range[1] * 100).toFixed(1)}%.<br>` +
    `Beat the fakes at the ${d.pct}th percentile (bar ${d.bar}, after ${d.looks} look${d.looks > 1 ? 's' : ''}).<br>` +
    `Strongest stretch: rows ${d.rows[0]}–${d.rows[1]} (${(d.t0 * 100).toFixed(0)}–${(d.t1 * 100).toFixed(0)}% through).<br>` +
    `<span style="color:#8a8f99">Knobs frozen: ${Object.entries(d.settings).map(([k, v]) => k + ' ' + (v > 0 ? '+' : '') + v).join(', ')}</span>`;
  t = (d.t0 + d.t1) / 2; playing = false; document.getElementById('play').textContent = '▶'; });
requestAnimationFrame(frame);
// some embedded views never fire animation frames; fall back to a plain timer so the globe still moves
let fallback = false;
setTimeout(() => { if (!ticked) { fallback = true; setInterval(() => { frame(performance.now()); }, 33); } }, 900);
</script>
"""


def render(payload: dict) -> str:
    return HTML.replace("__PAYLOAD__", json.dumps(payload))
