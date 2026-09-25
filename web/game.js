// Browser port of fruit_ninja_cam: same game rules as src/fruit_ninja_cam/game.py,
// MediaPipe Tasks Vision (JS) for hand tracking, Canvas 2D for rendering.

const VISION_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// --- config (mirrors config.py) ---------------------------------------------
const CFG = {
  REF_HEIGHT: 720,
  INDEX_FINGERTIP: 8,
  TRAIL_MAX_POINTS: 16,
  TRAIL_MAX_AGE: 0.35,
  STARTING_LIVES: 3,
  GRAVITY: 1400,
  SPAWN_INTERVAL_START: 1.1,
  SPAWN_INTERVAL_MIN: 0.45,
  SPAWN_INTERVAL_DECAY: 0.985,
  BOMB_CHANCE: 0.18,
  FRUIT_R_MIN: 36,
  FRUIT_R_MAX: 52,
  BOMB_R: 40,
  APEX_MIN: 0.04,
  APEX_MAX: 0.18,
  VX_SPREAD_FRAC: 0.22,
  MIN_SLICE_SPEED: 450,
  SLICE_SCORE: 10,
  COMBO_WINDOW: 0.6,
  COMBO_BONUS: 5,
  BLADE_WIDTH: 15,
};

const GOLD = "#fac456";
const MINT = "#bef096";
const DANGER = "#ff4448";
const BLADE_GLOW = "#5abeff";

// Colours converted from theme.py (BGR → RGB).
const STYLES = {
  Apple: { skin: "#d62e30", shadow: "#80181e", flesh: "#faeece", juice: "#ec605a", core: "#ecd6b2", rind: "#d83c3c", seeds: true, stem: true, leaf: true, gloss: 0.75 },
  Orange: { skin: "#fc8a18", shadow: "#b0540c", flesh: "#ffba60", juice: "#ff9e28", core: "#ffd896", rind: "#ffe8be", segments: 8, speckle: 0.5, stem: true, gloss: 0.45 },
  Banana: { skin: "#f4d648", shadow: "#ba8c1e", flesh: "#faeeba", juice: "#f6e278", rind: "#e8c846", shape: "crescent", gloss: 0.5 },
  Watermelon: { skin: "#3a943e", shadow: "#20521c", flesh: "#f4ecc4", juice: "#e84648", core: "#e04246", rind: "#76be60", stripes: "#286022", seeds: true, gloss: 0.6 },
  Grape: { skin: "#8a369c", shadow: "#541a60", flesh: "#d6e2b2", juice: "#963ea8", rind: "#8c4696", shape: "cluster", stem: true, gloss: 0.8 },
  Lemon: { skin: "#f6d834", shadow: "#b28c18", flesh: "#fcf096", juice: "#f8e25a", core: "#fef6c4", rind: "#fef8ce", shape: "ellipse", segments: 7, speckle: 0.4, gloss: 0.5 },
};
const FRUIT_NAMES = Object.keys(STYLES);
const CLUSTER = [[0, -0.52, 0.42], [-0.46, -0.16, 0.44], [0.46, -0.16, 0.44], [-0.24, 0.34, 0.46], [0.24, 0.34, 0.46], [0, -0.02, 0.46], [0, 0.72, 0.34]];

// --- dom / canvas -------------------------------------------------------------
const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const controls = document.getElementById("controls");
const statusEl = document.getElementById("status");
const btnCam = document.getElementById("play-cam");
const btnPointer = document.getElementById("play-pointer");

let W = 0, H = 0, DPR = 1, K = 1; // K scales pixel sizes from the 720p reference
function resize() {
  DPR = Math.min(2, window.devicePixelRatio || 1);
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = Math.round(W * DPR);
  canvas.height = Math.round(H * DPR);
  K = Math.max(0.6, Math.min(1.5, Math.min(W, H) / CFG.REF_HEIGHT));
  spriteCache.clear();
}
window.addEventListener("resize", resize);

const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const now = () => performance.now() / 1000;

let bestScore = 0;
try { bestScore = Number(localStorage.getItem("fnc-best")) || 0; } catch { /* storage unavailable */ }

// --- sprites -----------------------------------------------------------------
const spriteCache = new Map();

function makeCanvas(size) {
  const c = document.createElement("canvas");
  c.width = c.height = Math.ceil(size * DPR);
  const g = c.getContext("2d");
  g.scale(DPR, DPR);
  return [c, g];
}

function silhouette(g, st, c, r) {
  g.beginPath();
  if (st.shape === "cluster") {
    for (const [dx, dy, br] of CLUSTER) { g.moveTo(c + dx * r + br * r, c + dy * r); g.arc(c + dx * r, c + dy * r, br * r, 0, Math.PI * 2); }
  } else if (st.shape === "ellipse") {
    const a = (-18 * Math.PI) / 180;
    g.ellipse(c, c, r, r * 0.78, a, 0, Math.PI * 2);
    for (const s of [-1, 1]) {
      const px = c + s * r * 0.95 * Math.cos(a), py = c + s * r * 0.95 * Math.sin(a);
      g.moveTo(px + r * 0.16, py); g.arc(px, py, r * 0.16, 0, Math.PI * 2);
    }
  } else if (st.shape !== "crescent") {
    g.arc(c, c, r, 0, Math.PI * 2);
  }
}

function fillSilhouette(g, st, c, r, color) {
  g.fillStyle = color;
  g.strokeStyle = color;
  if (st.shape === "crescent") {
    g.lineWidth = r * 0.52; g.lineCap = "round";
    g.beginPath(); g.ellipse(c, c - r * 0.3, r * 0.98, r * 0.92, 0, (28 * Math.PI) / 180, (152 * Math.PI) / 180); g.stroke();
  } else {
    silhouette(g, st, c, r); g.fill();
  }
}

// Shading on top of the silhouette: key light upper-left, core shadow lower-right, spec + rim.
function shade(g, c, r, cx, cy, rr, gloss, dark = 0.55) {
  const lx = cx - rr * 0.38, ly = cy - rr * 0.42;
  let grad = g.createRadialGradient(lx, ly, rr * 0.05, cx, cy, rr * 1.15);
  grad.addColorStop(0, "rgba(255,255,255,0.28)");
  grad.addColorStop(0.45, "rgba(255,255,255,0)");
  grad.addColorStop(0.8, `rgba(0,0,0,${dark * 0.6})`);
  grad.addColorStop(1, `rgba(0,0,0,${dark})`);
  g.fillStyle = grad; g.fillRect(0, 0, c * 2, c * 2);
  grad = g.createRadialGradient(lx, ly, 0, lx, ly, rr * 0.34);
  grad.addColorStop(0, `rgba(255,255,255,${0.85 * gloss})`);
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad; g.fillRect(0, 0, c * 2, c * 2);
  grad = g.createRadialGradient(cx, cy, rr * 0.82, cx, cy, rr * 1.05);
  grad.addColorStop(0, "rgba(255,255,255,0)");
  grad.addColorStop(1, "rgba(200,220,255,0.18)");
  g.fillStyle = grad; g.fillRect(0, 0, c * 2, c * 2);
}

function drawStem(g, c, r, leaf) {
  const top = c - r * 0.92;
  g.strokeStyle = "#523418"; g.lineWidth = Math.max(2, r * 0.11); g.lineCap = "round";
  g.beginPath(); g.moveTo(c + r * 0.02, top + r * 0.16); g.lineTo(c - r * 0.1, top - r * 0.34); g.stroke();
  if (leaf) {
    g.save(); g.translate(c + r * 0.3, top - r * 0.24); g.rotate((-28 * Math.PI) / 180);
    g.fillStyle = "#4a9c2e"; g.beginPath(); g.ellipse(0, 0, r * 0.34, r * 0.16, 0, 0, Math.PI * 2); g.fill();
    g.strokeStyle = "#347420"; g.lineWidth = Math.max(1, r * 0.05); g.stroke(); g.restore();
  }
}

function fruitSprite(name, r) {
  const key = `f:${name}:${r}`;
  if (spriteCache.has(key)) return spriteCache.get(key);
  const st = STYLES[name];
  const size = r * 2 * 1.42, c = size / 2;
  const [cv, g] = makeCanvas(size);
  fillSilhouette(g, st, c, r, st.skin);
  g.globalCompositeOperation = "source-atop";
  if (st.stripes) {
    g.fillStyle = st.stripes;
    for (let i = -2; i <= 2; i++) {
      const off = i * r * 0.46;
      g.beginPath(); g.ellipse(c + off, c, Math.max(2, r * 0.13), r * 1.05, (off / r) * 0.16, 0, Math.PI * 2); g.fill();
    }
  }
  if (st.speckle) {
    for (let i = 0; i < r * 6; i++) {
      g.fillStyle = Math.random() < 0.5 ? `rgba(0,0,0,${0.12 * st.speckle})` : `rgba(255,255,255,${0.12 * st.speckle})`;
      g.fillRect(c + rand(-r, r), c + rand(-r, r), 1.2, 1.2);
    }
  }
  if (st.shape === "cluster") {
    for (const [i, [dx, dy, br]] of CLUSTER.entries()) {
      g.save();
      g.beginPath(); g.arc(c + dx * r, c + dy * r, br * r, 0, Math.PI * 2); g.clip();
      g.fillStyle = st.skin; g.fill();
      g.fillStyle = i % 2 ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"; g.fill();
      shade(g, c, r, c + dx * r, c + dy * r, br * r, st.gloss, 0.45);
      g.restore();
      g.beginPath(); g.arc(c + dx * r, c + dy * r, br * r, 0, Math.PI * 2);
      g.strokeStyle = "rgba(30,0,40,0.4)"; g.lineWidth = Math.max(1, r * 0.04); g.stroke();
    }
  } else if (st.shape === "crescent") {
    shade(g, c, r, c, c - r * 0.1, r * 1.05, st.gloss, 0.5);
    g.fillStyle = "rgba(60,40,10,0.55)";
    for (const ang of [28, 152]) {
      const a = (ang * Math.PI) / 180;
      g.beginPath(); g.arc(c + r * 0.98 * Math.cos(a), c - r * 0.3 + r * 0.92 * Math.sin(a), r * 0.24, 0, Math.PI * 2); g.fill();
    }
  } else {
    shade(g, c, r, c, c, r, st.gloss);
  }
  g.globalCompositeOperation = "source-over";
  if (st.stem) drawStem(g, c, r, st.leaf);
  const sprite = { canvas: cv, size };
  spriteCache.set(key, sprite);
  return sprite;
}

function bombSprite(r) {
  const key = `b:${r}`;
  if (spriteCache.has(key)) return spriteCache.get(key);
  const size = r * 2 * 1.42, c = size / 2;
  const [cv, g] = makeCanvas(size);
  g.fillStyle = "#262022"; g.beginPath(); g.arc(c, c, r, 0, Math.PI * 2); g.fill();
  g.globalCompositeOperation = "source-atop";
  shade(g, c, r, c, c, r, 1.0, 0.6);
  g.save(); g.translate(c, c); g.rotate((12 * Math.PI) / 180);
  g.strokeStyle = "#423a3e"; g.lineWidth = r * 0.1; g.beginPath(); g.ellipse(0, 0, r * 0.99, r * 0.3, 0, 0, Math.PI * 2); g.stroke();
  g.fillStyle = DANGER;
  for (let k = -2; k <= 2; k++) { g.beginPath(); g.ellipse(k * r * 0.42, k * r * 0.02, r * 0.09, r * 0.16, 0, 0, Math.PI * 2); g.fill(); }
  g.restore();
  g.globalCompositeOperation = "source-over";
  g.fillStyle = "#5c5254"; g.beginPath(); g.ellipse(c - r * 0.1, c - r * 0.86, r * 0.26, r * 0.16, (-18 * Math.PI) / 180, 0, Math.PI * 2); g.fill();
  const sprite = { canvas: cv, size };
  spriteCache.set(key, sprite);
  return sprite;
}

function cutFace(name, r) {
  const key = `c:${name}:${r}`;
  if (spriteCache.has(key)) return spriteCache.get(key);
  const st = STYLES[name];
  const size = r * 2.1, c = size / 2;
  const [cv, g] = makeCanvas(size);
  const disc = (rad, col) => { g.fillStyle = col; g.beginPath(); g.arc(c, c, rad, 0, Math.PI * 2); g.fill(); };
  disc(r, st.rind || st.shadow);
  disc(r * 0.9, st.flesh);
  if (st.core) disc(r * 0.74, st.core);
  if (st.segments) {
    g.strokeStyle = "rgba(255,255,255,0.55)"; g.lineWidth = Math.max(1, r * 0.045);
    for (let i = 0; i < st.segments; i++) {
      const a = (2 * Math.PI * i) / st.segments;
      g.beginPath(); g.moveTo(c, c); g.lineTo(c + r * 0.86 * Math.cos(a), c + r * 0.86 * Math.sin(a)); g.stroke();
    }
    g.fillStyle = "rgba(255,255,255,0.6)"; g.beginPath(); g.arc(c, c, r * 0.1, 0, Math.PI * 2); g.fill();
  }
  if (st.seeds) {
    g.fillStyle = "#221a1c";
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2 + 0.4, d = (0.3 + (i % 3) * 0.14) * r;
      g.beginPath(); g.ellipse(c + d * Math.cos(a), c + d * Math.sin(a), Math.max(1, r * 0.055), Math.max(2, r * 0.085), a, 0, Math.PI * 2); g.fill();
    }
  }
  const grad = g.createRadialGradient(c - r * 0.24, c - r * 0.3, 0, c - r * 0.24, c - r * 0.3, r * 0.6);
  grad.addColorStop(0, "rgba(255,255,255,0.22)"); grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad; g.fillRect(0, 0, size, size);
  const sprite = { canvas: cv, size };
  spriteCache.set(key, sprite);
  return sprite;
}

function blit(sprite, x, y, rot = 0, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(x, y);
  if (rot) ctx.rotate(rot);
  ctx.drawImage(sprite.canvas, -sprite.size / 2, -sprite.size / 2, sprite.size, sprite.size);
  ctx.restore();
}

// --- game logic (port of game.py) --------------------------------------------
const State = { MENU: "menu", PLAYING: "playing", GAME_OVER: "over" };

const game = {
  state: State.MENU,
  score: 0,
  lives: CFG.STARTING_LIVES,
  fruits: [],
  spawnInterval: CFG.SPAWN_INTERVAL_START,
  nextSpawnAt: 0,
  lastSliceAt: 0,
  combo: 0,
  events: [],

  start(t) {
    this.state = State.PLAYING;
    this.score = 0;
    this.lives = CFG.STARTING_LIVES;
    this.fruits = [];
    this.spawnInterval = CFG.SPAWN_INTERVAL_START;
    this.nextSpawnAt = t + 0.4;
    this.lastSliceAt = 0;
    this.combo = 0;
  },

  gravity() { return CFG.GRAVITY * (H / CFG.REF_HEIGHT); },

  update(dt, trail, t) {
    this.events = [];
    if (this.state !== State.PLAYING) return;
    if (t >= this.nextSpawnAt) {
      this.fruits.push(this.makeProjectile());
      this.nextSpawnAt = t + this.spawnInterval;
      this.spawnInterval = Math.max(CFG.SPAWN_INTERVAL_MIN, this.spawnInterval * CFG.SPAWN_INTERVAL_DECAY);
    }
    const g = this.gravity();
    for (const f of this.fruits) { f.vy += g * dt; f.x += f.vx * dt; f.y += f.vy * dt; f.rot += f.vrot * dt; }
    this.resolveSlices(trail, t);
    if (this.state !== State.PLAYING) return;
    this.cullMissed();
    if (t - this.lastSliceAt > CFG.COMBO_WINDOW) this.combo = 0;
  },

  makeProjectile() {
    const isBomb = Math.random() < CFG.BOMB_CHANCE;
    const x = rand(W * 0.15, W * 0.85);
    const y = H + 40 * K;
    const spread = W * CFG.VX_SPREAD_FRAC;
    let vx = rand(-spread, spread);
    if (x < W * 0.35) vx = Math.abs(vx);
    else if (x > W * 0.65) vx = -Math.abs(vx);
    const apexY = H * rand(CFG.APEX_MIN, CFG.APEX_MAX);
    const vy = -Math.sqrt(2 * this.gravity() * Math.max(80, y - apexY));
    const base = { x, y, vx, vy, rot: 0, vrot: rand(-1.6, 1.6) };
    if (isBomb) return { ...base, r: Math.round(CFG.BOMB_R * K), name: "BOMB", isBomb: true, vrot: rand(-0.6, 0.6) };
    const name = FRUIT_NAMES[Math.floor(Math.random() * FRUIT_NAMES.length)];
    return { ...base, r: Math.round(rand(CFG.FRUIT_R_MIN, CFG.FRUIT_R_MAX) * K), name, isBomb: false };
  },

  resolveSlices(trail, t) {
    if (trail.length < 2) return;
    for (const f of this.fruits) {
      if (f.dead) continue;
      const angle = trailHits(trail, f);
      if (angle === null) continue;
      f.dead = true;
      if (f.isBomb) {
        this.state = State.GAME_OVER;
        this.events.push({ ...f, bomb: true });
        this.fruits = this.fruits.filter((q) => !q.dead);
        return;
      }
      this.combo = t - this.lastSliceAt <= CFG.COMBO_WINDOW ? this.combo + 1 : 1;
      this.lastSliceAt = t;
      const points = CFG.SLICE_SCORE + Math.max(0, this.combo - 1) * CFG.COMBO_BONUS;
      this.score += points;
      this.events.push({ ...f, points, angle, combo: this.combo });
    }
    this.fruits = this.fruits.filter((q) => !q.dead);
  },

  cullMissed() {
    const kept = [];
    for (const f of this.fruits) {
      if (f.y - f.r > H + 10 * K) {
        if (!f.isBomb) {
          this.lives -= 1;
          fx.lifeLost = 1;
          if (this.lives <= 0) { this.lives = 0; this.state = State.GAME_OVER; }
        }
        continue;
      }
      kept.push(f);
    }
    this.fruits = kept;
  },
};

function segmentCircleHit(ax, ay, bx, by, cx, cy, r) {
  const abx = bx - ax, aby = by - ay, acx = cx - ax, acy = cy - ay;
  const len2 = abx * abx + aby * aby;
  if (len2 <= 1e-6) return Math.hypot(acx, acy) <= r;
  const t = clamp((acx * abx + acy * aby) / len2, 0, 1);
  return Math.hypot(ax + t * abx - cx, ay + t * aby - cy) <= r;
}

function trailHits(trail, f) {
  for (let i = 1; i < trail.length; i++) {
    const a = trail[i - 1], b = trail[i];
    const dt = b.t - a.t;
    if (dt <= 0) continue;
    if (Math.hypot(b.x - a.x, b.y - a.y) / dt < CFG.MIN_SLICE_SPEED * K) continue;
    if (segmentCircleHit(a.x, a.y, b.x, b.y, f.x, f.y, f.r)) return Math.atan2(b.y - a.y, b.x - a.x);
  }
  return null;
}

// --- input: blade trail from hand tracking or pointer ------------------------
const trail = [];
let tip = null;
let tipSeenAt = 0;

function pushTip(x, y, t) {
  tip = { x, y };
  tipSeenAt = t;
  trail.push({ x, y, t });
  while (trail.length > CFG.TRAIL_MAX_POINTS) trail.shift();
}

function ageTrail(t) {
  while (trail.length && t - trail[0].t > CFG.TRAIL_MAX_AGE) trail.shift();
  if (t - tipSeenAt > 0.25) tip = null;
}

canvas.addEventListener("pointermove", (e) => {
  if (e.pointerType === "mouse" || e.buttons || e.pointerType === "touch") {
    for (const ev of e.getCoalescedEvents ? e.getCoalescedEvents() : [e]) pushTip(ev.clientX, ev.clientY, now());
  }
});
canvas.addEventListener("pointerdown", (e) => {
  pushTip(e.clientX, e.clientY, now());
  if (game.state !== State.PLAYING && mode) begin();
});

// --- hand tracking -----------------------------------------------------------
let landmarker = null;
let mode = null; // "cam" | "pointer"
let lastVideoTime = -1;

async function initCamera() {
  statusEl.textContent = "Loading hand-tracking model…";
  const { FilesetResolver, HandLandmarker } = await import(`${VISION_URL}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${VISION_URL}/wasm`);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch {
    landmarker = await HandLandmarker.createFromOptions(fileset, opts("CPU"));
  }
  statusEl.textContent = "Waiting for camera permission…";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function videoRect() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(W / vw, H / vh);
  const dw = vw * s, dh = vh * s;
  return { dw, dh, ox: (W - dw) / 2, oy: (H - dh) / 2 };
}

function trackHand(t) {
  if (!landmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const res = landmarker.detectForVideo(video, performance.now());
  const hand = res.landmarks && res.landmarks[0];
  if (!hand) return;
  const p = hand[CFG.INDEX_FINGERTIP];
  const { dw, dh, ox, oy } = videoRect();
  pushTip(ox + (1 - p.x) * dw, oy + p.y * dh, t); // mirrored selfie view
}

// --- effects -----------------------------------------------------------------
const fx = { halves: [], particles: [], texts: [], shake: 0, flash: 0, lifeLost: 0, scoreShown: 0, scorePop: 0, overAt: 0 };

function spawnSliceFx(e) {
  const st = STYLES[e.name];
  const nx = -Math.sin(e.angle), ny = Math.cos(e.angle);
  const push = 150 * K;
  for (const side of [-1, 1]) {
    fx.halves.push({
      name: e.name, r: e.r, side, angle: e.angle,
      x: e.x, y: e.y, vx: e.vx * 0.5 + nx * push * side, vy: e.vy * 0.4 + ny * push * side - 60 * K,
      rot: 0, vrot: side * rand(2, 4), life: 1.6,
    });
  }
  for (let i = 0; i < 26; i++) {
    const a = rand(0, Math.PI * 2), sp = rand(80, 420) * K;
    fx.particles.push({ x: e.x, y: e.y, vx: Math.cos(a) * sp + e.vx * 0.3, vy: Math.sin(a) * sp - 120 * K, r: rand(2, 6) * K, color: st.juice, life: rand(0.4, 0.9), max: 0.9, g: 1 });
  }
  for (let i = 0; i < 10; i++) {
    const a = e.angle + rand(-0.5, 0.5) + (i % 2 ? Math.PI : 0), sp = rand(300, 700) * K;
    fx.particles.push({ x: e.x, y: e.y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, r: rand(1.5, 3) * K, color: "#fff6d8", life: rand(0.15, 0.35), max: 0.35, g: 0.2, add: true });
  }
  if (e.quiet) return;
  fx.texts.push({ x: e.x, y: e.y - e.r, text: `+${e.points}`, color: GOLD, life: 0.9 });
  if (e.combo >= 2) fx.texts.push({ x: e.x, y: e.y - e.r - 34 * K, text: `COMBO x${e.combo}`, color: MINT, life: 0.9, small: true });
  fx.shake = Math.max(fx.shake, 0.18);
}

function spawnBombFx(e) {
  fx.flash = 1;
  fx.shake = 0.7;
  for (let i = 0; i < 70; i++) {
    const a = rand(0, Math.PI * 2), sp = rand(60, 620) * K;
    const hot = i < 45;
    fx.particles.push({
      x: e.x, y: e.y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - (hot ? 0 : 80 * K),
      r: (hot ? rand(4, 12) : rand(14, 30)) * K, grow: hot ? 0 : 60 * K,
      color: hot ? ["#fff2b0", "#ffb640", "#ff6a1c"][i % 3] : "rgba(70,66,72,1)",
      life: hot ? rand(0.3, 0.8) : rand(0.8, 1.5), max: hot ? 0.8 : 1.5, g: hot ? 0.15 : -0.1, add: hot,
    });
  }
}

function updateFx(dt) {
  const g = game.gravity();
  for (const h of fx.halves) { h.vy += g * dt; h.x += h.vx * dt; h.y += h.vy * dt; h.rot += h.vrot * dt; h.life -= dt; }
  fx.halves = fx.halves.filter((h) => h.life > 0 && h.y - h.r < H + 60);
  for (const p of fx.particles) {
    p.vy += g * p.g * dt; p.vx *= 1 - 1.5 * dt; p.x += p.vx * dt; p.y += p.vy * dt; p.life -= dt;
    if (p.grow) p.r += p.grow * dt;
  }
  fx.particles = fx.particles.filter((p) => p.life > 0);
  for (const t of fx.texts) { t.y -= 60 * K * dt; t.life -= dt; }
  fx.texts = fx.texts.filter((t) => t.life > 0);
  fx.shake = Math.max(0, fx.shake - dt * 1.6);
  fx.flash = Math.max(0, fx.flash - dt * 2.5);
  fx.lifeLost = Math.max(0, fx.lifeLost - dt * 2);
  fx.scorePop = Math.max(0, fx.scorePop - dt * 3);
  if (fx.scoreShown < game.score) { fx.scoreShown = Math.min(game.score, fx.scoreShown + Math.max(1, (game.score - fx.scoreShown) * 10 * dt)); fx.scorePop = 1; }
  else fx.scoreShown = game.score;
}

// --- drawing -----------------------------------------------------------------
function drawBackdrop() {
  if (mode === "cam" && video.readyState >= 2) {
    const { dw, dh, ox, oy } = videoRect();
    ctx.save();
    ctx.filter = "brightness(0.62) saturate(0.7)";
    ctx.translate(W, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, ox, oy, dw, dh);
    ctx.restore();
  } else {
    const grad = ctx.createRadialGradient(W * 0.5, H * 0.42, 0, W * 0.5, H * 0.5, Math.max(W, H) * 0.75);
    grad.addColorStop(0, "#3a3840"); grad.addColorStop(1, "#141318");
    ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);
  }
  const v = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);
  v.addColorStop(0, "rgba(0,0,0,0)"); v.addColorStop(1, "rgba(0,0,0,0.55)");
  ctx.fillStyle = v; ctx.fillRect(0, 0, W, H);
}

function drawShadow(x, y, r) {
  const grad = ctx.createRadialGradient(x + r * 0.18, y + r * 0.3, 0, x + r * 0.18, y + r * 0.3, r * 1.2);
  grad.addColorStop(0, "rgba(0,0,0,0.35)"); grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad; ctx.fillRect(x - r * 1.4, y - r * 1.2, r * 3, r * 3);
}

function drawFruits(t) {
  for (const f of game.fruits) {
    drawShadow(f.x, f.y, f.r);
    if (f.isBomb) {
      ctx.save();
      const glow = ctx.createRadialGradient(f.x, f.y, f.r * 0.8, f.x, f.y, f.r * 1.6);
      glow.addColorStop(0, "rgba(255,60,60,0.25)"); glow.addColorStop(1, "rgba(255,60,60,0)");
      ctx.fillStyle = glow; ctx.fillRect(f.x - f.r * 2, f.y - f.r * 2, f.r * 4, f.r * 4);
      ctx.restore();
      blit(bombSprite(f.r), f.x, f.y, f.rot);
      const a = f.rot - Math.PI / 2 - 0.3;
      const sx = f.x + Math.cos(a) * f.r * 1.15, sy = f.y + Math.sin(a) * f.r * 1.15;
      const flick = 0.7 + 0.3 * Math.sin(t * 40);
      const sg = ctx.createRadialGradient(sx, sy, 0, sx, sy, 14 * K);
      sg.addColorStop(0, `rgba(255,250,210,${flick})`); sg.addColorStop(0.4, "rgba(255,170,60,0.6)"); sg.addColorStop(1, "rgba(255,120,40,0)");
      ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(sx, sy, 14 * K, 0, Math.PI * 2); ctx.fill();
    } else {
      blit(fruitSprite(f.name, f.r), f.x, f.y, f.rot);
    }
  }
}

function drawHalves() {
  for (const h of fx.halves) {
    const alpha = clamp(h.life / 0.4, 0, 1);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(h.x, h.y);
    ctx.rotate(h.angle + h.rot);
    ctx.beginPath();
    ctx.rect(-h.r * 1.5, h.side < 0 ? -h.r * 1.5 : 0, h.r * 3, h.r * 1.5);
    ctx.clip();
    const s = fruitSprite(h.name, h.r);
    ctx.save(); ctx.rotate(-h.angle); ctx.drawImage(s.canvas, -s.size / 2, -s.size / 2, s.size, s.size); ctx.restore();
    const face = cutFace(h.name, h.r);
    ctx.save(); ctx.scale(1, 0.34); ctx.drawImage(face.canvas, -face.size / 2, -face.size / 2, face.size, face.size); ctx.restore();
    ctx.restore();
  }
}

function drawParticles() {
  for (const p of fx.particles) {
    ctx.save();
    ctx.globalAlpha = clamp(p.life / p.max, 0, 1) * (p.add ? 1 : 0.9);
    if (p.add) ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = p.color;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
}

function smooth(pts) {
  let out = pts;
  for (let pass = 0; pass < 2; pass++) {
    if (out.length < 3) return out;
    const next = [out[0]];
    for (let i = 0; i < out.length - 1; i++) {
      const a = out[i], b = out[i + 1];
      next.push({ x: a.x * 0.75 + b.x * 0.25, y: a.y * 0.75 + b.y * 0.25 }, { x: a.x * 0.25 + b.x * 0.75, y: a.y * 0.25 + b.y * 0.75 });
    }
    next.push(out[out.length - 1]);
    out = next;
  }
  return out;
}

// Tapered polygon along the trail: zero width at the tail, full width at the tip.
function ribbon(pts, width) {
  const n = pts.length, left = [], right = [];
  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
    let dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    dx /= len; dy /= len;
    const w = (width / 2) * (i / (n - 1));
    left.push([pts[i].x - dy * w, pts[i].y + dx * w]);
    right.push([pts[i].x + dy * w, pts[i].y - dx * w]);
  }
  ctx.beginPath();
  ctx.moveTo(left[0][0], left[0][1]);
  for (const [x, y] of left) ctx.lineTo(x, y);
  const tip = pts[n - 1];
  ctx.arc(tip.x, tip.y, width / 2, Math.atan2(left[n - 1][1] - tip.y, left[n - 1][0] - tip.x), Math.atan2(right[n - 1][1] - tip.y, right[n - 1][0] - tip.x), true);
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(right[i][0], right[i][1]);
  ctx.closePath();
}

function drawBlade(t) {
  const pts = smooth(trail.map((p) => ({ x: p.x, y: p.y })));
  if (pts.length >= 2) {
    ctx.save();
    ctx.fillStyle = "rgba(90,190,255,0.45)";
    ctx.shadowColor = BLADE_GLOW; ctx.shadowBlur = 18 * K;
    ribbon(pts, CFG.BLADE_WIDTH * K * 1.6); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ribbon(pts, CFG.BLADE_WIDTH * K * 0.55); ctx.fill();
    ctx.restore();
  }
  if (!tip) return;
  const { x, y } = tip;
  ctx.save();
  const glow = ctx.createRadialGradient(x, y, 0, x, y, 26 * K);
  glow.addColorStop(0, "rgba(120,180,255,0.7)"); glow.addColorStop(1, "rgba(120,180,255,0)");
  ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x, y, 26 * K, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(x, y, 6 * K, 0, Math.PI * 2); ctx.fill();
  const ring = (15 + 2 * Math.sin(t * 6)) * K;
  ctx.strokeStyle = GOLD; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, ring, 0, Math.PI * 2); ctx.stroke();
  for (let i = 0; i < 4; i++) {
    const a = t * 2.4 + (i * Math.PI) / 2;
    ctx.beginPath();
    ctx.moveTo(x + Math.cos(a) * (ring + 4 * K), y + Math.sin(a) * (ring + 4 * K));
    ctx.lineTo(x + Math.cos(a) * (ring + 11 * K), y + Math.sin(a) * (ring + 11 * K));
    ctx.stroke();
  }
  ctx.restore();
}

function font(weight, px) { return `${weight} ${Math.round(px)}px Rubik, system-ui, sans-serif`; }

function panel(x, y, w, h, r = 18) {
  ctx.fillStyle = "rgba(18,16,26,0.55)";
  ctx.strokeStyle = "rgba(96,84,90,0.45)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.fill(); ctx.stroke();
}

function heart(cx, cy, s, filled) {
  ctx.beginPath();
  ctx.moveTo(cx, cy + s * 0.9);
  ctx.bezierCurveTo(cx - s * 1.6, cy - s * 0.1, cx - s * 0.7, cy - s * 1.2, cx, cy - s * 0.35);
  ctx.bezierCurveTo(cx + s * 0.7, cy - s * 1.2, cx + s * 1.6, cy - s * 0.1, cx, cy + s * 0.9);
  if (filled) { ctx.fillStyle = DANGER; ctx.fill(); }
  else { ctx.strokeStyle = "#5c5256"; ctx.lineWidth = 2; ctx.stroke(); }
}

function drawHud(t) {
  const s = Math.min(1, Math.max(0.7, W / 900));
  const pad = 16;
  panel(pad, pad, 190 * s, 76 * s);
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#acacb6"; ctx.font = font(700, 12 * s);
  ctx.fillText("SCORE", pad + 18 * s, pad + 26 * s);
  const pop = 1 + 0.2 * fx.scorePop;
  ctx.fillStyle = fx.scorePop > 0.05 ? GOLD : "#f5f5fa"; ctx.font = font(900, 32 * s * pop);
  ctx.fillText(Math.round(fx.scoreShown).toLocaleString(), pad + 18 * s, pad + 62 * s);

  const lw = (40 + CFG.STARTING_LIVES * 34) * s;
  const lx = W - lw - pad;
  panel(lx, pad, lw, 76 * s);
  ctx.fillStyle = "#acacb6"; ctx.font = font(700, 12 * s);
  ctx.fillText("LIVES", lx + 18 * s, pad + 26 * s);
  for (let i = 0; i < CFG.STARTING_LIVES; i++) {
    const wob = game.lives === 1 && i === 0 ? 2.5 * Math.sin(t * 9) : 0;
    heart(lx + 32 * s + i * 34 * s, pad + 52 * s + wob, 11 * s, i < game.lives);
  }

  if (game.state === State.PLAYING && game.combo >= 2) {
    const pulse = 1 + 0.06 * Math.sin(t * 12);
    ctx.textAlign = "center"; ctx.fillStyle = MINT; ctx.font = font(900, 26 * s * pulse);
    ctx.fillText(`COMBO x${game.combo}`, W / 2, pad + 44 * s);
  }
  if (fx.lifeLost > 0) {
    ctx.fillStyle = `rgba(255,50,60,${0.22 * fx.lifeLost})`;
    ctx.fillRect(0, 0, W, H);
  }
}

function drawTexts() {
  ctx.textAlign = "center";
  for (const tx of fx.texts) {
    ctx.save();
    ctx.globalAlpha = clamp(tx.life / 0.5, 0, 1);
    ctx.font = font(900, (tx.small ? 18 : 30) * K);
    ctx.lineWidth = 5; ctx.strokeStyle = "rgba(20,14,10,0.7)"; ctx.lineJoin = "round";
    ctx.strokeText(tx.text, tx.x, tx.y);
    ctx.fillStyle = tx.color; ctx.fillText(tx.text, tx.x, tx.y);
    ctx.restore();
  }
}

function glowTitle(text, x, y, px, color) {
  ctx.save();
  ctx.font = font(900, px); ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.shadowColor = color; ctx.shadowBlur = px * 0.5;
  ctx.fillStyle = color; ctx.fillText(text, x, y);
  ctx.shadowBlur = 0; ctx.lineWidth = Math.max(2, px * 0.06); ctx.strokeStyle = "rgba(40,24,6,0.8)";
  ctx.strokeText(text, x, y); ctx.fillText(text, x, y);
  ctx.restore();
}

function drawMenu(t) {
  ctx.fillStyle = "rgba(0,0,0,0.42)"; ctx.fillRect(0, 0, W, H);
  const s = Math.min(1.2, Math.max(0.55, W / 1100));
  const cy = H * 0.4;
  const icons = ["Watermelon", "Orange", "Apple", "Lemon", "Grape"];
  icons.forEach((n, i) => {
    const r = Math.round(28 * s);
    const x = W / 2 + (i - 2) * r * 3.1;
    blit(fruitSprite(n, r), x, cy - 130 * s + Math.sin(t * 2 + i) * 5 * s, Math.sin(t + i) * 0.15);
  });
  glowTitle("FRUIT NINJA", W / 2, cy - 30 * s, 70 * s, GOLD);
  ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#e8e8ee"; ctx.font = font(900, 22 * s);
  ctx.fillText("C  A  M", W / 2, cy + 26 * s);
  const bw = 330 * s;
  const grad = ctx.createLinearGradient(W / 2 - bw / 2, 0, W / 2 + bw / 2, 0);
  grad.addColorStop(0, "rgba(90,190,255,0)"); grad.addColorStop(1, "rgba(90,190,255,1)");
  ctx.strokeStyle = grad; ctx.lineWidth = 5 * s; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(W / 2 - bw / 2, cy + 48 * s); ctx.lineTo(W / 2 + bw / 2, cy + 42 * s); ctx.stroke();
  ctx.fillStyle = "#f2f2f6"; ctx.font = font(500, 18 * s);
  ctx.fillText(mode === "cam" ? "Raise your index finger and slash" : "Swipe to slice, and don't touch the bombs", W / 2, cy + 88 * s);
  if (mode) {
    ctx.font = font(700, 18 * s); ctx.fillStyle = "#fff";
    ctx.fillText("PRESS SPACE OR TAP TO PLAY", W / 2, cy + 124 * s);
  }
  if (bestScore > 0) {
    ctx.fillStyle = "#acacb6"; ctx.font = font(700, 14 * s);
    ctx.fillText(`BEST  ${bestScore.toLocaleString()}`, W / 2, cy + 156 * s);
  }
}

function drawGameOver(t) {
  const since = t - fx.overAt;
  ctx.fillStyle = `rgba(0,0,0,${Math.min(0.5, since)})`; ctx.fillRect(0, 0, W, H);
  if (since < 0.5) return;
  const s = Math.min(1.2, Math.max(0.55, W / 1100));
  const cy = H * 0.42;
  glowTitle("GAME OVER", W / 2, cy - 50 * s, 64 * s, "#ff5c60");
  ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#b2b2be"; ctx.font = font(700, 15 * s);
  ctx.fillText("FINAL SCORE", W / 2, cy + 20 * s);
  ctx.fillStyle = GOLD; ctx.font = font(900, 54 * s);
  ctx.fillText(game.score.toLocaleString(), W / 2, cy + 74 * s);
  ctx.fillStyle = "#acacb6"; ctx.font = font(700, 14 * s);
  ctx.fillText(game.score >= bestScore && game.score > 0 ? "NEW BEST!" : `BEST  ${bestScore.toLocaleString()}`, W / 2, cy + 104 * s);
  ctx.fillStyle = "#fff"; ctx.font = font(700, 17 * s);
  ctx.fillText("SPACE or TAP to slice again", W / 2, cy + 144 * s);
}

// --- main loop ---------------------------------------------------------------
let lastT = now();
let demoAt = 0;

function begin() {
  if (game.state === State.GAME_OVER && now() - fx.overAt < 0.8) return; // avoid instant restart mid-swipe
  game.start(now());
  fx.halves = []; fx.particles = []; fx.texts = []; fx.scoreShown = 0;
  controls.classList.add("hidden");
}

function demoTrail(t) {
  // Idle menu swipe so the title screen shows off the blade before anyone plays.
  if (mode || trail.length) return;
  if (t > demoAt) demoAt = t + 2.2;
  const p = 1 - (demoAt - t) / 1.1;
  if (p < 0 || p > 1) return;
  pushTip(W * (0.66 + 0.18 * p), H * (0.72 - 0.22 * Math.sin(p * Math.PI)), t);
}

function frame() {
  const t = now();
  const dt = Math.min(0.05, t - lastT);
  lastT = t;

  if (mode === "cam") trackHand(t);
  demoTrail(t);
  ageTrail(t);

  const wasPlaying = game.state === State.PLAYING;
  game.update(dt, trail, t);
  for (const e of game.events) (e.bomb ? spawnBombFx : spawnSliceFx)(e);
  if (wasPlaying && game.state === State.GAME_OVER) {
    fx.overAt = t;
    for (const f of game.fruits) if (!f.isBomb) spawnSliceFx({ ...f, points: 0, angle: 0, combo: 0, quiet: true });
    game.fruits = [];
    if (game.score > bestScore) {
      bestScore = game.score;
      try { localStorage.setItem("fnc-best", String(bestScore)); } catch { /* storage unavailable */ }
    }
  }
  updateFx(dt);

  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.save();
  if (fx.shake > 0) {
    const m = fx.shake * 14 * K;
    ctx.translate(rand(-m, m), rand(-m, m));
  }
  drawBackdrop();
  drawFruits(t);
  drawHalves();
  drawParticles();
  drawTexts();
  ctx.restore();
  if (fx.flash > 0) { ctx.fillStyle = `rgba(255,236,200,${fx.flash * 0.8})`; ctx.fillRect(0, 0, W, H); }
  if (game.state !== State.MENU) drawHud(t);
  if (game.state === State.MENU) drawMenu(t);
  if (game.state === State.GAME_OVER) drawGameOver(t);
  drawBlade(t);

  requestAnimationFrame(frame);
}

// --- wiring ------------------------------------------------------------------
btnPointer.addEventListener("click", (e) => {
  e.stopPropagation();
  mode = "pointer";
  begin();
});

btnCam.addEventListener("click", async (e) => {
  e.stopPropagation();
  btnCam.disabled = btnPointer.disabled = true;
  try {
    await initCamera();
    mode = "cam";
    statusEl.textContent = "";
    begin();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Couldn't start the camera. You can still play with mouse or touch.";
    btnPointer.disabled = false;
  } finally {
    btnCam.disabled = false;
  }
});

window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && mode) { e.preventDefault(); if (game.state !== State.PLAYING) begin(); }
  if ((e.key === "Escape" || e.key === "q") && game.state !== State.MENU) {
    game.state = State.MENU;
    game.fruits = [];
  }
});

resize();
requestAnimationFrame(frame);
