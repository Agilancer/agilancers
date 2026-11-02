// Agilancer viewer: the hero with every UAL2 animation, a WebGPU-simulated
// lab coat, and swappable lance / shield variants.
import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { CoatCloth } from './cloth/CoatCloth.js';
import { attachArms, ARMS } from './arms.js';

const q = new URLSearchParams(location.search);
const CAPTURE = q.has('capture');              // deterministic stepping for headless screenshots
const BASE = q.get('base') || './assets/models/';

// ------------------------------------------------------------------ renderer / scene
const renderer = new THREE.WebGPURenderer({ antialias: true, forceWebGL: q.get('gl') === '1' });
renderer.setPixelRatio(Math.min(devicePixelRatio, innerWidth < 700 ? 1.5 : 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('app').appendChild(renderer.domElement);
await renderer.init();
const isGPU = !!renderer.backend.isWebGPUBackend;
const badge = document.getElementById('backend');
badge.textContent = isGPU ? 'WebGPU' : 'WebGL2 fallback';
badge.classList.toggle('gpu', isGPU);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1c1f25);
scene.fog = new THREE.Fog(0x1c1f25, 9, 22);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.5;

const key = new THREE.DirectionalLight(0xfff0dd, 2.6);
key.position.set(2.2, 4.0, 3.0);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -2.2, right: 2.2, top: 2.6, bottom: -0.4, near: 0.5, far: 12 });
key.shadow.bias = -0.0004; key.shadow.normalBias = 0.025; key.shadow.radius = 3;
scene.add(key, key.target);
const rim = new THREE.DirectionalLight(0xaecbff, 1.4); rim.position.set(-3, 2.6, -3.2); scene.add(rim);
scene.add(new THREE.HemisphereLight(0xdfe6f0, 0x2a2622, 0.35));

// ground: soft disc + subtle grid so root motion reads
const groundTex = (() => {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#34373e'; g.fillRect(0, 0, 512, 512);
  g.strokeStyle = 'rgba(255,255,255,0.07)'; g.lineWidth = 2;
  for (let i = 0; i <= 512; i += 64) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 512); g.stroke(); g.beginPath(); g.moveTo(0, i); g.lineTo(512, i); g.stroke(); }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(12, 12); t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8; return t;
})();
const ground = new THREE.Mesh(new THREE.CircleGeometry(12, 96).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ map: groundTex, roughness: 0.92 }));
ground.receiveShadow = true; scene.add(ground);

const camera = new THREE.PerspectiveCamera(32, innerWidth / innerHeight, 0.05, 60);
camera.position.set(2.4, 1.5, 4.1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true; controls.minDistance = 0.6; controls.maxDistance = 12; controls.maxPolarAngle = Math.PI * 0.53;

addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

// ------------------------------------------------------------------ hero
const loader = new GLTFLoader();

// Hosts that do not serve .glb get "<file>.b64.txt" (set window.AGILANCER_B64 in the page).
const B64 = new Uint8Array(128); 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'.split('').forEach((c, i) => { B64[c.charCodeAt(0)] = i; });
function decodeBase64(txt) {
  txt = txt.replace(/[^A-Za-z0-9+/]/g, '');
  const n = txt.length, out = new Uint8Array(Math.floor(n * 3 / 4));
  let o = 0;
  for (let i = 0; i + 3 < n; i += 4) {
    const v = (B64[txt.charCodeAt(i)] << 18) | (B64[txt.charCodeAt(i + 1)] << 12) | (B64[txt.charCodeAt(i + 2)] << 6) | B64[txt.charCodeAt(i + 3)];
    out[o++] = v >> 16; out[o++] = (v >> 8) & 255; out[o++] = v & 255;
  }
  const rem = n % 4;
  if (rem) {
    const i = n - rem;
    const v = (B64[txt.charCodeAt(i)] << 18) | (B64[txt.charCodeAt(i + 1)] << 12) | (rem > 2 ? B64[txt.charCodeAt(i + 2)] << 6 : 0);
    out[o++] = v >> 16; if (rem > 2) out[o++] = (v >> 8) & 255;
  }
  return out.buffer.slice(0, o);
}
async function loadModel(file) {
  if (!window.AGILANCER_B64 && !q.has('b64')) return loader.loadAsync(BASE + file);
  const res = await fetch(BASE + file + '.b64.txt');
  if (!res.ok) throw new Error(`could not load ${file} (${res.status})`);
  return loader.parseAsync(decodeBase64(await res.text()), BASE);
}
const setLoading = (t) => { document.getElementById('loading-text').textContent = t; };
setLoading('Loading hero…');
const gltf = await loadModel('agilancer_hero.glb');
const hero = gltf.scene;
scene.add(hero);
let clothSource = null;
hero.traverse((o) => {
  if (o.isMesh) {
    o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false;
    const m = o.material;
    if (m.name === 'M_Hair') { m.envMapIntensity = 0.5; }
    if (m.name === 'M_Head' || m.name === 'M_Arms') { m.roughness = 0.6; }
    if (m.normalMap) m.normalScale.set(0.8, 0.8);
  }
  if (o.userData?.cloth) clothSource = o;
});
const skinned = [];
hero.traverse((o) => { if (o.isSkinnedMesh) skinned.push(o); });
const skeleton = skinned[0].skeleton;
const bone = (name) => skeleton.bones.find((b) => b.name === name);
const rootBone = bone('root');
const pelvisBone = bone('pelvis');

// ------------------------------------------------------------------ cloth
hero.updateMatrixWorld(true);
let cloth = null;
if (clothSource) {
  cloth = new CoatCloth({ renderer, clothMesh: clothSource, useGPU: isGPU && q.get('cpucloth') !== '1' });
  scene.add(cloth.mesh);
  clothSource.visible = false;
}

// ------------------------------------------------------------------ lance & shield
setLoading('Loading lance & shield…');
let arms = null;
try {
  const wg = await loadModel('agilancer_arms.glb');
  arms = attachArms(wg.scene, skinned[0]);
} catch (e) { console.warn('arms not available', e); document.getElementById('sec-weapons').style.display = 'none'; }

// ------------------------------------------------------------------ animation
const mixer = new THREE.AnimationMixer(hero);
const clipsInPlace = gltf.animations.filter((c) => c.name !== 'A_TPose' || q.has('tpose'));
let clipsRM = null;
const GROUPS = [
  ['Lance & shield combat', /^(Sword_|Shield_|Idle_Shield|Melee_|OverhandThrow)/],
  ['Movement', /^(NinjaJump|Slide|ClimbUp|Walk|Zombie_Walk|Hit_|LayToIdle)/],
  ['Idles & gestures', /^(Idle_|Yes|Consume|Chest_)/],
  ['Work & misc', /./],
];
let current = null, currentName = q.get('anim') || 'Idle_Shield_Loop';
let rootMotion = false, playing = true, speed = 1, autoAdvance = false;

function clipByName(name) {
  const list = rootMotion && clipsRM ? clipsRM : gltf.animations;
  return list.find((c) => c.name === name) || gltf.animations.find((c) => c.name === name);
}

function play(name, fade = 0.25) {
  const clip = clipByName(name);
  if (!clip) return;
  currentName = name;
  const action = mixer.clipAction(clip);
  action.reset();
  action.setLoop(THREE.LoopRepeat, Infinity);
  action.play();
  if (current && current !== action) current.crossFadeTo(action, fade, false);
  current = action;
  document.getElementById('anim-name').textContent = name.replace(/_/g, ' ');
  document.querySelectorAll('.anim').forEach((b) => b.classList.toggle('active', b.dataset.name === name));
}

const order = [];
function buildList(filter = '') {
  const list = document.getElementById('anim-list');
  list.innerHTML = '';
  order.length = 0;
  const used = new Set();
  for (const [label, re] of GROUPS) {
    const items = clipsInPlace.filter((c) => re.test(c.name) && !used.has(c.name) && c.name.toLowerCase().includes(filter.toLowerCase()));
    if (!items.length) continue;
    const h = document.createElement('div'); h.className = 'grp'; h.textContent = label; list.appendChild(h);
    for (const c of items) {
      used.add(c.name); order.push(c.name);
      const b = document.createElement('button');
      b.className = 'anim' + (c.name === currentName ? ' active' : '');
      b.dataset.name = c.name;
      b.innerHTML = `${c.name.replace(/_/g, ' ')} <small>${c.duration.toFixed(1)}s</small>`;
      b.onclick = () => play(c.name);
      list.appendChild(b);
    }
  }
}
buildList();
document.getElementById('anim-search').oninput = (e) => buildList(e.target.value);
const step = (d) => { const i = order.indexOf(currentName); play(order[(i + d + order.length) % order.length]); };
document.getElementById('btn-next').onclick = () => step(1);
document.getElementById('btn-prev').onclick = () => step(-1);
document.getElementById('btn-play').onclick = (e) => { playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; };
const rngSpeed = document.getElementById('rng-speed');
rngSpeed.oninput = () => { speed = +rngSpeed.value; document.getElementById('out-speed').textContent = speed.toFixed(2) + '×'; };
document.getElementById('chk-auto').onchange = (e) => { autoAdvance = e.target.checked; };
document.getElementById('chk-rm').onchange = async (e) => {
  rootMotion = e.target.checked;
  if (rootMotion && !clipsRM) {
    e.target.disabled = true;
    const rm = await loadModel('UAL2_Standard_RM.glb');
    clipsRM = rm.animations;
    e.target.disabled = false;
  }
  mixer.stopAllAction(); current = null; play(currentName, 0);
  if (cloth) cloth.reset();
};

// cloth UI
const chkCloth = document.getElementById('chk-cloth');
chkCloth.onchange = () => {
  if (!cloth) return;
  cloth.enabled = chkCloth.checked;
  cloth.mesh.visible = cloth.enabled; clothSource.visible = !cloth.enabled;
  if (cloth.enabled) cloth.reset();
};
const rngWind = document.getElementById('rng-wind');
rngWind.oninput = () => { document.getElementById('out-wind').textContent = (+rngWind.value).toFixed(1); };
const rngIter = document.getElementById('rng-iter');
rngIter.oninput = () => { document.getElementById('out-iter').textContent = rngIter.value; if (cloth) cloth.params.iterations = +rngIter.value; };
document.getElementById('btn-reset').onclick = () => cloth?.reset();
let colDbg = null;
document.getElementById('chk-col').onchange = (e) => {
  if (!cloth) return;
  if (!colDbg) colDbg = cloth.colliderDebug(scene);
  colDbg.visible = e.target.checked;
};

// arms UI
if (arms) {
  const selL = document.getElementById('sel-lance'), selS = document.getElementById('sel-shield');
  ARMS.forEach((v, i) => { selL.add(new Option(v.label, v.key)); selS.add(new Option(v.label, v.key)); });
  selL.value = q.get('lance') || 'marble'; selS.value = q.get('shield') || 'marble';
  arms.show(selL.value, selS.value);
  selL.onchange = selS.onchange = () => arms.show(selL.value, selS.value);
  document.getElementById('chk-weapons').onchange = (e) => arms.group.visible = e.target.checked;
}

// ------------------------------------------------------------------ loop
if (q.has('rm')) {                      // start with root motion (used by the capture harness)
  const rm = await loadModel('UAL2_Standard_RM.glb');
  clipsRM = rm.animations; rootMotion = true; document.getElementById('chk-rm').checked = true;
}
if (q.has('wire')) hero.traverse((o) => { if (o.isMesh) { o.material = o.material.clone(); o.material.wireframe = true; } });
play(currentName, 0);
mixer.update(+(q.get('t') || 0));
if (q.has('speed')) speed = +q.get('speed');
hero.updateMatrixWorld(true);
cloth?.reset();

const clock = new THREE.Clock();
const lastRoot = new THREE.Vector3();
const tmp = new THREE.Vector3();
const camOffset = new THREE.Vector3();
let fpsAcc = 0, fpsN = 0, lastStats = 0;
pelvisBone.getWorldPosition(lastRoot);

function simulate(dt) {
  const prevTime = current ? current.time : 0;
  mixer.update(playing ? dt * speed : 0);
  if (autoAdvance && current && current.time < prevTime) step(1);
  hero.updateMatrixWorld(true);
  // root motion: when the clip wraps, the hips jump back -> carry the cloth along
  pelvisBone.getWorldPosition(tmp);
  const jump = tmp.clone().sub(lastRoot);
  if (cloth && jump.lengthSq() > 0.35 * 0.35) cloth.translate(jump.x, jump.y, jump.z);
  if (rootMotion) {
    camOffset.set(tmp.x - controls.target.x, 0, tmp.z - controls.target.z);
    controls.target.add(camOffset); camera.position.add(camOffset);
  }
  lastRoot.copy(tmp);
  if (cloth?.enabled) {
    const w = +rngWind.value;
    const a = performance.now() * 0.00012;
    cloth.params.wind[0] = Math.cos(a) * w; cloth.params.wind[2] = -Math.abs(Math.sin(a)) * w * 0.6 - w * 0.6;
    cloth.update(playing ? dt * speed : 0);
    if (colDbg?.visible) cloth.updateDebug();
  }
}

function frame() {
  const dt = Math.min(clock.getDelta(), 1 / 15);
  if (!CAPTURE) simulate(dt);
  controls.update();
  renderer.render(scene, camera);
  fpsAcc += dt; fpsN++;
  if (fpsAcc - lastStats > 0.5) {
    document.getElementById('stats').textContent = `${(fpsN / fpsAcc).toFixed(0)} fps · coat ${cloth ? cloth.topo.N + ' particles on ' + (cloth.useGPU ? 'GPU compute' : 'CPU') : 'off'}`;
    fpsAcc = 0; fpsN = 0;
  }
}
renderer.setAnimationLoop(frame);
document.getElementById('loading').classList.add('done');
if (innerWidth < 700) document.getElementById('panel').classList.remove('open');
document.getElementById('panel-toggle').onclick = () => document.getElementById('panel').classList.toggle('open');

// ------------------------------------------------------------------ headless capture hooks
window.__app = { THREE, scene, camera, controls, hero, mixer, cloth, play, simulate, renderer, bone };
window.__advance = (seconds, fps = 60) => { const n = Math.round(seconds * fps); for (let i = 0; i < n; i++) simulate(1 / fps); };
if (CAPTURE) {
  // ?capture&views=front,back&settle=2 : step the sim deterministically, then expose named views
  document.getElementById('panel').style.display = 'none';
  const VIEWS = {
    front: [[0, 1.05, 4.4], [0, 0.95, 0]], back: [[0, 1.05, -4.4], [0, 0.95, 0]], left: [[4.4, 1.05, 0], [0, 0.95, 0]], right: [[-4.4, 1.05, 0], [0, 0.95, 0]],
    q34: [[2.5, 1.35, 3.5], [0, 0.95, 0]], q34b: [[-2.6, 1.35, -3.4], [0, 0.95, 0]], q34l: [[-2.6, 1.3, 3.4], [0, 0.95, 0]],
    face: [[0, 1.69, 0.62], [0, 1.68, 0]], face34: [[0.42, 1.71, 0.46], [0, 1.68, 0]], headback: [[-0.35, 1.82, -0.55], [0, 1.72, 0]],
    torso: [[0.3, 1.3, 1.6], [0, 1.15, 0]], low: [[1.6, 0.4, 2.6], [0, 0.8, 0]],
  };
  const follow = q.has('follow');
  window.__views = (q.get('views') || 'q34').split(',');
  window.__setView = async (k) => {
    k = k.split('~')[0];                  // '~n' suffix only keeps screenshot names unique
    if (k.startsWith('adv')) {            // adv<seconds>_<view>: advance the simulation, then frame a view
      const [a, v] = k.slice(3).split('_');
      window.__advance(+a);
      k = v;
    }
    const [p, t] = VIEWS[k] || JSON.parse(decodeURIComponent(k));
    const off = new THREE.Vector3();
    if (follow) { pelvisBone.getWorldPosition(off); off.y = 0; }
    camera.position.set(p[0] + off.x, p[1], p[2] + off.z); controls.target.set(t[0] + off.x, t[1], t[2] + off.z);
    controls.update(); await new Promise((r) => setTimeout(r, 150));
  };
  window.__frames = async (n, dt) => { for (let i = 0; i < n; i++) simulate(dt); };
  window.__advance(+(q.get('settle') || 2));
  if (q.has('pause')) playing = false;
}
window.__done = true;
