// Cloth topology for the hero's free-hanging lab coat.
//
// The GLB contains a skinned mesh `CoatCloth` whose vertices form a regular
// grid (rows x cols, row 0 = the seam with the rigid shoulder yoke). Each
// vertex carries its grid coordinate in the second UV set ("ClothGrid").
// Row 0 particles are pinned to the skinned yoke seam every frame; every
// other particle is driven only by the solver.
import * as THREE from 'three/webgpu';

export const MAX_NEIGHBORS = 12;

export function buildTopology(skinnedMesh) {
  const meta = JSON.parse(skinnedMesh.userData.cloth);
  const { rows, cols, ventCol, ventRow } = meta;
  const geo = skinnedMesh.geometry;
  const grid = geo.attributes.uv1;
  const pos = geo.attributes.position;
  const N = rows * cols;

  // particle -> representative vertex, vertex -> particle
  const vertexOfParticle = new Int32Array(N).fill(-1);
  const particleOfVertex = new Float32Array(pos.count);
  for (let v = 0; v < pos.count; v++) {
    const c = Math.round(grid.getX(v));
    // Blender's exporter flips V: stored = 1 - row
    const r = Math.round(1 - grid.getY(v));
    const p = r * cols + c;
    particleOfVertex[v] = p;
    if (vertexOfParticle[p] < 0) vertexOfParticle[p] = v;
  }
  const rest = new Float32Array(N * 3);
  const tmp = new THREE.Vector3();
  for (let p = 0; p < N; p++) {
    tmp.fromBufferAttribute(pos, vertexOfParticle[p]);
    rest.set([tmp.x, tmp.y, tmp.z], p * 3);
  }
  // row 0 is sewn to the yoke; the next rows are softly held to their skinned
  // positions so the coat leaves the shoulders smoothly instead of sagging away
  const pinned = new Float32Array(N);
  const softPin = new Float32Array(N);
  const SOFT = [0, 0.55, 0.30, 0.12];
  for (let c = 0; c < cols; c++) { pinned[c] = 1; for (let r = 1; r < SOFT.length; r++) softPin[r * cols + c] = SOFT[r]; }
  const ANCHOR_ROWS = SOFT.length;

  const nbrIdx = new Int32Array(N * MAX_NEIGHBORS).fill(-1);
  const nbrRest = new Float32Array(N * MAX_NEIGHBORS);
  const nbrStiff = new Float32Array(N * MAX_NEIGHBORS);
  const count = new Int32Array(N);
  const dist = (a, b) => Math.hypot(rest[a * 3] - rest[b * 3], rest[a * 3 + 1] - rest[b * 3 + 1], rest[a * 3 + 2] - rest[b * 3 + 2]);
  const id = (r, c) => r * cols + c;
  // the two coincident back-centre columns (ventCol, ventCol+1) are welded
  // above the vent and left free below it, so the vent flaps open
  const spans = (c0, c1) => c0 <= ventCol && c1 > ventCol;
  const link = (a, b, stiff, restLen) => {
    for (const [i, j] of [[a, b], [b, a]]) {
      if (count[i] >= MAX_NEIGHBORS) continue;
      const k = i * MAX_NEIGHBORS + count[i]++;
      nbrIdx[k] = j; nbrRest[k] = restLen ?? dist(a, b); nbrStiff[k] = stiff;
    }
  };
  const STRUCT = 1.0, SHEAR = 0.6, BEND = 0.34;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = id(r, c);
      if (c + 1 < cols) {
        if (c === ventCol) { if (r < ventRow) link(i, id(r, c + 1), STRUCT, 0); }
        else link(i, id(r, c + 1), STRUCT);
      }
      if (r + 1 < rows) link(i, id(r + 1, c), STRUCT);
      if (r + 1 < rows && c + 1 < cols && (!spans(c, c + 1) || r + 1 < ventRow)) {
        link(i, id(r + 1, c + 1), SHEAR);
        link(id(r, c + 1), id(r + 1, c), SHEAR);
      }
      if (c + 2 < cols && (!spans(c, c + 2) || r < ventRow)) link(i, id(r, c + 2), BEND);
      if (r + 2 < rows) link(i, id(r + 2, c), BEND);
    }
  }
  // tethers: max distance from each particle to the pinned particle at the top of its column
  const tether = new Float32Array(N * 2);
  for (let c = 0; c < cols; c++) {
    let acc = 0;
    for (let r = 0; r < rows; r++) {
      if (r > 0) acc += dist(id(r, c), id(r - 1, c));
      tether[id(r, c) * 2] = id(0, c);
      tether[id(r, c) * 2 + 1] = acc;
    }
  }
  // max distance from the animated (skinned) coat surface, by row: 0 at the seam, ~8 cm on the chest,
  // ~20 cm at the waist, ~50 cm at the hem (the UE/Unity cloth "max distance" paint), plus a backstop
  // that keeps the cloth from passing behind that surface into the body
  const maxDist = new Float32Array(N);
  for (let r = 0; r < rows; r++) {
    const t = Math.max(r - (ANCHOR_ROWS - 1), 0) / Math.max(rows - ANCHOR_ROWS, 1);
    for (let c = 0; c < cols; c++) maxDist[id(r, c)] = r < ANCHOR_ROWS ? 0.01 : 0.02 + 0.48 * Math.pow(t, 1.25);
  }
  // grid normals in the rest pose point outward or inward depending on winding: find which
  let out = 0;
  for (let r = 1; r < rows - 1; r++) for (let c = 1; c < cols - 1; c++) {
    const i = id(r, c);
    const n = gridNormal(rest, rows, cols, r, c);
    // outward ~ away from the vertical body axis through (0, y, 0)
    out += n[0] * rest[i * 3] + n[2] * rest[i * 3 + 2] > 0 ? 1 : -1;
  }
  const normalSign = out >= 0 ? 1 : -1;
  return { meta, rows, cols, N, rest, pinned, softPin, anchorRows: ANCHOR_ROWS, nbrIdx, nbrRest, nbrStiff, tether, vertexOfParticle, particleOfVertex, maxDist, normalSign };
}

/** Unit normal of a (rows x cols) particle grid at (r, c) from central differences. */
export function gridNormal(p, rows, cols, r, c, out = [0, 0, 0]) {
  const L = r * cols + Math.max(c - 1, 0), R = r * cols + Math.min(c + 1, cols - 1);
  const U = Math.max(r - 1, 0) * cols + c, D = Math.min(r + 1, rows - 1) * cols + c;
  const ex = p[R * 3] - p[L * 3], ey = p[R * 3 + 1] - p[L * 3 + 1], ez = p[R * 3 + 2] - p[L * 3 + 2];
  const fx = p[D * 3] - p[U * 3], fy = p[D * 3 + 1] - p[U * 3 + 1], fz = p[D * 3 + 2] - p[U * 3 + 2];
  const nx = ey * fz - ez * fy, ny = ez * fx - ex * fz, nz = ex * fy - ey * fx;
  const l = Math.hypot(nx, ny, nz) || 1;
  out[0] = nx / l; out[1] = ny / l; out[2] = nz / l;
  return out;
}

// Capsule colliders in bind-pose model space, rigidly attached to bones.
// [bone, ax, ay, az, bx, by, bz, radius]
export const COLLIDERS = [
  ['spine_03', -0.047, 1.390, -0.019, 0.047, 1.390, -0.019, 0.150],
  ['spine_03', -0.027, 1.340, -0.015, 0.027, 1.340, -0.015, 0.172],
  ['spine_02', -0.036, 1.280, -0.006, 0.036, 1.280, -0.006, 0.160],
  ['spine_02', -0.027, 1.220, 0.013, 0.027, 1.220, 0.013, 0.126],
  ['spine_02', -0.027, 1.160, 0.012, 0.027, 1.160, 0.012, 0.125],
  ['spine_01', -0.028, 1.100, 0.013, 0.028, 1.100, 0.013, 0.126],
  ['spine_01', -0.028, 1.040, 0.006, 0.028, 1.040, 0.006, 0.127],
  ['pelvis', -0.044, 0.980, 0.000, 0.044, 0.980, 0.000, 0.126],
  ['pelvis', -0.060, 0.920, -0.007, 0.060, 0.920, -0.007, 0.121],
  ['thigh_l', 0.094, 0.880, -0.001, 0.093, 0.550, 0.013, 0.098],
  ['calf_l', 0.093, 0.530, 0.008, 0.094, 0.160, -0.032, 0.072],
  ['thigh_r', -0.094, 0.880, -0.001, -0.093, 0.550, 0.013, 0.098],
  ['calf_r', -0.093, 0.530, 0.008, -0.094, 0.160, -0.033, 0.072],
  ['upperarm_l', 0.225, 1.442, -0.065, 0.455, 1.441, -0.070, 0.066],
  ['upperarm_r', -0.225, 1.442, -0.065, -0.455, 1.441, -0.070, 0.066],
  ['lowerarm_l', 0.480, 1.441, -0.070, 0.725, 1.441, -0.066, 0.045],
  ['lowerarm_r', -0.480, 1.441, -0.070, -0.725, 1.441, -0.066, 0.045],
  // buttocks (the tail used to slip into the hip/thigh fold)
  ['pelvis', -0.065, 0.900, -0.035, 0.065, 0.900, -0.035, 0.110],
  // the shield plate on the left forearm (two flat-ish capsules along its height)
  ['lowerarm_l', 0.530, 1.525, -0.270, 0.530, 1.525, 0.130, 0.060],
  ['lowerarm_l', 0.670, 1.525, -0.270, 0.670, 1.525, 0.130, 0.060],
];
