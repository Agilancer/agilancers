// CPU reference solver (used on the WebGL2 fallback backend). Same algorithm
// as the WebGPU compute path: Verlet integration, Jacobi-averaged distance
// constraints, column tethers, capsule collisions, ground plane.
import { MAX_NEIGHBORS } from './topology.js';

export class ClothCPU {
  constructor(topo, params) {
    this.topo = topo;
    this.params = params;
    const N = topo.N;
    this.pos = new Float32Array(N * 3);
    this.prev = new Float32Array(N * 3);
    this.tmp = new Float32Array(N * 3);
    this.normals = new Float32Array(N * 3);
    this.anchor0 = new Float32Array(N * 3);
    this.anchor1 = new Float32Array(N * 3);
    this.colliders = new Float32Array(64 * 8);
    this.numColliders = 0;
  }

  setState(positions) { this.pos.set(positions); this.prev.set(positions); }
  setAnchors(a0, a1) { this.anchor0.set(a0); this.anchor1.set(a1); }
  setColliders(data, n) { this.colliders.set(data); this.numColliders = n; }
  translate(dx, dy, dz) {
    for (const arr of [this.pos, this.prev]) for (let i = 0; i < arr.length; i += 3) { arr[i] += dx; arr[i + 1] += dy; arr[i + 2] += dz; }
  }

  step(dt, alpha, time) {
    const { N, pinned, nbrIdx, nbrRest, nbrStiff, tether } = this.topo;
    const P = this.params;
    const pos = this.pos, prev = this.prev;
    const g = P.gravity, damp = P.damping, air = P.airDrag;
    const wx = P.wind[0], wy = P.wind[1], wz = P.wind[2];
    // integrate
    for (let i = 0; i < N; i++) {
      const o = i * 3;
      if (pinned[i] > 0.5) {
        for (let k = 0; k < 3; k++) { const a = this.anchor0[o + k] + (this.anchor1[o + k] - this.anchor0[o + k]) * alpha; pos[o + k] = a; prev[o + k] = a; }
        continue;
      }
      const gust = 1 + 0.45 * Math.sin(time * 2.3 + pos[o] * 3.1 + pos[o + 1] * 1.7) * Math.sin(time * 1.3 + pos[o + 2] * 2.2);
      for (let k = 0; k < 3; k++) {
        const x = pos[o + k];
        let v = (x - prev[o + k]) * damp;
        const w = (k === 0 ? wx : k === 1 ? wy : wz) * gust;
        v += (w * dt - v) * air;           // relax toward the wind/air velocity (drag)
        prev[o + k] = x;
        pos[o + k] = x + v + (k === 1 ? g * dt * dt : 0);
      }
      const sp = this.topo.softPin[i];
      if (sp > 0) for (let k = 0; k < 3; k++) { const a = this.anchor0[o + k] + (this.anchor1[o + k] - this.anchor0[o + k]) * alpha; pos[o + k] += (a - pos[o + k]) * sp; }
    }
    // constraints (Jacobi, ping-pong)
    let src = pos, dst = this.tmp;
    for (let it = 0; it < P.iterations; it++) {
      for (let i = 0; i < N; i++) {
        const o = i * 3;
        let x = src[o], y = src[o + 1], z = src[o + 2];
        if (pinned[i] > 0.5) { dst[o] = x; dst[o + 1] = y; dst[o + 2] = z; continue; }
        let cx = 0, cy = 0, cz = 0, wsum = 0;
        for (let k = 0; k < MAX_NEIGHBORS; k++) {
          const j = nbrIdx[i * MAX_NEIGHBORS + k];
          if (j < 0) break;
          const q = j * 3;
          const dx = src[q] - x, dy = src[q + 1] - y, dz = src[q + 2] - z;
          const len = Math.max(Math.hypot(dx, dy, dz), 1e-6);
          const s = nbrStiff[i * MAX_NEIGHBORS + k];
          const share = pinned[j] > 0.5 ? 1.0 : 0.5;
          const f = (len - nbrRest[i * MAX_NEIGHBORS + k]) / len * s * share;
          cx += dx * f; cy += dy * f; cz += dz * f; wsum += s;
        }
        const w = P.relax / Math.max(wsum * 0.5, 1e-6);
        x += cx * w; y += cy * w; z += cz * w;
        // tether to the pinned particle at the top of the column
        const a = tether[i * 2] * 3, maxd = tether[i * 2 + 1] * P.tetherSlack;
        let tx = x - src[a], ty = y - src[a + 1], tz = z - src[a + 2];
        const tl = Math.hypot(tx, ty, tz);
        if (tl > maxd) { const f = maxd / tl; x = src[a] + tx * f; y = src[a + 1] + ty * f; z = src[a + 2] + tz * f; }
        // a coat hangs from its seam: never above it along the body's up axis (no flipping over the shoulders)
        { const U = P.up, h = (x - src[a]) * U[0] + (y - src[a + 1]) * U[1] + (z - src[a + 2]) * U[2] - P.aboveSeam;
          if (h > 0) { x -= U[0] * h; y -= U[1] * h; z -= U[2] * h; } }
        // capsules
        for (let c = 0; c < this.numColliders; c++) {
          const C = this.colliders, b = c * 8;
          const ax = C[b], ay = C[b + 1], az = C[b + 2], r = C[b + 3] + P.thickness;
          const bx = C[b + 4] - ax, by = C[b + 5] - ay, bz = C[b + 6] - az;
          const px = x - ax, py = y - ay, pz = z - az;
          const bb = bx * bx + by * by + bz * bz;
          const h = Math.min(Math.max((px * bx + py * by + pz * bz) / Math.max(bb, 1e-9), 0), 1);
          const qx = px - bx * h, qy = py - by * h, qz = pz - bz * h;
          const d = Math.hypot(qx, qy, qz);
          if (d < r) { const f = r / Math.max(d, 1e-6); x = ax + bx * h + qx * f; y = ay + by * h + qy * f; z = az + bz * h + qz * f; }
        }
        if (y < P.ground) y = P.ground;
        dst[o] = x; dst[o + 1] = y; dst[o + 2] = z;
      }
      const t = src; src = dst; dst = t;
    }
    if (src !== pos) pos.set(src);
  }

  computeNormals() {
    const { rows, cols } = this.topo;
    const p = this.pos, n = this.normals;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const L = r * cols + Math.max(c - 1, 0), R = r * cols + Math.min(c + 1, cols - 1);
        const U = Math.max(r - 1, 0) * cols + c, D = Math.min(r + 1, rows - 1) * cols + c;
        const ex = p[R * 3] - p[L * 3], ey = p[R * 3 + 1] - p[L * 3 + 1], ez = p[R * 3 + 2] - p[L * 3 + 2];
        const fx = p[D * 3] - p[U * 3], fy = p[D * 3 + 1] - p[U * 3 + 1], fz = p[D * 3 + 2] - p[U * 3 + 2];
        let nx = ey * fz - ez * fy, ny = ez * fx - ex * fz, nz = ex * fy - ey * fx;
        const l = Math.hypot(nx, ny, nz) || 1;
        const o = (r * cols + c) * 3;
        n[o] = nx / l; n[o + 1] = ny / l; n[o + 2] = nz / l;
      }
    }
  }
}
