// WebGPU compute solver for the coat (three.js TSL). One thread per
// particle; constraints are solved Jacobi-style with ping-pong buffers so
// every kernel only writes its own particle.
import * as THREE from 'three/webgpu';
import {
  Fn, instancedArray, uniform, instanceIndex, If, Loop, vec3, vec4, float, int,
  select, max, min, clamp, dot, length, cross, normalize, mix, sin, attribute,
  transformNormalToView, faceDirection,
} from 'three/tsl';
import { MAX_NEIGHBORS } from './topology.js';

export const MAX_COLLIDERS = 24;

export class ClothGPU {
  constructor(renderer, topo, params) {
    this.renderer = renderer;
    this.topo = topo;
    this.params = params;
    const N = topo.N;
    const v4 = (n) => instancedArray(new Float32Array(n * 4), 'vec4');
    this.posA = v4(N); this.posB = v4(N); this.prev = v4(N); this.normals = v4(N);
    this.anchor0 = v4(N); this.anchor1 = v4(N); this.tnorm = v4(N);
    this.target = v4(N);                                        // xyz = animated position this substep, w = max distance
    const md4 = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) md4[i * 4] = topo.maxDist[i];
    this.maxDist = instancedArray(md4, 'vec4');                 // x = max distance from the animated surface
    this.colliders = v4(MAX_COLLIDERS * 2);
    const pin = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) { pin[i * 4] = topo.pinned[i]; pin[i * 4 + 1] = topo.tether[i * 2]; pin[i * 4 + 2] = topo.tether[i * 2 + 1]; pin[i * 4 + 3] = topo.softPin[i]; }
    this.pinTether = instancedArray(pin, 'vec4');            // x pinned, y tether anchor id, z tether length, w soft pin
    this.nbrIdx = instancedArray(Int32Array.from(topo.nbrIdx), 'int');
    const rs = new Float32Array(N * MAX_NEIGHBORS * 2);
    for (let k = 0; k < N * MAX_NEIGHBORS; k++) { rs[k * 2] = topo.nbrRest[k]; rs[k * 2 + 1] = topo.nbrStiff[k]; }
    this.nbrRS = instancedArray(rs, 'vec2');

    this.u = {
      dt: uniform(1 / 120), alpha: uniform(1), time: uniform(0),
      gravity: uniform(params.gravity), damping: uniform(params.damping), air: uniform(params.airDrag),
      wind: uniform(new THREE.Vector3(...params.wind)), relax: uniform(params.relax),
      thickness: uniform(params.thickness), ground: uniform(params.ground), slack: uniform(params.tetherSlack),
      numColliders: uniform(0, 'int'), shift: uniform(new THREE.Vector3()),
      up: uniform(new THREE.Vector3(0, 1, 0)), aboveSeam: uniform(params.aboveSeam), backstop: uniform(params.backstop ?? 0.012),
    };
    this._build();
  }

  _build() {
    const { N, rows, cols } = this.topo;
    const u = this.u;
    const { posA, posB, prev, anchor0, anchor1, pinTether, nbrIdx, nbrRS, colliders, normals, tnorm, maxDist, target } = this;

    this.kIntegrate = Fn(() => {
      const i = instanceIndex;
      const pt = pinTether.element(i);
      // the solve kernel may bind at most 8 storage buffers: hand it one target buffer
      target.element(i).assign(vec4(mix(anchor0.element(i).xyz, anchor1.element(i).xyz, u.alpha), maxDist.element(i).x));
      If(pt.x.greaterThan(0.5), () => {
        const a = mix(anchor0.element(i).xyz, anchor1.element(i).xyz, u.alpha);
        posA.element(i).assign(vec4(a, 0));
        prev.element(i).assign(vec4(a, 0));
      }).Else(() => {
        const x = posA.element(i).xyz.toVar();
        const v = x.sub(prev.element(i).xyz).mul(u.damping).toVar();
        const gust = float(1).add(sin(u.time.mul(2.3).add(x.x.mul(3.1)).add(x.y.mul(1.7))).mul(sin(u.time.mul(1.3).add(x.z.mul(2.2)))).mul(0.45));
        v.addAssign(u.wind.mul(gust).mul(u.dt).sub(v).mul(u.air));
        prev.element(i).assign(vec4(x, 0));
        const xn = x.add(v).add(vec3(0, u.gravity.mul(u.dt).mul(u.dt), 0)).toVar();
        const a = mix(anchor0.element(i).xyz, anchor1.element(i).xyz, u.alpha);
        xn.assign(mix(xn, a, pt.w));
        posA.element(i).assign(vec4(xn, 0));
      });
    })().compute(N);

    const solve = (src, dst) => Fn(() => {
      const i = instanceIndex;
      const pt = pinTether.element(i);
      const x = src.element(i).xyz.toVar();
      If(pt.x.lessThan(0.5), () => {
        const corr = vec3(0).toVar();
        const wsum = float(0).toVar();
        const base = int(i).mul(MAX_NEIGHBORS);
        Loop(MAX_NEIGHBORS, ({ i: k }) => {
          const j = nbrIdx.element(base.add(k));
          If(j.greaterThanEqual(0), () => {
            const rsj = nbrRS.element(base.add(k));
            const d = src.element(j).xyz.sub(x);
            const len = max(length(d), 1e-6);
            const share = select(pinTether.element(j).x.greaterThan(0.5), float(1.0), float(0.5));
            corr.addAssign(d.mul(len.sub(rsj.x).div(len).mul(rsj.y).mul(share)));
            wsum.addAssign(rsj.y);
          });
        });
        x.addAssign(corr.mul(u.relax.div(max(wsum.mul(0.5), 1e-6))));
        // tether
        const a = src.element(int(pt.y)).xyz;
        const td = x.sub(a).toVar();
        const tl = length(td);
        const maxd = pt.z.mul(u.slack);
        If(tl.greaterThan(maxd), () => { x.assign(a.add(td.mul(maxd.div(tl)))); });
        // never above the seam along the body's up axis (no flipping over the shoulders)
        const hUp = dot(x.sub(a), u.up).sub(u.aboveSeam);
        If(hUp.greaterThan(0), () => { x.subAssign(u.up.mul(hUp)); });
        // max distance from the animated coat surface + backstop
        const tg = target.element(i);
        const tgt = tg.xyz;
        const dd = x.sub(tgt).toVar();
        const dl = length(dd);
        const md = tg.w;
        If(dl.greaterThan(md), () => { dd.assign(dd.mul(md.div(dl))); });
        const tn = tnorm.element(i).xyz;
        const sd = dot(dd, tn).add(u.backstop);
        If(sd.lessThan(0), () => { dd.subAssign(tn.mul(sd)); });
        x.assign(tgt.add(dd));
        // capsules: push out on the side the particle came from
        Loop(MAX_COLLIDERS, ({ i: c }) => {
          If(int(c).lessThan(u.numColliders), () => {
            const A = colliders.element(int(c).mul(2));
            const B = colliders.element(int(c).mul(2).add(1)).xyz;
            const r = A.w.add(u.thickness);
            const ba = B.sub(A.xyz);
            const pa = x.sub(A.xyz);
            const h = clamp(dot(pa, ba).div(max(dot(ba, ba), 1e-9)), 0, 1);
            const axisP = A.xyz.add(ba.mul(h));
            const q = x.sub(axisP).toVar();
            const dq = length(q);
            If(dq.lessThan(r), () => {
              // exit toward the coat's animated surface (it lies outside the body): no tunnelling to the far side
              const o = tgt.sub(axisP).toVar();
              const od = length(o);
              const dir = select(od.greaterThan(r.mul(0.5)), o.div(max(od, 1e-9)), q.div(max(dq, 1e-6)));
              x.assign(axisP.add(dir.mul(r)));
            });
          });
        });
        x.y.assign(max(x.y, u.ground));
      });
      dst.element(i).assign(vec4(x, 0));
    })().compute(N);
    this.kShift = Fn(() => {
      const i = instanceIndex;
      posA.element(i).assign(vec4(posA.element(i).xyz.add(u.shift), 0));
      prev.element(i).assign(vec4(prev.element(i).xyz.add(u.shift), 0));
    })().compute(N);
    this.kSolveAB = solve(posA, posB);
    this.kSolveBA = solve(posB, posA);

    this.kNormals = Fn(() => {
      const i = int(instanceIndex);
      const r = i.div(cols);
      const c = i.sub(r.mul(cols));
      const L = r.mul(cols).add(max(c.sub(1), 0));
      const R = r.mul(cols).add(min(c.add(1), cols - 1));
      const U = max(r.sub(1), 0).mul(cols).add(c);
      const D = min(r.add(1), rows - 1).mul(cols).add(c);
      const e = posA.element(R).xyz.sub(posA.element(L).xyz);
      const f = posA.element(D).xyz.sub(posA.element(U).xyz);
      normals.element(i).assign(vec4(normalize(cross(e, f)), 0));
    })().compute(N);
  }

  _upload(node, data, stride3 = true) {
    const arr = node.value.array;
    if (stride3) { for (let i = 0, n = data.length / 3; i < n; i++) { arr[i * 4] = data[i * 3]; arr[i * 4 + 1] = data[i * 3 + 1]; arr[i * 4 + 2] = data[i * 3 + 2]; } }
    else arr.set(data);
    node.value.needsUpdate = true;
  }

  setState(positions) { this._upload(this.posA, positions); this._upload(this.prev, positions); }
  setAnchors(a0, a1, tn) { this._upload(this.anchor0, a0); this._upload(this.anchor1, a1); if (tn) this._upload(this.tnorm, tn); }
  setColliders(data, n) {
    this._upload(this.colliders, data.subarray(0, MAX_COLLIDERS * 8), false);
    this.u.numColliders.value = Math.min(n, MAX_COLLIDERS);
  }
  translate(dx, dy, dz) {
    this.u.shift.value.set(dx, dy, dz);
    this.renderer.compute(this.kShift);
  }

  step(dt, alpha, time) {
    const P = this.params, u = this.u;
    u.dt.value = dt; u.alpha.value = alpha; u.time.value = time;
    u.gravity.value = P.gravity; u.damping.value = P.damping; u.air.value = P.airDrag;
    u.wind.value.set(...P.wind); u.relax.value = P.relax; u.thickness.value = P.thickness;
    u.slack.value = P.tetherSlack; u.ground.value = P.ground;
    u.up.value.set(...P.up); u.aboveSeam.value = P.aboveSeam; u.backstop.value = P.backstop ?? 0.012;
    const r = this.renderer;
    r.compute(this.kIntegrate);
    const iters = P.iterations + (P.iterations % 2);   // even -> result ends in posA
    for (let k = 0; k < iters; k += 2) { r.compute(this.kSolveAB); r.compute(this.kSolveBA); }
  }

  computeNormals() { this.renderer.compute(this.kNormals); }

  /** Material position/normal nodes that read the simulated particles. */
  applyToMaterial(material) {
    const posA = this.posA, normals = this.normals;
    material.positionNode = Fn(({ material: m }) => {
      const id = int(attribute('pidx', 'float'));
      const n = normals.element(id).xyz;
      m.normalNode = transformNormalToView(n).toVarying('vClothNormal').normalize().mul(faceDirection);
      return posA.element(id).xyz;
    })();
  }

  async readPositions() {
    const buf = await this.renderer.getArrayBufferAsync(this.posA.value);
    return new Float32Array(buf);
  }
}
