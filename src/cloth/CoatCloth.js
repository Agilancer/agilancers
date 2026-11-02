// Runtime for the hero's free-hanging lab coat.
//
// The coat's rigid parts (shoulder yoke, collar, sleeves, rolled cuffs) are
// ordinary skinned meshes driven by the animations. Everything below the yoke
// seam - chest panels, sides, back and the long tail with its vent - is this
// cloth: only the seam row follows the skeleton (shoulders, shoulder blades
// and, under each arm, the upper arm), the rest is pure physics.
import * as THREE from 'three/webgpu';
import { buildTopology, COLLIDERS } from './topology.js';
import { ClothCPU } from './ClothCPU.js';
import { ClothGPU, MAX_COLLIDERS } from './ClothGPU.js';

export const DEFAULT_CLOTH_PARAMS = {
  gravity: -9.81,
  damping: 0.988,        // per 1/120 s substep
  airDrag: 0.04,         // pull toward wind/air velocity
  wind: [0, 0, 0],
  relax: 1.0,
  iterations: 16,
  thickness: 0.012,
  ground: 0.01,
  tetherSlack: 1.02,
};

export class CoatCloth {
  constructor({ renderer, clothMesh, useGPU, params = {} }) {
    this.renderer = renderer;
    this.source = clothMesh;
    this.params = { ...DEFAULT_CLOTH_PARAMS, ...params, wind: [...(params.wind || DEFAULT_CLOTH_PARAMS.wind)] };
    this.topo = buildTopology(clothMesh);
    this.useGPU = useGPU;
    this.solver = useGPU ? new ClothGPU(renderer, this.topo, this.params) : new ClothCPU(this.topo, this.params);
    this.fixedDt = 1 / 120;
    this.accum = 0;
    this.time = 0;
    this.enabled = true;

    const N = this.topo.N;
    this._a0 = new Float32Array(N * 3);
    this._a1 = new Float32Array(N * 3);
    this._colData = new Float32Array(MAX_COLLIDERS * 8);
    this._v = new THREE.Vector3();
    this._m = new THREE.Matrix4();

    // colliders: bone + endpoints expressed in that bone's bind space
    const skel = clothMesh.skeleton;
    this.colliders = COLLIDERS.map(([bone, ax, ay, az, bx, by, bz, r]) => {
      const bi = skel.bones.findIndex((b) => b.name === bone);
      const toBone = new THREE.Matrix4().multiplyMatrices(skel.boneInverses[bi], clothMesh.bindMatrix);
      return { bone: skel.bones[bi], a: new THREE.Vector3(ax, ay, az).applyMatrix4(toBone), b: new THREE.Vector3(bx, by, bz).applyMatrix4(toBone), r };
    });

    // render mesh: same triangles/UVs, positions come from the particles
    const g = new THREE.BufferGeometry();
    const src = clothMesh.geometry;
    g.setIndex(src.index.clone());
    g.setAttribute('position', src.attributes.position.clone());
    g.setAttribute('normal', src.attributes.normal.clone());
    g.setAttribute('uv', src.attributes.uv.clone());
    g.setAttribute('pidx', new THREE.BufferAttribute(this.topo.particleOfVertex, 1));
    const srcMat = clothMesh.material;
    const mat = new THREE.MeshStandardNodeMaterial({
      map: srcMat.map, roughness: srcMat.roughness ?? 0.86, metalness: 0, side: THREE.DoubleSide,
      normalMap: useGPU ? null : srcMat.normalMap, color: srcMat.color,
    });
    if (useGPU) this.solver.applyToMaterial(mat);
    this.mesh = new THREE.Mesh(g, mat);
    this.mesh.name = 'CoatClothSim';
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.matrixAutoUpdate = false;     // particles live in world space
  }

  _skinned(out, rowsOnly) {
    const { vertexOfParticle, cols, N, anchorRows } = this.topo;
    const n = rowsOnly ? cols * anchorRows : N;
    const m = this.source;
    for (let p = 0; p < n; p++) {
      m.getVertexPosition(vertexOfParticle[p], this._v);
      this._v.applyMatrix4(m.matrixWorld);
      out[p * 3] = this._v.x; out[p * 3 + 1] = this._v.y; out[p * 3 + 2] = this._v.z;
    }
  }

  _updateColliders() {
    const d = this._colData;
    this.colliders.forEach((c, i) => {
      const A = this._v.copy(c.a).applyMatrix4(c.bone.matrixWorld);
      d[i * 8] = A.x; d[i * 8 + 1] = A.y; d[i * 8 + 2] = A.z; d[i * 8 + 3] = c.r;
      const B = this._v.copy(c.b).applyMatrix4(c.bone.matrixWorld);
      d[i * 8 + 4] = B.x; d[i * 8 + 5] = B.y; d[i * 8 + 6] = B.z;
    });
    this.solver.setColliders(d, this.colliders.length);
    return d;
  }

  /** Snap the cloth to the skinned (animation-only) pose. */
  reset() {
    const all = new Float32Array(this.topo.N * 3);
    this._skinned(all, false);
    this.solver.setState(all);
    this._a1.set(all.subarray(0, this._a1.length));
    this._skinned(this._a1, true);
    this._a0.set(this._a1);
    this._writeCPU();
  }

  /** Move the whole cloth (e.g. when root motion wraps back to the origin). */
  translate(dx, dy, dz) { this.solver.translate(dx, dy, dz); for (const a of [this._a0, this._a1]) for (let i = 0; i < this.topo.cols * this.topo.anchorRows * 3; i += 3) { a[i] += dx; a[i + 1] += dy; a[i + 2] += dz; } }

  update(frameDt) {
    frameDt = Math.min(frameDt, 1 / 20);
    this._a0.set(this._a1);
    this._skinned(this._a1, true);
    this._updateColliders();
    this.solver.setAnchors(this._a0, this._a1);
    this.accum += frameDt;
    const n = Math.min(Math.floor(this.accum / this.fixedDt), 6);
    this.accum -= n * this.fixedDt;
    if (this.accum > this.fixedDt) this.accum = 0;
    for (let k = 0; k < n; k++) {
      this.time += this.fixedDt;
      this.solver.step(this.fixedDt, (k + 1) / n, this.time);
    }
    this.solver.computeNormals();
    this._writeCPU();
  }

  _writeCPU() {
    if (this.useGPU) return;
    const s = this.solver, pv = this.topo.particleOfVertex;
    const pa = this.mesh.geometry.attributes.position, na = this.mesh.geometry.attributes.normal;
    for (let v = 0; v < pv.length; v++) {
      const p = pv[v] * 3;
      pa.setXYZ(v, s.pos[p], s.pos[p + 1], s.pos[p + 2]);
      na.setXYZ(v, s.normals[p], s.normals[p + 1], s.normals[p + 2]);
    }
    pa.needsUpdate = true; na.needsUpdate = true;
  }

  colliderDebug(scene) {
    const group = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color: 0x33ddff, wireframe: true, transparent: true, opacity: 0.35 });
    this._dbg = this.colliders.map((c) => {
      const len = 1;
      const m = new THREE.Mesh(new THREE.CapsuleGeometry(c.r, len, 6, 12), mat);
      group.add(m);
      return m;
    });
    scene.add(group);
    this._dbgGroup = group;
    return group;
  }

  updateDebug() {
    if (!this._dbg) return;
    const d = this._colData, up = new THREE.Vector3(0, 1, 0);
    this._dbg.forEach((m, i) => {
      const A = new THREE.Vector3(d[i * 8], d[i * 8 + 1], d[i * 8 + 2]);
      const B = new THREE.Vector3(d[i * 8 + 4], d[i * 8 + 5], d[i * 8 + 6]);
      const len = A.distanceTo(B);
      m.position.copy(A).add(B).multiplyScalar(0.5);
      m.quaternion.setFromUnitVectors(up, B.clone().sub(A).normalize());
      m.scale.set(1, 1, 1);
      m.geometry.dispose();
      m.geometry = new THREE.CapsuleGeometry(this.colliders[i].r, len, 4, 10);
    });
  }
}
