// Lance (right hand) and shield (left forearm). The arms GLB holds one
// Lance_<variant> and one Shield_<variant> node per design, authored in the
// hero's bind-pose space; here they are parented to the hand / forearm
// bones through the inverse bind matrices so they follow every animation.
import * as THREE from 'three/webgpu';

export const ARMS = [
  { key: 'iron', label: 'Weathered Iron' }, { key: 'silver', label: 'Polished Silver' },
  { key: 'gold', label: 'Royal Gold' }, { key: 'crimson', label: 'Crimson Shadow' },
  { key: 'marble', label: 'White Marble' }, { key: 'azure', label: 'Azure Frost' },
  { key: 'verdant', label: 'Verdant Stag' }, { key: 'amethyst', label: 'Amethyst Crystal' },
  { key: 'solar', label: 'Solar Ivory' }, { key: 'carbon', label: 'Carbon Copper' },
];

export function attachArms(armsScene, skinnedMesh) {
  const skel = skinnedMesh.skeleton;
  const socket = (boneName) => {
    const i = skel.bones.findIndex((b) => b.name === boneName);
    const g = new THREE.Group();
    g.name = 'socket_' + boneName;
    g.matrixAutoUpdate = false;
    g.matrix.multiplyMatrices(skel.boneInverses[i], skinnedMesh.bindMatrix);
    skel.bones[i].add(g);
    return g;
  };
  const hand = socket('hand_r');
  const forearm = socket('lowerarm_l');
  const lances = {}, shields = {};
  for (const child of [...armsScene.children]) {
    const [kind, key] = child.name.split('_');
    child.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false; } });
    if (kind === 'Lance') { hand.add(child); lances[key] = child; }
    if (kind === 'Shield') { forearm.add(child); shields[key] = child; }
  }
  const api = {
    lances, shields,
    show(lanceKey, shieldKey) {
      for (const [k, o] of Object.entries(lances)) o.visible = k === lanceKey;
      for (const [k, o] of Object.entries(shields)) o.visible = k === shieldKey;
    },
    group: {
      set visible(v) { hand.visible = v; forearm.visible = v; },
      get visible() { return hand.visible; },
    },
  };
  return api;
}
