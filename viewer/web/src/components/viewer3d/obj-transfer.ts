// An OBJ the loader worker parsed (`workers/loader.worker.ts`), rebuilt as the objects `OBJLoader` would have made.
//
// Same classes, names, geometry groups, and default materials, so a template from the worker and one parsed in the
// page are indistinguishable downstream. Rebuilding is cheap: the arrays arrive transferred, so nothing is copied.

import * as THREE from "three";

import type { SerializedObj } from "@/workers/loader-protocol";

export function objectFromSerialized(serialized: SerializedObj): THREE.Group {
  const container = new THREE.Group();
  for (const drawable of serialized.children) {
    const geometry = new THREE.BufferGeometry();
    for (const [name, attribute] of Object.entries(drawable.attributes)) {
      geometry.setAttribute(name, new THREE.BufferAttribute(attribute.array, attribute.itemSize, attribute.normalized));
    }
    for (const group of drawable.groups) geometry.addGroup(group.start, group.count, group.materialIndex);
    const [x0, y0, z0, x1, y1, z1] = drawable.box;
    geometry.boundingBox = new THREE.Box3(new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y1, z1));
    const [cx, cy, cz, radius] = drawable.sphere;
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(cx, cy, cz), radius);
    const materials = drawable.materials.map((spec) => {
      const material = drawable.kind === "LineSegments" ? new THREE.LineBasicMaterial()
        : drawable.kind === "Points" ? new THREE.PointsMaterial({ size: 1, sizeAttenuation: false })
          : new THREE.MeshPhongMaterial();
      material.name = spec.name;
      if (material instanceof THREE.MeshPhongMaterial) material.flatShading = spec.flatShading;
      material.vertexColors = spec.vertexColors;
      return material;
    });
    const material = materials.length > 1 ? materials : materials[0];
    const object = drawable.kind === "LineSegments" ? new THREE.LineSegments(geometry, material)
      : drawable.kind === "Points" ? new THREE.Points(geometry, material)
        : new THREE.Mesh(geometry, material);
    object.name = drawable.name;
    container.add(object);
  }
  return container;
}
