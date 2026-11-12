// The baker's rays on the GPU: a stack-based walk of the BVH (bvh.rs) in compute, no hardware ray
// tracing needed. Three passes read the same scene:
//   direct: per texel, the sun (stratified over the texel and the sun's disc, and a hard ray from each
//           cell for the distance field) and ambient occlusion (cosine-weighted, near hits occlude fully)
//   gather: per texel, the light arriving from every direction: the sky where a ray escapes, the light
//           the surface it meets sends back (its albedo times what reaches it, read from the atlas)
//   volume: per probe (given as a texel: its position), the same over the whole sphere, projected on L1 spherical harmonics, and the sun

struct Node { min: vec3f, first: u32, max: vec3f, count: u32 }
struct Tri { v0: vec4f, e1: vec4f, e2: vec4f }
struct Texel { pos: vec4f, normal: vec4f, dx: vec4f, dy: vec4f }
struct Params {
  sun_dir: vec4f,      // xyz towards the sun, w: tan of the disc's radius
  sun_irr: vec4f,      // rgb
  zenith: vec4f,
  horizon: vec4f,
  ground: vec4f,       // rgb; w: how close a back face says a point is inside a solid (m)
  fallback: vec4f,     // rgb: the irradiance assumed on a surface the atlas has no light for
  counts: vec4u,       // sun grid side, ao rays, gather rays, items in this batch
  reach: vec4f,        // ao reach, gather reach, lift off the surface, seed
  atlas: vec4u,        // atlas width, height, first item of the batch, -
}

@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read> materials: array<vec4f>;   // albedo rgb, opacity
@group(0) @binding(3) var<uniform> P: Params;
@group(0) @binding(4) var<storage, read> texels: array<Texel>;
@group(0) @binding(5) var<storage, read_write> out: array<vec4f>;
@group(0) @binding(6) var<storage, read> charts: array<vec4f>;      // two rows a chart: atlas x and y of a world point
@group(0) @binding(7) var<storage, read> light: array<vec4f>;       // per atlas texel: irradiance rgb, w: 1 where known

const PI: f32 = 3.14159265;
const FAR: f32 = 1e30;
const NONE: u32 = 0xffffffffu;
// how far a texel inside thin geometry is pushed out to be sampled (m; bake.rs PUSH)
const PUSH: f32 = 0.03;

struct Hit { t: f32, tri: u32, back: bool }

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand(a: u32, b: u32) -> f32 { return f32(pcg(a ^ pcg(b + bitcast<u32>(P.reach.w)))) / 4294967296.0; }

// Van der Corput radical inverse: with i / n, a Hammersley point
fn radical(i: u32) -> f32 { return f32(reverseBits(i)) / 4294967296.0; }

fn frame(n: vec3f) -> mat3x3f {
  let a = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(a, n));
  return mat3x3f(t, cross(n, t), n);
}

fn cosine(u: f32, v: f32, f: mat3x3f) -> vec3f {
  let r = sqrt(u);
  let a = v * 2.0 * PI;
  return normalize(f * vec3f(r * cos(a), r * sin(a), sqrt(max(0.0, 1.0 - u))));
}

fn sky(d: vec3f) -> vec3f {
  if (d.y < 0.0) { return P.ground.rgb; }
  return mix(P.horizon.rgb, P.zenith.rgb, d.y);
}

// does the ray pass a triangle of this material (leaves: only part of the light is stopped)?
fn stops(tri: u32, ray: u32) -> bool {
  let op = materials[bitcast<u32>(tris[tri].v0.w)].w;
  return op >= 1.0 || rand(tri * 9781u, ray) < op;
}

fn box(o: vec3f, inv: vec3f, mn: vec3f, mx: vec3f, tmax: f32) -> f32 {
  let a = (mn - o) * inv;
  let b = (mx - o) * inv;
  let lo = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
  let hi = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
  return select(FAR, max(lo, 0.0), hi >= max(lo, 0.0) && lo < tmax);
}

// the nearest hit within tmax (any: the first found, for shadow rays)
fn trace(o: vec3f, d: vec3f, tmax: f32, any: bool, ray: u32) -> Hit {
  var hit = Hit(tmax, NONE, false);
  let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
  var stack: array<u32, 48>;
  var sp = 0u;
  var n = 0u;
  loop {
    let node = nodes[n];
    if (node.count > 0u) {
      for (var i = node.first; i < node.first + node.count; i++) {
        let tr = tris[i];
        let p = cross(d, tr.e2.xyz);
        let det = dot(tr.e1.xyz, p);
        if (abs(det) < 1e-12) { continue; }
        let id = 1.0 / det;
        let s = o - tr.v0.xyz;
        let u = dot(s, p) * id;
        if (u < 0.0 || u > 1.0) { continue; }
        let q = cross(s, tr.e1.xyz);
        let v = dot(d, q) * id;
        if (v < 0.0 || u + v > 1.0) { continue; }
        let t = dot(tr.e2.xyz, q) * id;
        if (t > 1e-4 && t < hit.t && stops(i, ray)) {
          hit = Hit(t, i, det < 0.0);
          if (any) { return hit; }
        }
      }
    } else {
      let a = nodes[node.first];
      let b = nodes[node.first + 1u];
      let ta = box(o, inv, a.min, a.max, hit.t);
      let tb = box(o, inv, b.min, b.max, hit.t);
      let near = select(node.first + 1u, node.first, ta <= tb);
      let far = select(node.first, node.first + 1u, ta <= tb);
      let tn = min(ta, tb);
      let tf = max(ta, tb);
      if (tn < FAR) {
        if (tf < FAR && sp < 48u) { stack[sp] = far; sp++; }
        n = near;
        continue;
      }
    }
    if (sp == 0u) { break; }
    sp--;
    n = stack[sp];
  }
  return hit;
}

// the light a ray's hit sends back towards it: the surface's albedo over pi times what reaches it
fn bounced(o: vec3f, d: vec3f, h: Hit) -> vec3f {
  let tr = tris[h.tri];
  let albedo = materials[bitcast<u32>(tr.v0.w)].rgb;
  let c = bitcast<u32>(tr.e1.w);
  var e = P.fallback.rgb;
  if (c != NONE) {
    let p = o + d * h.t;
    let x = dot(charts[c * 2u], vec4f(p, 1.0));
    let y = dot(charts[c * 2u + 1u], vec4f(p, 1.0));
    let ix = clamp(i32(floor(x)), 0, i32(P.atlas.x) - 1);
    let iy = clamp(i32(floor(y)), 0, i32(P.atlas.y) - 1);
    let l = light[u32(iy) * P.atlas.x + u32(ix)];
    if (l.w > 0.5) { e = l.rgb; }
  }
  return albedo / PI * e;
}

// the sample points across a texel tried when its centre is inside a solid (index 1 to 8): its
// four quarters, then the middles of its four edges (a strip of face showing along one side)
fn offset(q: u32) -> vec2f {
  if (q <= 4u) { return vec2f(select(-0.3, 0.3, (q & 1u) == 1u), select(-0.3, 0.3, q > 2u)); }
  let e = array<vec2f, 4>(vec2f(-0.45, 0.0), vec2f(0.45, 0.0), vec2f(0.0, -0.45), vec2f(0.0, 0.45));
  return e[q - 5u];
}

// does a point see mostly back faces close round it (it is inside a solid)?
fn inside(o: vec3f, n: vec3f, key: u32) -> bool {
  let f = frame(n);
  var back = 0.0;
  for (var r = 0u; r < 16u; r++) {
    let d = cosine((f32(r) + 0.5) / 16.0, radical(r), f);
    let h = trace(o, d, P.ground.w, false, key + r + 7000u);
    if (h.tri != NONE && h.back) { back += 1.0; }
  }
  return back / 16.0 > bitcast<f32>(P.atlas.w);
}

@compute @workgroup_size(64)
fn direct(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= P.counts.w) { return; }
  let tx = texels[i];
  let n = tx.normal.xyz;
  let key = (P.atlas.z + i) * 7919u;
  let g = P.counts.x;
  let sun = P.sun_dir.xyz;
  let sf = frame(sun);
  var soft = 0.0;
  var hard = 0u;
  let lift = n * P.reach.z;
  // where the texel is sampled: its centre, or, where the centre lies inside a solid (a strip of face
  // showing above the ground, a face half under a wall), the first of four points across it that does
  // not; the sun's cells then shrink round that point
  var at = tx.pos.xyz;
  var pick = 0u;
  var span = 1.0;
  if (inside(at + lift, n, key)) {
    for (var q = 1u; q < 9u; q++) {
      let c = offset(q);
      let p = tx.pos.xyz + tx.dx.xyz * c.x + tx.dy.xyz * c.y;
      if (!inside(p + lift, n, key + q * 31u)) {
        at = p;
        pick = q;
        span = 0.4;
        break;
      }
    }
    // still inside: a face of thin geometry (a sheet, a plank) whose texel reaches past it, sampled
    // pushed out off the surface; under a thick solid (a wall) the point stays inside
    if (pick == 0u) {
      let p = tx.pos.xyz + n * PUSH;
      if (!inside(p + lift, n, key + 331u)) {
        at = p;
        pick = 9u;
        span = 0.4;
      }
    }
  }
  if (dot(sun, n) > 0.0) {
    for (var s = 0u; s < g * g; s++) {
      let cell = ((vec2f(f32(s % g), f32(s / g)) + 0.5) / f32(g) - 0.5) * span;
      let o = at + tx.dx.xyz * cell.x + tx.dy.xyz * cell.y + lift;
      // a hard ray from the cell's centre to the sun's: the distance field's sample
      if (trace(o, sun, FAR, true, key + s).tri == NONE) { hard |= 1u << s; }
      // a soft one to the cell's own part of the disc (a Fibonacci spiral, turned per texel)
      let k = (s + u32(rand(key, 17u) * f32(g * g))) % (g * g);
      let a = f32(k) * 2.399963;
      let r = P.sun_dir.w * sqrt((f32(k) + 0.5) / f32(g * g));
      let d = normalize(sun + sf[0] * cos(a) * r + sf[1] * sin(a) * r);
      if (trace(o, d, FAR, true, key + s + 1000u).tri == NONE) { soft += 1.0; }
    }
    soft /= f32(g * g);
  }
  // ambient occlusion: Hammersley points shifted per texel (Cranley-Patterson)
  let f = frame(n);
  let o = at + lift;
  let rays = P.counts.y;
  let s1 = rand(key, 3u);
  let s2 = rand(key, 5u);
  var occ = 0.0;
  var back = 0.0;
  for (var r = 0u; r < rays; r++) {
    let d = cosine(fract((f32(r) + 0.5) / f32(rays) + s1), fract(radical(r) + s2), f);
    let h = trace(o, d, P.reach.x, false, key + r + 2000u);
    if (h.tri == NONE) { continue; }
    let k = h.t / P.reach.x;
    occ += 1.0 - k * k;
    if (h.back && h.t < P.ground.w) { back += 1.0; }
  }
  out[i] = vec4f(soft, bitcast<f32>(hard | (pick << 24u)), occ / f32(rays), back / f32(rays));
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= P.counts.w) { return; }
  let tx = texels[i];
  let n = tx.normal.xyz;
  let f = frame(n);
  let key = (P.atlas.z + i) * 7919u;
  let rays = P.counts.z;
  let s1 = rand(key, 11u);
  let s2 = rand(key, 13u);
  var sum = vec3f(0.0);
  var back = 0.0;
  for (var r = 0u; r < rays; r++) {
    // across the texel, so the bounced light is the texel's, not its centre's
    let j = vec2f(rand(key + r, 19u), rand(key + r, 23u)) - 0.5;
    let o = tx.pos.xyz + tx.dx.xyz * j.x + tx.dy.xyz * j.y + n * P.reach.z;
    let d = cosine(fract((f32(r) + 0.5) / f32(rays) + s1), fract(radical(r) + s2), f);
    let h = trace(o, d, P.reach.y, false, key + r + 4000u);
    if (h.tri == NONE) { sum += sky(d); continue; }
    if (h.back) { back += 1.0; continue; }
    sum += bounced(o, d, h);
  }
  // cosine-weighted: irradiance is pi times the mean radiance
  out[i] = vec4f(sum * PI / f32(rays), back / f32(rays));
}

@compute @workgroup_size(64)
fn volume(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= P.counts.w) { return; }
  // a probe: its position, and its index in the volume (w of the normal, as bits)
  let probe = texels[i];
  let o = probe.pos.xyz;
  let key = bitcast<u32>(probe.normal.w) * 7919u + 77u;
  let rays = P.counts.z;
  let s1 = rand(key, 29u);
  let s2 = rand(key, 31u);
  var l0 = vec3f(0.0);
  var l1x = vec3f(0.0);
  var l1y = vec3f(0.0);
  var l1z = vec3f(0.0);
  var back = 0.0;
  for (var r = 0u; r < rays; r++) {
    // uniform over the sphere
    let u = fract((f32(r) + 0.5) / f32(rays) + s1);
    let v = fract(radical(r) + s2);
    let z = 1.0 - 2.0 * u;
    let rr = sqrt(max(0.0, 1.0 - z * z));
    let a = v * 2.0 * PI;
    let d = vec3f(rr * cos(a), z, rr * sin(a));
    let h = trace(o, d, P.reach.y, false, key + r);
    var l = sky(d);
    if (h.tri != NONE) {
      if (h.back) { back += 1.0; }
      l = select(bounced(o, d, h), vec3f(0.0), h.back);
    }
    // L1 projection (the 4 pi / rays weight below)
    l0 += l * 0.282095;
    l1x += l * 0.488603 * d.x;
    l1y += l * 0.488603 * d.y;
    l1z += l * 0.488603 * d.z;
  }
  let w = 4.0 * PI / f32(rays);
  // the sun: how much of its disc the point sees
  let g = P.counts.x;
  let sf = frame(P.sun_dir.xyz);
  var sun = 0.0;
  for (var s = 0u; s < g * g; s++) {
    let a = f32(s) * 2.399963;
    let r = P.sun_dir.w * sqrt((f32(s) + 0.5) / f32(g * g));
    let d = normalize(P.sun_dir.xyz + sf[0] * cos(a) * r + sf[1] * sin(a) * r);
    if (trace(o, d, FAR, true, key + s + 9000u).tri == NONE) { sun += 1.0; }
  }
  let base = i * 4u;
  out[base] = vec4f(l0 * w, sun / f32(g * g));
  out[base + 1u] = vec4f(l1x * w, back / f32(rays));
  out[base + 2u] = vec4f(l1y * w, 0.0);
  out[base + 3u] = vec4f(l1z * w, 0.0);
}
