# BrokkrSculpt — Cut Tool Plan

Status: proposed. Supersedes nothing; the current plane cut ships and keeps working throughout.
Author's note on numbers: every figure is either **sourced** (file:line, or a cited URL) or explicitly labelled **estimate**.

---

## 1. The problem, in the maintainer's words

From the handoff:

> "A better cut tool, like ZBrush's. The current cut is a single straight plane through everything. Wanted: something that can take a curve or a lasso."

And in session:

> "Something that makes it easier to cut off things that are sticking out of the main body."

### What we have

`crates/brokkr-core/src/clip.rs:66-82` — `ClipPlane` is a point plus a unit normal: an infinite half-space. `Volume::clip` writes `max(old, plane.distance(at) / voxel_size)` per voxel through `cut_voxel` (clip.rs:197-202), so the removed side is driven to `OUTSIDE` and the cut face closes as a property of the arithmetic. Bricks are classified three ways against the narrow band — `Keeps` (never read), `Removes` (dropped whole), `Crosses` (promoted dense, resolved per voxel) — which is the only reason a cut through a 40 MB scan does not become a gigabyte (clip.rs:17-28, 240-339). It is mask-aware and feathered, it is one undo `Entry` for the whole gesture across every body, and it acts on what is *drawn* (`Document::display_visibility`, solo included), so hiding a body excludes it and solo narrows the cut.

That machinery is good and none of it is being replaced.

### What is actually wrong

1. **It is unbounded in two of three directions.** The plane is infinite in-plane and infinite in depth. Circling is impossible; you get one straight line and everything on one side of it, forever, through every drawn body. This is the single most-complained-about behaviour in every application surveyed (Meshmixer's infinite plane, Blender's Box/Lasso Trim, Nomad's Trim — "Trim will cut through the whole object").
2. **It cannot express the shape the maintainer asked for.** A curve or a lasso has no `ClipPlane` representation. `finish_cut` (app.rs:5277-5299) is literally two rays and a cross product; the construction does not generalise by adding parameters.
3. **There is no preview at all.** `DragKind::Cutting` falls into a do-nothing arm on `PointerEvent::Moved` (app.rs:8102-8106), and `refresh_overlay` (app.rs:4487-4521) builds only the brush cursor ring. The comment at app.rs:8097 claiming "the cut line is only drawn while it is being dragged" is **false**. Today the user drags a destructive, irreversible-except-by-undo line across their model seeing nothing but a brush ring that implies a press would sculpt. The side convention is empirical, recorded from observation in `a_left_to_right_drag_removes_a_consistent_side` (app.rs:11292-11331), and the user has no way to know it before releasing.
4. **`live_tool_label` (app.rs:4796) hard-codes `"PLANE CUT"`**, and the arm sentence (app.rs:8781) promises "drag a line across the model". Both become lies the moment the tool takes a shape.
5. **Removing something *in front of* something else is not expressible.** For "cut off the bit sticking out", the only guard is a hand-painted mask on the far side. That is the workaround every other application also ships, and it is the one the maintainer is trying to avoid.

---

## 2. What ZBrush and Nomad actually do

The vocabulary matters because the rest of this document uses it. ZBrush's four brush families share one identical gesture — Ctrl+Shift, drag, release — and differ **only in what they do with the resulting curve**. Maxon states this outright for the Clip family: "only the Stroke type makes them different."

**CLIP** — *deforms, deletes nothing.*
`help.maxon.net/zbr/en-us/Content/html/user-guide/3d-modeling/hard-surface/clip-brushes/clip-brushes.html`: "These Clip brushes do not change the topology of your model; they only push the polygons." Every vertex on the shaded side is projected onto the swept curve. This is the most-used member of the family and the source of its loudest complaint: a ZBrushCentral thread ("clip brush leaves residue") has a user clip an arm off and report "the arm seems to sort of melt but not disappear?", answered canonically by Marcus_civis — "The Clip brushes don't remove polygons but squash them towards the curve." Every clipped polygon survives as a zero-area sliver welded to the cut plane, which then poisons downstream booleans and exports.

**TRIM** — *deletes, then caps.*
`.../clip-brushes/trim-curve/trim-curve.html`: polygons on the shadowed side are "totally remove[d] ... rather than simply pushing them toward the curve", and ZBrush rebuilds a surface across the opening "using the optimal number of polygons necessary to close the hole". Four documented failure modes: the capping routine "is designed for creating flat caps" so curved strokes degrade; the trim brushes are **not symmetrical** (ZBC 317772: "the trim brushes are not symmetrical for some reason"); they refuse meshes with multiple subdivision levels; and they fill holes the model was supposed to have (ZBC 335189 — "the hole where the spinal chord goes gets filled in").

**SLICE** — *partitions, keeps both, changes no form.*
`.../clip-brushes/slice/slice.html`: "simply slices the model's geometry and creates a different PolyGroup on each side of the drawn curve." Nothing is removed; a polygroup boundary is created and every downstream selection operation can consume it. It is the composition point of the whole family — a slice is a *reusable selection*.

**KNIFE** (2021.7) — *cuts along the exact stroke and closes it, with symmetry.*
`.../clip-brushes/knife-brushes/knife-brushes.html`: "designed to accurately cut and close a mesh along the precise line of the stroke", honours symmetry "unlike the Trim brushes", but "cannot cut holes through the center of meshes" — the stroke must enter and leave the silhouette.

**Nomad Sculpt** (`nomadsculpt.com/manual/tools`) makes the discard/keep split a *tool* choice, not a checkbox: **Trim** discards the selection, **Split** "will keep the selection as a new object". Both take the same shape strip (Lasso, Polygon, Line, Rect, Ellipse, Flip) which is shared by six consumers — Trim, Split, Project, Selection Mask, Facegroup and Hide. The gap left by a cut is a first-class four-way choice (Boolean via Emmett Lalish's Manifold library / Legacy / Fill / None) because there is no single right answer on a mesh. Its documented weaknesses are the ones we care about: it cuts through the whole object with no depth limit; mask-respect depends on which fill mode is active (`forum.nomadsculpt.com/t/trim-tool-cutting-through-mask/6145`, fixed by a changelog line "trim: protect masked area if flip mode is activated"); and corners round off — "unable to cut straight corners", with the developer confirming a proper polygonal boolean is needed (`forum.nomadsculpt.com/t/better-trim-tool/7945`).

### The mapping onto Brokkr

This is the important part, and it is why we are not building three tools.

| ZBrush verb | Brokkr equivalent | Status |
|---|---|---|
| Trim (delete + cap) | `max(field, cutter)` on the SDF | **Already shipped.** The cap is not a routine; it is what `max` does. There is no sliver residue, no flat-cap bias, no hole to fill. |
| Slice / Split (keep both) | `Document::split_masked` (split.rs:753) | **Already shipped.** One parallel pass, no connectivity walk, one undo entry. |
| Clip (push flat) | — | **Deliberately not built.** On an SDF there are no polygons to push; it would have to be simulated, and it would import the single behaviour users most reliably hate. |

**What was missing was never the verbs. It was the shape.** Every mature tool surveyed factors the region shape out of the operation and reuses one shape layer across many consumers; Brokkr has the operations and has one shape (an infinite plane).

---

## 3. The decision

**Recommendation: generalise the cut from one `ClipPlane` to a slice of them, `&[ClipPlane]`, and remove their intersection. Ship it as one `Tool::Cut`, one drag, one undo entry, with a live overlay preview and a depth-limited cutter by default.**

This is the spine of the "Lasso Cut" design, which was the strongest of the three (judge scores 5 / 6.5 / 5.5, against 4 / 6 / 4 and 4 / 5 / 5). Onto it we graft:

- **Depth limiting, in v1** — from "Sweep" and "Cutter". The winning design deferred it; that is wrong. "Cut off the thing sticking out" is the literal request, and a depth cap is geometrically just two more planes in the same slice. Deferring it ships the exact complaint every surveyed application has.
- **The ctrl → feathered mask → `split_masked` route, promoted and moved early** — from "Cutter". This is the keep-both half, it lands on the layers model rather than fighting it, and it reuses shipped code. The winning design had it as the last phase and explicitly labelled it the first thing to cut if the week ran short. That is backwards for a project that presents bodies as Photoshop layers.
- **Loose-piece reporting via the shipped read-only `Document::split_plan`** — from "Sweep". A cut that severs a body currently leaves two shells in one `Volume` with no indication, which is precisely the Meshmixer/Nomad failure the research identified as the one to beat. `split_plan` is read-only, parallel and already tested.

**What we are rejecting, and why:**

- **"Sweep"'s four verbs (Remove / Keep / Split / Select) on one gesture.** `Sense::Keep` is the flip arrow the same design also ships, reached by a second widget with no stated precedence — enumeration wearing composition's clothes, in a document that argues against enumeration four times. Worse, `Keep`'s removal region is the *complement* of the sweep, which contradicts the AABB gate the same design specifies as its performance story; a Keep cut would leave an axis-aligned brick-quantised wall of survivor material that is watertight, manifold and exactly wrong. Its persistent modal editing session also drops the one-shot property that app.rs:5370 explicitly records as the reason a stray click cannot remove half the model.
- **Non-convex removal via a union of convex pieces (`max` of `min`) — both "Sweep" and "Cutter" propose it, and it is verified broken.** At a point deep inside the union but on an internal decomposition seam, the value is exactly 0 rather than large-positive. Consequence: the brick's interval spans zero, so it classifies `Crosses` not `Removes`; it is promoted dense (128 KB, brick.rs:12-36), cloned for undo (another 128 KB), and then refused by `Brick::is_collapsible` (brick.rs:110-176, which only collapses at exactly `OUTSIDE` or `INSIDE`) — so a permanently resident 128 KB brick is inserted back into a region the user just deleted, once per seam brick, invisibly, with the exported mesh perfectly correct. A 20-point lasso earcuts to ~17 shared diagonals, each sweeping a brick sheet through the model depth. This directly falsifies the stated "no interior tile is promoted" guarantee and is the shape of the known mesh-pool-sized-for-the-wrong-input failure. **We are shipping convex-only in v1** and naming the correct fix for v2 (see §6).
- **"Cutter"'s auto depth slab as specified** ("one band past the deepest exit"). With `NARROW_BAND = 3.0` (brick.rs:12-36), that unconditionally eats 0.75 mm at a 0.25 mm voxel out of whatever is behind. Judge-verified failure: a 0.9 mm wall 0.5 mm behind a spur is thinned to 0.65 mm with a sharp step, and `is_printable()` returns true. The depth rule below is clamped against the next surface instead.
- **"Sweep"'s centroid raycast for the depth cap.** The centroid of a concave stroke is *outside the stroke*. Lasso a crescent around a curved fin and the centroid ray lands on the torso behind it, so the cap goes at the torso's back face and the tool removes the torso — the exact complaint it was added to fix, on exactly the strokes that motivated adding curves.

---

## 4. The design

### 4.1 Gestures

**Arming.** The existing CUT button in the tool strip, or a new bare key `c` — one row in `viewport::shortcut` (viewport.rs:666-742), which stays a pure function; the modal guard remains in `Brokkr::on_key`. No new `Tool` variant, no new strip button. The tool strip is already ~8 px over what fits a 768-high window with no scroll by panel.rs:1838-1854's own recorded arithmetic, and that arithmetic explicitly notes "one screenshot at 768 settles it properly and none has been taken".

**One left drag.** The whole path is accumulated (not just endpoints) into a `cut_path: Vec<Vec2>` field on `Brokkr`, beside `stamp_centres`. It cannot live in `Drag`, which is `Copy` and read by value at app.rs:7990 / 8065 / 8087.

**Three readings of the stroke, inferred, always previewed:**

| Reading | Condition | Result |
|---|---|---|
| **LINE** | simplifies to 2 points, or max deviation from the chord < `LASSO_DEVIATION_PX` | today's cut, bit-for-bit — one `ClipPlane`, the existing code path, unchanged |
| **CURVE** | open stroke (ends further apart than `CLOSE_RADIUS_PX`) | the stroke's points **plus two extension points** far out along the end segments, hulled |
| **LASSO** | ends within `CLOSE_RADIUS_PX` | the convex hull of the points |

The CURVE reading is the fix for a real defect in the winning design, which had only two readings. A shallow arc across a shoulder — the canonical TrimCurve gesture, the first thing anyone tries after being told the tool takes a curve — has a convex hull that is a *thin crescent*, so the design as written would gouge a sliver out of the middle of the shoulder and leave everything above the curve standing. Adding the extension points makes an open stroke partition the view the way ZBrush does when it extrapolates a short stroke "to the edge, following the final path of your stroke".

**The removed region is always convex** — the intersection of the hull's edge half-spaces with the depth slab. For a closed stroke that is the hull interior; for an open stroke it is the extended side. A stroke that reverses its turn direction (an S) is hulled, which removes more than drawn. This is shown in the preview and it is the same limitation ZBrush documents for ClipCurve: an S-shaped stroke "will reverse twice and produce an unexpected result".

**Modifiers, both visible before release:**
- **Shift held at release** — cut all the way through (no depth slab). The preview shading changes *while shift is held*, so the verb is visible rather than inferred from key state at the moment of release. The research names silent mode flips as a top pitfall (ZBrush's ALT-at-release "with little visible difference"); this design ships one modifier, and it is previewed.
- **Ctrl held at release** — write a feathered mask over the region instead of removing anything. This generalises the route that already exists (app.rs:5307-5313, `mask_halfspace`).

**Escape** cancels a live path without disarming (the treatment the gizmo already gets at app.rs:8709-8718); Escape with no live path disarms, at the same rung `Tool::Cut` occupies today. **Committing disarms** — `self.tool = Tool::Sculpt` stays exactly where it is at app.rs:5375, whose comment records that *this assignment*, not the Escape arm, carries the one-shot property.

**`live_tool_label` becomes shape-aware**: `PLANE CUT` / `CURVE CUT` / `LASSO CUT`. The mechanism already exists and was built for this — `(Tool::Mask, true) => "MASK — BLUR"`. The arm sentence at app.rs:8781 stops promising a line.

### 4.2 Geometry: screen gesture → SDF cutter → voxel write

**The operation.** Today: `new = max(old, d(x))`, an intersection of the solid with the half-space `d <= 0`. Generalise the removed region to a convex polyhedron, the intersection of N half-spaces `{ d_i(x) > 0 }`. The kept region is its complement, whose signed distance is `-max_i(-d_i) = min_i d_i`. So:

```
cut = min_i d_i(at) / voxel_size
new = cut_voxel(old, cut, free)          // clip.rs:197-202, UNCHANGED
```

Two properties fall out, and both were verified by the judges rather than asserted:

- **Bit identity is free.** `min` over one element *is* the element, so for `planes.len() == 1` the expression is character-for-character today's arithmetic. `an_unmasked_cut_writes_exactly_what_the_plain_max_wrote` (clip.rs:916) passes unchanged, and `cut_voxel`'s `free >= 1.0` branch — which exists because the blend differs in the last bit for 65,040 of five million random pairs (clip.rs:189-196) — is not touched.
- **Masking commutes.** The blend `old + (t - old) * free` is monotone increasing in `t` for `free > 0`, so `min_i blend(t_i) == blend(min_i t_i)`. Compute the min *first*, hand the single value to the existing `cut_voxel`. No second copy of the blend, no second copy of the clamp, and the mask stays uniformly honoured across every shape — which is exactly the inconsistency Nomad shipped and had to patch.

**Screen to world, with the bug the winning design inherited.** `finish_cut` today does:

```rust
let (eye, first) = self.ray_through(from);
let (_, second)  = self.ray_through(to);
ClipPlane::new(eye, second.cross(first))
```

`OrbitCamera::ray` (camera.rs:338-346) returns the **near-plane intersection**, not `camera.eye()`. That is correct today *only by accident*: the near point `P0` lies on the first ray, and the normal `d2 × d1` is perpendicular to both rays, so `(P0 - E)·n = 0` exactly. For a hull edge `k`, `n_k` is **not** perpendicular to `d0`, so plane `k` misses the true apex by `(P0 - E)·n_k`. `near() = distance * 0.01` (camera.rs:137) and framing distance is roughly `3 × radius`, so on a 100 mm model that is ~1.5 mm — **six voxels at 0.25 mm** — and it grows with model size. Each side plane is translated by a *different* amount, so the prism's apex becomes a small polyhedral gap and the region cut does not match the polygon drawn.

**Fix: use `camera.eye()` explicitly.** One token, plus a `debug_assert` that all rays share it, plus a test that compares the cut region against the drawn region *in world space* (see phase 3). This is worth calling out because it is the design's own headline promise — "there is no silent divergence between what you drew and what happens" — and it was false on the very first lasso, invisibly, because the preview is drawn by projecting the same screen points and is therefore screen-exact either way.

The construction also silently depends on the camera being **perspective** (all rays share an apex). `OrbitCamera` has no orthographic mode; the navcube's is separate (navcube.rs:99). With M planes instead of 2 this dependency becomes load-bearing, so it gets a comment and an assert rather than staying latent.

**Hull, decimation, and the corner bevel.** Monotone-chain convex hull (~40 lines, no dependency), then decimate to at most `MAX_CUT_PLANES = 16` vertices. The decimation rule is **not** "drop the vertex adding the least area" — it is **"drop the vertex with the smallest interior angle first, then the least area"**, for a reason:

`min_i d_i` is exact inside the polyhedron and exact outside a face, but outside a convex *edge* it over-estimates. At a wedge of interior angle θ, at radius r, the over-estimate is `r(1 - sin(θ/2))`. At the band edge (r = 3 voxels): **0.88 voxel at a right angle, 2.48 voxels at 20°.** The winning design quoted "~0.3 voxel at a right angle", which is understated by roughly 8×, and it treated acute corners as exotic when an outlier sample on a jittery hand-drawn loop *is* an acute hull vertex. Dropping an acute hull vertex only ever *shrinks* the removed region, so decimation errs toward removing less — the safe direction. Over-estimation itself errs toward removing *more*, i.e. it rounds convex cut corners rather than sharpening them, which is the direction you want: a knife edge at a cut is the printability failure a watertightness test cannot see.

**Depth, as two more planes.** A far cap plane with normal along the view direction, intersected into the same slice. Placement rule, on release:

1. Cast a small grid of rays through the hull's interior (a point proven interior — the hull's centroid *is* interior for a convex hull, unlike the concave-stroke case that broke "Sweep") across every drawn body, using `raycast`'s existing sphere-tracing skeleton (raycast.rs:43) extended to return the first solid span `(enter, exit)`.
2. Take the deepest `exit` over the rays that hit. Take the shallowest `enter` of any *subsequent* span (the next surface behind).
3. Place the far plane at `deepest_exit + margin`, **clamped** so it never comes within `margin` of the next surface. `margin = NARROW_BAND * voxel_size` (0.75 mm at 0.25 mm voxels, **sourced**: `NARROW_BAND = 3.0`, brick.rs:12-36).
4. If the gap between the two is smaller than `2 * margin`, do not depth-limit: cut through, and **say so in the status line** ("nothing behind it to spare — cut went through"). A tool that quietly thins the wall behind is worse than one that admits it cannot help.
5. If no ray finds a solid span (the lasso missed), there is no cap: the existing "the cut missed the model" outcome applies.

This is the direct fix for the failure the judges found in both other designs' depth rules. It is also what makes the cutter **bounded**, which restores `Document::clip`'s per-body box pre-gate (clip.rs:428-442) to something that actually rejects bodies rather than always answering "maybe".

**Brick classification.** `ClipPlane::range_over_box` (clip.rs:101-107) is reused verbatim, per plane. It stays `pub(crate)` and stays shared with `generate.rs`'s half-space mask, because its own doc records that a second copy would let the mask and the cut disagree about which bricks a plane touches.

- **KEEPS** if *any* single plane reports `farthest <= -band_mm`. Sound because `min_i d_i <= d_j` everywhere.
- **REMOVES** if *every* plane reports `nearest >= band_mm`. Sound because the brick is then saturated inside the polyhedron. A lasso *enclosing* a lump drops whole bricks — genuinely cheaper than the plane covering the same silhouette.
- **CROSSES** otherwise: promote dense, resolve per voxel.

The `Removes → Crosses` downgrade when `mask.protection_fill(coord) != Some(UNMASKED)` (clip.rs:254-258) is unchanged and still mandatory: a dropped brick has no voxels left to protect. `cut_would_change_a_voxel` (clip.rs:154-172) keeps its early return so `bricks_spared_by_mask` stays honest.

**Cost, stated honestly — the winning design got this wrong.** It claimed "a lasso is O(prism boundary), not O(body)". That is false as written: `clip_masked` opens with `let coords: Vec<BrickCoord> = self.brick_coords().collect();` (clip.rs:231) and walks **every** brick, serially, and per-brick classification goes from one `range_over_box` to up to 16+2. On the reference dragon (**sourced**: 22,119 bricks at 0.25 mm, split.rs module doc) that is ~390k plane-box tests where today there are 22k, before a voxel is touched. `EDIT_BUDGET` is 4 ms and `FRAME_BUDGET` is 16 ms (**sourced**: benches/budget.rs:24-26).

Two fixes, both phased:

- **An integer AABB pre-filter.** The depth-slabbed cutter has a finite world AABB. Filter the collected coord list by integer brick-index comparison before any plane arithmetic. The walk stays O(bricks in the body) with a trivial test per brick; only the *plane arithmetic* becomes boundary-proportional. Note this is a filter over `bricks.keys()`, not an iteration over the box — iterating the box index range would be worse on a sparse body, which is a mistake a sibling design made.
- **Parallel classify, serial apply (its own phase, its own days).** `generate::halfspace_mask` (generate.rs:342-383) runs the identical classification under `par_iter`, but it is `&self` filter-mapping into a fresh Vec, while `clip_masked` is `&mut self` interleaving `record_for_undo` / `take_brick` / `insert_brick` / `remove_brick` in a documented order. That `par_iter` **cannot be copied across**, which is what makes this a restructure rather than a one-liner. The split is: parallel read-only pass producing `Vec<(BrickCoord, Cut)>`, then the existing serial mutation loop consuming the verdicts. Transient memory is unchanged — verdicts, not bricks.

**Remesh cost.** Dirty marking stays clip.rs's: only bricks that actually changed, via `mark_dirty_voxel_range` (volume.rs:850-863), which expands ±1 voxel and therefore dirties the brick plus its 26 neighbours because a brick's apron reads one voxel into each. Worst case 27 meshes per changed brick, published into a pool of `MAX_BUFFERS = 8` × `VERTEX_CAPACITY = 11,000,000` (**sourced**: mesh_pool.rs:19-108).

`Volume::edit_voxels_where` is **not** used, for the reason clip.rs already declined it plus one more: its last line marks the *whole requested box* dirty (volume.rs:1085-1131), so a lasso over a spur's bounding box would correctly skip the field writes and then remesh thousands of bricks to produce a handful of triangles — and no existing test would see it.

**Preview is overlay geometry and only overlay geometry.** No field write during the drag. `remesh_dirty` (app.rs:3100-3145) is fully synchronous with no per-frame budget, through a `BlockAllocator` that never splits or merges blocks (mesh_pool.rs:110-250); per-motion-event brick churn is precisely the shape of the recorded 2026-08-22 incident (**sourced**: `MESH POOL FULL: 2755 bricks missing` with `live` at ~7.4M of 11M). The path is unprojected onto a camera-facing plane at the orbit target's depth and pushed into the existing sculpt overlay batch with `push_line` / `push_triangle` (overlay.rs:77-112). No fourth batch — renderer.rs:698-711 asks the next overlay to reuse the third.

**Degenerate input — three guards, each with its own sentence.** A destructive tool must refuse rather than act:

- Under `CLICK_SLOP_PX` (**sourced**: 4.0, app.rs:231) — unchanged: `"cut cancelled: drag a line across the model"`.
- **Minimum hull width, in voxels, not pixels.** A there-and-back stroke 100 px long and 2 px wide clears any area guard (~200 px²) while producing a lens perhaps 0.3 mm wide, inside which `min_i d_i` peaks at half the width; whether *any* voxel centre lands in the positive region depends on lattice phase. The slot appears intermittently or not at all — while bricks were promoted, written and counted, so the status reports success over a visibly unchanged model. The guard must therefore be on the hull's **minimum width at the model's depth, in voxels** (require ≥ 2), not its area in pixels. `"cut cancelled: that loop is thinner than the voxel grid"`.
- **All-or-nothing plane construction.** If any `ClipPlane::new` returns `None` (degenerate cross product), refuse the whole gesture. Silently dropping one side plane opens the polyhedron into an unbounded wedge.

### 4.3 What happens to the cut-off piece

**Default: it is destroyed.** `max(field, cutter)` drives it to `OUTSIDE`, the cut face closes as arithmetic, one undo entry. This is deliberate — the research records users naming "an extra step to delete the cut piece from the layers panel" as friction, and `split.rs`'s own doc records that split peaks at 2× the body (**sourced**: 2,038 MB held + 1,920 MB live on the 22k-brick dragon), so the 4.15 GiB dragon needs ~8.3 GiB against the 6 GiB `MAX_VOLUME_BYTES` ceiling and is *refused* — "the case that motivates split is the case the guard rejects."

**Keep-both is the ctrl route, and it is a first-class phase, not an afterthought.** Ctrl at release writes the cutter's own feathered distance as protection instead of removing material — `Volume::cutter_mask`, sharing `halfspace_mask`'s boundary-only brick strategy so interior bricks stay `MaskBrick::Uniform(PROTECTED)` and the existing `feathered()` smoothstep applies. `MaskRecipe` is **not** modified: it derives `Copy` and lives inside `Message`, so a slice needs a lifetime, an `Arc` costs `Copy`, and a fixed array puts ~400 bytes into every message. A `Brokkr::mask_cutter` sibling of `mask_halfspace` gets the same result, keeping its two important properties — skip a body whose generated mask `is_free()` so a stroke drawn elsewhere does not wipe a hand-painted mask, and one `Entry` of N `Change::WholeMask`.

That mask then feeds the shipped `Document::split_masked` (split.rs:753-816), which lifts the region off as its own body in one parallel pass with no connectivity walk, guarded by `split_masked_guard`. Circle the lump → it becomes a layer. That is the Photoshop model, and it is what Nomad's own community recommends over Trim for exactly this task.

**Feathering is not optional.** mask.rs:44-54 records the rule in one line — "Every path that writes the mask writes a feathered edge, never a step" — and names lasso masks writing hard values as the prior-art failure that broke Blender's border detection for years and folds geometry under the Move brush.

One thing to check while there, flagged by a judge and worth a test rather than a comment: `split_masked` thresholds at `MASKED_ENOUGH_TO_SPLIT = 128` with `>=` (split.rs:146), and `feathered(0.0)` is `smoothstep(0.5) * 255 = 127.5`, which rounds to exactly 128. The boundary lands on the selected side **by one rounding step**. Pin it with a test so a later change to `PROTECTED`, the smoothstep, or `>=` does not silently start slicing a one-voxel slot along the selection boundary.

**Loose pieces are reported.** After a cut that changed bricks, run `Document::split_plan` (split.rs:512-661 — read-only, parallel, sorted largest-first, sized in mm³ via `SIGNIFICANT_MM3 = 1.0`) behind a time gate, and append `— 2 loose pieces` to the status with a one-click separate. Do **not** write a second connectivity walk, and specifically not a brick-level one: split.rs:1207-1242 measures a brick walk finding 1 component on the dragon where the voxel walk finds 29 / 47 / 85 / 182 at four voxel sizes, and fusing parts of 4.6% and 4.0% on a real four-part model.

### 4.4 Mask and multi-body semantics

Everything carries over unchanged, and each is pinned by an existing test:

- **Visibility**: the gesture acts on `self.drawn_nodes()` → `Document::display_visibility(solo)`. Solo narrows the cut (app.rs:12182); a hidden body the cutter passes over comes back bit-identical (clip.rs:790). Never `saved_visibility`, never "the active body". `visible: &[bool]` stays indexed by **node position** over `doc.nodes()`, including folder rows, with the existing `debug_assert_eq!` on length.
- **Mask**: uniformly honoured in every shape and every modifier state — this is the inconsistency Nomad shipped and had to patch, and the commuting property in §4.2 is why we get it structurally rather than by discipline.
- **Undo**: one `Entry` of N `Change::Bricks` built inside `Document::clip_convex`, not assembled by the caller. A cut across four bodies restores all four or none. A cut that changes nothing records **no** entry and does not set `unsaved`.
- **Stroke recorder**: `Volume::begin_stroke` debug-asserts that no recorder is open and in release silently overwrites (volume.rs:1464-1472). Safe today only because `DragKind::Cutting` and `DragKind::Sculpt` are mutually exclusive; that stays true.

### 4.5 What the status line says

The existing four-outcome contract (app.rs:5322-5369) is preserved **including its priority order**, pinned by `a_cut_a_mask_blocked_names_the_mask_and_the_body_it_is_on` (app.rs:11384):

1. `bricks_spared_by_mask > 0` → names the mask **and the body**: "the mask on Left Ear blocked the cut"
2. `bodies_crossed > 0` → "the cut crossed 3 bodies and found nothing to remove"
3. otherwise → "the cut missed the model"
4. success → "cut 812 bricks" / "cut 812 bricks across 2 bodies"

Additions, all new sentences rather than edits to the above:

- shape and depth on success: `"cut 812 bricks inside the lasso, 12 mm deep"` / `"…, all the way through"`
- `"cut cancelled: that loop is thinner than the voxel grid"`
- `"cut cancelled: that stroke is too intricate — 16 sides is the limit"`
- `"nothing behind it to spare — cut went through"` (the depth clamp declining)
- `"— 2 loose pieces"` suffix from `split_plan`

Failure sentences must contain `"could not"` to render red (panel.rs:398); success sentences must not. There is no `set_status` helper and there must not be one — `record_status_change` (app.rs:5449) diffs the field once per frame into the breadcrumb trail precisely so a new call site cannot forget to log. Note its consequence: a status set and replaced within one frame is never recorded.

---

## 5. Phased work

**What the estimates assume:** one engineer familiar with `clip.rs`; existing test fixtures (dragon, two-lobe, defective scan) available; `scripts/drive.py` working on this KDE Wayland session; no CI toolchain surprises — this project has kept `main` red for two releases on local-clippy-green-but-CI-red before, so every phase ends with `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` against **CI's** toolchain, not the local one. Estimates are working days and include writing the tests named in each check. They do **not** include review turnaround.

Total: **~16 days.** The winning design estimated 6.75; the judges independently put it at roughly 2× optimistic and they were right — phase 0 needs instrumentation hooks that do not exist, and the parallelisation the design named as a one-line mitigation is a restructure of the crate's most invariant-dense function.

---

### Phase 0 — Baseline the cut that exists (1.5 days)

**Work.** `benches/budget.rs` and `scale.rs` contain no occurrence of `clip` — there is no measured baseline, so "as fast as today" is currently unfalsifiable. Add rows for: a plane through the middle of the dragon, a plane that misses, and — the one that matters — **thirty cuts in a row** on the defective-scan fixture. Record per run: bricks classified, bricks crossed, bricks removed, bricks dirtied, `Document::clip` wall time, following remesh wall time, peak recorder bytes, and `MeshPool` **vertices watermark** (not `live`). Timing the classify pass and the per-voxel pass separately needs hooks inside a private hot function that has none, so this is a small API change, not just bench rows — which is why it is not half a day.

**Check.** Numbers exist for all eight quantities and are written into clip.rs's module doc. The dirty-brick figure is compared by hand against (crossed bricks × 27) to confirm the `FxHashSet` dedup does what clip.rs:345 claims.

---

### Phase 1 — Core: `ClipPlane` → `&[ClipPlane]` (2.5 days)

**Work.** In `clip.rs`: add `Volume::clip_convex(&mut self, planes: &[ClipPlane])` and `Document::clip_convex(&mut self, planes, visible) -> CutOutcome`; keep `Volume::clip` / `Document::clip` as one-line wrappers so every existing call site and test is untouched. Add `cut_distance(planes, at)` (the min) and `classify(planes, centre, half, band_mm)` (any-keeps / all-removes, early exit). Add the integer AABB pre-filter on the coord list. Generalise `cut_would_change_a_voxel`, keeping its early return. **Do not modify** `ClipPlane`, `range_over_box`, `cut_voxel`, `MaskRecipe`, or the undo shape. Delete the vestigial `let _ = origin;` at clip.rs:348.

**Check.** The **entire existing clip.rs suite passes with zero edits** — that is the real gate; if any of it needed editing the wrapper is wrong. Specifically `an_unmasked_cut_writes_exactly_what_the_plain_max_wrote`, `a_cut_does_not_promote_untouched_interior_bricks_to_dense`, `a_cut_through_a_fully_masked_body_leaves_the_field_and_every_brick_alone`, `a_mask_spares_nothing_from_a_cut_that_had_nothing_left_to_remove`, `a_cut_across_two_bodies_is_one_undo_entry_that_restores_both`. New: `a_one_plane_convex_cut_is_the_plane_cut_bit_for_bit` via `testing::assert_same_field`; `a_lasso_removes_what_is_inside_it_and_nothing_outside`; `a_lasso_drops_whole_bricks_it_encloses` (assert `Cut::Removes` fired via a resident-bytes drop); `a_lasso_does_not_promote_untouched_interior_bricks_to_dense` (mirror of clip.rs:580 — uniform bricks drop by at most a handful, `resident_bytes` less than doubles). Phase 0's plane numbers unchanged within noise.

---

### Phase 2 — Parallel classify, serial apply (2 days)

**Work.** Split `clip_convex_masked` into a read-only parallel classification pass over the filtered coord list producing `Vec<(BrickCoord, Cut)>`, and the existing serial mutation loop consuming it. Preserve, in order and verbatim: `record_for_undo` **before** `take_brick` (the recorder reads prior contents out of the map), the `Removes → Crosses` mask downgrade, the fully-protected skip, `is_collapsible` collapse, and `mark_dirty_voxel_range`. This is its own phase because the obvious shortcut — copying `generate::halfspace_mask`'s `par_iter` — does not apply: that function is `&self` and builds a fresh Vec.

**Check.** Every phase-1 test still green, byte-identity included. Phase 0's plane bench improves or holds; a 16-plane lasso classify on the dragon comes in under `EDIT_BUDGET` (4 ms, **sourced**) or the phase reports the real figure and we decide explicitly. Transient peak memory measured and confirmed unchanged.

---

### Phase 3 — App: path, hull, guards, the eye fix (2.5 days)

**Work.** `cut_path: Vec<Vec2>` on `Brokkr`; cleared on press, appended in the `Moved` arm with `CUT_PATH_SPACING_PX = 2.0`, consumed at release. Douglas–Peucker simplify. `enum CutShape { Line, Curve, Lasso }` with the deviation and closure tests. Monotone-chain hull; extension points for open strokes; angle-first decimation to `MAX_CUT_PLANES = 16`. **Use `camera.eye()`, not `ray()`'s near point**, with a `debug_assert` that all rays share an apex and a comment naming the perspective dependency. The three degenerate guards. `Volume::clip_convex` wired up. Status sentences. `live_tool_label` shape-aware. `"c"` in `viewport::shortcut`. Mid-drag Escape rung.

**Check.** The four pinned existing tests untouched and green: `a_left_to_right_drag_removes_a_consistent_side` (the side convention is empirical — it is re-verified, not reasoned about), `a_click_with_the_cut_armed_does_nothing`, `escape_disarms_the_cut`, `solo_narrows_the_cut_to_the_body_it_is_showing`. New, headless: **`the_region_cut_matches_the_region_drawn`** — build a hull, cut a slab fixture, project the resulting cut boundary back to screen and assert it lies within half a voxel of the drawn polygon. This is the test that would have caught the near-plane bug and its absence is why that bug survived the design review. Plus: a thin there-and-back loop is refused with its own sentence; a shallow arc across a two-lobe fixture removes everything on the arc's outer side and not a crescent; shift forces the line; escape mid-path leaves the tool armed; a lasso disarms the tool like the line does.

---

### Phase 4 — The preview (2 days)

**Work.** New `crates/brokkr-app/src/app/cut_preview.rs`, mirroring `cursor.rs`. Project the path with `ray_through`, place at the orbit target's depth, draw the **decimated hull** (not the raw stroke — what is shown must be what is removed) as a closed loop with `push_line` plus a translucent fan with `push_triangle`, and the doomed side of a `Line` as a shaded quad. Called from `refresh_overlay` into the existing sculpt batch. Add `Tool::Cut` to `refresh_hover`'s early return so the brush ring stops implying a press would sculpt. `refresh_overlay()` from the `Moved` arm during `DragKind::Cutting` — overlay only. Delete the false comment at app.rs:8097.

**Check.** Headless: `cut_preview::build` emits a closed loop of exactly `hull.len()` segments and shades the correct side for a line. **Real, via `scripts/drive.py`** (see `docs/DRIVING-THE-APP.md`): drag a loop, screenshot mid-drag, confirm the polygon is drawn over the model and the shading is on the side that subsequently disappears; confirm the brush ring is gone. The preview cannot be verified by `cargo test` and this project has shipped "the data changed but the renderer was not told" before — `drive.py` has already found five defects the suite could not see.

---

### Phase 5 — Depth, on by default (2 days)

**Work.** `raycast::first_solid_spans(volume, origin, dir, max)` beside the existing march (raycast.rs:43), reusing the sphere-tracing skeleton, the empty-brick span skip and `MAX_STEPS`, returning the first two spans. Grid of rays through the hull centroid region across every drawn body. Far plane placement with the next-surface clamp and the decline-and-say-so branch. Shift disables. Preview shading switches while shift is held. Depth reported in the status.

**Check.** Fixture with a spur in front of a wall, **with the wall close enough that the naive rule would eat it**: the default lasso removes the spur and leaves the wall bit-identical via `assert_same_field`, and the wall's minimum thickness is measured before and after and is unchanged. Same lasso with shift removes both. Fixture where the spur is flush against the body: the cut goes through and the test asserts **the status says so** rather than asserting the tool was clever. `drive.py` screenshots of both shading states.

---

### Phase 6 — Ctrl selects, and the selection becomes a layer (1.5 days)

**Work.** `Volume::cutter_mask(planes, feather_mm)` in `generate.rs`, with `halfspace_mask` reimplemented on top of it so there is one implementation rather than two; `cutter_mask_demand` as the shared non-copying branch of `generated_mask_demand`, so the refusal arrives before the allocation. `Brokkr::mask_cutter` as a sibling of `mask_halfspace`, keeping the `is_free()` skip and the one-`Entry`-of-N-`WholeMask` shape. Ctrl-at-release routes to it. Status offers the existing `Message::BodySplitMasked` follow-up.

**Check.** `a_ctrl_lasso_protects_what_is_inside_it_and_nothing_outside` (field bit-identical, `assert_same_field`); `a_ctrl_lasso_that_misses_leaves_a_hand_painted_mask_alone`; the mask boundary has real intermediate values with no step; **`the_feather_midpoint_lands_on_the_selected_side`** pinning the 127.5 → 128 → `>=` rounding chain. End to end by hand: ctrl+lasso a protrusion, `BodySplitMasked`, confirm the lump arrives as its own row in the layer panel with the source intact in history and both bodies printable.

---

### Phase 7 — Printability, cost, loose pieces, and driving it for real (2 days)

**Work.** The printability sweep. Loose-piece reporting via `split_plan` behind a time budget. Make the `MESH POOL FULL` banner's remedy actionable (see risks). Lasso rows in the bench. Full `drive.py` session on a real defective scan. CI-toolchain clippy and fmt.

**Check — and this is the phase where a real defect in the winning design is fixed.**

`is_printable()` is `boundary_edges == 0 && inconsistent_edges == 0 && triangles > 0` (**sourced**: export.rs:134-136). It does **not** count `non_manifold_edges`, and it fails on an emptied body. So `oblique_lassos_at_many_angles_are_all_printable`, the design's headline gate, is structurally blind to the exact defect a 16-plane polyhedron adds: sixteen sharp cut-cut edges running the full depth plus their corner junctions — and clip.rs already records four-way edges at cut rims as a non-convergent problem that was *accepted* rather than solved (1/0-then-12/1/6/2 bad edges at 1..6 voxels of rounding). The lasso sweep must therefore assert **`non_manifold_edges` against a plane-cut baseline on the same fixture**, not just `is_printable()`.

Other checks: `a_lasso_cut_leaves_no_more_dense_bricks_than_the_plane_that_covers_it` — an **absolute** assertion on dirty-brick count and `MeshPool` watermark delta against phase 0's recorded numbers, replacing the winning design's proposed `a_lasso_dirties_no_more_bricks_than_the_plane_that_covers_it`, which is unfalsifiable: a plane pushes every `Removes` brick (half the body) into `touched`, against a small lasso's handful of wall bricks, so it passes by three orders of magnitude and gates nothing. The **thirty-cuts-in-a-row** bench from phase 0 must show the watermark not climbing monotonically past a stated bound. A synthetic cut that severs a dumbbell reports two pieces; `split_plan` over budget reports nothing rather than stalling the commit.

`drive.py`: circle a spur on a real defective scan, commit, export, inspect. Confirm the preview redraws smoothly, the shape switch at the deviation threshold is legible, the status matches what happened, and the pool banner does not appear. **If circling the lump is not faster than orbit-plus-straight-line was, the design failed and the honest thing is to say so here.**

---

## 6. What v1 deliberately does not do

**Non-convex removal regions.** A C-shaped lasso is hulled, so the gap is cut too. Bounded by drawing the hull rather than the stroke, so the divergence is visible before release, and by one-entry undo.
*Trigger:* the first real session where a hulled stroke destroys geometry the user meant to keep.
*The fix, when we take it:* **not** `max` over `min` of ear-clipped pieces — the judges verified that produces exactly-zero values on every internal seam, deep inside the removal region, which forces `Crosses` instead of `Removes`, allocates 128 KB, and is then refused by `is_collapsible` so it stays resident forever. On a 20-point lasso that is ~17 seams each sweeping a brick sheet through the model depth, invisible because the exported mesh is correct. The correct fix is an **exact 2D signed distance to the simplified polygon in the cutter's cross-section**, with the perspective divergence handled explicitly — which is a real piece of work and needs its own design, not a graft.

**Print connectors on split (plug / dowel / snap).** PrusaSlicer's are the reference and the research is right that a split with no registration is an unfinished split for a tool whose output gets printed.
*Trigger:* the first model split for printing that has to be glued. It is a feature of split, not of cutting, and it needs its own per-material tolerance story.

**A shape-selector strip (lasso / rect / ellipse / line).** The single most consistent structural finding across all four research reports, and every surveyed application ships one. Deferred purely on vertical space: panel.rs:1838-1854 records the tool strip already ~8 px over what fits 768 with no scroll, and panel.rs:1546-1559 records the tool card at 660 px where "anything further has to take something out rather than add to the number".
*Trigger:* the next tool-strip re-layout — **and the screenshot at 768 that panel.rs itself says nobody has taken.** Rect and ellipse are then different constructors for the same plane slice; the geometry does not change.

**An orthographic camera mode.** The real fix for perspective taper, and every surveyed community has invented an "orthographic first" ritual around its absence.
*Trigger:* the first symmetric pair of cuts that comes out asymmetric. It is its own project with navcube, framing and picking consequences, and `finish_cut`'s existing plane construction silently depends on perspective — that dependency is now written down rather than latent.

**A re-editable cut after commit.** One-shot plus undo. Explicitly do **not** reuse the `Gizmo` struct for this: it names a `NodeId`, holds a `Similarity`, and arming it costs a full second copy of the body's field through the bake pipeline with its own refusal path (app.rs:2592-2638). Copy `gizmo::world_per_pixel`, `to_pixels`, `distance_to_segment` and the `contains`-then-`pick` shape if a re-editable plane widget is ever wanted; do not copy the struct.
*Trigger:* users repeatedly undoing and redrawing a nearly-right cut.

**A clip / push-flat mode.** Not built, in any form. See §2.

**A minimum-wall-thickness printability check.** See open questions — this is a gap the suite already has, not one this plan opens, and it should not be smuggled in here.

---

## 7. How this composes with the bodies / booleans milestone

The planned milestone is: multiple bodies, primitives as new bodies, booleans between bodies, a gizmo for moving bodies, removing thin hair strands, removing disconnected stragglers.

- **One lattice, no resampler.** A `Document` holds one `voxel_size` and a `Volume` has no origin field (volume.rs:416-443), so brick `c` covers bit-identically the same world box in every body. `merge.rs`'s `union_from` is therefore a plain brick-by-brick `min` with a hard `assert_eq!` on voxel size, classified by `Fold` from one map lookup a side; subtraction `max(a, -b)` is the same walk with a different fold. The "bodies must share a lattice or be resampled" prerequisite is **already satisfied by construction**. Do not build a resampler for it.
- **Body-as-cutter is not a `&[ClipPlane]`, and pretending otherwise would be slower.** A sampled field has no closed form; forcing it through a per-voxel evaluator would replace `merge.rs`'s one-lookup-a-side classifier with a hash lookup per voxel. Body subtraction shares this plan's *tail* — `cut_voxel`, `is_collapsible`, apron-correct dirty marking, one-entry undo — and supplies its verdicts from `Fold`. Two cutter sources, one write path, stated rather than forced. That honest boundary is the one genuinely good idea in the "Cutter" design and it survives here.
- **Primitive-as-cutter is nearly free.** `primitive.rs:94` already has closed-form 1-Lipschitz distances for cube, sphere and cylinder, and `primitive.rs:201` already runs a Lipschitz version of the same three-way brick classification — the second copy this codebase does not need a third of. A placed primitive cutter is `removal(p) = -kind.distance(placement.inverse_transform_point(p))` with Lipschitz constant `1 / placement.min_scale()` (similarity.rs:191). It slots into the same `Volume::clip_convex` skeleton with a different bound function, once that skeleton is generic over `(removal, bound)` — which phase 1 makes cheap even though it does not do it.
- **Thin hair strands are a selection problem and already have a recipe.** `MaskRecipe::Thickness` (generate.rs:151) answers it; the lasso adds the ability to *scope* it.
- **Disconnected stragglers are shipped.** `Document::split_plan` / `split` (split.rs:512, 677) is a full per-voxel union-find with mm³ sizing. Phase 7 wires the reporting; the milestone wires the button.
- **Visibility is already the include/exclude control** — `Document::clip(plane, visible)` cuts every drawn body, which is precisely the layer-stack model Live Boolean users describe liking, without Live Boolean's footgun of visibility *also* selecting the operator.

**Direction of travel:** cut with a shape → mask with the same shape → `split_masked` → a body → boolean with other bodies. Every arrow in that chain is either shipped or is one phase of this plan.

---

## 8. Risks and open questions

**Accepted, not solved.**

- **Perspective taper.** The cutter is a frustum wedge, not a prism, so a small lasso takes more than its screen region at depth. Every surveyed tool has this; there is no orthographic mode to offer. The preview is drawn at focus depth and is screen-exact — it shows the region, not the divergence.
- **The bevel at convex cut edges.** `min_i d_i` over-estimates outside a convex edge by `r(1 - sin(θ/2))`: 0.88 voxel at a right angle, 2.48 voxels at 20°, at the band edge. Angle-first decimation bounds how acute a hull corner can be; the error direction rounds rather than sharpens, which is the direction printing wants. This is the SDF analogue of Nomad's corner-rounding complaint, bounded by the lattice instead of by mesh density.
- **`split_masked` refuses the largest scans.** 2× peak against a 6 GiB ceiling; "the case that motivates split is the case the guard rejects." The refusal is correct and names a voxel size that would work, but users will meet it. The named fix (move bricks rather than copy them — 99.3% belong to one output) is out of scope here.
- **The status line's failure surface grows.** Five new sentences. Each needs its own assertion or the breadcrumb trail quietly degrades, and a status set and replaced within one frame is never recorded (app.rs:5449).

**Live risks with mitigations.**

- **Repeated-cut pool fragmentation — the known gotcha, now reachable from ordinary editing.** `BlockAllocator` never splits or merges blocks; a freed block is reusable only by a request rounding to the same granule count (`GRANULARITY = 256`); the bump `watermark()`, not `live()`, is what fails an allocation. A cut monotonically *shrinks* the meshes it touches, so every changed brick orphans a block in a large granule class and takes a fresh one off the bump. `MeshPool::reset` — the only cure — is reachable **only** from `rebuild_everything` (resample / import / open / re-orient) and from no editing operation. Trim thirty spikes off a scan in one session and the plausible outcome is `MESH POOL FULL` with `live` far below capacity, whose banner then advises "delete a body or resample coarser" — neither of which is what the user wants after thirty successful trims — with reopening the file as the only real route. Mitigation: phase 0's thirty-cut bench makes it measurable, phase 7 asserts a watermark bound, and the banner gains an actionable remedy (a rebuild-view action calling `reset` + `mark_everything_dirty`) instead of advice that does not apply. **Open question:** should a cut trigger that reset automatically above a watermark/live ratio? It is a multi-second full remesh on a large document, so my inclination is user-initiated — but the threshold and whether it should ever be automatic are unsettled.
- **Per-voxel cost on crossing bricks goes up by the plane count.** Up to 18 dot products where there was one, bounded to crossing bricks and offset by whole-brick drops inside the polyhedron. Phase 0 exists because there is no baseline; phase 2 exists because the mitigation is a restructure, not a one-liner.
- **Shape inference misfires.** `LASSO_DEVIATION_PX` and `CLOSE_RADIUS_PX` are judgement, not proof. A hand-drawn "straight" drag across 800 px can easily clear 8 px of deviation, which would make today's one reliable behaviour conditional on hand steadiness. The threshold must be set from a real hand on a real stroke during phase 3, and it is the single most likely constant to need retuning after a day of use. `Line` is the safe default, so the threshold should be **generous**, not tight, and shift forces it.
- **Undo transient peak is unchanged but unexamined.** `record_for_undo` clones live bricks uncompressed until `end_stroke`, so a large cut holds hundreds of megabytes transiently against the 256 MB `DEFAULT_HISTORY_BUDGET`. Not a regression; phase 0 records it, and any refusal should quote `Brick::encoded_bytes` rather than a ratio.
- **The preview is the one part the suite cannot see.** Phase 4 is not padding.

**Open questions the research could not settle.**

1. **`is_printable()` has no thickness term.** It counts holes and winding only (export.rs:134). A cut that grazes a tapering surface, or a plane that barely clips a bulge, leaves a wafer that is watertight, manifold, and unprintable — and no test in this repo can see it. Targets circulating for a 0.4 mm nozzle are 0.8 mm absolute / 1.2 mm recommended for supported walls and 1.2 / 1.6 mm for the unsupported free edge a cut creates, but these are community figures, not a spec, and they vary by material and slicer. **This is a gap the current suite already has**, made more reachable by any tool that cuts more often. I recommend it as separate work, sized on its own, and I am explicitly *not* smuggling it into this plan — but it should be scheduled, because "printability is a hard requirement" is not currently true of the check that enforces it.
2. **Whether Nomad's Boolean trim mode honours a mask in the current build.** The 2.9 changelog claims "trim: boolean now support masking"; a February 2026 forum answer to a 2.9.3 user says the opposite. Unresolved, and it does not affect us — our masking is uniform by construction — but it is why "mask-respect must not depend on mode" is stated as a constraint rather than assumed.
3. **The `feathered(0.0)` → 128 → `MASKED_ENOUGH_TO_SPLIT >=` chain.** Correct today by one rounding step. Pinned by a test in phase 6, but whether the threshold *should* be a strict `>` with an explicit tie-break is a design question nobody has answered.
4. **The 768-px tool strip.** panel.rs:1838-1854 says the arithmetic puts it ~8 px over and that a screenshot would settle it. Nobody has taken it. This plan avoids needing the answer by adding no button — but the shape-selector strip in §6, and every future tool, is blocked on it.
