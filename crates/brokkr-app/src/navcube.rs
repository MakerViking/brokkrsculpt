// SPDX-License-Identifier: AGPL-3.0-or-later

//! The navigation cube in the corner of the viewport.
//!
//! Drawn through [`crate::cursor`]'s overlay pipeline in a pass of its own,
//! scissored to a corner box, with the same orientation as the model. Clicking a
//! part flies the camera to look from that direction.
//!
//! # Simpler than SindriCAD's, on purpose
//!
//! SindriCAD (`src/viewport/viewCube.ts`) builds twenty six pickable meshes — six
//! faces, twelve edges, eight corners — and raycasts them. Here a click
//! intersects **one box** and is then classified by how many of the hit point's
//! components are near an extreme: one is a face, two an edge, three a corner.
//! Same twenty six outcomes, no mesh soup, and it cannot drift out of step with
//! what was actually drawn.
//!
//! Note SindriCAD is Z-up CAD and this is Y-up, so the six directions below are
//! Brokkr's own rather than a port.

use brokkr_gpu::OverlayBatch;
use glam::{Mat4, Vec2, Vec3};

use crate::camera::OrbitCamera;
use crate::theme;

/// Edge length of the corner box, in logical pixels.
pub const SIZE_PX: f32 = 92.0;
/// Gap between the box and the top right of the viewport.
pub const MARGIN_PX: f32 = 10.0;

/// Half extent of the cube in its own space.
const HALF: f32 = 0.5;

/// How far a face's quad is inset from the cube's edge.
///
/// The gap is what makes the edges and corners look like the separate targets
/// they are, without drawing anything for them.
const INSET: f32 = 0.07;

/// How near an extreme a component has to be to count toward the classification.
///
/// At 0.35 of a 0.5 half extent, the middle 70% of a face picks the face and the
/// outer band picks the edge or corner it is nearest. Tuned to be forgiving:
/// a corner is a small target and missing it lands on a sensible edge.
const EXTREME: f32 = 0.35;

/// One thing on the cube that can be clicked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Part {
    /// Where the camera should sit, relative to the target. Unit length.
    pub direction: Vec3,
    /// How many components were extreme: 1 face, 2 edge, 3 corner.
    pub extremes: u8,
}

/// The six faces: outward normal, the two axes its label is drawn along, and
/// the letter.
///
/// Single letters rather than words, because there is no text renderer in
/// `brokkr-gpu` and these are drawn as line segments. `D` is down, which avoids
/// `B` having to mean both back and bottom.
const FACES: [(Vec3, Vec3, Vec3, char); 6] = [
    (Vec3::X, Vec3::NEG_Z, Vec3::Y, 'R'),
    (Vec3::NEG_X, Vec3::Z, Vec3::Y, 'L'),
    // Seen from above, the front of the model is at the bottom of the screen,
    // so the top face's label runs up toward the BACK. Mirrored for the bottom.
    // Both were the other way round first and the projection test caught it.
    (Vec3::Y, Vec3::X, Vec3::NEG_Z, 'T'),
    (Vec3::NEG_Y, Vec3::X, Vec3::Z, 'D'),
    (Vec3::Z, Vec3::X, Vec3::Y, 'F'),
    (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y, 'B'),
];

/// The corner box in widget local logical pixels: top left, then size.
pub fn corner_rect(viewport: Vec2) -> (Vec2, Vec2) {
    let size = Vec2::splat(SIZE_PX.min(viewport.x * 0.4).min(viewport.y * 0.4).max(1.0));
    (Vec2::new(viewport.x - size.x - MARGIN_PX, MARGIN_PX), size)
}

/// Whether a widget local point is inside the corner box.
pub fn contains(viewport: Vec2, at: Vec2) -> bool {
    let (origin, size) = corner_rect(viewport);
    at.x >= origin.x && at.y >= origin.y && at.x <= origin.x + size.x && at.y <= origin.y + size.y
}

/// The matrix the cube is drawn with.
///
/// Orthographic and centred on the origin, so the cube neither moves nor changes
/// size — only its orientation carries information. Rotation comes from the
/// camera, including its roll, so a rolled view tips the cube too.
pub fn view_projection(camera: &OrbitCamera) -> Mat4 {
    let direction = (camera.eye() - camera.target).normalize_or(Vec3::Z);
    let view = glam::camera::rh::view::look_at_mat4(direction * 4.0, Vec3::ZERO, camera.up());
    // A little over the cube's corner-to-centre distance (0.866) so a corner-on
    // view still fits with air around it.
    let reach = 1.05;
    let projection =
        glam::camera::rh::proj::directx::orthographic(-reach, reach, -reach, reach, 0.1, 10.0);
    projection * view
}

fn linear(colour: iced::Color, alpha: f32) -> [f32; 4] {
    let to_linear = |c: f32| c.powf(2.2);
    [to_linear(colour.r), to_linear(colour.g), to_linear(colour.b), alpha]
}

/// Line segments making up a letter, in a unit square with y up.
///
/// Blocky forms built from straight segments: at this size a curve would cost
/// vertices nobody can see, and a font atlas would be a dependency for six
/// glyphs.
fn glyph(letter: char) -> &'static [[(f32, f32); 2]] {
    const STEM: f32 = -0.28;
    const RIGHT: f32 = 0.28;
    match letter {
        'F' => {
            &[[(STEM, -0.5), (STEM, 0.5)], [(STEM, 0.5), (RIGHT, 0.5)], [(STEM, 0.0), (0.16, 0.0)]]
        }
        'B' => &[
            [(STEM, -0.5), (STEM, 0.5)],
            [(STEM, 0.5), (RIGHT, 0.5)],
            [(STEM, 0.0), (RIGHT, 0.0)],
            [(STEM, -0.5), (RIGHT, -0.5)],
            [(RIGHT, 0.5), (RIGHT, 0.0)],
            [(RIGHT, 0.0), (RIGHT, -0.5)],
        ],
        'L' => &[[(STEM, 0.5), (STEM, -0.5)], [(STEM, -0.5), (RIGHT, -0.5)]],
        'R' => &[
            [(STEM, -0.5), (STEM, 0.5)],
            [(STEM, 0.5), (RIGHT, 0.5)],
            [(STEM, 0.0), (RIGHT, 0.0)],
            [(RIGHT, 0.5), (RIGHT, 0.0)],
            [(0.0, 0.0), (RIGHT, -0.5)],
        ],
        'T' => &[[(-0.32, 0.5), (0.32, 0.5)], [(0.0, 0.5), (0.0, -0.5)]],
        'D' => &[
            [(STEM, -0.5), (STEM, 0.5)],
            [(STEM, 0.5), (0.12, 0.5)],
            [(STEM, -0.5), (0.12, -0.5)],
            [(0.12, 0.5), (RIGHT, 0.26)],
            [(RIGHT, 0.26), (RIGHT, -0.26)],
            [(RIGHT, -0.26), (0.12, -0.5)],
        ],
        _ => &[],
    }
}

/// Build the cube for one frame.
///
/// `hovered` is the part under the pointer, whose face is lit so it is obvious
/// what a click would do.
pub fn build(batch: &mut OverlayBatch, camera: &OrbitCamera, hovered: Option<Part>) {
    batch.clear();

    let towards_viewer = (camera.eye() - camera.target).normalize_or(Vec3::Z);
    let face_colour = linear(theme::PANEL_2, 0.92);
    let lit_colour = linear(theme::ACCENT, 0.92);
    let edge_colour = linear(theme::LINE_STRONG, 1.0);
    let label_colour = linear(theme::TEXT_DIM, 1.0);
    let lit_label = linear(theme::ON_ACCENT, 1.0);

    for (normal, u, v, letter) in FACES {
        // A face pointing away is still drawn: the solid pipeline writes depth,
        // so the near faces cover it, and skipping it here would open a hole
        // whenever the cube is seen corner on.
        let lit =
            hovered.is_some_and(|part| part.extremes == 1 && part.direction.dot(normal) > 0.9);
        let reach = HALF - INSET;
        let centre = normal * HALF;
        let corner = |su: f32, sv: f32| centre + u * (reach * su) + v * (reach * sv);

        batch.push_quad(
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
            if lit { lit_colour } else { face_colour },
        );

        // A rim, so a face reads as a face rather than as a flat colour where
        // two of them meet at a shallow angle.
        for (from, to) in [
            ((-1.0, -1.0), (1.0, -1.0)),
            ((1.0, -1.0), (1.0, 1.0)),
            ((1.0, 1.0), (-1.0, 1.0)),
            ((-1.0, 1.0), (-1.0, -1.0)),
        ] {
            batch.push_line(corner(from.0, from.1), corner(to.0, to.1), edge_colour);
        }

        // The label, only on faces turned toward the viewer: on a face seen
        // edge on it is unreadable smearing, and on a face pointing away it
        // would read mirrored.
        if normal.dot(towards_viewer) > 0.2 {
            let scale = reach * 0.55;
            // Lifted clear of the face so the depth bias does not have to fight
            // the quad it sits on.
            let plane = centre + normal * 0.012;
            for segment in glyph(letter) {
                let point = |p: (f32, f32)| plane + u * (p.0 * scale) + v * (p.1 * scale);
                batch.push_line(
                    point(segment[0]),
                    point(segment[1]),
                    if lit { lit_label } else { label_colour },
                );
            }
        }
    }
}

/// The label a direction carries on the cube.
///
/// The same vocabulary the cube's own faces are lettered with, so the menu that
/// opens on a face names things the way the face beside it does. `Down` rather
/// than `Bottom` for the same reason the cube uses `D`: `B` already means back.
pub fn facing_label(facing: brokkr_core::Facing) -> &'static str {
    match facing {
        brokkr_core::Facing::Right => "Right",
        brokkr_core::Facing::Left => "Left",
        brokkr_core::Facing::Up => "Top",
        brokkr_core::Facing::Down => "Down",
        brokkr_core::Facing::Front => "Front",
        brokkr_core::Facing::Back => "Back",
    }
}

/// Which part of the cube a widget local point is over, if any.
pub fn pick(camera: &OrbitCamera, viewport: Vec2, at: Vec2) -> Option<Part> {
    if !contains(viewport, at) {
        return None;
    }
    let (origin, size) = corner_rect(viewport);
    // Normalised device coordinates within the corner box. Screen y grows down
    // and clip space y grows up, hence the flip.
    let ndc =
        Vec2::new((at.x - origin.x) / size.x * 2.0 - 1.0, 1.0 - (at.y - origin.y) / size.y * 2.0);

    let inverse = view_projection(camera).inverse();
    let near = inverse * glam::Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
    let far = inverse * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
    let near = near.truncate() / near.w;
    let far = far.truncate() / far.w;
    let direction = (far - near).normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }

    classify(box_hit(near, direction)?)
}

/// Where a ray enters the cube, by the slab method.
fn box_hit(origin: Vec3, direction: Vec3) -> Option<Vec3> {
    let mut enter = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        let d = direction[axis];
        let o = origin[axis];
        if d.abs() < 1.0e-6 {
            // Parallel to this pair of faces: a miss unless it is already
            // between them.
            if o.abs() > HALF {
                return None;
            }
            continue;
        }
        let (mut low, mut high) = ((-HALF - o) / d, (HALF - o) / d);
        if low > high {
            std::mem::swap(&mut low, &mut high);
        }
        enter = enter.max(low);
        exit = exit.min(high);
        if enter > exit {
            return None;
        }
    }
    Some(origin + direction * enter)
}

/// Turn a hit point on the cube's surface into a view direction.
fn classify(hit: Vec3) -> Option<Part> {
    let mut direction = Vec3::ZERO;
    let mut extremes = 0u8;
    for axis in 0..3 {
        if hit[axis].abs() >= EXTREME {
            direction[axis] = hit[axis].signum();
            extremes += 1;
        }
    }
    // Every point on the surface has at least one component at ±HALF, so this
    // can only happen for a hit that was not on the surface at all.
    (extremes > 0).then(|| Part { direction: direction.normalize(), extremes })
}

/// The yaw and pitch that put the camera on a direction.
///
/// Pitch is clamped the way [`OrbitCamera::orbit_radians`] clamps it, so a top
/// or bottom face lands just short of the pole rather than collapsing the view
/// matrix. Straight up has no meaningful yaw, so the current one is kept —
/// otherwise clicking Top would also spin the model.
pub fn orientation(direction: Vec3, current_yaw: f32) -> (f32, f32) {
    let d = direction.normalize_or(Vec3::Z);
    let pitch = d.y.clamp(-1.0, 1.0).asin();
    let horizontal = Vec2::new(d.x, d.z);
    let yaw = if horizontal.length() < 1.0e-4 { current_yaw } else { d.x.atan2(d.z) };
    (yaw, pitch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> OrbitCamera {
        OrbitCamera::framing(Vec3::ZERO, 30.0)
    }

    const VIEWPORT: Vec2 = Vec2::new(1280.0, 720.0);

    fn centre_of_cube() -> Vec2 {
        let (origin, size) = corner_rect(VIEWPORT);
        origin + size * 0.5
    }

    #[test]
    fn the_box_sits_in_the_top_right_with_a_margin() {
        let (origin, size) = corner_rect(VIEWPORT);
        assert!((origin.x + size.x - (VIEWPORT.x - MARGIN_PX)).abs() < 1.0e-4);
        assert!((origin.y - MARGIN_PX).abs() < 1.0e-4);
        assert!(contains(VIEWPORT, centre_of_cube()));
    }

    /// A click anywhere but the corner belongs to the viewport, and stealing it
    /// would break sculpting over most of the screen.
    #[test]
    fn a_click_outside_the_box_picks_nothing() {
        for at in [
            Vec2::new(10.0, 10.0),
            Vec2::new(VIEWPORT.x * 0.5, VIEWPORT.y * 0.5),
            Vec2::new(VIEWPORT.x - 5.0, VIEWPORT.y - 5.0),
        ] {
            assert!(!contains(VIEWPORT, at), "{at:?} should be outside");
            assert!(pick(&camera(), VIEWPORT, at).is_none());
        }
    }

    /// The box shrinks rather than overflowing a small viewport, or it would
    /// cover the model on a narrow window.
    #[test]
    fn the_box_stays_inside_a_tiny_viewport() {
        let tiny = Vec2::new(120.0, 90.0);
        let (origin, size) = corner_rect(tiny);
        assert!(size.x <= tiny.x * 0.4 + 1.0);
        assert!(origin.x >= 0.0 && origin.y >= 0.0);
        assert!(origin.x + size.x <= tiny.x);
    }

    /// The centre of the box looks straight down the view axis, so it must pick
    /// a single face, and that face must be the one facing the camera.
    #[test]
    fn the_middle_of_the_box_picks_the_face_towards_the_camera() {
        for (yaw, pitch, expected) in [
            (0.0, 0.0, Vec3::Z),
            (std::f32::consts::FRAC_PI_2, 0.0, Vec3::X),
            (std::f32::consts::PI, 0.0, Vec3::NEG_Z),
            (-std::f32::consts::FRAC_PI_2, 0.0, Vec3::NEG_X),
            (0.0, 1.4, Vec3::Y),
            (0.0, -1.4, Vec3::NEG_Y),
        ] {
            let camera = OrbitCamera { yaw, pitch, ..camera() };
            let part = pick(&camera, VIEWPORT, centre_of_cube())
                .expect("the middle of the box should hit the cube");
            assert_eq!(part.extremes, 1, "the middle of a face is a face");
            assert!(
                part.direction.dot(expected) > 0.99,
                "at yaw {yaw} pitch {pitch} expected {expected:?}, got {:?}",
                part.direction
            );
        }
    }

    /// One box and a classification stands in for SindriCAD's twenty six
    /// meshes, so the classification is the thing to pin.
    #[test]
    fn the_classifier_separates_faces_edges_and_corners() {
        let face = classify(Vec3::new(0.0, 0.1, HALF)).unwrap();
        assert_eq!(face.extremes, 1);
        assert!(face.direction.dot(Vec3::Z) > 0.99);

        let edge = classify(Vec3::new(0.0, HALF, HALF)).unwrap();
        assert_eq!(edge.extremes, 2);
        assert!(edge.direction.dot(Vec3::new(0.0, 1.0, 1.0).normalize()) > 0.99);

        let corner = classify(Vec3::new(HALF, HALF, HALF)).unwrap();
        assert_eq!(corner.extremes, 3);
        assert!(corner.direction.dot(Vec3::splat(1.0).normalize()) > 0.99);

        // Every direction is unit length, since the camera is placed along it.
        for hit in
            [Vec3::new(0.2, -0.1, HALF), Vec3::new(-HALF, HALF, 0.0), Vec3::new(-HALF, -HALF, HALF)]
        {
            let part = classify(hit).unwrap();
            assert!((part.direction.length() - 1.0).abs() < 1.0e-5);
        }
    }

    /// Exactly on the threshold is the case a naive implementation gets wrong in
    /// a way nobody notices until a click does something surprising.
    #[test]
    fn a_hit_exactly_on_the_threshold_counts_as_extreme() {
        let part = classify(Vec3::new(EXTREME, 0.0, HALF)).unwrap();
        assert_eq!(part.extremes, 2, "on the boundary should already be an edge");

        let inside = classify(Vec3::new(EXTREME - 0.01, 0.0, HALF)).unwrap();
        assert_eq!(inside.extremes, 1, "just inside should still be the face");
    }

    #[test]
    fn every_part_gives_a_camera_orientation_that_holds_together() {
        for x in [-1.0f32, 0.0, 1.0] {
            for y in [-1.0f32, 0.0, 1.0] {
                for z in [-1.0f32, 0.0, 1.0] {
                    let direction = Vec3::new(x, y, z);
                    if direction == Vec3::ZERO {
                        continue;
                    }
                    let (yaw, pitch) = orientation(direction.normalize(), 0.3);
                    assert!(yaw.is_finite() && pitch.is_finite(), "{direction:?}");

                    let camera = OrbitCamera { yaw, pitch, ..camera() };
                    let basis = camera.view().to_cols_array();
                    assert!(
                        basis.iter().all(|v| v.is_finite()),
                        "{direction:?} collapsed the view"
                    );
                    assert!((camera.right().length() - 1.0).abs() < 1.0e-4);
                    assert!(camera.right().dot(camera.up()).abs() < 1.0e-4);
                }
            }
        }
    }

    /// Clicking Top should tip the view over the model, not also spin it: a
    /// straight up direction has no yaw to read, so the current one is kept.
    #[test]
    fn a_top_view_keeps_the_yaw_it_already_had() {
        let (yaw, pitch) = orientation(Vec3::Y, 1.234);
        assert_eq!(yaw, 1.234);
        assert!(pitch > 1.0, "a top view should look down from high up");

        let (yaw, pitch) = orientation(Vec3::NEG_Y, -0.5);
        assert_eq!(yaw, -0.5);
        assert!(pitch < -1.0);
    }

    #[test]
    fn the_cube_draws_six_faces_with_rims() {
        let mut batch = OverlayBatch::default();
        build(&mut batch, &camera(), None);
        // Six quads of two triangles each.
        assert_eq!(batch.surfaces.len(), 6 * 6);
        // Four rim segments per face, plus glyphs on whichever faces are turned
        // toward the viewer.
        assert!(batch.lines.len() >= 6 * 4 * 2);
    }

    #[test]
    fn hovering_a_face_lights_only_that_face() {
        let camera = camera();
        let mut plain = OverlayBatch::default();
        build(&mut plain, &camera, None);

        let hovered = pick(&camera, VIEWPORT, centre_of_cube()).unwrap();
        let mut lit = OverlayBatch::default();
        build(&mut lit, &camera, Some(hovered));

        let changed =
            plain.surfaces.iter().zip(&lit.surfaces).filter(|(a, b)| a.colour != b.colour).count();
        // Exactly one quad's worth of vertices, and no more.
        assert_eq!(changed, 6, "hovering should light one face, not {}", changed / 6);
    }

    /// An edge or a corner is a view direction, not a face, so nothing should
    /// light up: lighting the nearest face would say the wrong thing about what
    /// a click will do.
    #[test]
    fn hovering_an_edge_lights_no_face() {
        let camera = camera();
        let mut plain = OverlayBatch::default();
        build(&mut plain, &camera, None);

        let mut lit = OverlayBatch::default();
        build(
            &mut lit,
            &camera,
            Some(Part { direction: Vec3::new(1.0, 1.0, 0.0).normalize(), extremes: 2 }),
        );
        assert!(
            plain.surfaces.iter().zip(&lit.surfaces).all(|(a, b)| a.colour == b.colour),
            "an edge hover lit a face"
        );
    }

    /// Which way round each face's label goes is the one thing here that cannot
    /// be derived by staring at it — a mirrored R and a skewed T both shipped in
    /// the first version. So it is measured: project the label's own axes
    /// through the cube's matrix, with the camera square onto that face, and
    /// require +u to go right and +v to go up on screen.
    #[test]
    fn every_face_label_reads_the_right_way_round() {
        for (normal, u, v, letter) in FACES {
            let (yaw, pitch) = orientation(normal, 0.0);
            let camera = OrbitCamera { yaw, pitch: pitch.clamp(-1.5, 1.5), roll: 0.0, ..camera() };
            let matrix = view_projection(&camera);
            let project = |p: Vec3| {
                let clip = matrix * p.extend(1.0);
                Vec2::new(clip.x / clip.w, clip.y / clip.w)
            };

            let centre = normal * HALF;
            let origin = project(centre);
            let along_u = project(centre + u * 0.3) - origin;
            let along_v = project(centre + v * 0.3) - origin;

            assert!(
                along_u.x > 0.05,
                "{letter}: the label runs backwards, +u went {along_u:?} on screen"
            );
            assert!(
                along_v.y > 0.05,
                "{letter}: the label is upside down or sideways, +v went {along_v:?}"
            );
            // And the two axes are not collapsed onto each other, which would
            // mean the label is drawn edge on.
            assert!(
                along_u.x.abs() > along_u.y.abs(),
                "{letter}: +u is more vertical than horizontal: {along_u:?}"
            );
        }
    }

    #[test]
    fn every_face_letter_has_a_glyph_and_unknown_letters_draw_nothing() {
        for (_, _, _, letter) in FACES {
            assert!(!glyph(letter).is_empty(), "{letter} has no glyph");
        }
        assert!(glyph('%').is_empty());
    }
}
