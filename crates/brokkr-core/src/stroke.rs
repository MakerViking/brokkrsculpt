// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stroke path interpolation.
//!
//! Pointer samples arrive far apart during a fast drag, often further apart
//! than the brush is wide. Stamping once per event would leave a dotted trail
//! rather than a continuous cut, and the faster the user works the worse it
//! gets.
//!
//! So a stroke remembers where it last stamped and walks the straight line to
//! each new sample at a fixed spacing. The leftover distance stays on the
//! stroke rather than being rounded away, which keeps the spacing even across
//! event boundaries instead of clustering stamps at every sample.

use glam::Vec3;

/// Most stamps one pointer event may produce.
///
/// A cursor that jumps across the model, or a stall that batches a second of
/// motion into one event, would otherwise turn a single event into thousands of
/// stamps and drop a frame. Losing stroke fidelity in that case is the better
/// trade.
pub const MAX_STAMPS_PER_EVENT: usize = 64;

/// The state of one continuous drag.
#[derive(Debug, Default, Clone)]
pub struct Stroke {
    last: Option<Vec3>,
    /// Unit vector along the last segment walked, or zero before the stroke
    /// has moved far enough to have one.
    ///
    /// Kept as a field rather than returned alongside the stamps because every
    /// stamp one `advance` produces shares it: the walk is a straight line, so
    /// one direction covers the lot and the output stays a plain list of
    /// points.
    direction: Vec3,
}

impl Stroke {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a stroke, which always stamps once at the starting point.
    pub fn begin(&mut self, at: Vec3, out: &mut Vec<Vec3>) {
        self.last = Some(at);
        // A stroke that has not moved has no direction yet. Guessing one would
        // comb the first stamp of every stroke an arbitrary way.
        self.direction = Vec3::ZERO;
        out.push(at);
    }

    /// True while a drag is in progress.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.last.is_some()
    }

    pub fn end(&mut self) {
        self.last = None;
        self.direction = Vec3::ZERO;
    }

    /// Which way the stroke is travelling, once it has moved far enough to
    /// know. `None` at the start of a stroke and after it ends.
    ///
    /// Patterns that comb read this, so hair lies the way the brush was
    /// dragged rather than along a fixed world axis.
    #[inline]
    pub fn direction(&self) -> Option<Vec3> {
        (self.direction != Vec3::ZERO).then_some(self.direction)
    }

    /// Walk from the last stamp to `to`, appending a stamp centre every
    /// `spacing` world units.
    ///
    /// Appends nothing when the pointer has not yet moved a full spacing, and
    /// keeps the remainder for next time.
    pub fn advance(&mut self, to: Vec3, spacing: f32, out: &mut Vec<Vec3>) {
        let Some(from) = self.last else {
            self.begin(to, out);
            return;
        };
        debug_assert!(spacing > 0.0, "spacing must be positive");

        let delta = to - from;
        let distance = delta.length();
        if distance < spacing || !distance.is_finite() {
            return;
        }

        let steps = ((distance / spacing).floor() as usize).min(MAX_STAMPS_PER_EVENT);
        let unit = delta / distance;
        self.direction = unit;
        let step = unit * spacing;
        let mut at = from;
        for _ in 0..steps {
            at += step;
            out.push(at);
        }
        self.last = Some(at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beginning_a_stroke_stamps_once_where_it_started() {
        let mut stroke = Stroke::new();
        let mut out = Vec::new();
        stroke.begin(Vec3::new(1.0, 2.0, 3.0), &mut out);
        assert_eq!(out, vec![Vec3::new(1.0, 2.0, 3.0)]);
        assert!(stroke.is_active());
    }

    #[test]
    fn a_long_drag_is_broken_into_evenly_spaced_stamps() {
        let mut stroke = Stroke::new();
        let mut out = Vec::new();
        stroke.begin(Vec3::ZERO, &mut out);
        out.clear();

        stroke.advance(Vec3::new(10.0, 0.0, 0.0), 2.0, &mut out);
        assert_eq!(out.len(), 5, "a 10 unit drag at 2 unit spacing is 5 stamps");
        for (index, point) in out.iter().enumerate() {
            assert!((point.x - (index as f32 + 1.0) * 2.0).abs() < 1.0e-5, "uneven at {index}");
        }
    }

    #[test]
    fn a_movement_shorter_than_the_spacing_stamps_nothing_yet() {
        let mut stroke = Stroke::new();
        let mut out = Vec::new();
        stroke.begin(Vec3::ZERO, &mut out);
        out.clear();

        stroke.advance(Vec3::new(0.5, 0.0, 0.0), 2.0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn leftover_distance_carries_over_instead_of_bunching_stamps() {
        // The reason the remainder is kept: a slow drag delivers many small
        // events, and rounding each one down to the nearest stamp would either
        // drop the motion entirely or restart the spacing from every sample.
        let mut stroke = Stroke::new();
        let mut out = Vec::new();
        stroke.begin(Vec3::ZERO, &mut out);
        out.clear();

        // Ten events of 0.5 units each, at spacing 2, should give 2 stamps at
        // exactly 2 and 4 units along.
        for step in 1..=10 {
            stroke.advance(Vec3::new(step as f32 * 0.5, 0.0, 0.0), 2.0, &mut out);
        }
        assert_eq!(out.len(), 2, "got {out:?}");
        assert!((out[0].x - 2.0).abs() < 1.0e-5);
        assert!((out[1].x - 4.0).abs() < 1.0e-5);
    }

    #[test]
    fn a_cursor_jump_is_capped_rather_than_stalling_the_frame() {
        let mut stroke = Stroke::new();
        let mut out = Vec::new();
        stroke.begin(Vec3::ZERO, &mut out);
        out.clear();

        stroke.advance(Vec3::new(100_000.0, 0.0, 0.0), 0.1, &mut out);
        assert_eq!(out.len(), MAX_STAMPS_PER_EVENT);
    }

    #[test]
    fn advancing_without_beginning_starts_the_stroke() {
        let mut stroke = Stroke::new();
        let mut out = Vec::new();
        stroke.advance(Vec3::new(5.0, 0.0, 0.0), 1.0, &mut out);
        assert_eq!(out, vec![Vec3::new(5.0, 0.0, 0.0)]);
        assert!(stroke.is_active());
    }

    #[test]
    fn ending_a_stroke_resets_it() {
        let mut stroke = Stroke::new();
        let mut out = Vec::new();
        stroke.begin(Vec3::ZERO, &mut out);
        stroke.end();
        assert!(!stroke.is_active());

        // The next drag starts fresh rather than interpolating from the old
        // stroke's last point, which would drag a line across the model.
        out.clear();
        stroke.advance(Vec3::new(50.0, 0.0, 0.0), 1.0, &mut out);
        assert_eq!(out.len(), 1);
    }
}
