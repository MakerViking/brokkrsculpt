// SPDX-License-Identifier: AGPL-3.0-only

//! The timeline: stored views on a strip, and flying between them.
//!
//! ZBrush's Timeline, and the same idea. A key is the camera and the brush
//! settings at one moment, stored at a position along a strip; the playhead
//! runs along it and the camera follows. It is what turns "let me check that
//! from the front" into one click, and a row of keys into a turntable.
//!
//! # What a key holds, and what playback puts back
//!
//! A key stores a whole [`View`] -- camera, mirror planes, brush radius and
//! strength -- and **jumping to one restores all of it**. That is what makes a
//! key a stored working setup rather than only a camera angle.
//!
//! **Playback moves the camera and nothing else.** Watching a form turn should
//! not reach over and change the brush out from under the hand holding it, and
//! a mirror plane switching on halfway through a fly-through would be alarming
//! rather than useful. The two behaviours differ on purpose; see
//! [`Timeline::pose_at`].
//!
//! # Why the position is a fraction and not a time
//!
//! A key sits at `0..=1` along the strip. What that means in seconds is
//! [`PLAY_MS`], which is the application's business -- the file stores where
//! keys are relative to each other, so changing the playback speed later
//! cannot re-time everybody's saved keys.

use brokkr_core::{Keyframe, View};

use crate::camera::OrbitCamera;

/// How long the playhead takes to cross the whole strip.
///
/// Six seconds is a slow turn rather than a swoop -- long enough to read a
/// form, short enough to sit through twice.
const PLAY_MS: f32 = 6_000.0;

/// How close a click has to be to a key to count as hitting it, in pixels.
///
/// The same problem the right-click menu has, and the same answer: a pointer
/// is not precise, and a key is a small diamond. Generous enough to hit
/// without aiming, tight enough that clicking the gap between two keys adds
/// one rather than dragging a neighbour.
const HIT_SLOP_PX: f32 = 8.0;

/// Width the strip is assumed to be until the first layout reports otherwise.
///
/// Only ever wrong for one frame, and only for hit testing, which cannot
/// happen before the strip has been drawn.
const ASSUMED_WIDTH_PX: f32 = 600.0;

/// Stored views, and the state of scrubbing through them.
#[derive(Debug, Default)]
pub(crate) struct Timeline {
    /// In ascending `at` order, which everything here relies on.
    pub keys: Vec<Keyframe>,
    /// Where the playhead sits, `0..=1`.
    pub playhead: f32,
    pub playing: bool,
    /// The key a press picked up, if the pointer is dragging one.
    dragging: Option<usize>,
    /// The key under the pointer, so the strip can light it before it is
    /// clicked.
    pub hovered: Option<usize>,
    /// Where the pointer last was along the strip, `0..=1`.
    ///
    /// Kept because `mouse_area`'s `on_press` carries no position -- only
    /// `on_move` does -- so a press acts on wherever the last move reported.
    /// A pointer always moves onto the strip before it is pressed there.
    hover_at: Option<f32>,
    /// Pixel width of the strip, reported by the layout.
    width: f32,
}

/// What a press on the strip turned out to mean.
///
/// Returned rather than acted on, because every one of these is something only
/// the application can do -- it owns the camera, the brush, and the flag that
/// says the document has unsaved changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pressed {
    /// A key was hit: go to it, and a drag from here re-times it.
    WentTo(usize),
    /// Empty strip: a key was added there, holding the current view.
    Added(usize),
}

impl Timeline {
    pub fn new() -> Self {
        Self { width: ASSUMED_WIDTH_PX, ..Self::default() }
    }

    /// Replace every key, as a file being opened does.
    ///
    /// Playback stops: it was running through a different model's keys.
    pub fn adopt(&mut self, keys: Vec<Keyframe>) {
        self.keys = keys;
        self.keys.sort_by(|a, b| a.at.total_cmp(&b.at));
        self.playhead = 0.0;
        self.playing = false;
        self.dragging = None;
        self.hovered = None;
    }

    /// Tell the strip how wide it is being drawn, so a pixel can be turned
    /// into a position along it.
    pub fn resized(&mut self, width: f32) {
        if width.is_finite() && width > 1.0 {
            self.width = width;
        }
    }

    /// A pixel offset from the strip's left edge, as a position along it.
    fn position_of(&self, x: f32) -> f32 {
        (x / self.width).clamp(0.0, 1.0)
    }

    /// Note where the pointer is, and which key it is over.
    pub fn hover(&mut self, x: f32) {
        let at = self.position_of(x);
        self.hover_at = Some(at);
        self.hovered = self.key_near(at);
        if let Some(index) = self.dragging {
            self.retime(index, at);
        }
    }

    pub fn leave(&mut self) {
        self.hover_at = None;
        self.hovered = None;
        self.dragging = None;
    }

    pub fn release(&mut self) {
        self.dragging = None;
    }

    /// The key within [`HIT_SLOP_PX`] of a position, nearest first.
    fn key_near(&self, at: f32) -> Option<usize> {
        let slop = HIT_SLOP_PX / self.width;
        self.keys
            .iter()
            .enumerate()
            .map(|(index, key)| (index, (key.at - at).abs()))
            .filter(|(_, distance)| *distance <= slop)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index)
    }

    /// A press on the strip: hit a key to go to it, or add one where there is
    /// none.
    ///
    /// `view` is what a new key stores, so the caller passes the current one.
    pub fn press(&mut self, view: View) -> Option<Pressed> {
        let at = self.hover_at?;
        self.playing = false;
        self.playhead = at;

        if let Some(index) = self.key_near(at) {
            self.dragging = Some(index);
            return Some(Pressed::WentTo(index));
        }

        let index = self.insert(Keyframe { at, view });
        // A new key is picked up straight away, so the same gesture that
        // creates one can slide it into place.
        self.dragging = Some(index);
        Some(Pressed::Added(index))
    }

    /// A right press on the strip removes the key under it.
    pub fn remove_under_pointer(&mut self) -> Option<usize> {
        let at = self.hover_at?;
        let index = self.key_near(at)?;
        self.keys.remove(index);
        self.dragging = None;
        self.hovered = None;
        Some(index)
    }

    /// Store a key at `at`, keeping the list sorted, and return where it went.
    fn insert(&mut self, key: Keyframe) -> usize {
        let index = self.keys.partition_point(|existing| existing.at < key.at);
        self.keys.insert(index, key);
        index
    }

    /// Move a key along the strip, keeping the list sorted.
    ///
    /// Re-sorting means the dragged key's index can change under the drag, so
    /// the tracked index moves with it. Getting this wrong drops the drag the
    /// moment a key is pulled past its neighbour, which is exactly when a user
    /// is paying attention.
    fn retime(&mut self, index: usize, at: f32) {
        let Some(key) = self.keys.get_mut(index) else {
            self.dragging = None;
            return;
        };
        key.at = at;
        let key = self.keys.remove(index);
        let moved = self.insert(key);
        self.dragging = Some(moved);
        self.playhead = at;
    }

    pub fn dragged_key(&self) -> Option<usize> {
        self.dragging
    }

    /// The view at a position along the strip, interpolated between the two
    /// keys it falls between.
    ///
    /// `None` when there are no keys at all. Before the first key and after the
    /// last there is nothing to interpolate toward, so the nearest key's own
    /// view is the answer -- a playhead off the end of the keys holds still
    /// rather than flying somewhere nobody stored.
    ///
    /// The brush and mirror fields come from the *preceding* key rather than
    /// being blended. Radius would blend perfectly well; mirror planes are
    /// booleans and would have to snap somewhere regardless, and having half
    /// the view step while the other half slides is harder to predict than
    /// having all of it step.
    pub fn pose_at(&self, at: f32) -> Option<View> {
        let last = self.keys.len().checked_sub(1)?;
        let first = &self.keys[0];
        if at <= first.at {
            return Some(first.view);
        }
        let final_key = &self.keys[last];
        if at >= final_key.at {
            return Some(final_key.view);
        }

        // The first key at or past `at`; the one before it is its partner.
        let next = self.keys.partition_point(|key| key.at < at);
        let (before, after) = (&self.keys[next - 1], &self.keys[next]);
        let span = after.at - before.at;
        // Two keys stacked on the same spot: there is no span to travel along,
        // and dividing by it would give an infinity that reaches the camera.
        let t = if span > f32::EPSILON { (at - before.at) / span } else { 0.0 };
        // Eased, matching the navigation cube's flights, so a fly-through
        // settles into each key rather than cornering through it.
        let eased = t * t * (3.0 - 2.0 * t);

        Some(View {
            camera_target: before.view.camera_target.lerp(after.view.camera_target, eased),
            camera_distance: lerp(before.view.camera_distance, after.view.camera_distance, eased),
            // The short way round, so a turntable never unwinds most of a
            // circle to reach a heading a few degrees away.
            camera_yaw: before.view.camera_yaw
                + OrbitCamera::shortest_angle_delta(before.view.camera_yaw, after.view.camera_yaw)
                    * eased,
            camera_pitch: lerp(before.view.camera_pitch, after.view.camera_pitch, eased),
            camera_roll: lerp(before.view.camera_roll, after.view.camera_roll, eased),
            ..before.view
        })
    }

    /// Start or stop playback.
    ///
    /// Pressing play at the end starts again from the beginning, because the
    /// alternative is a button that visibly does nothing.
    pub fn toggle_play(&mut self) {
        if self.keys.len() < 2 {
            self.playing = false;
            return;
        }
        self.playing = !self.playing;
        if self.playing && self.playhead >= self.last_key_at() {
            self.playhead = self.keys[0].at;
        }
    }

    fn last_key_at(&self) -> f32 {
        self.keys.last().map_or(1.0, |key| key.at)
    }

    /// Advance the playhead. Returns the pose to fly to, if it moved.
    ///
    /// Playback runs from the first key to the last rather than across the
    /// whole strip, so the leading and trailing empty stretches are not dead
    /// time a viewer has to sit through.
    pub fn advance(&mut self, elapsed_ms: f32) -> Option<View> {
        if !self.playing {
            return None;
        }
        // Clamped the way a camera flight clamps it: a frame that took a long
        // time, because the window was hidden or a remesh ran long, must not
        // teleport the playhead past several keys.
        self.playhead += elapsed_ms.clamp(0.0, 50.0) / PLAY_MS;
        let end = self.last_key_at();
        if self.playhead >= end {
            self.playhead = end;
            self.playing = false;
        }
        self.pose_at(self.playhead)
    }

    /// Where each key sits along the strip, for drawing.
    pub fn positions(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.keys.iter().enumerate().map(|(index, key)| (index, key.at))
    }

    pub fn width(&self) -> f32 {
        self.width
    }
}

fn lerp(from: f32, to: f32, t: f32) -> f32 {
    from + (to - from) * t
}
