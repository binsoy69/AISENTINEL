# AISENTINEL Thresholds

This file explains the behavior-related thresholds in
`config/front_node.ini.example` and `config/mid_node.ini.example`. Both node
templates currently use the same threshold values. Local `front_node.ini` or
`mid_node.ini` files can override them.

Changes usually take effect after restarting the affected node process.

General tuning rule:

- Higher confidence thresholds usually reduce false positives but increase
  missed detections.
- Lower confidence thresholds usually catch more weak signals but increase
  noise.
- Higher sustain/cooldown values reduce alert spam but make the system slower
  to react.
- Lower sustain/cooldown values react faster but can create repeated or noisy
  incidents.

## Detector / Motion Mode

These values only matter when `[detector] mode` is not `front_runtime`. With
`front_runtime`, the full Hailo behavior runtime is used instead.

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `motion_threshold` | `24.0` | Needs stronger frame difference before motion incident. | More sensitive to small movement or lighting changes. |
| `motion_min_area_ratio` | `0.012` | More of the frame must move. | Smaller moving areas can trigger. |
| `motion_cooldown_sec` | `8.0` | Fewer repeated motion incidents. | Repeated incidents can happen sooner. |
| `annotated_banner_ttl_sec` | `4.0` | Alert banner stays longer on stream. | Banner disappears faster. |

## Inference

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `pose_confidence` | `0.70` | Fewer person/pose detections; fewer false positives, more missed students. | More detections; may include weak or noisy poses. |
| `object_confidence` | `0.25` | Fewer object detections overall. | More object detections, including weaker ones. |

## Object Detection Confidence Interaction

When `detector.mode = front_runtime`, object detection uses both the base
inference threshold and the class-specific object thresholds.

`[inference] object_confidence` is the first/base cutoff used when creating the
object model detector. Any object below this value can be discarded before the
behavior logic sees it.

After that, `[object_detection] phone_confidence` and
`cheat_sheet_confidence` are applied as class-specific filters.

The practical threshold is therefore the stricter value:

```text
effective phone threshold       = max(object_confidence, phone_confidence)
effective cheat_sheet threshold = max(object_confidence, cheat_sheet_confidence)
```

With the current template values:

```ini
[inference]
object_confidence = 0.25

[object_detection]
phone_confidence = 0.25
cheat_sheet_confidence = 0.30
```

The effective thresholds are:

```text
phone       >= 0.25
cheat_sheet >= 0.30
```

If `object_confidence = 0.50`, then both phone and cheat-sheet detections would
effectively need confidence `>= 0.50`, even if their class-specific values are
lower.

## Tracking

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `iou_threshold` | `0.30` | Student tracking needs tighter bbox overlap; identities may drop during movement. | Easier reacquisition, but higher risk of identity mixups. |
| `max_lost` | `60` | Keeps missing tracks longer; better for occlusion, slower to forget absent students. | Drops missing tracks faster. |

## Head Behavior

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `head_tilt_angle_deg` | `30.0` | Requires stronger sideways head tilt. | Flags smaller tilts. |
| `head_turn_ratio` | `0.18` | Requires larger nose/shoulder offset for head turn. | More sensitive to looking sideways. |
| `shoulder_turn_angle_deg` | `20.0` | Requires stronger shoulder rotation. | Flags smaller body turns. |
| `sustained_sec` | `2.5` | Behavior must last longer before incident. | Faster alerts, with more false-positive risk. |
| `event_cooldown_sec` | `10.0` | Same student's repeated head events are spaced out more. | Repeated head events can trigger sooner. |
| `keypoint_confidence` | `0.30` | Uses only stronger pose keypoints. | Uses weaker keypoints; noisier but less likely to miss. |

## Passing Papers

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `event_cooldown_sec` | `10.0` | Fewer repeated passing-paper alerts for the same pair. | Repeats sooner. |
| `keypoint_confidence` | `0.30` | Wrist/keypoint signals must be stronger. | More wrist signals accepted, but noisier. |
| `row_tolerance_px` | `80` | Students farther apart vertically can count as same row. | Stricter same-row matching. |
| `reference_bbox_height` | `300.0` | Changes perspective scaling baseline. Higher makes scaled pixel thresholds smaller for typical students. | Lower makes scaled thresholds larger. |
| `wrist_proximity_px` | `160` | Wrists can be farther apart and still count as interaction. | Wrists must be closer. |
| `min_interaction_sec` | `0.03` | Wrist proximity must last longer. | Very brief proximity can trigger. |

## Hands Under Table

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `hand_confidence` | `0.30` | Hand detections must be stronger; more missed hands. | More weak hand detections accepted. |
| `person_confidence` | `0.50` | Person detections must be stronger. | More weak person detections accepted. |
| `hands_missing_sustain_sec` | `3.0` | Hands must be missing longer before alert. | Alerts faster. |
| `event_cooldown_sec` | `10.0` | Fewer repeated alerts. | Repeats sooner. |
| `min_visible_hands` | `2` | Requires more visible hands to count as safe. | `1` would treat one visible hand as enough. |
| `hand_assoc_margin_px` | `60` | Hands farther outside student bbox can still associate. | Association must be closer. |
| `smooth_window_frames` | `12` | More smoothing; slower but steadier. | Faster response, more jitter. |
| `smooth_missing_ratio` | `0.60` | More frames in the window must show missing hands. | Easier to confirm missing hands. |
| `student_absent_reset_sec` | `2.0` | Waits longer before resetting missing-student state. | Resets faster when student disappears. |
| `table_edge_near_px` | `35` | Larger table-edge zone for detecting hand disappearance. | Stricter edge proximity. |
| `edge_disappear_arm_sec` | `0.75` | Longer allowed time after edge contact for hand disappearance. | Hand must disappear sooner after edge contact. |

## Object Detection

These class-specific thresholds are applied after the base
`[inference] object_confidence` filter.

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `person_confidence` | `0.50` | Stronger person detections required for object association. | More weak person detections accepted. |
| `phone_confidence` | `0.25` | Phone must be more confidently detected. | More phone detections, with more false-positive risk. |
| `cheat_sheet_confidence` | `0.30` | Cheat sheet must be more confidently detected. | More cheat-sheet detections, with more false-positive risk. |
| `event_cooldown_sec` | `10.0` | Fewer repeated phone/cheat-sheet alerts. | Repeats sooner. |
| `assoc_iou_thresh` | `0.05` | Object must overlap student more to associate. | Easier object-to-student association, with more wrong-association risk. |

## Evidence

These values do not decide whether behavior happened, but they affect evidence
size, quality, and how much context is saved around incidents.

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `pre_event_frames` | `2` | More frames before the event are saved. | Less pre-event context. |
| `post_event_frames` | `2` | More frames after the event are saved. | Less post-event context. |
| `gif_frame_count` | `5` | More GIF frames, larger evidence files. | Shorter GIFs, smaller files. |
| `gif_max_width` | `640` | Wider/sharper GIFs, more storage and CPU. | Smaller GIFs, less detail. |
| `gif_fps` | `4` | Faster GIF playback. | Slower GIF playback. |

## Spam Suppression

These values suppress duplicate incidents for the same student/behavior during
one session.

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `duplicate_suppression_sec` | `60.0` | Same behavior/student is suppressed longer. | Repeated incidents appear sooner. |
| `clear_required_sec` | `3.0` | Behavior must disappear longer before it can alert again. | Alerts can re-arm faster. |

## Sound Sensor

These values apply to KY-037 + ADS1015 sound monitoring when
`[sound_sensor] enabled = true`.

| Value | Current | Effect if increased | Effect if decreased |
|---|---:|---|---|
| `enabled` | `false` | `true` starts sound monitoring during sessions. | `false` disables sound telemetry/incidents. |
| `alert_threshold_db` | `55.0` | Louder sound required before a noise incident. | Quieter sounds can trigger incidents. |
| `incident_cooldown_sec` | `10.0` | Noise incidents repeat less often. | Noise incidents can repeat sooner. |
| `sample_interval` | `0.002` | Samples less frequently; lower CPU/I2C load but less detail. | Samples more frequently; more detail but more load. |
| `window_seconds` | `1.0` | Smoother readings, slower response. | Faster readings, more jitter. |
| `i2c_bus` | `1` | Must match Raspberry Pi I2C bus. Wrong value breaks sensor reads. | Must match Raspberry Pi I2C bus. Wrong value breaks sensor reads. |
| `i2c_address` | `0x48` | Must match ADS1015 address. Wrong value breaks sensor reads. | Must match ADS1015 address. Wrong value breaks sensor reads. |
| `adc_channel` | `0` | Must match sensor wiring channel. Wrong value reads the wrong input. | Must match sensor wiring channel. Wrong value reads the wrong input. |
| `full_scale` | `4.096` | Wider voltage range, less resolution. | Narrower range, more resolution but clipping risk. |
| `data_rate` | `1600` | More ADC samples per second, more responsive. | Fewer samples per second, less responsive. |
