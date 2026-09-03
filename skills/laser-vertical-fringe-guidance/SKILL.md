---
name: laser-vertical-fringe-guidance
description: Guide the current Michelson interferometer from live image analysis to clear, stable vertical laser fringes before white-light alignment. Use for laser fringe acquisition, straightening, spacing adjustment, or diagnosing why the live frame is not ready; do not use this device-specific knob mapping for another interferometer.
---

# Laser Vertical Fringe Guidance

Use the latest `vision.fringe_guidance` snapshot or call the read-only
`laser_fringe_analyze` tool before each instruction. Coordinates are image pixels with the
origin at the full-frame upper-left, x rightward and y downward. `tilt_deg` is relative to
vertical: positive is clockwise in the displayed image. Color is camera appearance only;
never infer optical path difference, phase, fringe order, or thickness from it.

## Decide from current evidence

Follow `laser_vertical_alignment.stage` unless newer visual evidence contradicts it:

- `acquire`: do not prescribe either tilt-knob rotation. Ask the experimenter to overlap the two main
  return spots and bring at least three continuous bright fringes into the ROI.
- `stabilize` or `improve_view`: do not turn a knob. Wait for stability or improve ROI,
  focus, and exposure until fresh centerlines and a credible angle are available.
- `straighten`: use only the upper knob on the upper-left rear side of the moving mirror.
  From behind the moving mirror, positive tilt needs this upper knob counterclockwise;
  negative tilt needs it clockwise. Give one approximately 1/16-turn
  step, then stop and reanalyse. If `abs(tilt_deg)` increases, return to the prior position
  and reverse direction.
- `improve_spacing`: keep the upper knob fixed and use only the lower knob on the
  lower-right rear side of the moving mirror after the stripes are near vertical.
  More than 10 reliable bright stripes in the ROI is too dense: turn this lower knob
  clockwise about 1/16 turn from behind the mirror. Fewer than 4 is too sparse: turn it counterclockwise
  about 1/16 turn. Stop when 4-10 reliable bright stripes remain; back off if spacing moves
  the wrong way or tilt grows. If spacing quality is invalid, improve the view before this adjustment.
- `ready`: stop turning, lock the adjustment, and save the laser reference frame before
  switching sources.

## Response contract

Give only one manual adjustment per turn in this order: current observation, one action,
expected visual change, and stop/recheck condition. Identify a fringe by its `id`, color,
`position.center_px`, and centerline shape when useful. Compare frames using absolute tilt,
normal spacing, centerline residual/curvature, brightness, motion, and reliable fringe count.
Do not treat display `correction_deg` as a mirror rotation.

The positional knob mapping is calibrated only when viewing this instrument from behind the moving
mirror. If viewpoint, mirror mount, or knob labels are uncertain, request a tiny reversible
trial instead of asserting a direction. Never claim to have turned a knob, never drive a
motor through this skill, and stop for abnormal resistance, end-of-travel, lost fringes,
severe saturation, vibration, or any laser-eye hazard.

Declare the laser preparation complete only when fresh stable frames show 4-10
continuous bright centerlines, `abs(tilt_deg) <= 3`, valid spacing quality, and no severe
blur or motion. This is preparation for white-light search, not proof of zero optical path.
