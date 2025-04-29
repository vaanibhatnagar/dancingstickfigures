"""
Animation module for dance movements.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Stick figure geometry constants
class StickFigureParams:
    def __init__(self):
        self.hip_base_y = 1.0
        self.hip_x = 0.4
        self.foot1 = (0.45, 0.0)
        self.foot2 = (0.55, 0.0)
        self.head_offset = (0.7, 1.5)
        self.amplitude = 0.15
        self.shoulder_frac = 0.3
        self.bend_strength = 0.2
        self.knee_frac = 0.5
        self.eye_dx, self.eye_dy = 0.08, 0.1
        self.mouth_w, self.mouth_dy = 0.2, 0.1


def create_training_animation(
    pattern,
    beat_times,
    beat_types,
    best_phis,
    fitness_history,
    total_time,
    generations,
    fps=30,
):
    """Create animation showing the training process and final result."""
    stick_params = StickFigureParams()
    total_frames = int(total_time * fps)
    frames_per_gen = total_frames / generations

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle("Genetic Algorithm Dance Training", fontsize=16)

    # Left panel setup (stick figure)
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 3)
    ax1.axis("off")
    ax1.plot(*stick_params.foot1, "k_", ms=20)
    ax1.plot(*stick_params.foot2, "k_", ms=20)

    (th1,) = ax1.plot([], [], lw=3)  # thigh 1
    (sh1,) = ax1.plot([], [], lw=3)  # shin 1
    (th2,) = ax1.plot([], [], lw=3)  # thigh 2
    (sh2,) = ax1.plot([], [], lw=3)  # shin 2
    (back_line,) = ax1.plot([], [], lw=3)  # spine
    (larm,) = ax1.plot([], [], lw=2)  # left arm
    (rarm,) = ax1.plot([], [], lw=2)  # right arm
    head = plt.Circle((0, 0), 0.3, fill=False, lw=2)
    ax1.add_patch(head)
    (e1,) = ax1.plot([], [], "o", ms=4)  # eye 1
    (e2,) = ax1.plot([], [], "o", ms=4)  # eye 2
    (mouth,) = ax1.plot([], [], lw=1)  # mouth
    (k1dot,) = ax1.plot([], [], "o", ms=6)  # knee 1
    (k2dot,) = ax1.plot([], [], "o", ms=6)  # knee 2
    (beat_dot,) = ax1.plot([], [], "o", color="red", ms=8)  # beat indicator
    action_text = ax1.text(0.75, 2.8, "", fontsize=16, ha="center")
    type_text = ax1.text(0.75, 2.6, "", fontsize=14, ha="center", color="blue")

    # Right panel setup (learning curve)
    ax2.set_xlim(0, generations - 1)
    ymin, ymax = min(fitness_history), max(fitness_history)
    ax2.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Max Fitness")
    ax2.set_title("Learning Curve")
    (curve,) = ax2.plot([], [], lw=2)

    def init_train():
        for ln in [
            th1,
            sh1,
            th2,
            sh2,
            back_line,
            larm,
            rarm,
            e1,
            e2,
            mouth,
            k1dot,
            k2dot,
            beat_dot,
            curve,
        ]:
            ln.set_data([], [])
        head.set_center((0, 0))
        action_text.set_text("")
        type_text.set_text("")
        return [
            th1,
            sh1,
            th2,
            sh2,
            back_line,
            larm,
            rarm,
            head,
            e1,
            e2,
            mouth,
            k1dot,
            k2dot,
            beat_dot,
            action_text,
            type_text,
            curve,
        ]

    def animate_train(frame):
        """
        Animates the training process for the stick figure dance.

        Parameters:
        frame (int): The current frame number in the animation.

        Returns:
        list: A list of artists modified in this frame, for blitting.

        Description:
        This function updates the positions and states of various body parts of the stick figure
        based on the current frame. It distinguishes between fast and slow beats, setting the
        corresponding positions for the twerking or waving animations. The function also updates the
        face features, beat indicator, action text, and the learning curve on the right panel.
        """

        t = frame / fps
        gen = min(generations - 1, int(frame / frames_per_gen))
        phi = best_phis[gen]
        idx = np.searchsorted(beat_times, t) - 1
        idx = np.clip(idx, 0, len(beat_times) - 2)
        dt = pattern[idx]
        raw = np.sin(2 * np.pi * (t - beat_times[idx]) / dt + phi[idx])

        # Fast beat: twerk
        if beat_types[idx]:
            hy = stick_params.hip_base_y + stick_params.amplitude * raw
            hpt = (
                stick_params.hip_x + stick_params.head_offset[0],
                hy + stick_params.head_offset[1],
            )
            head.set_center(hpt)

            k1 = (
                (stick_params.hip_x + stick_params.foot1[0]) * stick_params.knee_frac,
                (hy + stick_params.foot1[1]) * stick_params.knee_frac
                + stick_params.bend_strength,
            )
            k2 = (
                (stick_params.hip_x + stick_params.foot2[0]) * stick_params.knee_frac,
                (hy + stick_params.foot2[1]) * stick_params.knee_frac
                + stick_params.bend_strength,
            )

            th1.set_data([stick_params.hip_x, k1[0]], [hy, k1[1]])
            sh1.set_data([k1[0], stick_params.foot1[0]], [k1[1], stick_params.foot1[1]])
            th2.set_data([stick_params.hip_x, k2[0]], [hy, k2[1]])
            sh2.set_data([k2[0], stick_params.foot2[0]], [k2[1], stick_params.foot2[1]])
            back_line.set_data([stick_params.hip_x, hpt[0]], [hy, hpt[1]])

            sx = (
                stick_params.hip_x
                + (hpt[0] - stick_params.hip_x) * stick_params.shoulder_frac
            )
            sy = hy + (hpt[1] - hy) * stick_params.shoulder_frac
            larm.set_data([sx, k1[0]], [sy, k1[1]])
            rarm.set_data([sx, k2[0]], [sy, k2[1]])
            k1dot.set_data([k1[0]], [k1[1]])
            k2dot.set_data([k2[0]], [k2[1]])
        # Slow beat: wave
        else:
            hy = stick_params.hip_base_y
            hpt = (
                stick_params.hip_x + stick_params.head_offset[0],
                hy + stick_params.head_offset[1],
            )
            head.set_center(hpt)

            k1 = (
                (stick_params.hip_x + stick_params.foot1[0]) * stick_params.knee_frac,
                (hy + stick_params.foot1[1]) * stick_params.knee_frac
                + stick_params.bend_strength,
            )
            k2 = (
                (stick_params.hip_x + stick_params.foot2[0]) * stick_params.knee_frac,
                (hy + stick_params.foot2[1]) * stick_params.knee_frac
                + stick_params.bend_strength,
            )

            th1.set_data([stick_params.hip_x, k1[0]], [hy, k1[1]])
            sh1.set_data([k1[0], stick_params.foot1[0]], [k1[1], stick_params.foot1[1]])
            th2.set_data([stick_params.hip_x, k2[0]], [hy, k2[1]])
            sh2.set_data([k2[0], stick_params.foot2[0]], [k2[1], stick_params.foot2[1]])
            back_line.set_data([stick_params.hip_x, hpt[0]], [hy, hpt[1]])

            sx = (
                stick_params.hip_x
                + (hpt[0] - stick_params.hip_x) * stick_params.shoulder_frac
            )
            sy = hy + (hpt[1] - hy) * stick_params.shoulder_frac
            angle = -np.pi / 2 + np.pi * (raw + 1) / 2
            x2 = sx + np.cos(angle)
            y2 = sy + np.sin(angle)
            larm.set_data([sx, x2], [sy, y2])
            rarm.set_data([sx, k2[0]], [sy, k2[1]])
            k1dot.set_data([], [])
            k2dot.set_data([], [])

        # Face
        e1.set_data([hpt[0] - stick_params.eye_dx], [hpt[1] + stick_params.eye_dy])
        e2.set_data([hpt[0] + stick_params.eye_dx], [hpt[1] + stick_params.eye_dy])
        mouth.set_data(
            [hpt[0] - stick_params.mouth_w / 2, hpt[0] + stick_params.mouth_w / 2],
            [hpt[1] - stick_params.mouth_dy] * 2,
        )

        # Beat indicator
        if np.any(np.isclose(t, beat_times, atol=1 / fps)):
            beat_dot.set_data([1.3], [2.8])
        else:
            beat_dot.set_data([], [])

        # Determine action
        action = "Twerk" if beat_types[idx] else "Wave"
        action_text.set_text(action)
        type_text.set_text("Fast beat" if beat_types[idx] else "Slow beat")

        # Growth curve
        xs = np.arange(gen + 1)
        ys = fitness_history[: gen + 1]
        curve.set_data(xs, ys)

        return [
            th1,
            sh1,
            th2,
            sh2,
            back_line,
            larm,
            rarm,
            head,
            e1,
            e2,
            mouth,
            k1dot,
            k2dot,
            beat_dot,
            action_text,
            type_text,
            curve,
        ]

    anim = animation.FuncAnimation(
        fig1,
        animate_train,
        init_func=init_train,
        frames=total_frames,
        interval=1000 / fps,
        blit=True,
    )

    return fig1, anim


def create_final_animation(
    pattern, beat_times, beat_types, best_phi, total_time, fps=30
):
    """Create final animation with the best solution."""
    stick_params = StickFigureParams()
    total_frames = int(total_time * fps)

    fig2, axf = plt.subplots(figsize=(6, 5))
    fig2.suptitle("Optimized Dance Animation", fontsize=16)
    axf.set_xlim(0, 1.5)
    axf.set_ylim(0, 3)
    axf.axis("off")
    axf.plot(*stick_params.foot1, "k_", ms=20)
    axf.plot(*stick_params.foot2, "k_", ms=20)

    (f_th1,) = axf.plot([], [], lw=3)
    (f_sh1,) = axf.plot([], [], lw=3)
    (f_th2,) = axf.plot([], [], lw=3)
    (f_sh2,) = axf.plot([], [], lw=3)
    (f_back,) = axf.plot([], [], lw=3)
    (f_larm,) = axf.plot([], [], lw=2)
    (f_rarm,) = axf.plot([], [], lw=2)
    f_head = plt.Circle((0, 0), 0.3, fill=False, lw=2)
    axf.add_patch(f_head)
    (f_e1,) = axf.plot([], [], "o", ms=4)
    (f_e2,) = axf.plot([], [], "o", ms=4)
    (f_mouth,) = axf.plot([], [], lw=1)
    (f_k1,) = axf.plot([], [], "o", ms=6)
    (f_k2,) = axf.plot([], [], "o", ms=6)
    (f_beat,) = axf.plot([], [], "o", color="red", ms=8)
    action_text = axf.text(0.75, 2.8, "", fontsize=16, ha="center")
    type_text = axf.text(0.75, 2.6, "", fontsize=14, ha="center", color="blue")

    def init_final():
        for ln in [
            f_th1,
            f_sh1,
            f_th2,
            f_sh2,
            f_back,
            f_larm,
            f_rarm,
            f_e1,
            f_e2,
            f_mouth,
            f_k1,
            f_k2,
            f_beat,
        ]:
            ln.set_data([], [])
        f_head.set_center((0, 0))
        action_text.set_text("")
        type_text.set_text("")
        return [
            f_th1,
            f_sh1,
            f_th2,
            f_sh2,
            f_back,
            f_larm,
            f_rarm,
            f_head,
            f_e1,
            f_e2,
            f_mouth,
            f_k1,
            f_k2,
            f_beat,
            action_text,
            type_text,
        ]

    def animate_final(frame):
        t = frame / fps
        idx = np.searchsorted(beat_times, t) - 1
        idx = np.clip(idx, 0, len(beat_times) - 2)
        dt = pattern[idx]
        raw = np.sin(2 * np.pi * (t - beat_times[idx]) / dt + best_phi[idx])

        if beat_types[idx]:
            hy = stick_params.hip_base_y + stick_params.amplitude * raw
            hpt = (
                stick_params.hip_x + stick_params.head_offset[0],
                hy + stick_params.head_offset[1],
            )
            f_head.set_center(hpt)

            k1 = (
                (stick_params.hip_x + stick_params.foot1[0]) * stick_params.knee_frac,
                (hy + stick_params.foot1[1]) * stick_params.knee_frac
                + stick_params.bend_strength,
            )
            k2 = (
                (stick_params.hip_x + stick_params.foot2[0]) * stick_params.knee_frac,
                (hy + stick_params.foot2[1]) * stick_params.knee_frac
                + stick_params.bend_strength,
            )

            f_th1.set_data([stick_params.hip_x, k1[0]], [hy, k1[1]])
            f_sh1.set_data(
                [k1[0], stick_params.foot1[0]], [k1[1], stick_params.foot1[1]]
            )
            f_th2.set_data([stick_params.hip_x, k2[0]], [hy, k2[1]])
            f_sh2.set_data(
                [k2[0], stick_params.foot2[0]], [k2[1], stick_params.foot2[1]]
            )
            f_back.set_data([stick_params.hip_x, hpt[0]], [hy, hpt[1]])

            sx = (
                stick_params.hip_x
                + (hpt[0] - stick_params.hip_x) * stick_params.shoulder_frac
            )
            sy = hy + (hpt[1] - hy) * stick_params.shoulder_frac
            f_larm.set_data([sx, k1[0]], [sy, k1[1]])
            f_rarm.set_data([sx, k2[0]], [sy, k2[1]])
            f_k1.set_data([k1[0]], [k1[1]])
            f_k2.set_data([k2[0]], [k2[1]])
        else:
            hy = stick_params.hip_base_y
            hpt = (
                stick_params.hip_x + stick_params.head_offset[0],
                hy + stick_params.head_offset[1],
            )
            f_head.set_center(hpt)

            k1 = (
                (stick_params.hip_x + stick_params.foot1[0]) * stick_params.knee_frac,
                (hy + stick_params.foot1[1]) * stick_params.knee_frac
                + stick_params.bend_strength,
            )
            k2 = (
                (stick_params.hip_x + stick_params.foot2[0]) * stick_params.knee_frac,
                (hy + stick_params.foot2[1]) * stick_params.knee_frac
                + stick_params.bend_strength,
            )

            f_th1.set_data([stick_params.hip_x, k1[0]], [hy, k1[1]])
            f_sh1.set_data(
                [k1[0], stick_params.foot1[0]], [k1[1], stick_params.foot1[1]]
            )
            f_th2.set_data([stick_params.hip_x, k2[0]], [hy, k2[1]])
            f_sh2.set_data(
                [k2[0], stick_params.foot2[0]], [k2[1], stick_params.foot2[1]]
            )
            f_back.set_data([stick_params.hip_x, hpt[0]], [hy, hpt[1]])

            sx = (
                stick_params.hip_x
                + (hpt[0] - stick_params.hip_x) * stick_params.shoulder_frac
            )
            sy = hy + (hpt[1] - hy) * stick_params.shoulder_frac
            angle = -np.pi / 2 + np.pi * (raw + 1) / 2
            x2 = sx + np.cos(angle)
            y2 = sy + np.sin(angle)
            f_larm.set_data([sx, x2], [sy, y2])
            f_rarm.set_data([sx, k2[0]], [sy, k2[1]])
            f_k1.set_data([], [])
            f_k2.set_data([], [])

        # Determine action
        action = "Twerk" if beat_types[idx] else "Wave"
        action_text.set_text(action)
        type_text.set_text("Fast beat" if beat_types[idx] else "Slow beat")

        # Face
        f_e1.set_data([hpt[0] - stick_params.eye_dx], [hpt[1] + stick_params.eye_dy])
        f_e2.set_data([hpt[0] + stick_params.eye_dx], [hpt[1] + stick_params.eye_dy])
        f_mouth.set_data(
            [hpt[0] - stick_params.mouth_w / 2, hpt[0] + stick_params.mouth_w / 2],
            [hpt[1] - stick_params.mouth_dy] * 2,
        )

        # Beat indicator
        if np.any(np.isclose(t, beat_times, atol=1 / fps)):
            f_beat.set_data([1.3], [2.8])
        else:
            f_beat.set_data([], [])

        return [
            f_th1,
            f_sh1,
            f_th2,
            f_sh2,
            f_back,
            f_larm,
            f_rarm,
            f_head,
            f_e1,
            f_e2,
            f_mouth,
            f_k1,
            f_k2,
            f_beat,
            action_text,
            type_text,
        ]

    anim = animation.FuncAnimation(
        fig2,
        animate_final,
        init_func=init_final,
        frames=total_frames,
        interval=1000 / fps,
        blit=True,
    )
    return fig2, anim
