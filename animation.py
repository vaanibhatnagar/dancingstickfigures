"""
This module contains the StickFigureAnimator class, which is responsible for
creating and animating a stick figure that dances to a metronome beat.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sounddevice as sd


class StickFigureAnimator:
    def __init__(
        self, beat_times, beat_types, pattern, bpm, total_time, metronome, sample_rate
    ):
        """
        Initialize the StickFigureAnimator class.

        Parameters
        ----------
        beat_times : array_like
            Cumulative times of each beat.
        beat_types : array_like
            Array indicating fast beats (True) and slow beats (False).
        pattern : array_like
            Sequence of intervals for each beat in the pattern.
        bpm : int
            Beats per minute for the pattern.
        total_time : float
            Total duration of the pattern.
        metronome : array_like
            The generated metronome audio data.
        sample_rate : int
            Sample rate of the generated audio.

        Attributes
        ----------
        hip_base_y : float
            Y-coordinate of the hip base.
        hip_x : float
            X-coordinate of the hip.
        foot1, foot2 : tuple
            Coordinates of the two feet.
        head_offset : tuple
            Offset of the head from the hip.
        amplitude : float
            Amplitude of the dance moves.
        shoulder_frac : float
            Fraction of the arm to use for the shoulder.
        bend_strength : float
            Strength of the bend in the elbow.
        knee_frac : float
            Fraction of the leg to use for the knee.
        eye_dx, eye_dy : float
            Size of the eyes.
        mouth_w, mouth_dy : float
            Size of the mouth.
        played : bool
            Flag indicating whether the animation has been played.
        start_time : float
            Timestamp of when the animation was started.
        frames_per_gen : int
            Number of frames to animate per generation.
        raws : list
            List of raw animation frames.
        """
        self.beat_times = beat_times
        self.beat_types = beat_types
        self.pattern = pattern
        self.bpm = bpm
        self.total_time = total_time
        self.metronome = metronome
        self.sample_rate = sample_rate
        self.M = len(beat_times)
        self.fps = 30
        self.total_frames = int(total_time * self.fps)

        # Stick figure geometry parameters
        self.hip_base_y = 1.0
        self.hip_x = 0.4
        self.foot1 = (0.45, 0.0)
        self.foot2 = (0.55, 0.0)
        self.head_offset = (0.7, 1.5)
        self.amplitude = 0.15
        self.shoulder_frac = 0.3
        self.bend_strength = 0.2
        self.knee_frac = 0.75
        self.eye_dx, self.eye_dy = 0.08, 0.1
        self.mouth_w, self.mouth_dy = 0.2, 0.1

        # Global variables
        self.played = False
        self.start_time = None
        self.frames_per_gen = self.total_frames / 150  # generations
        self.raws = []

    def create_training_animation(self, best_genomes, fitness_history, final_moves):
        """
        Create an animation depicting the training process of the stick figure dancer.

        Parameters
        ----------
        best_genomes : list of arrays
            The best phase values for each generation in the genetic algorithm.
        fitness_history : list of floats
            The fitness scores of the best genome for each generation.
        final_moves : array_like
            Boolean array indicating which beats are considered fast (True) or slow (False).

        Returns
        -------
        fig1 : matplotlib.figure.Figure
            The figure object containing the animation.
        anim1 : matplotlib.animation.FuncAnimation
            The animation object showcasing the stick figure's dance and learning curve.
        """

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left panel setup
        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(0, 3)
        ax1.axis("off")
        ax1.plot(*self.foot1, "k_", ms=20)
        ax1.plot(*self.foot2, "k_", ms=20)
        (th1,) = ax1.plot([], [], lw=3)
        (sh1,) = ax1.plot([], [], lw=3)
        (th2,) = ax1.plot([], [], lw=3)
        (sh2,) = ax1.plot([], [], lw=3)
        (back_line,) = ax1.plot([], [], lw=3)
        (larm,) = ax1.plot([], [], lw=2)
        (rarm,) = ax1.plot([], [], lw=2)
        head = plt.Circle((0, 0), 0.3, fill=False, lw=2)
        ax1.add_patch(head)
        (e1,) = ax1.plot([], [], "o", ms=4)
        (e2,) = ax1.plot([], [], "o", ms=4)
        (mouth,) = ax1.plot([], [], lw=1)
        (k1dot,) = ax1.plot([], [], "o", ms=6)
        (k2dot,) = ax1.plot([], [], "o", ms=6)
        (beat_dot,) = ax1.plot([], [], "o", color="red", ms=8)
        action_text = ax1.text(0.75, 2.8, "", fontsize=16, ha="center")
        type_text = ax1.text(0.75, 2.6, "", fontsize=14, ha="center", color="blue")

        # Right panel setup
        ax2.set_xlim(0, len(fitness_history) - 1)
        ymin, ymax = min(fitness_history), max(fitness_history)
        ax2.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Max Fitness")
        ax2.set_title("Learning Curve")
        (curve,) = ax2.plot([], [], lw=2)
        action_text = ax1.text(0.25, 2.8, "", fontsize=16, ha="center")
        type_text = ax1.text(0.5, 2.6, "", fontsize=14, ha="center", color="blue")

        def init_train():
            self.start_time = time.perf_counter()
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
            Animation function for training phase.

            Parameters
            ----------
            frame : int
                Frame number in the animation.

            Returns
            -------
            artists : list
                List of artists to be updated in the animation.
            """
            t = time.perf_counter() - self.start_time
            gen = min(len(best_genomes) - 1, int(frame / self.frames_per_gen))
            phi = best_genomes[gen]
            idx = np.searchsorted(self.beat_times, t) - 1
            idx = np.clip(idx, 0, self.M - 2)
            dt = self.pattern[idx]
            raw = np.sin(2 * np.pi * (t - self.beat_times[idx]) / dt + phi[idx])
            self.raws.append(raw)

            # Fast beat: twerk
            if final_moves[idx]:
                hy = self.hip_base_y + self.amplitude * raw
                hpt = (self.hip_x + self.head_offset[0], hy + self.head_offset[1])
                head.set_center(hpt)
                k1 = (
                    (self.hip_x + self.foot1[0]) * self.knee_frac,
                    (hy + self.foot1[1]) * self.knee_frac + self.bend_strength,
                )
                k2 = (
                    (self.hip_x + self.foot2[0]) * self.knee_frac,
                    (hy + self.foot2[1]) * self.knee_frac + self.bend_strength,
                )
                th1.set_data([self.hip_x, k1[0]], [hy, k1[1]])
                sh1.set_data([k1[0], self.foot1[0]], [k1[1], self.foot1[1]])
                th2.set_data([self.hip_x, k2[0]], [hy, k2[1]])
                sh2.set_data([k2[0], self.foot2[0]], [k2[1], self.foot2[1]])
                back_line.set_data([self.hip_x, hpt[0]], [hy, hpt[1]])
                sx = self.hip_x + (hpt[0] - self.hip_x) * self.shoulder_frac
                sy = hy + (hpt[1] - hy) * self.shoulder_frac
                larm.set_data([sx, k1[0]], [sy, k1[1]])
                rarm.set_data([sx, k2[0]], [sy, k2[1]])
                k1dot.set_data([k1[0]], [k1[1]])
                k2dot.set_data([k2[0]], [k2[1]])
            # Slow beat: wave
            else:
                hy = self.hip_base_y
                hpt = (self.hip_x + self.head_offset[0], hy + self.head_offset[1])
                head.set_center(hpt)
                k1 = (
                    (self.hip_x + self.foot1[0]) * self.knee_frac,
                    (hy + self.foot1[1]) * self.knee_frac + self.bend_strength,
                )
                k2 = (
                    (self.hip_x + self.foot2[0]) * self.knee_frac,
                    (hy + self.foot2[1]) * self.knee_frac + self.bend_strength,
                )
                th1.set_data([self.hip_x, k1[0]], [hy, k1[1]])
                sh1.set_data([k1[0], self.foot1[0]], [k1[1], self.foot1[1]])
                th2.set_data([self.hip_x, k2[0]], [hy, k2[1]])
                sh2.set_data([k2[0], self.foot2[0]], [k2[1], self.foot2[1]])
                back_line.set_data([self.hip_x, hpt[0]], [hy, hpt[1]])
                sx = self.hip_x + (hpt[0] - self.hip_x) * self.shoulder_frac
                sy = hy + (hpt[1] - hy) * self.shoulder_frac
                angle = -np.pi / 2 + np.pi * (raw + 1) / 2
                x2 = sx + np.cos(angle)
                y2 = sy + np.sin(angle)
                larm.set_data([sx, x2], [sy, y2])
                rarm.set_data([sx, k2[0]], [sy, k2[1]])
                k1dot.set_data([], [])
                k2dot.set_data([], [])

            # Face
            e1.set_data([hpt[0] - self.eye_dx], [hpt[1] + self.eye_dy])
            e2.set_data([hpt[0] + self.eye_dx], [hpt[1] + self.eye_dy])
            mouth.set_data(
                [hpt[0] - self.mouth_w / 2, hpt[0] + self.mouth_w / 2],
                [hpt[1] - self.mouth_dy] * 2,
            )

            # Beat indicator
            if np.any(np.isclose(t, self.beat_times, atol=1 / self.fps)):
                beat_dot.set_data([1.3], [2.8])
            else:
                beat_dot.set_data([], [])

            # Determine action
            action = "Twerk" if final_moves[idx] else "Wave"
            action_text.set_text(action)
            type_text.set_text("Fast beat" if final_moves[idx] else "Slow beat")

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

        anim1 = animation.FuncAnimation(
            fig1,
            animate_train,
            init_func=init_train,
            frames=self.total_frames,
            interval=1000 / self.fps,
            blit=True,
        )

        return fig1, anim1

    def create_final_animation(
        self,
        final_moves,
        best_genome,
        pattern,
        beat_times,
        beat_types,
        best_phi,
        total_time,
        fps=30,
    ):
        """
        Create and return the final animation of the stick figure synchronized with the
        provided beat pattern and optimal genome solution.

        Parameters
        ----------
        final_moves : array_like
            A list indicating which moves are final.
        best_genome : array_like
            The best genome solution from the genetic algorithm.
        pattern : array_like
            Sequence of intervals for each beat in the pattern.
        beat_times : array_like
            Cumulative times of each beat.
        beat_types : array_like
            Array indicating fast beats (True) and slow beats (False).
        best_phi : array_like
            Optimal phase values for the stick figure movements.
        total_time : float
            Total duration of the animation.
        fps : int, optional
            Frames per second for the animation, default is 30.

        Returns
        -------
        fig2 : matplotlib.figure.Figure
            The figure object for the animation.
        anim2 : matplotlib.animation.FuncAnimation
            The animation object controlling the stick figure animation.
        """

        total_frames = int(total_time * fps)

        fig2, axf = plt.subplots(figsize=(6, 5))
        axf.set_xlim(0, 1.5)
        axf.set_ylim(0, 3)
        axf.axis("off")
        axf.plot(*self.foot1, "k_", ms=20)
        axf.plot(*self.foot2, "k_", ms=20)
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

        def init_final():
            self.start_time = time.perf_counter()
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
            ]

        def animate_final(frame):
            """
            Animation function for final phase.

            Parameters
            ----------
            frame : int
                Frame number in the animation.

            Returns
            -------
            artists : list
                List of artists to be updated in the animation.
            """
            t = time.perf_counter() - self.start_time
            idx = np.searchsorted(self.beat_times, t) - 1
            idx = np.clip(idx, 0, self.M - 2)
            dt = self.pattern[idx]
            raw = np.sin(2 * np.pi * (t - self.beat_times[idx]) / dt + best_genome[idx])

            if final_moves[idx]:
                hy = self.hip_base_y + self.amplitude * raw
                hpt = (self.hip_x + self.head_offset[0], hy + self.head_offset[1])
                f_head.set_center(hpt)
                k1 = (
                    (self.hip_x + self.foot1[0]) * self.knee_frac,
                    (hy + self.foot1[1]) * self.knee_frac + self.bend_strength,
                )
                k2 = (
                    (self.hip_x + self.foot2[0]) * self.knee_frac,
                    (hy + self.foot2[1]) * self.knee_frac + self.bend_strength,
                )
                f_th1.set_data([self.hip_x, k1[0]], [hy, k1[1]])
                f_sh1.set_data([k1[0], self.foot1[0]], [k1[1], self.foot1[1]])
                f_th2.set_data([self.hip_x, k2[0]], [hy, k2[1]])
                f_sh2.set_data([k2[0], self.foot2[0]], [k2[1], self.foot2[1]])
                f_back.set_data([self.hip_x, hpt[0]], [hy, hpt[1]])
                sx = self.hip_x + (hpt[0] - self.hip_x) * self.shoulder_frac
                sy = hy + (hpt[1] - hy) * self.shoulder_frac
                f_larm.set_data([sx, k1[0]], [sy, k1[1]])
                f_rarm.set_data([sx, k2[0]], [sy, k2[1]])
                f_k1.set_data([k1[0]], [k1[1]])
                f_k2.set_data([k2[0]], [k2[1]])

                # Determine action
                action_text.set_text("Twerk")
            else:
                hy = self.hip_base_y
                hpt = (self.hip_x + self.head_offset[0], hy + self.head_offset[1])
                f_head.set_center(hpt)
                k1 = (
                    (self.hip_x + self.foot1[0]) * self.knee_frac,
                    (hy + self.foot1[1]) * self.knee_frac + self.bend_strength,
                )
                k2 = (
                    (self.hip_x + self.foot2[0]) * self.knee_frac,
                    (hy + self.foot2[1]) * self.knee_frac + self.bend_strength,
                )
                f_th1.set_data([self.hip_x, k1[0]], [hy, k1[1]])
                f_sh1.set_data([k1[0], self.foot1[0]], [k1[1], self.foot1[1]])
                f_th2.set_data([self.hip_x, k2[0]], [hy, k2[1]])
                f_sh2.set_data([k2[0], self.foot2[0]], [k2[1], self.foot2[1]])
                f_back.set_data([self.hip_x, hpt[0]], [hy, hpt[1]])
                sx = self.hip_x + (hpt[0] - self.hip_x) * self.shoulder_frac
                sy = hy + (hpt[1] - hy) * self.shoulder_frac
                angle = -np.pi / 2 + np.pi * (raw + 1) / 2
                x2 = sx + np.cos(angle)
                y2 = sy + np.sin(angle)
                f_larm.set_data([sx, x2], [sy, y2])
                f_rarm.set_data([sx, k2[0]], [sy, k2[1]])
                f_k1.set_data([], [])
                f_k2.set_data([], [])

                # Determine action
                action_text.set_text("Wave")

            # face
            f_e1.set_data([hpt[0] - self.eye_dx], [hpt[1] + self.eye_dy])
            f_e2.set_data([hpt[0] + self.eye_dx], [hpt[1] + self.eye_dy])
            f_mouth.set_data(
                [hpt[0] - self.mouth_w / 2, hpt[0] + self.mouth_w / 2],
                [hpt[1] - self.mouth_dy] * 2,
            )

            # beat indicator
            if np.any(np.isclose(t, self.beat_times, atol=1 / self.fps)):
                f_beat.set_data([1.3], [2.8])
            else:
                f_beat.set_data([], [])

            # Play audio on first frame
            if frame == 0 and not self.played:
                try:
                    sd.play(
                        self.metronome.astype(np.float32), samplerate=self.sample_rate
                    )
                except Exception:
                    pass
                self.played = True

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
            ]

        anim2 = animation.FuncAnimation(
            fig2,
            animate_final,
            init_func=init_final,
            frames=total_frames,
            interval=1000 / fps,
            blit=True,
        )
        return fig2, anim2
