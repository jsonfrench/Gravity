import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from matplotlib.patches import Circle

class OrbitalPlots:
    def __init__(self, positions_list, ratio_vals, corr_vals, times_years,
                 xlim=1, ylim=1,
                 mov_avg_len=19, prominence_val=0.05):
        """
        positions_list : list of np.ndarray
            Each array shape (n_steps, 2), for each body.
        ratio_vals : np.ndarray
            Array of ratio values (same length as times_years or shorter).
        corr_vals : np.ndarray
            Array of cosine correlation values (same length as times_years or shorter).
        times_years : np.ndarray
            Time array (in years).
        xlim, ylim : tuple
            Plot limits for both figures.
        mov_avg_len : int
            Length of moving average smoothing.
        prominence_val : float
            Peak detection prominence.
        """

        self.positions_list = positions_list
        self.ratio_vals = ratio_vals
        self.corr_vals = corr_vals
        self.times_years = times_years
        self.xlim = xlim
        self.ylim = ylim
        self.mov_avg_len = mov_avg_len
        self.prominence_val = prominence_val

        # global animation parameters
        self.idx = 0
        self.paused = True

        # --- Derived arrays ---
        self.ratio_vals_smooth = np.convolve(ratio_vals, np.ones(mov_avg_len)/mov_avg_len, mode='valid')
        self.corr_vals_smooth = np.convolve(corr_vals, np.ones(mov_avg_len)/mov_avg_len, mode='valid')
        self.times_ratio = times_years[:len(self.ratio_vals_smooth)]

        # --- Peak detection ---
        self.ratio_peaks_all, _ = find_peaks(self.ratio_vals_smooth, prominence=prominence_val)
        cos_peaks_all, _ = find_peaks(self.corr_vals_smooth, prominence=prominence_val)
        self.cos_peaks = np.array([cp for cp in cos_peaks_all if self.corr_vals_smooth[cp] >= 0.9], dtype=int)

        print(f"Initialized OrbitalPlots with {len(positions_list)} orbits.")
        print(f"Found {len(self.ratio_peaks_all)} ratio peaks, {len(self.cos_peaks)} cosine peaks ≥ 0.9.")

    # ============================================================
    # =============== FIGURE 1: Orbital Motion ===================
    # ============================================================
    def create_orbit_figure(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-self.xlim, self.xlim)
        ax.set_ylim(-self.ylim, self.ylim)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Orbital Motion")

        # Distinct colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.positions_list)))

        # Plot paths and markers
        markers = []
        for i, pos in enumerate(self.positions_list):
            ax.plot(pos[:, 0], pos[:, 1], '-', alpha=0.3, color=colors[i], label=f"Body {i}")
            (marker_line,) = ax.plot([], [], 'o', color=colors[i], markersize=6)
            markers.append(marker_line)

        # Slider & button
        ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Index', 0, len(self.positions_list[0]) - 1, valinit=0, valstep=1)

        ax_button = plt.axes([0.90, 0.2, 0.12, 0.05])
        self.button = Button(ax_button, 'Play/Pause')

        self.paused = True
        self.current_idx = 0

        def toggle(event):
            self.paused = not self.paused

        self.button.on_clicked(toggle)

        def slider_update(val):
            self.current_idx = int(self.slider.val)
            update(self.current_idx)

        self.slider.on_changed(slider_update)

        def update(i):
            for j, pos in enumerate(self.positions_list):
                markers[j].set_data([pos[i, 0]], [pos[i, 1]])


        def animate(frame):
            if not self.paused:
                self.current_idx = (self.current_idx + 1) % len(self.positions_list[0])
                self.slider.set_val(self.current_idx)
                update(self.current_idx)

        self.anim1 = FuncAnimation(fig, animate, frames=len(self.times_years),
                           interval=20, repeat=True)
        ax.legend()
        # fig.show()  # ← remove this
        update(0)
    def show_plots(self):
        plt.show()
    # ============================================================
    # =========== FIGURE 2: Ratio + Cosine Animation =============
    # ============================================================
    def create_ratio_cosine_figure(self):
        fig2, (ax_orbit, ax_combined) = plt.subplots(2, 1, figsize=(7, 9))
        plt.subplots_adjust(bottom=0.35, hspace=0.35)  # ← make room for extra slider

        # Orbit panel
        ax_orbit.set_xlim(-self.xlim, self.xlim)
        ax_orbit.set_ylim(-self.ylim, self.ylim)
        ax_orbit.set_aspect('equal')
        ax_orbit.set_title("Orbital Motion")
        ax_orbit.grid(True)
        self._mars_circle=None
        if hasattr(self,'mars_ref_radius') and np.isfinite(self.mars_ref_radius):
            sx,sy=self.positions_list[0][0]  # Sun at first frame
            self._mars_circle=Circle((sx,sy), float(self.mars_ref_radius), fill=False, ls='--', lw=1.5, alpha=0.8, ec='orange')
            ax_orbit.add_patch(self._mars_circle)


        colors = plt.cm.tab10(np.linspace(0, 1, len(self.positions_list)))
        self._flashlight_lines=[]; self._flashlight_active=False
        markers = []
        for i, pos in enumerate(self.positions_list):
            ax_orbit.plot(pos[:, 0], pos[:, 1], '-', alpha=0.3, color=colors[i], label=f"Body {i}")
            (marker_line,) = ax_orbit.plot([], [], 'o', color=colors[i], markersize=6)
            markers.append(marker_line)

        #Flash Light Toggle Button
        ax_flash = fig2.add_axes([0.88, 0.24, 0.10, 0.05])
        self.button_flash = Button(ax_flash, 'Flashlight')
        self.button_flash.on_clicked(lambda _ : self.toggle_flashlight(ax_orbit, colors))

        # Combined plot
        ax_combined.set_xlim(self.times_ratio[0], self.times_ratio[-1])
        ax_combined.set_title("Accel Ratio (Green) and Cosine (Magenta)")
        ax_combined.set_xlabel("Time (years)")
        ax_combined.set_ylabel("Cosine", color='m')
        ax_combined.set_ylim(-1.1, 1.1)
        corr_line2, = ax_combined.plot([], [], 'm-')

        ax_ratio2 = ax_combined.twinx()
        ax_ratio2.set_ylabel("Accel Ratio", color='g')
        ax_ratio2.set_ylim(
            np.nanmin(self.ratio_vals_smooth) * 0.9,
            np.nanmax(self.ratio_vals_smooth) * 1.1
        )
        ratio_line2, = ax_ratio2.plot([], [], 'g-')
        peak_dots2, = ax_ratio2.plot([], [], 'ro', markersize=5)
        cos_peak_dots2, = ax_combined.plot([], [], 'ro', markersize=5)
        time_marker2 = ax_combined.axvline(self.times_ratio[0], color='k', ls='--')

        # State
        self.paused2 = True
        self.current_idx2 = 0
        self._last_idx2 = -1
        self.ratio_peaks_seen = set()
        self.cos_values_at_ratio_peaks = []
        self.speed_factor = 1.0  # new variable

        self.pause_events = []  # list of (peak_idx, slider_idx, delta_steps)

        # === Controls ===
        ax_slider2 = plt.axes([0.15, 0.17, 0.65, 0.03])
        self.slider2 = Slider(ax_slider2, 'Time idx', 0, len(self.times_ratio) - 1, valinit=0, valstep=1)

        plt.subplots_adjust(bottom=0.35, right=0.86, hspace=0.35)  # add right margin
        ax_button2 = fig2.add_axes([0.88, 0.17, 0.10, 0.05])       # attach to fig2 and push right
        self.button2 = Button(ax_button2, 'Play/Pause')

        # --- NEW speed slider ---
        ax_speed = plt.axes([0.15, 0.10, 0.65, 0.03])
        self.speed_slider = Slider(ax_speed, 'Speed ×', 0.1, 10.0, valinit=1.0, valstep=0.1)

        def toggle2(event):
            self.paused2 = not self.paused2

        self.button2.on_clicked(toggle2)

        def slider2_update(val):
            self.current_idx2 = int(self.slider2.val)
            update_fig2(self.current_idx2)

        self.slider2.on_changed(slider2_update)

        def speed_update(val):
            self.speed_factor = self.speed_slider.val

        self.speed_slider.on_changed(speed_update)

        # --- Update Function ---
        def update_fig2(idx):
            for j, pos in enumerate(self.positions_list):
                markers[j].set_data([pos[idx, 0]], [pos[idx, 1]])

            if self._mars_circle is not None:
                sx,sy=self.positions_list[0][idx]
                self._mars_circle.center=(sx,sy)

            corr_line2.set_data(self.times_ratio[:idx+1], self.corr_vals_smooth[:idx+1])
            ratio_line2.set_data(self.times_ratio[:idx+1], self.ratio_vals_smooth[:idx+1])
            time_marker2.set_xdata([self.times_ratio[idx], self.times_ratio[idx]])

            # --- Flashlight update based on stored Mars confidence positions ---
            if getattr(self, '_flashlight_active', False):
                self._update_flashlight(idx, ax_orbit)

            # Detect crossings of ratio peaks between last index and current index (handles wrap-around)
            peaks = np.asarray(self.ratio_peaks_all, dtype=int)
            if self._last_idx2 == -1:
                crossed_mask = (peaks <= idx)
            else:
                if idx >= self._last_idx2:
                    crossed_mask = (peaks > self._last_idx2) & (peaks <= idx)
                else:
                    crossed_mask = (peaks > self._last_idx2) | (peaks <= idx)
            new_ratio_peaks = [int(p) for p in peaks[crossed_mask] if p not in self.ratio_peaks_seen]

            for p in new_ratio_peaks:
                nearest_idx = (self.cos_peaks[np.argmin(np.abs(self.cos_peaks - p))]
                            if len(self.cos_peaks) > 0 else None)
                delta = (nearest_idx - p) if nearest_idx is not None else None
                print(f"------------Peak Number {len(self.ratio_peaks_seen)+1}------------")
                status = (f"{abs(delta)} timesteps to nearest cosine peak"
                        if delta is not None else "no nearby cosine peak")
                print(f"Ratio peak idx={p}, nearest cosine peak idx={nearest_idx}, {status}")

                cos_val = self.corr_vals_smooth[p]
                self.cos_values_at_ratio_peaks.append(cos_val)
                self.ratio_peaks_seen.add(p)

                # --- NEW: record & print slider pause index and step delta ---
                slider_pause_idx = int(idx)
                delta_steps = int(slider_pause_idx - p)  # simple difference (no wrap)
                self.pause_events.append((int(p), slider_pause_idx, delta_steps))
                # Prep flashlight state right when we pause, so the button has data
                try: self._flashlight_setup_from_last_event()
                except Exception as _e: pass

                print(f"PAUSE: slider_idx={slider_pause_idx}, ratio_peak_idx={p}, Δsteps={delta_steps}")

                cos_array = np.array(self.cos_values_at_ratio_peaks)
                mean_cos = np.mean(cos_array)
                std_cos = np.std(cos_array)
                angles = np.degrees(np.arccos(np.clip(cos_array, -1, 1)))

                print(f"Mean cosine={mean_cos:.4f}, Std={std_cos:.4f}, "
                    f"Mean angle={np.mean(angles):.2f}°, Std angle={np.std(angles):.2f}°")
                self.paused2 = True

                # Recompute Mars confidence positions based on updated pause_events.
                try:
                    ep = self.estimate_earth_period(idx)
                    self.track_mars_after_synodic_period(ep)
                except Exception:
                    pass

            self._last_idx2 = idx
            peak_dots2.set_data(self.times_ratio[self.ratio_peaks_all], self.ratio_vals_smooth[self.ratio_peaks_all])
            if len(self.cos_peaks) > 0:
                cos_peak_dots2.set_data(self.times_ratio[self.cos_peaks], self.corr_vals_smooth[self.cos_peaks])
            else:
                cos_peak_dots2.set_data([], [])
        
        # --- Animation ---
        def animate2(frame):
            if not self.paused2:
                step = int(10 * self.speed_factor)  # speed factor affects simulation step size
                self.current_idx2 = (self.current_idx2 + step) % len(self.times_ratio)
                self.slider2.set_val(self.current_idx2)
                update_fig2(self.current_idx2)

        self.anim2 = FuncAnimation(fig2, animate2, frames=len(self.times_ratio),
                                interval=20, repeat=True)
        ax_orbit.legend()
        update_fig2(0)

    # ============================================================
    # =================== Earth Period Estimate ==================
    # ============================================================
    def estimate_earth_period(self, idx):
        """
        Approximate Earth's orbital period (in years) using positions up to the current slider index.
        idx : int
            Time index (inclusive) along times_years/positions_list to use for the estimate.
        """
        if idx < 1:
            return None

        idx = min(idx, len(self.times_years) - 1)

        # Vector from Sun to Earth over time
        rel = self.positions_list[1][:idx+1] - self.positions_list[0][:idx+1]
        angles = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))

        net_angle = angles[-1] - angles[0]
        rotations = net_angle / (2 * np.pi)
        if rotations <= 0:
            return None

        return (self.times_years[idx] - self.times_years[0]) / rotations

    # ============================================================
    # =========== Synodic Period Estimates from stops ============
    # ============================================================
    def synodic_period_estimates(self):
        """
        Use only recorded slider stop events to compute gaps between stops.
        Stores and returns index differences, real-time differences (years),
        and their mean/std. Returns None if fewer than 2 stops exist.
        """
        events = getattr(self, "pause_events", [])
        if len(events) < 2:
            self.synodic_index_diffs = []
            self.synodic_time_diffs = []
            self.synodic_mean = None
            self.synodic_std = None
            return None

        slider_indices = np.array([int(e[1]) for e in events], dtype=int)
        idx_diffs = np.diff(slider_indices)

        times = self.times_ratio[slider_indices]
        time_diffs = np.diff(times)

        self.synodic_index_diffs = idx_diffs
        self.synodic_time_diffs = time_diffs
        self.synodic_mean = float(np.mean(time_diffs))
        self.synodic_std = float(np.std(time_diffs))

        return {
            "count": len(slider_indices),
            "index_differences": idx_diffs,
            "time_differences": time_diffs,
            "mean": self.synodic_mean,
            "std": self.synodic_std,
        }

    # ============================================================
    # =========== Mars Period Estimate (from synodic) ============
    # ============================================================
    def mars_period_estimate(self, earth_period_years):
        """
        Estimate Mars' orbital period using the mean synodic period and Earth period.
        Computes a Mars period for each synodic interval, stores samples,
        and returns dict with mean/std/samples; None if synodic stats are unavailable.
        """
        stats = self.synodic_period_estimates()
        if not stats or stats["mean"] is None or earth_period_years is None:
            return None

        E = float(earth_period_years)  # Earth period (years)
        samples = []
        for S in np.atleast_1d(stats["time_differences"]):
            S = float(S)
            denom = (1.0 / E) - (1.0 / S)
            if denom == 0:
                continue
            samples.append(1.0 / denom)

        if len(samples) == 0:
            return None

        samples_arr = np.asarray(samples, dtype=float)
        self.mars_period_samples = samples_arr
        mean_val = float(np.mean(samples_arr))
        std_val = float(np.std(samples_arr))

        return {"mean": mean_val, "std": std_val, "samples": samples_arr}

    # ============================================================
    # ======= Mars Orbital Radius Estimate (Kepler's 3rd) ========
    # ============================================================
    def mars_radius_estimates(self, earth_period_years, sun_mass=1.9885e30, G=6.67430e-11):
        """
        Use Mars period samples to estimate orbital radius via Kepler's 3rd law.
        Assumes Earth-period-based Mars period samples are in years.
        Returns dict with mean/std/samples (meters), or None if unavailable.
        """
        period_info = self.mars_period_estimate(earth_period_years)
        if not period_info or "samples" not in period_info:
            return None

        samples_years = np.asarray(period_info["samples"], dtype=float)
        if samples_years.size == 0:
            return None

        sec_per_year = 365.25 * 24 * 3600.0
        P_sec = samples_years * sec_per_year

        # Kepler 3rd: P^2 = 4π^2 a^3 / (G M) => a = [G M (P/2π)^2]^(1/3)
        factor = G * sun_mass
        radii = (factor * (P_sec / (2 * np.pi))**2) ** (1.0 / 3.0)

        self.mars_radius_samples = radii
        mean_r = float(np.mean(radii))
        std_r = float(np.std(radii))
        
        # 99% confidence interval
        cmin = mean_r - 2.6 * std_r
        cmax = mean_r + 2.6 * std_r

        return {
            "mean": mean_r,
            "std": std_r,
            "samples": radii,
            "conf_min": cmin,
            "conf_max": cmax,
        }

    # ============================================================
    # =========== Mars Angular Velocity Estimates (omega) =========
    # ============================================================
    def mars_omega_estimates(self, earth_period_years):
        """
        Compute angular velocity (rad/s) for each Mars period sample.
        Stores samples and returns a dict with mean/std/samples, or None if unavailable.
        """
        period_info = self.mars_period_estimate(earth_period_years)
        if not period_info or "samples" not in period_info:
            return None

        samples_years = np.asarray(period_info["samples"], dtype=float)
        if samples_years.size == 0:
            return None

        sec_per_year = 365.25 * 24 * 3600.0
        omegas = 2 * np.pi / (samples_years * sec_per_year)

        self.mars_omega_samples = omegas
        mean_omega = float(np.mean(omegas))
        std_omega = float(np.std(omegas))

        # 99% confidence interval
        conf_min = mean_omega - 2.6 * std_omega
        conf_max = mean_omega + 2.6 * std_omega

        return {
            "mean": mean_omega,
            "std": std_omega,
            "samples": omegas,
            "conf_min": conf_min,
            "conf_max": conf_max,
        }

    # ============================================================
    # =========== Tracking Mars After Synodic Period =============
    # ============================================================
    def track_mars_after_synodic_period(self, earth_period_years):
        """
        For each pause_event segment, assume Mars is collinear with Sun–Earth at the pause
        (opposite Earth). Using 95% confidence bounds on omega and radius, generate the four
        possible Mars positions (rmin/ rmax × omegamin/ omegamax) for each timestep until the
        next pause. Stores segments in self.mars_conf_position_segments.
        """
        # restart the list every time we recompute tracking
        self.mars_conf_position_segments = []

        rad_info = self.mars_radius_estimates(earth_period_years)
        omega_info = self.mars_omega_estimates(earth_period_years)
        if not rad_info or not omega_info:
            return None

        # if confidence bounds collapsed, bail
        if any(v is None for v in (rad_info.get("conf_min"), rad_info.get("conf_max"),
                                   omega_info.get("conf_min"), omega_info.get("conf_max"))):
            return None

        r_lo = float(rad_info.get("conf_min", 0.0))
        r_hi = float(rad_info.get("conf_max", 0.0))
        # enforce non-negative, ordered radii; avoid degenerate zero span
        r_min = max(0.0, min(r_lo, r_hi))
        r_max = max(r_min, max(r_lo, r_hi))
        if r_max <= 0:
            return None
        if r_min <= 0:
            r_min = 0.1 * r_max  # give a small span if lower bound collapsed

        o_lo = float(omega_info.get("conf_min", 0.0))
        o_hi = float(omega_info.get("conf_max", 0.0))
        # enforce positive angular speeds and ordering
        o_min = min(abs(o_lo), abs(o_hi))
        o_max = max(abs(o_lo), abs(o_hi))
        if o_max <= 0:
            return None
        if o_min <= 0:
            o_min = 0.1 * o_max

        events = sorted(getattr(self, "pause_events", []), key=lambda e: e[1])
        if len(events) == 0:
            # fallback: synthesize from ratio peaks so we can visualize something
            if len(self.ratio_peaks_all) >= 2:
                events = [(int(p), int(p), 0) for p in self.ratio_peaks_all]
            else:
                return None

        sec_per_year = 365.25 * 24 * 3600.0
        segments = []
        for i, ev in enumerate(events):
            start_idx = int(ev[1])
            if start_idx >= len(self.times_ratio):
                continue
            end_idx = int(events[i + 1][1]) if i + 1 < len(events) else len(self.times_ratio) - 1
            end_idx = min(end_idx, len(self.times_ratio) - 1)
            if end_idx <= start_idx:
                continue

            sun_pos = self.positions_list[0][start_idx]
            earth_pos = self.positions_list[1][start_idx]
            dir_vec = earth_pos - sun_pos
            norm = np.linalg.norm(dir_vec)
            if norm == 0:
                continue
            u_dir = dir_vec / norm  # assume Mars is collinear on the same side as Earth during pause event
            theta0 = np.arctan2(u_dir[1], u_dir[0])

            times_seg = self.times_ratio[start_idx : end_idx + 1]
            dt_years = times_seg - times_seg[0]
            dt_sec = dt_years * sec_per_year

            omegas = [o_min, o_min, o_max, o_max]
            radii = [r_min, r_max, r_min, r_max]

            positions = np.zeros((len(times_seg), 4, 2))
            for j in range(4):
                theta = theta0 + omegas[j] * dt_sec
                positions[:, j, 0] = sun_pos[0] + radii[j] * np.cos(theta)
                positions[:, j, 1] = sun_pos[1] + radii[j] * np.sin(theta)

            segments.append(
                {"start_idx": start_idx, "end_idx": end_idx, "positions": positions}
            )

        self.mars_conf_position_segments = segments
        return segments

    # ============================================================
    # =========== Flashlight / Telescope Utilities ===============
    # ============================================================
    def _ensure_flashlight_lines(self, ax_orbit, count=4):
        """
        Create and store the flashlight line artists if they don't exist.
        """
        lines = getattr(self, "_flashlight_lines", [])
        missing = max(0, count - len(lines))
        for _ in range(missing):
            (line,) = ax_orbit.plot([], [], color='yellow', lw=1.5, alpha=0.8)
            line.set_visible(False)
            lines.append(line)
        self._flashlight_lines = lines
        return lines

    def _hide_flashlight_lines(self):
        for line in getattr(self, "_flashlight_lines", []):
            line.set_visible(False)

    def toggle_flashlight(self, ax_orbit, colors=None):
        """
        Toggle the Mars confidence flashlight on/off.
        """
        self._flashlight_active = not getattr(self, "_flashlight_active", False)
        if not self._flashlight_active:
            self._hide_flashlight_lines()
            return

        self._ensure_flashlight_lines(ax_orbit, count=4)

        # Update immediately with current slider index if available; hide if no data
        ok = False
        try:
            # ensure segments exist
            ep_all = self.estimate_earth_period(len(self.times_years) - 1)
            self.track_mars_after_synodic_period(ep_all)
            idx = int(getattr(self, "current_idx2", 0))
            ok = self._update_flashlight(idx, ax_orbit)
        except Exception:
            ok = False
        if not ok:
            self._hide_flashlight_lines()

    def _update_flashlight(self, idx, ax_orbit):
        """
        Plot yellow lines from Earth to each possible Mars position for this timestep.
        Returns True if updated, False otherwise.
        """
        if not getattr(self, "_flashlight_active", False):
            return False

        lines = self._ensure_flashlight_lines(ax_orbit, count=4)

        segments = getattr(self, "mars_conf_position_segments", [])
        if not segments:
            # try to recompute if we have an Earth period estimate
            try:
                ep = self.estimate_earth_period(len(self.times_years) - 1)
                self.track_mars_after_synodic_period(ep)
                segments = getattr(self, "mars_conf_position_segments", [])
            except Exception:
                segments = []
        if not segments:
            # fallback: aim opposite Earth with max radius if available
            rad_info = self.mars_radius_estimates(self.estimate_earth_period(idx))
            if not rad_info or rad_info.get("conf_max") is None:
                self._hide_flashlight_lines()
                return False
            radii = [val for val in (rad_info.get("conf_min"), rad_info.get("conf_max")) if val and val > 0]
            earth_pos = self.positions_list[1][idx]
            sun_pos = self.positions_list[0][idx]
            dir_vec = earth_pos - sun_pos
            norm = np.linalg.norm(dir_vec)
            if norm == 0 or len(radii) == 0:
                self._hide_flashlight_lines()
                return False
            u_dir = dir_vec / norm
            pos_candidates = np.array([sun_pos + r * u_dir for r in radii])
        else:
            # Find segment containing idx; if none, pick nearest and clamp
            seg = None
            for s in segments:
                if s["start_idx"] <= idx <= s["end_idx"]:
                    seg = s
                    break
            if seg is None:
                seg = min(segments, key=lambda s: min(abs(idx - s["start_idx"]), abs(idx - s["end_idx"])))

            t_idx = idx - seg["start_idx"]
            # clamp to valid range
            t_idx = max(0, min(t_idx, len(seg["positions"]) - 1))

            pos_candidates = seg["positions"][t_idx]  # shape (4,2)
        earth_pos = self.positions_list[1][idx]

        pos_candidates = np.atleast_2d(pos_candidates)
        # Grow line list if we have more candidates than lines
        if len(lines) < len(pos_candidates):
            extra = len(pos_candidates) - len(lines)
            self._ensure_flashlight_lines(ax_orbit, count=len(pos_candidates))
            lines = self._flashlight_lines

        ok = False
        for i, line in enumerate(lines):
            if i < len(pos_candidates):
                target = pos_candidates[i]
                line.set_data([earth_pos[0], target[0]], [earth_pos[1], target[1]])
                line.set_visible(True)
                line.set_zorder(0.5)
                ok = True
            else:
                line.set_visible(False)
        if not ok:
            self._hide_flashlight_lines()
        return ok

    # ============================================================
    # =========== Plot stored Mars confidence positions ==========
    # ============================================================
    def plot_mars_conf_positions(self):
        """
        Create a static plot showing Earth, Mars, and all stored Mars confidence positions
        from track_mars_after_synodic_period(). Returns (fig, ax) or None if no data.
        """
        segs = getattr(self, "mars_conf_position_segments", None)
        if not segs:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_title("Mars Confidence Positions")
            ax.text(0.5, 0.5, "No Mars confidence positions available.", ha='center', va='center')
            ax.axis('off')
            return fig, ax

        all_pts = []
        for seg in segs:
            pts = seg.get("positions")
            if pts is not None:
                all_pts.append(pts.reshape(-1, 2))
        if not all_pts:
            print("No Mars confidence positions available. Run track_mars_after_synodic_period first.")
            return None

        all_pts = np.vstack(all_pts)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Mars Confidence Positions")
        ax.set_xlim(-self.xlim, self.xlim)
        ax.set_ylim(-self.ylim, self.ylim)

        # Plot Earth and Mars actual trajectories for reference
        if len(self.positions_list) > 1:
            ax.plot(self.positions_list[1][:, 0], self.positions_list[1][:, 1], 'b-', alpha=0.2, label="Earth path")
        if len(self.positions_list) > 2:
            ax.plot(self.positions_list[2][:, 0], self.positions_list[2][:, 1], 'r-', alpha=0.2, label="Mars path")

        ax.scatter(all_pts[:, 0], all_pts[:, 1], c='gold', s=5, alpha=0.6, label="Mars conf points")
        ax.legend()
        return fig, ax
