import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from matplotlib.patches import Circle
from matplotlib.patches import Wedge

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
        self._flashlight_patch=None; self._flashlight_active=False
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

            # --- Flashlight (general slider time): recompute for t_now - t* ---
            if getattr(self,'_flashlight_active',False) and self._flashlight_patch is not None:
                t_now_y = float(self.times_ratio[idx]); dty = t_now_y - getattr(self,'_t_star_y', t_now_y)
                theta_center = getattr(self,'_theta_star',0.0) + getattr(self,'_omega_rel',0.0)*dty
                dtheta = abs(getattr(self,'_omega_max',0.0))*abs(dty)
                theta1, theta2 = sorted((np.degrees(theta_center - dtheta), np.degrees(theta_center + dtheta)))
                ex,ey = self.positions_list[1][idx]   # flashlight originates at Earth (current index)
                self._flashlight_patch.set_center((ex,ey))
                self._flashlight_patch.set_radius(getattr(self,'_R_wedge', self.xlim))
                self._flashlight_patch.set_theta1(theta1); self._flashlight_patch.set_theta2(theta2)
                self._flashlight_patch.set_zorder(0.5)

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
    # ================ Synodic Period Calculation ================
    # ============================================================
    def peak_stats_and_kepler(self, which='ratio', unit='years', mu=None, sun_idx=0, earth_idx=1):
        import numpy as np
        if which not in ('ratio','cosine'): raise ValueError("which must be 'ratio' or 'cosine'")
        peaks = self.ratio_peaks_all if which=='ratio' else getattr(self,'cos_peaks',np.array([]))
        if peaks is None or len(peaks)<2:
            return {'count':0,'separations':np.array([]),'median':np.nan,'mean':np.nan,'std':np.nan,'unit':unit,'which':which,
                    'earth_T_kepler_days':np.nan,'earth_T_kepler_years':np.nan,'synodic_median_years':np.nan,
                    'mars_T_from_synodic_years':np.nan,'mars_T_from_synodic_days':np.nan}
        tY = self.times_ratio.astype(float)                          # years
        sepY = np.diff(tY[peaks])                                    # years between consecutive peaks
        conv = 1.0 if unit=='years' else 365.25 if unit=='days' else 365.25*24.0 if unit=='hours' else (_ for _ in ()).throw(ValueError("unit must be 'years','days','hours'"))
        sep = sepY*conv
        stats = {'count':len(sep),'separations':sep,'median':float(np.nanmedian(sep)),'mean':float(np.nanmean(sep)),'std':float(np.nanstd(sep)),'unit':unit,'which':which}
        # --- Kepler Earth period from ⟨r⟩ (requires mu = G*M_sun) ---
        if mu is not None:
            r = self.positions_list[earth_idx]-self.positions_list[sun_idx]   # meters
            a_mean = float(np.nanmean(np.linalg.norm(r,axis=1)))              # meters
            T_earth_s = 2*np.pi*np.sqrt(a_mean**3/mu)                         # seconds
            T_earth_days = T_earth_s/86400.0; T_earth_years = T_earth_days/365.25
        else:
            T_earth_days = np.nan; T_earth_years = np.nan
        stats.update({'earth_T_kepler_days':float(T_earth_days),'earth_T_kepler_years':float(T_earth_years)})
        # --- Mars sidereal period from synodic median (use ratio-peak spacing as S) ---
        S_years = float(np.nanmedian(sepY))                                   # synodic in years
        stats['synodic_median_years'] = S_years
        if np.isfinite(T_earth_years) and np.isfinite(S_years) and S_years>0:
            # decide superior vs inferior planet branch
            # superior (e.g., Mars): S >= P_E ⇒ P_M = 1/(1/P_E - 1/S)
            # inferior (e.g., Venus): S <  P_E ⇒ P = 1/(1/P_E + 1/S)
            if S_years >= T_earth_years:
                denom = (1.0/T_earth_years - 1.0/S_years)
            else:
                denom = (1.0/T_earth_years + 1.0/S_years)
            P_mars_years = 1.0/denom if denom!=0 else np.nan
            P_mars_days = P_mars_years*365.25
        else:
            P_mars_years = np.nan; P_mars_days = np.nan
        stats.update({'mars_T_from_synodic_years':float(P_mars_years),'mars_T_from_synodic_days':float(P_mars_days)})
        # cache (optional)
        if which=='ratio': self._ratio_peak_stats=stats
        else: self._cos_peak_stats=stats
        return stats
    def mars_distance_from_period(self, mu=None, period_days=None):
        if mu is None or period_days is None or not np.isfinite(period_days) or period_days<=0:
            return np.nan
        T = period_days * 86400.0
        r = ((T**2 * mu) / (4*np.pi**2))**(1/3)
        return float(r)
    def _flashlight_setup_from_last_event(self, idx_window=6, safety=1.1, sun_idx=0, earth_idx=1, mars_idx=2):
        if not hasattr(self,'pause_events') or len(self.pause_events)==0: return False
        p, slider_idx, delta_steps = self.pause_events[-1]                    # last recorded event
        # lag in *steps* was measured when event happened; convert to years
        if len(self.times_ratio)<2: return False
        dt_step_y = float(self.times_ratio[1]-self.times_ratio[0])
        t_pause_y = float(self.times_ratio[slider_idx%len(self.times_ratio)])
        t_star_y  = t_pause_y - np.median([e[2] for e in self.pause_events])*dt_step_y  # robust t*
        # geocentric angle series θ(t) for Earth→Mars
        d = (self.positions_list[mars_idx]-self.positions_list[earth_idx])[:len(self.times_ratio)]
        theta = np.unwrap(np.arctan2(d[:,1], d[:,0]))
        # index near t*
        i_star = int(np.clip(np.searchsorted(self.times_ratio, t_star_y)-1, 1, len(self.times_ratio)-2))
        k = int(max(2, idx_window))
        lo, hi = max(1, i_star-k), min(len(theta)-2, i_star+k)
        # instantaneous relative angular rate (central diff) and a conservative bound in rad/year
        omega_rel = float((theta[i_star+1]-theta[i_star-1])/(2*dt_step_y))
        omega_max = float(np.max(np.abs(np.diff(theta[lo:hi+1])))/dt_step_y)*safety
        # flashlight radius: prefer Kepler radius if set, else fall back to observed max Mars distance
        if hasattr(self,'mars_ref_radius') and np.isfinite(self.mars_ref_radius):
            R = float(self.mars_ref_radius*1.05)
        else:
            R = float(1.05*np.nanmax(np.linalg.norm(self.positions_list[mars_idx]-self.positions_list[sun_idx],axis=1)))
        # stash state
        self._t_star_y = t_star_y; self._theta_star = float(theta[i_star]); self._omega_rel = omega_rel; self._omega_max = omega_max; self._R_wedge = R
        return True
    
    def toggle_flashlight(self, ax_orbit, colors, earth_idx=1):
        import numpy as np
        if getattr(self,'_flashlight_active',False):
            try:
                if self._flashlight_patch: self._flashlight_patch.remove()
            except Exception: pass
            self._flashlight_patch=None; self._flashlight_active=False; return
        
        if not self._flashlight_setup_from_last_event():
            # Fallback: try to build a provisional setup from the current slider index
            try:
                idx = int(getattr(self,'current_idx2', 0))
                self._provisional_flashlight_setup(idx)
            except Exception:
                print("[flashlight] No pause_events yet; press Play until the first auto-pause, then try again.")
                return
            
        # create the wedge once; angles/radius/center will be updated each frame
        ex,ey = self.positions_list[earth_idx][0]
        self._flashlight_patch = Wedge(center=(ex,ey), r=self._R_wedge, theta1=0, theta2=0, width=None, ec=colors[earth_idx], lw=1.5, ls='--', alpha=0.85, fill=True)
        self._flashlight_patch.set_alpha(0.17); ax_orbit.add_patch(self._flashlight_patch)
        self._flashlight_active=True
        # initialize wedge geometry immediately (works even if paused)
        try:
            idx = int(getattr(self,'current_idx2', 0))
            t_now_y = float(self.times_ratio[idx]); dty = t_now_y - getattr(self,'_t_star_y', t_now_y)
            theta_center = getattr(self,'_theta_star',0.0) + getattr(self,'_omega_rel',0.0)*dty
            dtheta = abs(getattr(self,'_omega_max',0.0))*abs(dty)
            theta1, theta2 = sorted((np.degrees(theta_center - dtheta), np.degrees(theta_center + dtheta)))
            ex,ey = self.positions_list[1][idx]
            self._flashlight_patch.set_center((ex,ey))
            self._flashlight_patch.set_radius(getattr(self,'_R_wedge', self.xlim))
            self._flashlight_patch.set_theta1(theta1); self._flashlight_patch.set_theta2(theta2)
            self._flashlight_patch.set_zorder(0.5)
            ax_orbit.figure.canvas.draw_idle()
        except Exception:
            pass

    def _provisional_flashlight_setup(self, idx, idx_window=6, safety=1.1, sun_idx=0, earth_idx=1, mars_idx=2):
        # Use the most recent peak at or before idx (if any); otherwise use idx itself.
        peaks = np.asarray(self.ratio_peaks_all, dtype=int)
        if len(self.times_ratio)<2: raise RuntimeError("not enough samples")
        t_now_y = float(self.times_ratio[int(np.clip(idx,0,len(self.times_ratio)-1))])
        prev_peaks = peaks[peaks<=idx] if len(peaks)>0 else np.array([],int)
        p = int(prev_peaks[-1]) if len(prev_peaks)>0 else int(idx)
        # emulate a zero-lag event at current slider location
        self.pause_events = getattr(self,'pause_events', [])
        self.pause_events.append((p, idx, 0))
        if not self._flashlight_setup_from_last_event(): raise RuntimeError("provisional setup failed")
