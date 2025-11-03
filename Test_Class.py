import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from matplotlib.patches import Wedge

class OrbitalPlots:
    def __init__(self, positions_list, ratio_vals, corr_vals, times_years,
                 xlim=1, ylim=1,
                 mov_avg_len=19, prominence_val=0.05):
        self.positions_list = positions_list
        self.ratio_vals = ratio_vals
        self.corr_vals = corr_vals
        self.times_years = times_years
        self.xlim = xlim
        self.ylim = ylim
        self.mov_avg_len = mov_avg_len
        self.prominence_val = prominence_val

        # Derived arrays
        self.ratio_vals_smooth = np.convolve(ratio_vals, np.ones(mov_avg_len)/mov_avg_len, mode='valid')
        self.corr_vals_smooth = np.convolve(corr_vals, np.ones(mov_avg_len)/mov_avg_len, mode='valid')
        self.times_ratio = times_years[:len(self.ratio_vals_smooth)]

        # Peak detection
        self.ratio_peaks_all, _ = find_peaks(self.ratio_vals_smooth, prominence=prominence_val)
        cos_peaks_all, _ = find_peaks(self.corr_vals_smooth, prominence=prominence_val)
        self.cos_peaks = np.array([cp for cp in cos_peaks_all if self.corr_vals_smooth[cp] >= 0.9], dtype=int)

        self.pause_events = []
        self._flashlight_patch = None
        self._flashlight_active = False

        print(f"Initialized OrbitalPlots with {len(positions_list)} orbits.")
        print(f"Found {len(self.ratio_peaks_all)} ratio peaks, {len(self.cos_peaks)} cosine peaks ≥ 0.9.")

    # ============================================================
    # =========== UNIFIED SPLIT-VIEW ANIMATION ===================
    # ============================================================
    def create_ratio_cosine_figure(self):
        fig, (ax_orbit, ax_combined) = plt.subplots(
            1, 2, figsize=(12, 7),
            gridspec_kw={'width_ratios': [1.3, 1.0]}
        )

        # Reserve slightly more space for the UI below the plots
        plt.subplots_adjust(bottom=0.33, wspace=0.3)

        # Orbit subplot
        ax_orbit.set_xlim(-self.xlim, self.xlim)
        ax_orbit.set_ylim(-self.ylim, self.ylim)
        ax_orbit.set_aspect('equal')
        ax_orbit.set_title("Orbital Motion")
        ax_orbit.grid(True)

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.positions_list)))
        markers = []
        for i, pos in enumerate(self.positions_list):
            ax_orbit.plot(pos[:, 0], pos[:, 1], '-', alpha=0.3, color=colors[i], label=f"Body {i}")
            (marker_line,) = ax_orbit.plot([], [], 'o', color=colors[i], markersize=6)
            markers.append(marker_line)
        ax_orbit.legend()

        # Ratio/Cosine subplot
        ax_combined.set_xlim(self.times_ratio[0], self.times_ratio[-1])
        ax_combined.set_title("Accel Ratio (Green) and Cosine (Magenta)")
        ax_combined.set_xlabel("Time (years)")
        ax_combined.set_ylabel("Cosine", color='m')
        ax_combined.set_ylim(-1.1, 1.1)
        corr_line, = ax_combined.plot([], [], 'm-')

        ax_ratio = ax_combined.twinx()
        ax_ratio.set_ylabel("Accel Ratio", color='g')
        ax_ratio.set_ylim(np.nanmin(self.ratio_vals_smooth)*0.9, np.nanmax(self.ratio_vals_smooth)*1.1)
        ratio_line, = ax_ratio.plot([], [], 'g-')
        peak_dots, = ax_ratio.plot([], [], 'ro', markersize=5)
        cos_dots, = ax_combined.plot([], [], 'ro', markersize=5)
        time_marker = ax_combined.axvline(self.times_ratio[0], color='k', ls='--')

        # ======== Controls (aligned below both plots) ========
        # Place sliders lower, directly under each subplot
        ax_slider = plt.axes([0.15, 0.17, 0.30, 0.035])
        self.slider = Slider(ax_slider, 'Time idx', 0, len(self.times_ratio)-1, valinit=0, valstep=1)

        ax_speed = plt.axes([0.55, 0.17, 0.30, 0.035])
        self.speed_slider = Slider(ax_speed, 'Speed ×', 0.1, 10.0, valinit=1.0, valstep=0.1)

        # Flashlight button under orbit plot (left)
        ax_flash = plt.axes([0.15, 0.08, 0.10, 0.06])
        self.button_flash = Button(ax_flash, 'Flashlight')

        # Peak/Play buttons under ratio plot (right)
        ax_prev = plt.axes([0.55, 0.08, 0.10, 0.06])
        self.button_prev = Button(ax_prev, '← Peak')

        ax_next = plt.axes([0.66, 0.08, 0.10, 0.06])
        self.button_next = Button(ax_next, '→ Peak')

        ax_play = plt.axes([0.77, 0.08, 0.10, 0.06])
        self.button_play = Button(ax_play, 'Play/Pause')

        ax_auto = plt.axes([0.88, 0.08, 0.10, 0.06])
        self.button_auto = Button(ax_auto, 'Auto')
        
        # ======== States ========
        self.paused = True
        self.current_idx = 0
        self.speed_factor = 1.0
        self._last_idx = -1
        self.auto_cycle = False
        self.ratio_peaks_seen = set()

        # ======== Callbacks ========
        self.button_play.on_clicked(lambda e: self.toggle_pause())
        self.button_next.on_clicked(lambda e: self.jump_to_peak(next=True))
        self.button_prev.on_clicked(lambda e: self.jump_to_peak(next=False))
        self.button_flash.on_clicked(lambda _: self.toggle_flashlight(ax_orbit, colors))
        self.button_auto.on_clicked(lambda e: self.toggle_auto_cycle())

        self.slider.on_changed(lambda val: update(int(self.slider.val)))
        self.speed_slider.on_changed(lambda val: setattr(self, 'speed_factor', self.speed_slider.val))
        fig.canvas.mpl_connect('key_press_event', self._on_keypress)

        # ======== Update Function ========
        def update(i):
            for j, pos in enumerate(self.positions_list):
                markers[j].set_data([pos[i, 0]], [pos[i, 1]])
            corr_line.set_data(self.times_ratio[:i+1], self.corr_vals_smooth[:i+1])
            ratio_line.set_data(self.times_ratio[:i+1], self.ratio_vals_smooth[:i+1])
            time_marker.set_xdata([self.times_ratio[i], self.times_ratio[i]])

            if getattr(self, '_flashlight_active', False) and self._flashlight_patch is not None:
                t_now_y = float(self.times_ratio[i])
                dty = t_now_y - getattr(self, '_t_star_y', t_now_y)
                theta_center = getattr(self, '_theta_star', 0.0) + getattr(self, '_omega_rel', 0.0)*dty
                dtheta = abs(getattr(self, '_omega_max', 0.0))*abs(dty)
                theta1, theta2 = sorted((np.degrees(theta_center - dtheta), np.degrees(theta_center + dtheta)))
                ex, ey = self.positions_list[1][i]
                self._flashlight_patch.set_center((ex, ey))
                self._flashlight_patch.set_theta1(theta1)
                self._flashlight_patch.set_theta2(theta2)

            peaks = np.asarray(self.ratio_peaks_all, dtype=int)
            if self._last_idx == -1:
                crossed = (peaks <= i)
            else:
                crossed = (peaks > self._last_idx) & (peaks <= i)
            for p in peaks[crossed]:
                if p not in self.ratio_peaks_seen:
                    self.ratio_peaks_seen.add(p)
                    print(f"---- Peak {len(self.ratio_peaks_seen)} at idx {p} ----")
                    self.pause_events.append((int(p), int(i), int(i - p)))
                    self._flashlight_setup_from_last_event()
                    if not self.auto_cycle:
                        self.paused = True

            self._last_idx = i
            peak_dots.set_data(self.times_ratio[self.ratio_peaks_all], self.ratio_vals_smooth[self.ratio_peaks_all])
            cos_dots.set_data(self.times_ratio[self.cos_peaks], self.corr_vals_smooth[self.cos_peaks])

        # ======== Animation ========
        def animate(frame):
            if not self.paused:
                step = int(5 * self.speed_factor)
                self.current_idx = (self.current_idx + step) % len(self.times_ratio)
                self.slider.set_val(self.current_idx)
                update(self.current_idx)

        self.anim = FuncAnimation(fig, animate, frames=len(self.times_ratio),
                                  interval=20, repeat=True)
        update(0)
        plt.show()
    
    # ======== Utility Methods ========
    def toggle_pause(self): self.paused = not self.paused
    def toggle_auto_cycle(self): self.auto_cycle = not self.auto_cycle

    def jump_to_peak(self, next=True):
        if len(self.ratio_peaks_all) == 0: return
        peaks = self.ratio_peaks_all
        current = self.current_idx
        if next:
            idx = peaks[np.searchsorted(peaks, current, side='right') % len(peaks)]
        else:
            idx = peaks[np.searchsorted(peaks, current, side='left') - 1]
        self.current_idx = int(idx)
        self.slider.set_val(self.current_idx)
        print(f"Jumped to {'next' if next else 'previous'} peak at idx {idx}")

    def _on_keypress(self, event):
        if event.key == ' ': self.toggle_pause()
        elif event.key == 'right': self.jump_to_peak(True)
        elif event.key == 'left': self.jump_to_peak(False)
        elif event.key.lower() == 'f':
            print("[Hotkey] Flashlight toggle")
            self.toggle_flashlight(event.canvas.figure.axes[0], plt.cm.tab10(np.linspace(0,1,len(self.positions_list))))

    # ============================================================
    # ================ Flashlight Logic ==========================
    # ============================================================
    def _flashlight_setup_from_last_event(self, idx_window=6, safety=1.1,
                                          sun_idx=0, earth_idx=1, mars_idx=2):
        if not self.pause_events: return False
        p, slider_idx, delta_steps = self.pause_events[-1]
        dt_y = float(self.times_ratio[1]-self.times_ratio[0])
        t_pause_y = float(self.times_ratio[slider_idx % len(self.times_ratio)])
        t_star_y = t_pause_y - delta_steps*dt_y
        d = (self.positions_list[mars_idx]-self.positions_list[earth_idx])[:len(self.times_ratio)]
        theta = np.unwrap(np.arctan2(d[:,1], d[:,0]))
        i_star = int(np.clip(np.searchsorted(self.times_ratio, t_star_y)-1, 1, len(self.times_ratio)-2))
        lo, hi = max(1, i_star-idx_window), min(len(theta)-2, i_star+idx_window)
        omega_rel = float((theta[i_star+1]-theta[i_star-1])/(2*dt_y))
        omega_max = float(np.max(np.abs(np.diff(theta[lo:hi+1])))/dt_y)*safety
        R = float(1.05*np.nanmax(np.linalg.norm(self.positions_list[mars_idx]-self.positions_list[sun_idx],axis=1)))
        self._t_star_y=t_star_y; self._theta_star=float(theta[i_star])
        self._omega_rel=omega_rel; self._omega_max=omega_max; self._R_wedge=R
        return True

    def toggle_flashlight(self, ax_orbit, colors, earth_idx=1):
        # Turn OFF
        if getattr(self, '_flashlight_active', False):
            try: 
                if self._flashlight_patch: self._flashlight_patch.remove()
            except Exception:
                pass
            self._flashlight_patch = None
            self._flashlight_active = False
            ax_orbit.figure.canvas.draw_idle()
            return

        # Turn ON
        if not self._flashlight_setup_from_last_event():
            # Fallback: build provisional setup from current slider index
            try:
                idx = int(getattr(self, 'current_idx', 0))
                self._provisional_flashlight_setup(idx)
            except Exception:
                print("[flashlight] No event yet — play to a peak or move the slider, then try again.")
                return

        # Create wedge and apply geometry immediately at current slider position
        ex, ey = self.positions_list[earth_idx][int(getattr(self, 'current_idx', 0))]
        self._flashlight_patch = Wedge(center=(ex, ey), r=float(getattr(self, '_R_wedge', self.xlim)),
                                    theta1=0, theta2=0, ec=colors[earth_idx],
                                    lw=1.5, ls='--', alpha=0.15, fill=True)
        ax_orbit.add_patch(self._flashlight_patch)
        self._flashlight_active = True

        # apply geometry now so it's visible and correctly oriented even if paused
        try:
            i = int(getattr(self, 'current_idx', 0))
            self._apply_flashlight_geometry(i)
        except Exception:
            pass

        ax_orbit.figure.canvas.draw_idle()

    def _apply_flashlight_geometry(self, i, earth_idx=1):
        """
        Update wedge center and angles for the flashlight at time-index i.
        Ensures angles are wrapped to [0, 360) to match Matplotlib's Wedge convention.
        """
        if not getattr(self, '_flashlight_active', False) or self._flashlight_patch is None:
            return

        # center at Earth's current position
        ex, ey = self.positions_list[earth_idx][i]
        self._flashlight_patch.set_center((ex, ey))

        # compute angular center and half-width
        t_now_y = float(self.times_ratio[i])
        dty = t_now_y - float(getattr(self, '_t_star_y', t_now_y))
        theta_center = float(getattr(self, '_theta_star', 0.0) + getattr(self, '_omega_rel', 0.0)*dty)  # radians
        dtheta = abs(float(getattr(self, '_omega_max', 0.0))) * abs(dty)  # radians

        # wrap to [0, 360)
        theta_deg = (np.degrees(theta_center) % 360.0)
        dtheta_deg = np.degrees(dtheta)

        theta1 = (theta_deg - dtheta_deg) % 360.0
        theta2 = (theta_deg + dtheta_deg) % 360.0

        # set geometry
        self._flashlight_patch.set_theta1(theta1)
        self._flashlight_patch.set_theta2(theta2)
        self._flashlight_patch.set_zorder(0.5)


    def _provisional_flashlight_setup(self, idx, idx_window=6, safety=1.1, sun_idx=0, earth_idx=1, mars_idx=2):
        """
        Fallback setup if no peak-pause events exist yet:
        uses current index 'idx' as the reference time.
        """
        if len(self.times_ratio) < 2:
            raise RuntimeError("not enough samples")

        # geocentric angle Earth->Mars
        d = (self.positions_list[mars_idx] - self.positions_list[earth_idx])[:len(self.times_ratio)]
        theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))  # radians

        i_star = int(np.clip(idx, 1, len(theta) - 2))
        dt_y = float(self.times_ratio[1] - self.times_ratio[0])
        lo, hi = max(1, i_star - idx_window), min(len(theta) - 2, i_star + idx_window)

        omega_rel = float((theta[i_star + 1] - theta[i_star - 1]) / (2 * dt_y))  # rad / year
        omega_max = float(np.max(np.abs(np.diff(theta[lo:hi + 1]))) / dt_y) * safety

        # radius: use max Mars–Sun distance seen
        R = float(1.05 * np.nanmax(np.linalg.norm(
            self.positions_list[mars_idx] - self.positions_list[sun_idx], axis=1)))

        # store state for live updates
        self._t_star_y = float(self.times_ratio[i_star])
        self._theta_star = float(theta[i_star])
        self._omega_rel = omega_rel
        self._omega_max = omega_max
        self._R_wedge = R

    # ============================================================
    # ================ Kepler Helper Methods ====================
    # ============================================================
    def peak_stats_and_kepler(self, which='ratio', unit='days', mu=None, sun_idx=0, earth_idx=1):
        peaks=self.ratio_peaks_all if which=='ratio' else getattr(self,'cos_peaks',np.array([]))
        if len(peaks)<2: return {'count':0}
        tY=self.times_ratio.astype(float); sepY=np.diff(tY[peaks])
        conv=365.25 if unit=='days' else 1.0; sep=sepY*conv
        stats={'count':len(sep),'median':float(np.nanmedian(sep)),
               'mean':float(np.nanmean(sep)),'std':float(np.nanstd(sep)),'unit':unit}
        if mu is not None:
            r=self.positions_list[earth_idx]-self.positions_list[sun_idx]
            a_mean=np.nanmean(np.linalg.norm(r,axis=1))
            T_s=2*np.pi*np.sqrt(a_mean**3/mu); T_days=T_s/86400.0; T_years=T_days/365.25
        else: T_days=np.nan; T_years=np.nan
        stats.update({'earth_T_kepler_days':T_days,'earth_T_kepler_years':T_years})
        S_years=np.nanmedian(sepY); stats['synodic_median_years']=S_years
        if np.isfinite(T_years) and np.isfinite(S_years):
            denom=(1/T_years-1/S_years) if S_years>=T_years else (1/T_years+1/S_years)
            P_years=1/denom if denom!=0 else np.nan; P_days=P_years*365.25
        else: P_years=np.nan; P_days=np.nan
        stats.update({'mars_T_from_synodic_years':P_years,'mars_T_from_synodic_days':P_days})
        return stats

    def mars_distance_from_period(self, mu=None, period_days=None):
        if mu is None or period_days is None or not np.isfinite(period_days) or period_days<=0: return np.nan
        T=period_days*86400.0; r=((T**2*mu)/(4*np.pi**2))**(1/3); return float(r)
