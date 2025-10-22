import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

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

        ax_button = plt.axes([0.82, 0.045, 0.1, 0.04])
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

        self.anim = FuncAnimation(fig, animate, frames=len(self.times_years),
                          interval=20, repeat=True)
        plt.legend()
        fig.show()


        update(0)


    # ============================================================
    # =========== FIGURE 2: Ratio + Cosine Animation =============
    # ============================================================
    def create_ratio_cosine_figure(self):
        fig2, (ax_orbit, ax_combined) = plt.subplots(2, 1, figsize=(7, 9))
        plt.subplots_adjust(bottom=0.25, hspace=0.35)

        # Orbit panel
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

        # Combined plot
        ax_combined.set_xlim(self.times_ratio[0], self.times_ratio[-1])
        ax_combined.set_title("Accel Ratio (Green) and Cosine (Magenta)")
        ax_combined.set_xlabel("Time (years)")
        ax_combined.set_ylabel("Cosine", color='m')
        ax_combined.set_ylim(-1.1, 1.1)
        corr_line2, = ax_combined.plot([], [], 'm-')

        ax_ratio2 = ax_combined.twinx()
        ax_ratio2.set_ylabel("Accel Ratio", color='g')
        ax_ratio2.set_ylim(np.nanmin(self.ratio_vals_smooth)*0.9,
                           np.nanmax(self.ratio_vals_smooth)*1.1)
        ratio_line2, = ax_ratio2.plot([], [], 'g-')
        peak_dots2, = ax_ratio2.plot([], [], 'ro', markersize=5)
        cos_peak_dots2, = ax_combined.plot([], [], 'ro', markersize=5)
        time_marker2 = ax_combined.axvline(self.times_ratio[0], color='k', ls='--')

        # State
        self.paused2 = True
        self.current_idx2 = 0
        self.ratio_peaks_seen = set()
        self.cos_values_at_ratio_peaks = []

        # Button + slider
        ax_slider2 = plt.axes([0.15, 0.12, 0.65, 0.03])
        slider2 = Slider(ax_slider2, 'Time idx', 0, len(self.times_ratio) - 1, valinit=0, valstep=1)
        ax_button2 = plt.axes([0.82, 0.11, 0.1, 0.04])
        button2 = Button(ax_button2, 'Play/Pause')

        def toggle2(event):
            self.paused2 = not self.paused2

        button2.on_clicked(toggle2)

        def slider2_update(val):
            self.current_idx2 = int(slider2.val)
            update_fig2(self.current_idx2)

        slider2.on_changed(slider2_update)

        # --- Update Function ---
        def update_fig2(idx):
            for j, pos in enumerate(self.positions_list):
                markers[j].set_data([pos[idx, 0]], [pos[idx, 1]])


            corr_line2.set_data(self.times_ratio[:idx+1], self.corr_vals_smooth[:idx+1])
            ratio_line2.set_data(self.times_ratio[:idx+1], self.ratio_vals_smooth[:idx+1])
            time_marker2.set_xdata([self.times_ratio[idx], self.times_ratio[idx]])

            # Detect new ratio peaks
            new_ratio_peaks = [p for p in self.ratio_peaks_all if p <= idx and p not in self.ratio_peaks_seen]
            for p in new_ratio_peaks:
                nearest_idx = self.cos_peaks[np.argmin(np.abs(self.cos_peaks - p))] if len(self.cos_peaks) > 0 else None
                delta = (nearest_idx - p) if nearest_idx is not None else None
                status = f"{abs(delta)} timesteps to nearest cosine peak" if delta is not None else "no nearby cosine peak"
                print(f"Ratio peak idx={p}, nearest cosine peak idx={nearest_idx}, {status}")

                cos_val = self.corr_vals_smooth[p]
                self.cos_values_at_ratio_peaks.append(cos_val)
                self.ratio_peaks_seen.add(p)

                cos_array = np.array(self.cos_values_at_ratio_peaks)
                mean_cos = np.mean(cos_array)
                std_cos = np.std(cos_array)
                angles = np.degrees(np.arccos(np.clip(cos_array, -1, 1)))
                print(f"Mean cosine={mean_cos:.4f}, Std={std_cos:.4f}, Mean angle={np.mean(angles):.2f}°, Std angle={np.std(angles):.2f}°")

                self.paused2 = True  # Pause at each ratio peak

            peak_dots2.set_data(self.times_ratio[self.ratio_peaks_all], self.ratio_vals_smooth[self.ratio_peaks_all])
            if len(self.cos_peaks) > 0:
                cos_peak_dots2.set_data(self.times_ratio[self.cos_peaks], self.corr_vals_smooth[self.cos_peaks])
            else:
                cos_peak_dots2.set_data([], [])

        def animate2(frame):
            if not self.paused2:
                step = 10
                self.current_idx2 = (self.current_idx2 + step) % len(self.times_ratio)
                slider2.set_val(self.current_idx2)
                update_fig2(self.current_idx2)

        self.anim = FuncAnimation(fig2, animate2, frames=len(self.times_years),
                          interval=20, repeat=True)
        plt.legend()
        fig2.show()


        update_fig2(0)


    def show_plots(self):
        plt.show()