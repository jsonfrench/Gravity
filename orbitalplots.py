# OrbitalPlots class file

# computations
import numpy as np

# display/animation
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

# Signal analysis packages
from scipy.signal import find_peaks, peak_widths, stft
from numpy.fft import rfft, rfftfreq
import pywt

# change display backend for animations
import matplotlib
matplotlib.use('Qt5Agg')

class OrbitalPlots:
    def __init__(self, positions_list, ratio_vals, corr_vals, times_years,
                 xlim=1, ylim=1,
                 mov_avg_len=19, prominence_val=0.05):
        self.positions_list = positions_list
        # clean the ratio array to remove any instances of nan?
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

        # Derived arrays
        self.ratio_vals_smooth = np.convolve(ratio_vals, np.ones(mov_avg_len)/mov_avg_len, mode='valid')
        self.corr_vals_smooth = np.convolve(corr_vals, np.ones(mov_avg_len)/mov_avg_len, mode='valid')
        self.times_ratio = times_years[:len(self.ratio_vals_smooth)]

        # Peak detection (full arrays)
        self.ratio_peaks_all, _ = find_peaks(self.ratio_vals_smooth, prominence=prominence_val)
        cos_peaks_all, _ = find_peaks(self.corr_vals_smooth, prominence=prominence_val)
        self.cos_peaks = np.array([cp for cp in cos_peaks_all if self.corr_vals_smooth[cp] >= 0.9], dtype=int)

        # Initialize state for ratio figure
        self.paused2 = True
        self.current_idx2 = 0
        self.ratio_peaks_seen = set()
        self.cos_values_at_ratio_peaks = []
        self.timesteps_per_frame = 1  # controlled by speed slider

        print(f"Initialized OrbitalPlots with {len(positions_list)} orbits.")
        print(f"Found {len(self.ratio_peaks_all)} ratio peaks, {len(self.cos_peaks)} cosine peaks ≥ 0.9.")

    # ========================= ORBIT FIGURE =========================
    def create_orbit_figure(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-self.xlim, self.xlim)
        ax.set_ylim(-self.ylim, self.ylim)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Orbital Motion")

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.positions_list)))
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

        def toggle(event):
            self.paused = not self.paused
        self.button.on_clicked(toggle)

        def slider_update(val):
            self.idx = int(self.slider.val)
            update(self.idx)
        self.slider.on_changed(slider_update)

        def update(i):
            for j, pos in enumerate(self.positions_list):
                markers[j].set_data([pos[i, 0]], [pos[i, 1]])

        def animate(frame):
            if not self.paused:
                self.idx = (self.idx + 1) % len(self.positions_list[0])
                self.slider.set_val(self.idx)
                update(self.idx)

        self.anim = FuncAnimation(fig, animate, frames=len(self.times_years), interval=20, repeat=True)
        plt.legend()
        update(0)
        # fig.show()

    # ========================= ORBIT FIGURE with earth overlay =========================
    # needs an initial index to slice the overlay_positions array with len(short_time_array)
    def create_orbit_figure_earth_overlay(self, initial_index, short_time_array, overlay_positions):
        # Figure setup
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-self.xlim, self.xlim)
        ax.set_ylim(-self.ylim, self.ylim)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Orbital Motion with Short-Time Overlay")

        L = len(short_time_array)
        temp_positions_list = self.positions_list
        # slice overlay_positions from initial_index to initial_index + L
        overlay_positions_short_time = overlay_positions[initial_index:initial_index+L]
        temp_positions_list.append(overlay_positions_short_time)
        

        # --- Update frame length to min nonempty length ---
        L = min(len(p) for p in temp_positions_list)

        # --- Plot all paths ---
        colors = plt.cm.tab10(np.linspace(0, 1, len(temp_positions_list)))
        markers = []
        for i, pos in enumerate(temp_positions_list):
            ax.plot(pos[:, 0], pos[:, 1], '-', alpha=0.3, color=colors[i], label=f"Body {i}")
            (marker_line,) = ax.plot([], [], 'o', color=colors[i], markersize=6)
            markers.append(marker_line)

        # --- Slider + button setup ---
        ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
        slider = Slider(ax_slider, 'Index', 0, L - 1, valinit=0, valstep=1)

        ax_button = plt.axes([0.82, 0.045, 0.1, 0.04])
        button = Button(ax_button, 'Play/Pause')

        paused = False

        def toggle(event):
            nonlocal paused
            paused = not paused

        def update(idx):
            for j, pos in enumerate(temp_positions_list):
                if idx < pos.shape[0]:
                    markers[j].set_data([pos[idx, 0]], [pos[idx, 1]])

        def slider_update(val):
            update(int(slider.val))

        slider.on_changed(slider_update)
        button.on_clicked(toggle)

        # --- Animation loop ---
        idx = 0

        def animate(frame):
            nonlocal idx
            if not paused and L > 0:
                idx = (idx + 1) % L
                slider.set_val(idx)

        ani = FuncAnimation(fig, animate, frames=L, interval=30, repeat=True)
        self.ani = ani  # keep reference alive

        plt.show()
        return ani

    # ==================== RATIO + COSINE FIGURE ====================
    def create_ratio_cosine_figure(self):
        fig2, (ax_orbit, ax_combined) = plt.subplots(2, 1, figsize=(7, 9))
        plt.subplots_adjust(bottom=0.35, hspace=0.35)

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
        self.timesteps_per_frame = 1  # default speed

        # --- Buttons and sliders ---
        ax_button2 = plt.axes([0.82, 0.25, 0.1, 0.04])
        button2 = Button(ax_button2, 'Play/Pause')
        button2.on_clicked(lambda event: setattr(self, 'paused2', not self.paused2))

        ax_slider2 = plt.axes([0.15, 0.25, 0.65, 0.03])
        slider2 = Slider(ax_slider2, 'Time idx', 0, len(self.times_ratio)-1, valinit=0, valstep=1)
        def slider2_update(val):
            self.current_idx2 = int(val)
            update_fig2(self.current_idx2)

        slider2.on_changed(slider2_update)

        ax_speed = plt.axes([0.15, 0.18, 0.65, 0.03])
        speed_slider = Slider(ax_speed, 'Speed', 1, 5, valinit=1, valstep=1)
        def speed_update(val):
            # Scale speed relative to total length
            total_steps = len(self.times_ratio)
            self.timesteps_per_frame = int(val * max(1, total_steps//500))
        speed_slider.on_changed(speed_update)

        # --- Update function ---
        def update_fig2(idx):
            for j, pos in enumerate(self.positions_list):
                markers[j].set_data([pos[idx, 0]], [pos[idx, 1]])

            corr_line2.set_data(self.times_ratio[:idx+1], self.corr_vals_smooth[:idx+1])
            ratio_line2.set_data(self.times_ratio[:idx+1], self.ratio_vals_smooth[:idx+1])
            time_marker2.set_xdata([self.times_ratio[idx], self.times_ratio[idx]])

            # Real-time ratio peak detection
            ratio_partial = self.ratio_vals_smooth[:idx+1]
            ratio_peaks_partial, _ = find_peaks(ratio_partial, prominence=self.prominence_val)
            new_peaks = [p for p in ratio_peaks_partial if p not in self.ratio_peaks_seen]
            for p in new_peaks:
                nearest_idx = self.cos_peaks[np.argmin(np.abs(self.cos_peaks - p))] if len(self.cos_peaks)>0 else None
                delta = (nearest_idx - p) if nearest_idx is not None else None
                status = f"{abs(delta)} timesteps to nearest cosine peak" if delta is not None else "no nearby cosine peak"
                print(f"Ratio peak time={self.times_years[p]}, nearest cosine peak idx={nearest_idx}, {status}")

                cos_val = self.corr_vals_smooth[p]
                self.cos_values_at_ratio_peaks.append(cos_val)
                self.ratio_peaks_seen.add(p)

                cos_array = np.array(self.cos_values_at_ratio_peaks)
                mean_cos = np.mean(cos_array)
                std_cos = np.std(cos_array)
                angles = np.degrees(np.arccos(np.clip(cos_array, -1, 1)))
                print(f"Mean cosine={mean_cos:.4f}, Std={std_cos:.4f}, Mean angle={np.mean(angles):.2f}°, Std angle={np.std(angles):.2f}°")

                # Auto-pause at peak
                self.paused2 = True

            # Update dots
            if len(self.ratio_peaks_seen) > 0:
                peak_dots2.set_data(self.times_ratio[list(self.ratio_peaks_seen)],
                                    self.ratio_vals_smooth[list(self.ratio_peaks_seen)])
            if len(self.cos_peaks) > 0:
                cos_peak_dots2.set_data(self.times_ratio[self.cos_peaks], self.corr_vals_smooth[self.cos_peaks])
            else:
                cos_peak_dots2.set_data([], [])

        # --- Animation ---
        def animate2(frame):
            if not self.paused2:
                self.current_idx2 += self.timesteps_per_frame
                if self.current_idx2 >= len(self.times_ratio):
                    self.current_idx2 = len(self.times_ratio) - 1
                # Temporarily disable slider callbacks to prevent interference with pause
                slider2.eventson = False
                slider2.set_val(self.current_idx2)
                slider2.eventson = True
                update_fig2(self.current_idx2)


        self.anim = FuncAnimation(fig2, animate2, frames=len(self.times_years),
                                interval=20, repeat=True)
        plt.legend()
        # fig2.show()
        update_fig2(0)

    def plot_ratio_cosine_with_synodic(self):
        """
        Plot the entire ratio and cosine arrays with peaks marked, show FFT-predicted
        synodic period, and draw vertical dashed lines at synodic intervals with a slider
        to shift their phase.
        """

        fig, ax = plt.subplots(figsize=(10,6))
        plt.subplots_adjust(bottom=0.2)

        # --- Smoothed arrays ---
        # ratio_smooth = np.convolve(self.ratio_vals, np.ones(self.mov_avg_len)/self.mov_avg_len, mode='valid')
        ratio_smooth = self.ratio_vals
        # corr_smooth = np.convolve(self.corr_vals, np.ones(self.mov_avg_len)/self.mov_avg_len, mode='valid')
        corr_smooth = self.corr_vals
        times = self.times_years[:len(ratio_smooth)]

        # height requirement for ratio peaks
        minimum_peak_height = 0.9 * (np.nanmax(ratio_smooth) - np.nanmin(ratio_smooth)) + np.nanmin(ratio_smooth)

        # --- Peaks ---
        ratio_peaks, _ = find_peaks(ratio_smooth, height=minimum_peak_height)
        cos_peaks, _ = find_peaks(corr_smooth, prominence=self.prominence_val)



        # --- Plot ratio and cosine ---
        ax.plot(times, ratio_smooth, 'g-', label='Ratio')
        ax.plot(times, corr_smooth, 'm-', label='Cosine')
        ax.plot(times[ratio_peaks], ratio_smooth[ratio_peaks], 'ro', label='Ratio Peaks')
        ax.plot(times[cos_peaks], corr_smooth[cos_peaks], 'bo', label='Cos Peaks')

        # --- FFT to predict synodic period ---
        ratio_centered = ratio_smooth - np.mean(ratio_smooth)
        N = len(ratio_centered)
        dt = times[1] - times[0]
        freqs = rfftfreq(N, dt)
        fft_mag = np.abs(rfft(ratio_centered))

        # Only consider frequencies <= 1/year
        mask = freqs <= 1
        fft_mag_masked = fft_mag[mask]


        fft_peaks_indices, _ = find_peaks(x=fft_mag_masked, height=0.5 * np.max(fft_mag_masked))
        peak_freq = freqs[mask][fft_peaks_indices[0]]

        synodic_period = 1 / peak_freq


        # if np.any(mask):
        #     freqs_masked = freqs[mask]
        #     fft_mag_masked = fft_mag[mask]
        #     idx_peak = np.argmax(fft_mag_masked[1:]) + 1  # skip DC
        #     synodic_period = 1 / freqs_masked[idx_peak]
        # else:
        #     synodic_period = np.nan
        #


        ax.set_title(f"Ratio & Cosine with Peaks\nPredicted Synodic Period ≈ {synodic_period:.2f} yr")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        # --- Dashed lines at synodic intervals ---
        num_lines = int(np.ceil((times[-1] - times[0]) / synodic_period))
        line_positions = np.array([i*synodic_period for i in range(num_lines)])
        lines = [ax.axvline(x=pos, color='k', ls='--') for pos in line_positions]

        # --- Slider to shift phase of dashed lines ---
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(ax_slider, 'Phase', 0, synodic_period, valinit=0)

        def update_phase(val):
            phase = slider.val
            for i, line in enumerate(lines):
                new_pos = (i*synodic_period + phase) % (times[-1] + synodic_period)
                line.set_xdata([new_pos, new_pos])
            fig.canvas.draw_idle()

        slider.on_changed(update_phase)

        # plt.show()

    def plot_fft(self):
        fft = rfft(self.ratio_vals_smooth - np.mean(self.ratio_vals_smooth))
        freq = rfftfreq(len(self.ratio_vals_smooth), d=(self.times_years[1]-self.times_years[0]))
        plt.plot(self.times_years[:len(self.ratio_vals_smooth)], self.ratio_vals_smooth - np.nanmean(self.ratio_vals_smooth))
        plt.title('ratio array vs time')
        # plt.show()
        plt.plot(freq, np.abs(fft))
        plt.title('magnitude spectrum of ratio vals')
        # plt.show()


    def plot_ratio_cosine_with_synodic_fft(self):
        """
        Plot the entire ratio and cosine arrays with peaks marked, show FFT-predicted
        synodic period, and add a subplot of the FFT magnitude between 0 and 1/year.
        Includes vertical dashed lines at predicted synodic intervals with a phase slider.
        """

        fig, (ax_main, ax_fft) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios':[2,1]})
        plt.subplots_adjust(bottom=0.2, hspace=0.35)

        # --- Smoothed arrays ---
        ratio_smooth = np.convolve(self.ratio_vals, np.ones(self.mov_avg_len)/self.mov_avg_len, mode='valid')
        corr_smooth = np.convolve(self.corr_vals, np.ones(self.mov_avg_len)/self.mov_avg_len, mode='valid')
        times = self.times_years[:len(ratio_smooth)]

        # --- Peaks ---
        ratio_peaks, _ = find_peaks(ratio_smooth, prominence=self.prominence_val)
        cos_peaks, _ = find_peaks(corr_smooth, prominence=self.prominence_val)

        mean_synodic_actual = np.mean(np.diff(self.times_years[ratio_peaks]))
        stdev_synodic_actual = np.std(np.diff(self.times_years[ratio_peaks]))

        # --- Plot ratio and cosine ---
        ax_main.plot(times, ratio_smooth, 'g-', label='Ratio')
        ax_main.plot(times, corr_smooth, 'm-', label='Cosine')
        ax_main.plot(times[ratio_peaks], ratio_smooth[ratio_peaks], 'ro', label='Ratio Peaks')
        ax_main.plot(times[cos_peaks], corr_smooth[cos_peaks], 'bo', label='Cos Peaks')


        # Only consider frequencies <= 1/year
        # --- FFT to predict synodic period ---
        ratio_centered = ratio_smooth - np.mean(ratio_smooth)
        N = len(ratio_centered)
        dt = times[1] - times[0]  # timestep in years

        fft_vals = rfft(ratio_centered)
        fft_mag = np.abs(fft_vals)
        freqs = rfftfreq(N, dt)  # frequencies in 1/year

        # Consider only positive frequencies <= 2/year
        mask_freq = 1
        mask = (freqs > 0) & (freqs <= mask_freq)
        freqs_masked = freqs[mask]
        fft_mag_masked = fft_mag[mask]

        fft_peaks_indices, _ = find_peaks(fft_mag_masked, height=0.5 * np.max(fft_mag_masked))
        # find the widths of the peaks in the FFT.  scipy.signal peak_widths returns arrays consisting
        # of [0] peak widths
        #    [1] width_heights
        #    [2] interpolated lefthand positions of horizontal lines intersecting peak at given rel_height
        #    [3] ---- " ---- righthand
        fft_peak_widths = peak_widths(x=fft_mag_masked, peaks=fft_peaks_indices, rel_height=1)

        # use interpolated positions of horizontal lines to estimate CoM of peak
        # print(freqs_masked[int(np.round(fft_peak_widths[3][0]))],freqs_masked[int(np.round(fft_peak_widths[3][1]))])

        first_peak_com_index = int(np.rint((fft_peak_widths[2][0]+fft_peak_widths[3][0])/2))
        lower_peak_width_index = int(fft_peak_widths[2][0])
        upper_peak_width_index = int(fft_peak_widths[3][0])
        freq_com_estimated = np.sum(fft_mag_masked[lower_peak_width_index:upper_peak_width_index] * freqs_masked[lower_peak_width_index:upper_peak_width_index]) / np.sum(fft_mag_masked[lower_peak_width_index:upper_peak_width_index])
        com_synodic = 1 / freq_com_estimated

        # com_synodic = 1/freqs_masked[first_peak_com_index]

        # freq_com_estimated = np.sum(freqs_masked[fft_peaks_indices] * fft_mag_masked[fft_peaks_indices]) / np.sum(fft_mag_masked[fft_peaks_indices])
        #
        # freq_com_estimated = np.sum(freqs * fft_mag) / np.sum(fft_mag)

        freq_com_estimated = com_synodic


        if len(freqs_masked) > 0:
            # idx_peak = np.argmax(fft_mag_masked)
            idx_peak = fft_peaks_indices[0]
            synodic_period = 1 / freqs_masked[idx_peak]
        else:
            synodic_period = np.nan

        ax_main.set_title(f'Predicted (Mean) SynP (1peak): {synodic_period:.6f} yr;'
                          f'\n ACTUAL: mSynP: {mean_synodic_actual:.6f}, stdev: {stdev_synodic_actual:.6f}'
                          f'\n 1/[CoM of 1st peak]: {com_synodic:.6f}')

        # Plot FFT magnitude
        ax_fft.clear()
        ax_fft.plot(freqs_masked, fft_mag_masked, 'b-')
        ax_fft.set_xlabel("Frequency (1/year)")
        ax_fft.set_ylabel("Magnitude")
        ax_fft.set_title(f"FFT Magnitude (0-1 / year), SynP from 1 peak: {synodic_period:.2f} yr")
        ax_fft.grid(True)

        # Highlight peak
        if not np.isnan(synodic_period):
            ax_fft.plot(freqs_masked[idx_peak], fft_mag_masked[idx_peak], 'ro', label='Predicted Synodic Frequency')
            ax_fft.legend()


        # --- Dashed lines at synodic (from single peak) intervals ---
        num_lines = int(np.ceil((times[-1] - times[0]) / synodic_period))
        line_positions = np.array([i*synodic_period for i in range(num_lines)])
        lines = [ax_main.axvline(x=pos, color='k', ls='--') for pos in line_positions]

        num_lines_com = int(np.ceil((times[-1] - times[0]) / freq_com_estimated))
        line_positions_com = np.array([i*freq_com_estimated for i in range(num_lines_com)])
        lines_com = [ax_main.axvline(x=pos, color='g', ls='--') for pos in line_positions_com]

        # --- Slider to shift phase of dashed lines ---
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(ax_slider, 'Phase', 0, synodic_period, valinit=0)

        def update_phase(val):
            phase = slider.val
            for i, line in enumerate(lines):
                new_pos = (i*synodic_period + phase) % (times[-1] + synodic_period)
                line.set_xdata([new_pos, new_pos])
            fig.canvas.draw_idle()

        slider.on_changed(update_phase)

        # --- FFT subplot ---
        ax_fft.plot(freqs_masked, fft_mag_masked, 'b-')
        ax_fft.set_xlabel("Frequency (1/year)")
        ax_fft.set_ylabel("FFT Magnitude")
        ax_fft.set_title("Magnitude of RFFT (0-1 / year)")
        ax_fft.grid(True)

        # --- Annotate predicted synodic period on FFT ---
        peak_freq = freqs_masked[idx_peak] if not np.isnan(synodic_period) else 0
        peak_mag = fft_mag_masked[idx_peak] if not np.isnan(synodic_period) else 0
        ax_fft.plot(peak_freq, peak_mag, 'ro', label='Predicted Synodic Frequency')
        ax_fft.legend()

        plt.show()
        # fig.show()

    def plot_ratio_wavelet(self, scale_min=1, scale_max=256):
        """
        Compute and display the Continuous Wavelet Transform (CWT)
        of the smoothed ratio array using a Morlet wavelet.
        Displays power spectrum as a function of time and period (in years).
        """

        # --- Data ---
        ratio_smooth = self.ratio_vals_smooth - np.nanmean(self.ratio_vals_smooth)
        times = self.times_ratio
        dt = times[1] - times[0]

        # --- Define wavelet parameters ---
        wavelet = 'cmor1.5-1.0'   # Complex Morlet, good balance of time/freq localization
        scales = np.arange(int(scale_min), int(scale_max))  # range of scales; increase max for finer freq resolution

        # --- Compute CWT ---
        coeffs, freqs = pywt.cwt(ratio_smooth, scales, wavelet, sampling_period=dt)
        power = np.abs(coeffs)**2
        period = 1 / freqs  # convert from frequency (1/yr) to period (years)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        T, P = np.meshgrid(times, period)

        im = ax.pcolormesh(T, P, power, shading='auto', cmap='viridis')
        ax.set_yscale('log')
        ax.set_ylabel("Period (years)")
        ax.set_xlabel("Time (years)")
        ax.set_title("Continuous Wavelet Transform (CWT) Power Spectrum of Ratio Array")
        fig.colorbar(im, ax=ax, label="Power")

        print(f"Wavelet period range: {period.min():.6f} to {period.max():.6f} years")

        ax.set_ylim(1, period.max())  # show only periods ≥ 1 year

        # --- Add reference lines for major peaks (optional) ---
        if len(self.ratio_peaks_all) > 0:
            for pk in self.times_ratio[self.ratio_peaks_all]:
                ax.axvline(pk, color='w', ls='--', lw=0.5, alpha=0.6)

        # plt.tight_layout()
        # plt.show()


    def plot_ratio_wavelet_wide(self, min_period=1.0, max_period=10.0, n_scales=256, wavelet='cmor1.5-1.0'):
        """
        Compute and display a 'wide' CWT that captures low-frequency (long-period) signals.
        Parameters:
            min_period : float  -> minimum period (years) to display (e.g. 1.0)
            max_period : float  -> maximum period (years) you want to capture (e.g. 10.0)
            n_scales   : int    -> number of scales between scale_min and scale_max (use 200-2000 for higher resolution)
            wavelet    : str    -> pywt wavelet name (complex Morlet like 'cmorB-C' recommended)

        This method chooses log-spaced scales so the transform has good resolution at long periods.
        """

        # --- Data and sampling ---
        ratio_smooth = self.ratio_vals_smooth - np.nanmean(self.ratio_vals_smooth)
        times = self.times_ratio
        dt = times[1] - times[0]  # sampling period in years

        # --- Wavelet and central frequency ---
        central_freq = pywt.central_frequency(wavelet)  # f0
        if central_freq <= 0:
            raise ValueError(f"Unexpected central_frequency {central_freq} for wavelet {wavelet}")

        # --- Compute scale range required to cover desired periods ---
        # Convert desired period range into corresponding scale range using:
        #   period = (scale * dt) / f0  => scale = period * f0 / dt
        scale_min = max(1.0, (min_period * central_freq) / dt)   # ensure >= 1
        scale_max = max(scale_min * 2, (max_period * central_freq) / dt)  # at least twice scale_min

        # Log-spaced scales are ideal for covering decades of period
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)

        # --- Compute CWT ---
        coeffs, freqs = pywt.cwt(ratio_smooth, scales, wavelet, sampling_period=dt)
        power = np.abs(coeffs)**2

        # convert scales -> period using same formula
        # NOTE: using pywt.central_frequency(wavelet) for consistent conversion
        period = (scales * dt) / central_freq

        # sanity print to diagnose 'empty plot' problems
        print(f"CWT period range (years): {period.min():.6e} → {period.max():.6e}")
        print(f"Requested display window: {min_period} → {max_period} years")
        if period.max() < min_period:
            print("WARNING: computed maximum period is smaller than requested min_period. Increase scale_max or n_scales.")

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(11, 6))
        T, P = np.meshgrid(times, period)
        im = ax.pcolormesh(T, P, power, shading='auto', cmap='viridis')

        ax.set_yscale('log')
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Period (years)")
        ax.set_title(f"CWT Power — wavelet={wavelet} — periods {period.min():.2e}–{period.max():.2e} yr")
        fig.colorbar(im, ax=ax, label='Power')

        # Clip the display to the requested window but only inside the available period range
        display_min = max(min_period, period.min())
        display_max = min(max_period, period.max())
        if display_min >= display_max:
            # fallback: show entire available range
            ax.set_ylim(period.min(), period.max())
        else:
            ax.set_ylim(display_min, display_max)

        # plt.tight_layout()
        # plt.show()


    def plot_ratio_stft(self, window_years=2.0, overlap=0.5, max_freq=1.0):
        """
        Compute and display the Short-Time Fourier Transform (STFT) of the smoothed ratio array
        to visualize time-varying frequency content.

        Parameters
        ----------
        window_years : float, optional
            Duration of each STFT window in years (default 2.0).
            Larger windows give better frequency resolution but poorer time resolution.
        overlap : float, optional
            Fractional overlap between consecutive windows (0–1). Default is 0.5.
        max_freq : float, optional
            Maximum frequency (in 1/year) to display in the spectrogram.
        """

        # --- Prepare data ---
        ratio = self.ratio_vals_smooth - np.nanmean(self.ratio_vals_smooth)
        times = self.times_ratio
        dt = times[1] - times[0]  # sampling interval in years

        # --- Convert window size to samples ---
        nperseg = int(window_years / dt)
        if nperseg < 4:
            raise ValueError("Window too small for data spacing — increase window_years.")

        noverlap = int(overlap * nperseg)

        # --- Compute STFT ---
        f, t, Zxx = stft(ratio, fs=1 / dt, nperseg=nperseg, noverlap=noverlap, window='hann')
        power = np.abs(Zxx) ** 2

        # --- Filter out high frequencies ---
        mask = f <= max_freq
        f = f[mask]
        power = power[mask, :]

        # --- Plot spectrogram ---
        fig, ax = plt.subplots(figsize=(10, 6))
        T, F = np.meshgrid(t + times[0], f)  # shift time axis to actual time
        im = ax.pcolormesh(T, F, power, shading='auto', cmap='viridis')

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Frequency (1/year)")
        ax.set_title(
            f"STFT Power Spectrum of Ratio Array\n(Window={window_years:.2f} yr, Overlap={overlap * 100:.0f}%)")
        ax.set_ylim(0, max_freq)
        ax.grid(True)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Power (Amplitude²)")

        # plt.tight_layout()
        # plt.show()


    def compute_local_synodic_radii(self):
        """
        Uses the class's existing peak list (self.peaks)
        to compute local synodic periods and estimate Mars'
        instantaneous orbital radius assuming near-circular orbits.

        Returns
        -------
        results : list of tuples
            Each entry:
            (
                (t_peak_i, t_peak_{i+1}),   # the two peak times
                synodic_period,             # Δt between peaks
                est_mars_radius             # a_M estimate from synodic assumption
            )
        """

        # Ensure peaks exist
        if len(self.ratio_peaks_all) < 2:
            print("Not enough peaks in self.peaks to compute local synodic periods.")
            return []

        times = self.times_years
        peak_times = times[self.ratio_peaks_all]

        results = []

        # Loop through consecutive peaks
        for i in range(len(peak_times) - 1):
            t1 = peak_times[i]
            t2 = peak_times[i + 1]

            syn = t2 - t1
            if syn <= 0:
                continue

            # ----- Estimate Mars radius from synodic period -----
            # 1/P_syn = |1 - 1/P_M|
            # Assume P_M > 1 (Mars slower)
            #   => 1/P_M = 1 - 1/P_syn
            invP_M = 1.0 - 1.0/syn

            if invP_M <= 0:
                # Happens for eccentric orbits near perihelion
                a_M = np.nan
            else:
                P_M = 1.0 / invP_M
                a_M = P_M ** (2.0/3.0)    # AU

            results.append(((t1, t2), syn, a_M))
            print(f"Local synodic period {i}: {syn} yrs; est. orbital rad: {a_M} au")
        

        return results

    def plot_local_synodic_radius_markers(self):
        """
        Plots actual Earth and Mars orbits and adds predicted Mars radius markers
        based on local synodic periods. Each marker is placed along the ray from
        Sun → Earth evaluated at the MIDPOINT of the synodic interval.
        Also fits an ellipse with Sun at origin to the predicted markers and plots it.
        """

        import scipy.optimize

        # ------------------------------------------------------------
        # 1. Compute synodic radius estimates
        # ------------------------------------------------------------
        syn_results = self.compute_local_synodic_radii()
        if len(syn_results) == 0:
            print("No synodic radius data available.")
            return

        # ------------------------------------------------------------
        # 2. Get actual orbits
        # ------------------------------------------------------------
        earth_xy = np.asarray(self.positions_list[1])   # Earth
        mars_xy  = np.asarray(self.positions_list[2])   # Mars
        times    = self.times_years

        # Compute Earth's mean orbital radius (simulation units)
        r_earth_mean = np.mean(np.sqrt(earth_xy[:,0]**2 + earth_xy[:,1]**2))

        # ------------------------------------------------------------
        # 3. Make the figure
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_aspect('equal', 'box')

        # Plot real orbits
        ax.plot(earth_xy[:,0], earth_xy[:,1], label="Earth (actual)", alpha=0.75)
        ax.plot(mars_xy[:,0], mars_xy[:,1], label="Mars (actual)", alpha=0.75)
        ax.plot(0, 0, 'yo', markersize=10, label="Sun")

        # ------------------------------------------------------------
        # 4. Add predicted synodic radii using midpoint placement
        # ------------------------------------------------------------
        pred_points = []

        for k, ((t1, t2), syn, est_aM) in enumerate(syn_results):
            if est_aM is None or np.isnan(est_aM):
                continue

            # Convert AU to simulation units
            pred_radius = est_aM * r_earth_mean

            # Midpoint time
            t_mid = 0.5 * (t1 + t2)

            # Closest Earth index to t_mid
            idx_mid = np.argmin(np.abs(times - t_mid))
            ex, ey = earth_xy[idx_mid]
            rE = np.sqrt(ex**2 + ey**2)
            if rE == 0:
                continue

            # Unit vector Sun→Earth at midpoint
            ux = -ex / rE
            uy = -ey / rE

            # Predicted Mars position along that ray
            px = pred_radius * ux
            py = pred_radius * uy
            pred_points.append([px, py])

            ax.plot(px, py, 'r*', markersize=11)
            ax.text(px, py, f"{k}", color='red', fontsize=9)

        pred_points = np.array(pred_points)
        if len(pred_points) >= 5:  # Need enough points to fit
            # ------------------------------------------------------------
            # 5. Fit an ellipse with Sun at origin (polar form)
            # ------------------------------------------------------------
            def ellipse_r(theta, a, e):
                return a * (1 - e**2) / (1 + e * np.cos(theta))

            theta = np.arctan2(pred_points[:,1], pred_points[:,0])
            r_obs = np.sqrt(pred_points[:,0]**2 + pred_points[:,1]**2)

            # Initial guess: a = mean(r_obs), e = 0.1
            p0 = [np.mean(r_obs), 0.1]
            bounds = ([0, 0], [np.inf, 0.9])  # positive semi-major, eccentricity < 1
            popt, _ = scipy.optimize.curve_fit(ellipse_r, theta, r_obs, p0=p0, bounds=bounds)
            a_fit, e_fit = popt

            # Generate fitted ellipse points for plotting
            theta_fit = np.linspace(0, 2*np.pi, 300)
            r_fit = ellipse_r(theta_fit, a_fit, e_fit)
            x_fit = r_fit * np.cos(theta_fit)
            y_fit = r_fit * np.sin(theta_fit)

            ax.plot(x_fit, y_fit, 'm--', label=f"Fitted ellipse (Sun at focus)")
            print(f"Regression ellipse eccentricity: {e_fit:.4f}")

            mars_radii = np.sqrt(mars_xy[:,0]**2 + mars_xy[:,1]**2)
            mars_theta = np.arctan2(mars_xy[:,1], mars_xy[:,0])

            # Fit ellipse in polar coordinates (Sun at origin)
            def ellipse_r(theta, a, e):
                return a*(1-e**2)/(1 + e*np.cos(theta))

            p0 = [np.mean(mars_radii), 0.1]
            bounds = ([0, 0], [np.inf, 0.9])
            popt, _ = scipy.optimize.curve_fit(ellipse_r, mars_theta, mars_radii, p0=p0, bounds=bounds)
            a_mars, e_mars = popt
            print(f"Mars actual eccentricity (from polar fit): {e_mars:.4f}")

        # ------------------------------------------------------------
        # 6. Finish
        # ------------------------------------------------------------
        ax.set_xlabel("x (simulation units)")
        ax.set_ylabel("y (simulation units)")
        ax.set_title("Actual Orbits + Synodic Predicted Mars Radii (midpoint Earth direction)")
        ax.legend()
        plt.tight_layout()
        plt.show()


        
    @staticmethod
    def fit_ellipse_least_squares(points):
        """
        Fit an ellipse to a set of 2D points using the Direct Least Squares method.
        Returns the ellipse parameters and eccentricity.

        Parameters
        ----------
        points : array-like, shape (n_points, 2)
            Array of (x, y) coordinates.

        Returns
        -------
        ellipse_params : dict
            Dictionary containing:
                - 'center' : (h, k)
                - 'axes'   : (a, b) semi-major and semi-minor axes
                - 'angle'  : rotation angle of the ellipse in radians
                - 'eccentricity' : e
        """
        x = points[:, 0]
        y = points[:, 1]

        # Build design matrix for conic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
        # Scatter matrix
        S = np.dot(D.T, D)
        # Constraint matrix
        C = np.zeros((6,6))
        C[0,2] = C[2,0] = 2
        C[1,1] = -1

        # Solve generalized eigenvalue problem
        import scipy.linalg
        eigvals, eigvecs = scipy.linalg.eig(S, C)
        # Pick the real solution
        cond = np.isreal(eigvals)
        a = np.real(eigvecs[:, cond][:,0])

        # Extract parameters
        A, B, C_, D_, E_, F_ = a

        # Compute center
        denom = B**2 - 4*A*C_
        h = (2*C_*D_ - B*E_) / denom
        k = (2*A*E_ - B*D_) / denom

        # Compute semi-axes
        up = 2*(A*E_**2 + C_*D_**2 + F_*B**2 - B*D_*E_ - 4*A*C_*F_)
        down1 = (B**2 - 4*A*C_)*( (C_ - A*np.sqrt(1 + (B**2)/((A-C_)**2))) )
        down2 = (B**2 - 4*A*C_)*( (C_ + A*np.sqrt(1 + (B**2)/((A-C_)**2))) )
        a_len = np.sqrt(np.abs(up/down1))
        b_len = np.sqrt(np.abs(up/down2))

        # Compute rotation angle
        angle = 0.5 * np.arctan2(B, A - C_)

        # Eccentricity
        e = np.sqrt(1 - (b_len**2 / a_len**2))

        return {
            "center": (h, k),
            "axes": (a_len, b_len),
            "angle": angle,
            "eccentricity": e
        }



    # ================= SHOW PLOTS =================
    def show_plots(self):
        plt.show()


'''
    def plot_local_synodic_radius_markers(self):
        """
        Plots actual Earth and Mars orbits and adds predicted Mars radius markers
        based on local synodic periods.  Each marker is placed along the ray from
        Sun → Earth evaluated at the MIDPOINT of the synodic interval.
        """

        # ------------------------------------------------------------
        # 1. Compute synodic radius estimates
        # ------------------------------------------------------------
        syn_results = self.compute_local_synodic_radii()
        if len(syn_results) == 0:
            print("No synodic radius data available.")
            return

        # ------------------------------------------------------------
        # 2. Get actual orbits
        # ------------------------------------------------------------
        earth_xy = np.asarray(self.positions_list[1])   # Earth
        mars_xy  = np.asarray(self.positions_list[2])   # Mars
        times    = self.times_years

        # Compute Earth's mean orbital radius (simulation units)
        r_earth_mean = np.mean(np.sqrt(earth_xy[:,0]**2 + earth_xy[:,1]**2))

        # ------------------------------------------------------------
        # 3. Make the figure
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_aspect('equal', 'box')

        # Plot real orbits
        ax.plot(earth_xy[:,0], earth_xy[:,1], label="Earth (actual)", alpha=0.75)
        ax.plot(mars_xy[:,0], mars_xy[:,1],   label="Mars (actual)", alpha=0.75)

        ax.plot(0, 0, 'yo', markersize=10, label="Sun")

        # ------------------------------------------------------------
        # 4. Add predicted synodic radii using midpoint placement
        # ------------------------------------------------------------
        for k, ((t1, t2), syn, est_aM) in enumerate(syn_results):

            if est_aM is None or np.isnan(est_aM):
                continue

            # Convert AU to simulation units
            pred_radius = est_aM * r_earth_mean

            # Midpoint time
            t_mid = 0.5 * (t1 + t2)

            # Closest Earth index to t_mid
            idx_mid = np.argmin(np.abs(times - t_mid))

            ex, ey = earth_xy[idx_mid]
            rE = np.sqrt(ex**2 + ey**2)
            if rE == 0:
                continue

            # Unit vector Sun→Earth at midpoint
            ux = -ex / rE
            uy = -ey / rE

            # Predicted Mars position along that ray
            px = pred_radius * ux
            py = pred_radius * uy

            ax.plot(px, py, 'r*', markersize=11)
            ax.text(px, py, f"{k}", color='red', fontsize=9)

        # ------------------------------------------------------------
        # 5. Finish
        # ------------------------------------------------------------
        ax.set_xlabel("x (simulation units)")
        ax.set_ylabel("y (simulation units)")
        ax.set_title("Actual Orbits + Synodic Predicted Mars Radii (midpoint Earth direction)")
        ax.legend()
        plt.tight_layout()
        plt.show()
'''