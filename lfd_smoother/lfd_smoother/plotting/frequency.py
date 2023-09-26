
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

from lfd_smoother.plotting.trajectory import TrajectoryStock

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.default'] = 'regular'


class FrequencyAnalysis:

    def __init__(self):
        pass

    def plot_frequency_domain(self, traj):
        ts = traj.ts
        ys = traj.ys
        yds = traj.yds
        ydds = traj.ydds
        yddds = traj.yddds
        dt = np.mean(np.diff(ts))

        # Perform the Fourier Transform
        ys_freq = fft(ys)
        yds_freq = fft(yds)
        ydds_freq = fft(ydds)
        yddds_freq = fft(yddds)

        # Get the frequencies for the Fourier Transform
        freq = fftfreq(len(ts), dt)

        # Plotting
        plt.figure(figsize=(14,10))

        plt.subplot(221)
        plt.plot(freq, np.abs(ys_freq))
        plt.title('Position')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')

        plt.subplot(222)
        plt.plot(freq, np.abs(yds_freq))
        plt.title('Velocity')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')

        plt.subplot(223)
        plt.plot(freq, np.abs(ydds_freq))
        plt.title('Acceleration')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')

        plt.subplot(224)
        plt.plot(freq, np.abs(yddds_freq))
        plt.title('Jerk')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    def run(self, demo_name):
        original_traj = TrajectoryStock()
        original_traj.import_from_lfd_storage("filter"+demo_name)

        smooth_traj = TrajectoryStock()
        smooth_traj.import_from_pydrake("smooth"+demo_name, original_traj.ts[-1])

        self.plot_frequency_domain(original_traj)
        self.plot_frequency_domain(smooth_traj)


