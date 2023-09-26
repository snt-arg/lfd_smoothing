
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle

from lfd_smoother.util.demonstration import Demonstration

import matplotlib.collections as mcoll
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.default'] = 'regular'

class ToleranceAnalysis:

    def __init__(self) -> None:
        pass

    def import_data(self, demo_name):
        with open("timing/{}.pickle".format(demo_name), 'rb') as file:
            self.ts_original = pickle.load(file)
        with open("timing_new/{}.pickle".format(demo_name), 'rb') as file:
            self.ss_new = pickle.load(file)
        with open("metadata/{}.pickle".format(demo_name), 'rb') as file:
            self.metadata = pickle.load(file)    
        with open("tolerances/{}.pickle".format(demo_name), 'rb') as file:
            self.tolerances = pickle.load(file) 
        
        self.demo = Demonstration()
        self.demo.read_from_json("waypoints/{}.json".format(demo_name))
        
        self.ts = self.metadata["ts"]
        self.ss = self.metadata["ss"]
        self.commands = self.metadata["commands"]
        self.cmds_raw = np.array(self.metadata.get("cmd_raw",[]))
        self.ss_original = np.array(self.ts_original) / self.ts_original[-1]
        # self.ts_original = np.array(self.ts_original) * float(smooth_traj.ts[-1]) / self.ts_original[-1]
        
        self.tolerances = np.array(self.tolerances)
        self.tol_trans = self.tolerances[:,0]
        self.tol_rot = self.tolerances[:,1]

        self.extract_commands()
        
    def extract_commands(self):
        func_s_command = interpolate.interp1d(self.ss, np.array(self.commands), kind='linear', fill_value="extrapolate")
        self.wp_commands = np.array(func_s_command(self.ss_original))

    def plot_waypoints(self, smooth_duration, correct_duration):       
        x, y, z = self.demo.positions.T
        
        plt.figure(figsize=(6, 1.6))
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')        

        plt.scatter(self.ss_original*smooth_duration, y, c='royalblue', marker='+', label=f'Before refinement', alpha=0.5)
        plt.scatter(self.ss_new*correct_duration, y, c='firebrick', marker='*', label=f'After refinement')

        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel('$y_{w_i}$ [m]', fontsize=12)
        plt.legend(fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_t_command(self):
        plt.plot(self.ts, self.commands, label='$s_r(t)$', color='firebrick')
        # Grids and Background
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')        
        plt.xlabel('Time [s]')
        plt.ylabel('Deceleration command R(t)')
        plt.legend()
        plt.tight_layout()
        plt.show()        

    def plot_t_s(self):
        plt.plot(self.ts, self.ss, label='$s_r(t)$', color='firebrick')
        
        v0 = 0.2
        t_new = np.linspace(0, 5, 100) 
        s_new = v0 * t_new
        
        plt.plot(t_new, s_new, label=f'$s(t) = V_0^r \cdot t$', linestyle='--', color='royalblue')
        # Grids and Background
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')        
        plt.xlabel('Time [s]]')
        plt.ylabel('Normalized time (s)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_tols_only(self):
        plt.figure(figsize=(7, 4))

        plt.subplot(2, 1, 2)
        plt.plot(self.ts, self.commands, label='$R(t)$', color='royalblue')
        plt.xlabel('Time [s]')
        plt.ylabel('Brake Command')

        plt.subplot(2, 1, 1)
        plt.fill_between(self.ss_new * self.ts[-1], -self.tol_trans, self.tol_trans, color='royalblue', alpha=0.2)
        plt.ylabel('Tolerance Bounds [m]')
        plt.tight_layout()
        plt.show()

    def plot_t_s_command(self):
        plt.figure(figsize=(6, 3))

        plt.subplot(2, 1, 2)
        plt.plot(self.ts, self.commands, label='$R(t)$', color='firebrick')
        plt.axvline(x=1, color='green', linestyle='-.')

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.xlabel('Time [s]')
        # plt.ylabel('R')
        plt.legend()

        plt.subplot(2, 1, 1)
        plt.plot(self.ts, self.ss, label='$s_r(t)$', color='firebrick')

        v0 = 0.2
        t_new = np.linspace(0, 5, 100)
        s_new = v0 * t_new
        plt.plot(t_new, s_new, label=f'$s(t) = V_0^r \cdot t$', linestyle='--', color='royalblue')
        plt.axvline(x=1, color='green', linestyle='-.')
        plt.text(0.2, 0.7, "$t^*$", fontsize=10, color="black", verticalalignment='top', bbox = dict(boxstyle="round", fc="tab:green", alpha=0.2))
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

        plt.legend()
        plt.tight_layout()
        plt.show()

    def map_range(self,value):
        input_range = [1, -1]  # input range
        output_range = [0,-1]  # desired output range

        # line equation parameters
        slope = (output_range[1] - output_range[0]) / (input_range[1] - input_range[0])
        intercept = output_range[0] - slope * input_range[0]

        return slope * value + intercept

    def plot_low_high_tol(self, smooth_traj, correct_traj):
        tol_default = 0.02
        
        x, y, z = self.demo.positions.T
        X, Y, Z = correct_traj.positions.T
        Ts = correct_traj.ts

        XX, YY, ZZ = smooth_traj.positions.T
        TTs = smooth_traj.ts 

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 3))

        l1, = axs[0].plot(Ts, X, label='$x_f^r$ - high tol.' , color='firebrick', linewidth=2)
        axs[0].axvline(x=Ts[-1], color='firebrick', linestyle='-.', linewidth=1.5, alpha=0.7)
        axs[0].text(Ts[-1]-0.17, 0.25, f'{Ts[-1]:.2f} s', fontsize=8, color="black", verticalalignment='top', 
                    bbox = dict(boxstyle="round", fc="firebrick", alpha=0.3))

        l2, = axs[0].plot(TTs, XX, label='$x_f^r$ - low tol.'  , color='royalblue', linewidth=2)
        axs[0].axvline(x=TTs[-1], color='royalblue', linestyle='-.', linewidth=1.5, alpha=0.7)
        axs[0].text(TTs[-1]-0.17, 0.25, f'{TTs[-1]:.2f} s', fontsize=8, color="black", verticalalignment='top', 
                    bbox = dict(boxstyle="round", fc="royalblue", alpha=0.3))

        l3 = axs[0].fill_between(self.ss_original * correct_traj.ts[-1], x - self.tol_trans, x + self.tol_trans, 
                                 color='firebrick', alpha=0.2, label="tolerance=5cm")
        l4 = axs[0].fill_between(self.ss_original * smooth_traj.ts[-1], x - tol_default, x + tol_default, 
                                 color='royalblue', alpha=0.2, label="tolerance=2cm")
        axs[0].grid(True, linestyle='--', linewidth=0.5, color='gray')


        velocities = np.array(correct_traj.velocities)
        times = np.array(correct_traj.ts)
        accelerations = np.gradient(velocities, times)
        jerks = np.gradient(accelerations, times)

        axs[1].plot(correct_traj.ts, np.abs(jerks), label='$x_f^r$ - high tol.', color='firebrick')
        axs[1].axhline(y=np.max(np.abs(jerks)), color="firebrick", linestyle='--', alpha=0.7)
        axs[1].text(0.75, np.max(np.abs(jerks)) - 3, f'Maximum: {np.max(np.abs(jerks)):.2f}', 
                    color="firebrick", verticalalignment='top')


        velocities = np.array(smooth_traj.velocities)
        times = np.array(smooth_traj.ts)
        accelerations = np.gradient(velocities, times)
        jerks = np.gradient(accelerations, times) 

        axs[1].plot(smooth_traj.ts, np.abs(jerks), label='$x_f^r$ - low tol.', color='royalblue')
        axs[1].axhline(y=np.max(np.abs(jerks)), color="royalblue", linestyle='--', alpha=0.7)
        axs[1].text(0.75, np.max(np.abs(jerks)) - 3, f'Maximum: {np.max(np.abs(jerks)):.2f}', 
                    color="royalblue", verticalalignment='top')
        axs[1].grid(True, linestyle='--', linewidth=0.5, color='gray')

        axs[1].set_xlabel('Time [s]', fontsize=11)
        axs[0].set_ylabel('$x_f^r$ $[m]$', fontsize=12)
        axs[1].set_ylabel('$||\dddot{p}_f^r||$ [$m/s^3$]', fontsize=12)

        legend = axs[0].legend(handles=[l3,l4], fontsize=9, markerscale=1, loc='upper left')
        axs[0].add_artist(legend)
        axs[1].legend(fontsize=9, markerscale=1)

        plt.tight_layout()
        plt.show()

    def plot_x_with_tols(self, correct_traj):
        self.ts_new = self.ss_new * correct_traj.ts[-1]
        x, y, z = self.demo.positions.T  
        X, Y, Z = correct_traj.positions.T

        colors = [
            "maroon",
            "firebrick",
            "darkorange",
            "limegreen",
            "limegreen",
            "teal",
            "teal", 
            "royalblue"
            ]
        cmap_name = "custom_div_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        fig, ax = plt.subplots(figsize=(6, 2))        
        n_segments = len(correct_traj.ts) - 1
        
        wps_interp = np.interp(
        np.linspace(0, 1, n_segments),
        self.ss_new,
        self.wp_commands
        )

        norm = plt.Normalize(wps_interp.min(), wps_interp.max())
        colors = cm(norm(wps_interp))

        points = np.array([correct_traj.ts, X]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcoll.LineCollection(segments, colors=colors, linewidth=2, zorder=2)
        ax.add_collection(lc)
        ax.autoscale()
        ax.fill_between(self.ts_new, x - self.tol_trans, x + self.tol_trans, 
                        color='slategray', alpha=0.4, zorder=1, label="tolerance bounds")
        
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('$x_f^r$ [m]', fontsize=12)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])

        fig.subplots_adjust(top = 0.7, bottom = 0.25, right=0.94, left=0.12,
                    hspace = 0, wspace = 0)

        cbar_ax = fig.add_axes([0.15, 0.84, 0.75, 0.05])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('$R$', rotation=0, labelpad=-37)
        ax.legend(loc='lower right')
        plt.show()


    def plot_traj_with_tols(self, correct_traj):

        self.ts_new = self.ss_new * correct_traj.ts[-1]
        x, y, z = self.demo.positions.T  
        X, Y, Z = correct_traj.positions.T

        colors = [
            "firebrick",
            "darkorange",
            "darkorange",
            "limegreen",
            "limegreen",
            "teal",
            "teal", 
            "royalblue"
            ]
        cmap_name = "custom_div_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        
        fig, axs = plt.subplots(3, 1, sharex=True)
        n_segments = len(correct_traj.ts) - 1
        wps_interp = np.interp(
        np.linspace(0, 1, n_segments),
        self.ss_new,
        self.wp_commands
        )

        norm = plt.Normalize(wps_interp.min(), wps_interp.max())
        colors = cm(norm(wps_interp))

        points = np.array([correct_traj.ts, X]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcoll.LineCollection(segments, colors=colors, linewidth=2)
        axs[0].add_collection(lc)
        axs[0].autoscale()
        axs[0].fill_between(self.ts_new, x - self.tol_trans, x + self.tol_trans, color='slategray', alpha=0.4)
        axs[0].grid(True, linestyle='--', linewidth=0.5, color='gray')
        axs[0].axvline(x=1.2, color='teal', linestyle='-.', linewidth=1.5, alpha=0.7)

        points = np.array([correct_traj.ts, Y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcoll.LineCollection(segments, colors=colors, linewidth=2)
        axs[1].add_collection(lc)
        axs[1].autoscale()
        axs[1].fill_between(self.ts_new, y - self.tol_trans, y + self.tol_trans, color='slategray', alpha=0.4)
        axs[1].grid(True, linestyle='--', linewidth=0.5, color='gray')  
        axs[1].axvline(x=1.2, color='teal', linestyle='-.', linewidth=1.5, alpha=0.7)

        points = np.array([correct_traj.ts, Z]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcoll.LineCollection(segments, colors=colors, linewidth=2)
        axs[2].add_collection(lc)
        axs[2].autoscale()
        axs[2].fill_between(self.ts_new, z - self.tol_trans, z + self.tol_trans, color='slategray', alpha=0.4)
        axs[2].grid(True, linestyle='--', linewidth=0.5, color='gray')
        axs[2].axvline(x=1.2, color='teal', linestyle='-.', linewidth=1.5, alpha=0.7)

        axs[2].set_xlabel('Time [s]')
        axs[0].set_ylabel('X [m]')
        axs[1].set_ylabel('Y [m]')
        axs[2].set_ylabel('Z [m]')

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])

        # Make space for the colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.88, 0.11, 0.02, 0.77])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.yaxis.set_ticks_position('left')
        cbar.set_label('$R(t)$', rotation=90, labelpad=15)

        for ax in axs:
            ax.legend()

        plt.show()
