
import math
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import interpolate
import pickle
from lfd_interface.srv import GetDemonstration

from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState


from pydrake.all import *
from lfd_smoother.util.fr3_drake import FR3Drake
from lfd_smoother.util.demonstration import Demonstration

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.collections as mcoll
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from matplotlib import transforms


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.default'] = 'regular'

class TrajectoryStock:

    def __init__(self):
        self.positions = None # To be filled later by Cartesian Analysis
        self.velocities = None # To be filled later by Cartesian Analysis

    def import_from_lfd_storage(self,name, t_scale=0):
        rospy.wait_for_service("get_demonstration")
        sc_lfd_storage = rospy.ServiceProxy("get_demonstration", GetDemonstration)
        resp = sc_lfd_storage(name=name)
        joint_trajectory = resp.Demonstration.joint_trajectory
        self._from_joint_trajectory(joint_trajectory, t_scale)
        self.normalize_t()

    def import_from_pydrake(self,name, t_scale=0):
        with open("traj/{}.pickle".format(name), 'rb') as file:
            traj = pickle.load(file)
        self._from_pydrake(traj,t_scale)
        self.normalize_t()

    def import_from_dmp_joint_trajectory(self, name, t_scale=0):
        with open("dmp/{}.pickle".format(name), 'rb') as file:
            joint_trajectory = pickle.load(file) 
        self._from_joint_trajectory(joint_trajectory, t_scale)
        self.normalize_t()

    def import_from_joint_trajectory(self,name):
        pass


    def _from_joint_trajectory(self, joint_trajectory, t_scale):
        n_time_steps = len(joint_trajectory.points)
        n_dim = len(joint_trajectory.joint_names)

        ys = np.zeros([n_time_steps,n_dim])
        ts = np.zeros(n_time_steps)
        yds = np.zeros([n_time_steps,n_dim])
        ydds = np.zeros([n_time_steps,n_dim])
        yddds = np.zeros([n_time_steps,n_dim])

        for (i,point) in enumerate(joint_trajectory.points):
            ys[i,:] = point.positions
            ts[i] = point.time_from_start.to_sec()
        
        if t_scale != 0:
            old_t = ts[-1]
            scale = t_scale/old_t
            ts = ts * scale       

        self.traj = [interpolate.CubicSpline(ts , y) for y in ys.T]
        traj_d = [f.derivative(1) for f in self.traj]
        traj_dd = [f.derivative(2) for f in self.traj]
        traj_ddd = [f.derivative(3) for f in self.traj]
        
        for (i,t) in enumerate(ts):
            yds[i,:] = np.array([f(t) for f in traj_d]).T
            ydds[i,:] = np.array([f(t) for f in traj_dd]).T
            yddds[i,:] = np.array([f(t) for f in traj_ddd]).T

        self.ts = ts
        self.ys = ys
        self.yds = yds
        self.ydds = ydds
        self.yddds = yddds

        self.value = self._value_spline


    def _from_pydrake(self, traj, t_scale):    
        position = []
        velocity = []
        acceleration = []
        jerk = []

        if t_scale != 0:
            old_t = traj.end_time() - traj.start_time()
            scale = t_scale/old_t
            basis = traj.basis()
            scaled_knots = [(knot * scale) for knot in basis.knots()]
            new_traj = BsplineTrajectory(
                BsplineBasis(basis.order(), scaled_knots),
                traj.control_points()) 
            traj = new_traj             

        self.ts = np.linspace(0, traj.end_time(), 1000)

        for t in self.ts:
            position.append(traj.value(t))
            velocity.append(traj.MakeDerivative().value(t))
            acceleration.append(traj.MakeDerivative(2).value(t))
            jerk.append(traj.MakeDerivative(3).value(t))

        self.ys = np.array(position).squeeze(axis=2)
        self.yds = np.array(velocity).squeeze(axis=2)
        self.ydds = np.array(acceleration).squeeze(axis=2)
        self.yddds = np.array(jerk).squeeze(axis=2)
        self.traj = traj

        self.value = self._value_pydrake
    

    def _value_spline(self, t):
        return np.array([f(t) for f in self.traj]).T
    
    def _value_pydrake(self, t):
        return self.traj.value(t)
    
    def normalize_t(self):
        self.ss = self.ts / self.ts[-1]

    def plot(self):

        ts = self.ts
        
        data = {
            "${q}_o$ [rad]": self.ys,
            "$\dot{q}_o$ [rad/s]": self.yds,
            "$\ddot{q}_o$ [rad/s${}^2$]": self.ydds,
            "$\dddot{q}_o$ [rad/s${}^3$]": self.yddds
        }
        
        fig, axs = plt.subplots(2, 2, figsize=(5,4))
        axs = axs.ravel()

        for idx, (label, values) in enumerate(data.items()):
            axs[idx].plot(ts, np.array(values))
            # axs[idx].set_title(label)
            axs[idx].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            axs[idx].set_ylabel(label, labelpad=-4, fontsize=12)
            
            if idx >= 2:  # Only set x-label for the bottom subplots
                axs[idx].set_xlabel('time [s]', labelpad=0)
            
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)
        plt.show()


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


class TorqueAnalysis:

    def __init__(self) -> None:
        
        self.torques = []
        self.velocities = []
        self.record_flag = False

        self.rate = rospy.Rate(1000)
        rospy.Subscriber("/joint_states", JointState, self.cb_record_joint_state)
        self.pub_robot = rospy.Publisher("/position_joint_controller/command", Float64MultiArray , queue_size=0)

    def cb_record_joint_state(self, data):
        if self.record_flag:
            self.torques.append(data.effort[:-2])
            self.velocities.append(data.velocity[:-2])
    
    def plot_torque(self):
        for idx, joint_torques in enumerate(zip(*self.torques)):
            plt.figure()
            plt.plot(joint_torques)
            plt.title(f'Joint {idx + 1} Torque Profile')
            plt.xlabel('Time step')
            plt.ylabel('Torque')
        plt.show()  

    def calculate_energy_consumption(self):
        energy = []
        for joint_torques, joint_velocities in zip(zip(*self.torques), zip(*self.velocities)):
            power = [abs(torque*velocity) for torque, velocity in zip(joint_torques, joint_velocities)]
            energy.append(sum(power))
        return sum(energy)
    
    def run(self, traj):
        self.record_flag = True
        t_start = rospy.Time.now().to_sec()
        t = 0  
        while (not rospy.is_shutdown()) and t < traj.ts[-2]:
            t = rospy.Time.now().to_sec() - t_start
            msg = Float64MultiArray()
            msg.data = traj.value(t)
            self.pub_robot.publish(msg)
            self.rate.sleep() 

        self.record_flag = False
        return self.calculate_energy_consumption()
    



class CartesianAnalysis:

    def __init__(self) -> None:
        self.robot = FR3Drake(franka_pkg_xml="/home/abrk/catkin_ws/src/lfd/lfd_smoothing/drake/franka_drake/package.xml",
                urdf_path="package://franka_drake/urdf/fr3_nohand.urdf")
        self.visualizer_context = self.robot.visualizer.GetMyContextFromRoot(self.robot.context)
        self.ee_body = self.robot.plant.GetBodyByName("fr3_link8")

    def to_position(self, q):
        self.robot.plant.SetPositions(self.robot.plant_context, q)
        transform = self.robot.plant.EvalBodyPoseInWorld(self.robot.plant_context, self.ee_body)
        return transform.translation()
    
    def to_velocity(self, q, qd):
        qqd = np.concatenate((q,qd), axis=0)
        self.robot.plant.SetPositionsAndVelocities(self.robot.plant_context, qqd)
        velocity = self.robot.plant.EvalBodySpatialVelocityInWorld(self.robot.plant_context, self.ee_body)
        return np.linalg.norm(velocity.translational())
    
    def to_torque(self, q, qd, qdd): # Does not work!
        qqd = np.concatenate((q,qd), axis=0)
        self.robot.plant.SetPositionsAndVelocities(self.robot.plant_context, qqd)
        force = MultibodyForces(self.robot.plant)
        force.SetZero()
        torq = self.robot.plant.CalcInverseDynamics(self.robot.plant_context, qdd, force)
        return torq
        
    
    def from_trajectory(self, traj):
        self.ts = traj.ts
        self.positions = []
        self.velocities = []
        for i in range(len(traj.ts)):
            self.positions.append(self.to_position(traj.ys[i]))
            self.velocities.append(self.to_velocity(traj.ys[i], traj.yds[i]))
        
        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        traj.positions = self.positions
        traj.velocities = self.velocities
    
    
    def plot(self):
        # Make subplots for X, Y, Z, and V
        fig, axs = plt.subplots(4)
        
        # plot x, y, z
        axs[0].plot(self.ts, self.positions[:, 0]) # plot x
        axs[0].set_title('X coordinate')
        axs[1].plot(self.ts, self.positions[:, 1]) # plot y
        axs[1].set_title('Y coordinate')
        axs[2].plot(self.ts, self.positions[:, 2]) # plot z
        axs[2].set_title('Z coordinate')
        
        # plot v
        axs[3].plot(self.ts, self.velocities)
        axs[3].set_title('V')
        
        # Set common labels
        fig.text(0.5, 0.04, 'Time', ha='center', va='center')
        fig.text(0.06, 0.5, 'Values', ha='center', va='center', rotation='vertical')
        
        plt.tight_layout()
        plt.show() 


    def plot_3d(self):
        # Make a figure and subplots for 3D trajectory, X, Y, Z, and V
        fig = plt.figure()
        
        # 3D trajectory plot
        ax_traj = fig.add_subplot(2, 2, 1, projection='3d')
        ax_traj.plot(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2])
        ax_traj.set_title('3D Trajectory')
        ax_traj.set_xlabel('X')
        ax_traj.set_ylabel('Y')
        ax_traj.set_zlabel('Z')
      

        plt.tight_layout()
        plt.show()



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
        # Plotting original waypoints
        plt.scatter(self.ss_original*smooth_duration, y, c='royalblue', marker='+', label=f'Before refinement', alpha=0.5)

        # Plotting corrected waypoints
        plt.scatter(self.ss_new*correct_duration, y, c='firebrick', marker='*', label=f'After refinement')

        # Adding labels, title, and legend
        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel('$y_{w_i}$ [m]', fontsize=12)
        # plt.title('Waypoint Comparisons')
        plt.legend(fontsize=9)

        # Adding grid for better readability
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

    def plot_t_s_command(self):
        plt.figure(figsize=(6, 3))

        # First plot (2x1 grid, first subplot)
        plt.subplot(2, 1, 2)
        plt.plot(self.ts, self.commands, label='$R(t)$', color='firebrick')
        plt.axvline(x=2.958, color='green', linestyle='-.')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.xlabel('Time [s]')
        plt.ylabel('R')
        plt.legend()

        # Second plot (2x1 grid, second subplot)
        plt.subplot(2, 1, 1)
        plt.plot(self.ts, self.ss, label='$s_r(t)$', color='firebrick')

        v0 = 0.2
        t_new = np.linspace(0, 5, 100)
        s_new = v0 * t_new
        plt.plot(t_new, s_new, label=f'$s(t) = V_0^r \cdot t$', linestyle='--', color='royalblue')
        
        plt.axvline(x=2.958, color='green', linestyle='-.')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        # plt.xlabel('Time [s]')
        plt.ylabel('s')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_normalized(self, smooth_traj, correct_traj):
        tol_default = 0.02
        
        x, y, z = self.demo.positions.T

        X, Y, Z = correct_traj.positions.T
        Ts = correct_traj.ts #/ correct_traj.ts[-1]

        XX, YY, ZZ = smooth_traj.positions.T
        TTs = smooth_traj.ts #/ smooth_traj.ts[-1]

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 3))

        l1, = axs[0].plot(Ts, X, label='$x_f^r$ - high tol.' , color='firebrick', linewidth=2)
        axs[0].axvline(x=Ts[-1], color='firebrick', linestyle='-.', linewidth=1.5, alpha=0.7)
        axs[0].text(Ts[-1]-0.17, 0.25, f'{Ts[-1]:.2f} s', fontsize=8, color="black", verticalalignment='top', bbox = dict(boxstyle="round", fc="firebrick", alpha=0.3))

        l2, = axs[0].plot(TTs, XX, label='$x_f^r$ - low tol.'  , color='royalblue', linewidth=2)
        axs[0].axvline(x=TTs[-1], color='royalblue', linestyle='-.', linewidth=1.5, alpha=0.7)
        axs[0].text(TTs[-1]-0.17, 0.25, f'{TTs[-1]:.2f} s', fontsize=8, color="black", verticalalignment='top', bbox = dict(boxstyle="round", fc="royalblue", alpha=0.3))

        l3 = axs[0].fill_between(self.ss_original * correct_traj.ts[-1], x - self.tol_trans, x + self.tol_trans, color='firebrick', alpha=0.2, label="tolerance=5cm")
        l4 = axs[0].fill_between(self.ss_original * smooth_traj.ts[-1], x - tol_default, x + tol_default, color='royalblue', alpha=0.2, label="tolerance=2cm")
        axs[0].grid(True, linestyle='--', linewidth=0.5, color='gray')


        velocities = np.array(correct_traj.velocities)
        times = np.array(correct_traj.ts)
        accelerations = np.gradient(velocities, times)
        jerks = np.gradient(accelerations, times)

        axs[1].plot(correct_traj.ts, np.abs(jerks), label='$x_f^r$ - high tol.', color='firebrick')
        axs[1].axhline(y=np.max(np.abs(jerks)), color="firebrick", linestyle='--', alpha=0.7)
        axs[1].text(0.75, np.max(np.abs(jerks)) - 3, f'Maximum: {np.max(np.abs(jerks)):.2f}', color="firebrick", verticalalignment='top')

        # bbox = dict(boxstyle="round", fc="blue")
        # tform = transforms.blended_transform_factory(axs[0].transData, axs[0].transAxes) 
        # axs[0].annotate('{:.2g}s'.format(np.max(np.abs(jerks))), 
        #                 xy=(1, np.max(np.abs(jerks))), 
        #                 bbox=bbox, 
        #                 xycoords=tform)



        velocities = np.array(smooth_traj.velocities)
        times = np.array(smooth_traj.ts)
        accelerations = np.gradient(velocities, times)
        jerks = np.gradient(accelerations, times) 

        axs[1].plot(smooth_traj.ts, np.abs(jerks), label='$x_f^r$ - low tol.', color='royalblue')
        axs[1].axhline(y=np.max(np.abs(jerks)), color="royalblue", linestyle='--', alpha=0.7)
        axs[1].text(0.75, np.max(np.abs(jerks)) - 3, f'Maximum: {np.max(np.abs(jerks)):.2f}', color="royalblue", verticalalignment='top')
        # ellipse = Ellipse((0.13, 40), width=0.3, height=80, edgecolor='green', facecolor='none', linestyle='--')
        # axs[1].add_patch(ellipse)
        axs[1].grid(True, linestyle='--', linewidth=0.5, color='gray')

        axs[1].set_xlabel('Time [s]', fontsize=11)
        axs[0].set_ylabel('$x_f^r$ $[m]$', fontsize=12)
        axs[1].set_ylabel('$||\dddot{p}_f^r||$ [$m/s^3$]', fontsize=12)

        legend = axs[0].legend(handles=[l3,l4], fontsize=9, markerscale=1, loc='upper left')
        axs[0].add_artist(legend)
        # axs[0].legend(handles=[l1,l2], fontsize=9, markerscale=1, loc='lower right')
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
        # colors = ["black", "blue", "cyan", "green", "yellow", "orange", "red", "magenta"]
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
        # ax.margins(0,0)
        # fig.subplots_adjust(top=0.72, bottom=0.25)

        cbar_ax = fig.add_axes([0.15, 0.84, 0.75, 0.05])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('$R$', rotation=0, labelpad=-37)
        ax.legend(loc='lower right')
        # fig.tight_layout(rect=[0, 1, 0, 0])
        plt.show()


        # # Make space for the colorbar
        # fig.subplots_adjust(right=0.8, bottom=0.30)
        # cbar_ax = fig.add_axes([0.9, 0.28, 0.015, 0.6]) # [left, bottom, width, height]
        # cbar = fig.colorbar(sm, cax=cbar_ax)
        # cbar_ax.yaxis.set_ticks_position('left')
        # cbar.set_label('$R$', rotation=90, labelpad=7)

        # ax.legend(loc='lower right')

        # # fig.tight_layout(rect=[0, 0, 0.8, 1])
        # plt.show()


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
        # colors = ["black", "blue", "cyan", "green", "yellow", "orange", "red", "magenta"]
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
        # cm = plt.get_cmap('turbo')
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

        # cbar_ax = fig.add_axes([0.15, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        # cbar = fig.colorbar(sm, cax=cbar_ax)

        # Make space for the colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.88, 0.11, 0.02, 0.77])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.yaxis.set_ticks_position('left')
        cbar.set_label('$R(t)$', rotation=90, labelpad=15)

        for ax in axs:
            ax.legend()

        # fig.tight_layout(rect=[0, 0, 0.8, 1])
        plt.show()

    def plot_traj_with_tols_3d(self, traj):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = self.demo.positions.T
        X, Y, Z = traj.positions.T

        # Compute tangent, normal and binormal for the trajectory
        # These are basic derivatives; for more accurate representations, more sophisticated techniques can be used
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        
        tangent = np.array([dx, dy, dz]).T
        tangent /= np.linalg.norm(tangent, axis=1)[:, np.newaxis]
        
        # Arbitrarily select [1, 1, 1] as another direction to produce the normal via cross product
        normal = np.cross(tangent, [1, 1, 1])
        normal /= np.linalg.norm(normal, axis=1)[:, np.newaxis]
        
        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal, axis=1)[:, np.newaxis]
        
        u = np.linspace(0, 2 * np.pi, 49)

        for i in range(len(x)):
            tol_radius = self.tol_trans  # Assuming a constant tolerance
            x_circ = x[i] + tol_radius * (np.cos(u) * normal[i, 0] + np.sin(u) * binormal[i, 0])
            y_circ = y[i] + tol_radius * (np.cos(u) * normal[i, 1] + np.sin(u) * binormal[i, 1])
            z_circ = z[i] + tol_radius * (np.cos(u) * normal[i, 2] + np.sin(u) * binormal[i, 2])
            
            ax.plot(x_circ, y_circ, z_circ, color='gray', alpha=0.2)
        
        # Plot the original and new trajectories
        ax.plot(x, y, z, label='Original Trajectory')
        # ax.plot(X, Y, Z, label='New Trajectory', color='r')

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        ax.legend()
        plt.show()



class JerkAnalysis:

    def __init__(self) -> None:
        pass


    def plot_with_high_jerk(self, ts, ys, jerks, bounds = None):
        colors = [
            "darkslateblue", 
            "teal",             
            "limegreen",
            "darkorange",
            "darkorange",
            "darkorange",
            "red",
            "red",
            "red",
            "red",
            "red",
            "red",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
            "darkred",
            "darkred"
        ]
        # colors = ["black", "blue", "cyan", "green", "yellow", "orange", "red", "magenta"]
        cmap_name = "custom_div_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        # Take the absolute value of jerks
        if bounds is None:
            bounds = np.abs(jerks)
        else:
            bounds = np.abs(bounds)

        abs_jerks = np.abs(jerks)

        # Create line segments and corresponding colors
        points = np.array([ts, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, bounds.max())
        colors = cm(norm(abs_jerks))  # Using the custom colormap
        
        # Plot trajectory with color-coded jerk
        fig, ax = plt.subplots(figsize=(5, 3))
        lc = mcoll.LineCollection(segments, colors=colors, linewidth=4)
        ax.add_collection(lc)
        ax.autoscale()
        
        fig.subplots_adjust(left=0.1)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)  # Using the custom colormap
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, location='left', use_gridspec=True, pad=0.12)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        # plt.xlabel('Time')
        # plt.ylabel('Trajectory')
        # plt.title('Trajectory with Jerk Magnitude as Color')

        # Set three rounded ticks for both x and y axes (beginning, middle, and end)
        ax.set_xticks([round(ts.min(), 1), round(np.mean(ts), 1), round(ts.max(), 1)])
        ax.set_yticks([round(ys.min(), 1), round(np.mean(ys), 1), round(ys.max(), 1)])

        # Set three rounded ticks for the colorbar (beginning, middle, and end)
        rounded_bound_max = math.floor(bounds.max() / 1000) * 1000
        cbar.set_ticks([0, rounded_bound_max / 2, rounded_bound_max])

        plt.show()


    def plot_with_low_jerk(self, ts, ys, jerks, bounds = None):
        colors = [
            "darkslateblue", 
            "teal",             
            "limegreen"
        ]
        # colors = ["black", "blue", "cyan", "green", "yellow", "orange", "red", "magenta"]
        cmap_name = "custom_div_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
        # Take the absolute value of jerks
        if bounds is None:
            bounds = np.abs(jerks)
        else:
            bounds = np.abs(bounds)

        abs_jerks = np.abs(jerks)

        # Create line segments and corresponding colors
        points = np.array([ts, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, bounds.max())
        colors = cm(norm(abs_jerks))  # Using the custom colormap
        
        # Plot trajectory with color-coded jerk
        fig, ax = plt.subplots(figsize=(5, 3))
        lc = mcoll.LineCollection(segments, colors=colors, linewidth=4)
        ax.add_collection(lc)
        ax.autoscale()
        
        fig.subplots_adjust(left=0.1)

        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)  # Using the custom colormap
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, location='left', use_gridspec=True, pad=0.12)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        # plt.xlabel('Time')
        # plt.ylabel('Trajectory')
        # plt.title('Trajectory with Jerk Magnitude as Color')

        # Set three rounded ticks for both x and y axes (beginning, middle, and end)
        ax.set_xticks([round(ts.min(), 1), round(np.mean(ts), 1), round(ts.max(), 1)])
        ax.set_yticks([round(ys.min(), 1), round(np.mean(ys), 1), round(ys.max(), 1)])

        # Set three rounded ticks for the colorbar (beginning, middle, and end)
        rounded_bound_max = math.floor(bounds.max() / 10) * 10
        cbar.set_ticks([0, rounded_bound_max / 2, rounded_bound_max])

        plt.show()


class DMPAnalysis:

    def __init__(self) -> None:
        pass

    def plot_3d(self, original_dmp, smooth_dmp):
        # Make a figure for 3D trajectory
        fig = plt.figure(figsize=(4, 4))
        
        # Create 3D axis
        ax_traj = fig.add_subplot(111, projection='3d')
        
        # Plot original DMP
        ax_traj.plot(original_dmp.positions[:, 0], original_dmp.positions[:, 1], original_dmp.positions[:, 2], label='$p_{o,dmp}$', linestyle='--', color='royalblue')
        
        # Plot smooth DMP
        ax_traj.plot(smooth_dmp.positions[:, 0], smooth_dmp.positions[:, 1], smooth_dmp.positions[:, 2], label='$p_{f,dmp}$', linestyle='-', color='firebrick')

        # ax_traj.plot(original_traj.positions[:, 0], original_traj.positions[:, 1], original_traj.positions[:, 2], label='Traj', linestyle='--', color='g')

        # ax_traj.set_title('3D Trajectory')
        ax_traj.set_xlabel('x [m]')
        ax_traj.set_ylabel('y [m]')
        ax_traj.set_zlabel('z [m]')
        
        # Add legend to differentiate the original and smooth DMP
        ax_traj.legend(fontsize=12)
        
        # plt.tight_layout()
        plt.show()

    # def plot_vel_acc_jerk(self, original_dmp, smooth_dmp):
    #     o_velocities = np.array(original_dmp.velocities)
    #     o_times = np.array(original_dmp.ts)
    #     o_accelerations = np.gradient(o_velocities, o_times)
    #     o_jerks = np.gradient(o_accelerations, o_times)

    #     s_velocities = np.array(smooth_dmp.velocities)
    #     s_times = np.array(smooth_dmp.ts)
    #     s_accelerations = np.gradient(s_velocities, s_times)
    #     s_jerks = np.gradient(s_accelerations, s_times)        

    def calc_abs_jerk(self, traj):
        # Calculate jerk for original_dmp
        velocities = np.array(traj.velocities)
        times = np.array(traj.ts)
        accelerations = np.gradient(velocities, times, edge_order=1)
        jerks = np.gradient(accelerations, times, edge_order=1)
        abs_jerks = np.abs(jerks)
        return abs_jerks


    def plot_abs_jerk(self,original_dmp, smooth_dmp, scaled_dmp):

        # Create plot for jerk comparison
        plt.figure(figsize=(5, 2))
        plt.plot(original_dmp.ts, self.calc_abs_jerk(original_dmp), label='Original DMP', linestyle='--', color='royalblue')
        plt.plot(scaled_dmp.ts, self.calc_abs_jerk(scaled_dmp), label='Original DMP - Scaled', linestyle='-.', color='turquoise')
        plt.plot(smooth_dmp.ts, self.calc_abs_jerk(smooth_dmp), label='Smooth DMP', linestyle='-', color='firebrick')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        # plt.title('Absolute Jerk Comparison')
        plt.xlabel('Normalized time s')
        plt.ylabel('|Jerk|')
        plt.legend()
        plt.tight_layout()
        plt.show()



    def plot_with_kin_lims(self,scaled_dmp):
        ts = scaled_dmp.ts
        data = {
            "vel [rad/s]": scaled_dmp.yds,
            "accl [rad/s${}^2$]": scaled_dmp.ydds,
            "jerk [rad/s${}^3$]": scaled_dmp.yddds
        }
        
        vel_limits = [2, 1.1, 1.5, 1.25, 3, 1.5, 3]
        acc_limits = [10, 10, 10, 10, 10, 10, 10]
        jerk_limits = [5000, 5000, 5000, 5000, 5000, 5000, 5000]
        
        limits = {
            "vel [rad/s]": vel_limits,
            "accl [rad/s${}^2$]": acc_limits,
            "jerk [rad/s${}^3$]": jerk_limits
        }
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        plt.figure(figsize=(8, 2.5))
        
        for idx, (key, values) in enumerate(data.items()):
            plt.subplot(1, 3, idx + 1)
            
            for i, dof in enumerate(range(7)):  # Assuming 7 DOF
                
                color = colors[i]
                limit = limits[key][i]
                
                plt.plot(ts, values[:, i], label=f'Joint {i+1}', linestyle="-", linewidth=2, color="royalblue", alpha=0.5)
                plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

                # Highlight Violations
                above_limit = np.where(values[:, i] > limit)
                plt.scatter(ts[above_limit], values[above_limit, i], color=color, marker='x' , zorder=3)
                
            # plt.axhline(limit, color='r', linestyle='--', label=f'{key} Limit')
            
            # plt.title(key)
            plt.xlabel('Time [s]')
            plt.ylabel(key, labelpad=-5)
            # plt.legend()
            
        plt.tight_layout()
        plt.show()


    def plot_vel_acc_kin_lims(self, scaled_dmp, smooth_dmp):
        # Different time stamps for scaled_dmp and smooth_dmp
        ts_scaled = scaled_dmp.ts
        ts_smooth = smooth_dmp.ts

        vel_limits = [2, 1.1, 1.5, 1.25, 3, 1.5, 3]
        acc_limits = [10, 10, 10, 10, 10, 10, 10]
        
        limits = {
            "vel [rad/s]": vel_limits,
            "accl [rad/s${}^2$]": acc_limits
        }
        
        colors = ['r', 'r', 'r', 'r', 'r', 'r', 'r']
        
        fig, axes = plt.subplots(2, 2, figsize=(5, 4))
        
        for idx, name in enumerate(['vel [rad/s]', 'accl [rad/s${}^2$]']):
            for plot_row, (dmp, ts) in enumerate([(scaled_dmp, ts_scaled), (smooth_dmp, ts_smooth)]):
                ax = axes[plot_row, idx]
                
                if name == 'vel [rad/s]':
                    values = dmp.yds
                else:  # 'accl [rad/s${}^2$]'
                    values = dmp.ydds

                for i, dof in enumerate(range(7)):  # Assuming 7 DOF
                    color = colors[i]
                    limit = limits[name][i]
                    
                    # Highlight Violations
                    above_limit = np.where(values[:, i] > limit)
                    ax.scatter(ts[above_limit], values[above_limit, i], label="violation", color=color, marker='o' , s=10,  zorder=3, alpha=0.7)
                    
                    # plot_color = colors[i] if above_limit[0].size > 0 else 'royalblue'
                    plot_color = 'royalblue'

                    ax.plot(ts, values[:, i], linestyle="-", label="trajectory",  linewidth=1.5, color=plot_color, alpha=0.7)
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

                    
                # ax.xlabel('Time [s]')
                # ax.ylabel(name, labelpad=-5)

        axes[0, 0].set_ylabel('$\dot{q}_{s,dmp}$ [rad/s]' , labelpad=-5, fontsize=12)
        axes[0, 1].set_ylabel('$\ddot{q}_{s,dmp}$ [rad/s${}^2$]', labelpad=-3, fontsize=12)
        axes[1, 0].set_ylabel('$\dot{q}_{f,dmp}$ [rad/s]', labelpad=-5, fontsize=12)
        axes[1, 1].set_ylabel('$\ddot{q}_{f,dmp}$ [rad/s${}^2$]', labelpad=-5, fontsize=12)

        axes[1, 0].set_xlabel('Time [s]', fontsize=11)
        axes[1, 1].set_xlabel('Time [s]', fontsize=11)

        # Get the handles and labels
        handles, labels = axes[0,1].get_legend_handles_labels()

        # Use a dictionary to remove duplicates
        unique = {label: handle for handle, label in zip(handles, labels)}

        plt.legend(unique.values(), unique.keys(), loc='upper right', fontsize=9, markerscale=1)
        plt.tight_layout()
        plt.show()

    def plot_compare_jerks(self, scaled_dmp, smooth_dmp):
        ts_scaled = scaled_dmp.ts
        ts_smooth = smooth_dmp.ts

        data = {
            "Scaled DMP $\dddot{q}$ [rad/s${}^3$]": scaled_dmp.yddds,
            "Smooth DMP $\dddot{q}$ [rad/s${}^3$]": smooth_dmp.yddds
        }

        fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
        ax2 =  ax1.twinx()
        color1 = "steelblue"
        color2 = (0.5, 0.0, 0.0)
        for label, values in data.items():
            # Assuming values is a 2D array where each column represents a DOF
            for dof in range(values.shape[1]):
                if label.startswith('Scaled'):
                    ax1.plot(ts_scaled, values[:, dof], color=color1, linestyle="-", linewidth=2, alpha=0.7)
                    ax1.tick_params(axis='y', labelcolor=color1)
                else: # Smooth
                    ax2.plot(ts_smooth, values[:, dof], color=color2, linewidth=2,  alpha=0.6)
                    ax2.tick_params(axis='y', labelcolor=color2)
                    # ymin, ymax = ax2.get_ylim()
                    ax2.set_ylim(-450, 450)

        y_max_scaled = np.max(scaled_dmp.yddds)
        y_min_scaled = np.min(scaled_dmp.yddds)
        y_max_smooth = np.max(smooth_dmp.yddds)
        y_min_smooth = np.min(smooth_dmp.yddds)

        ax1.axhline(y=y_max_scaled, color=color1, linestyle='--')  # you can adjust color and linestyle as needed
        ax1.text(0.6, y_max_scaled - 50, f'Maximum: {y_max_scaled:.2f}', color="black", verticalalignment='top', bbox = dict(boxstyle="round", fc=color1, alpha=0.3))
        ax1.axhline(y=y_min_scaled, color=color1, linestyle='--')  # you can adjust color and linestyle as needed
        ax1.text(0.6, y_min_scaled + 55, f'Minimum: {y_min_scaled:.2f}', color="black", verticalalignment='bottom', bbox = dict(boxstyle="round", fc=color1, alpha=0.3))

        ax2.text(0.6, y_max_smooth + 20 , f'Maximum: {y_max_smooth:.2f}', color="black", verticalalignment='bottom', bbox = dict(boxstyle="round", fc=color2, alpha=0.3))
        ax2.axhline(y=y_max_smooth, color=color2, linestyle='--')  # you can adjust color and linestyle as needed
        ax2.axhline(y=y_min_smooth, color=color2, linestyle='--')  # you can adjust color and linestyle as needed
        ax2.text(0.6, y_min_smooth - 20, f'Minimum: {y_min_smooth:.2f}', color="black", verticalalignment='top', bbox = dict(boxstyle="round", fc=color2, alpha=0.3))

        # ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax1.set_ylabel("$\dddot{q}_{s,dmp}$ [rad/s${}^3$]", labelpad=-4, color=color1, fontsize=14)
        ax2.set_ylabel("$\dddot{q}_{f,dmp}$ [rad/s${}^3$]", labelpad=4, color=color2, fontsize=14)
        ax1.set_xlabel('Time [s]', labelpad=0, fontsize=12)
        # ax1.legend(loc='upper right', fontsize='small')
        
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)
        plt.show()