
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
            "pos [rad]": self.ys,
            "vel [rad/s]": self.yds,
            "accl [rad/s${}^2$]": self.ydds,
            "jerk [rad/s${}^3$]": self.yddds
        }
        
        fig, axs = plt.subplots(2, 2, figsize=(5,4))
        axs = axs.ravel()

        for idx, (label, values) in enumerate(data.items()):
            axs[idx].plot(ts, np.array(values))
            # axs[idx].set_title(label)
            axs[idx].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            axs[idx].set_ylabel(label, labelpad=-4)
            
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

    def plot_t_command(self):
        plt.plot(self.ts, self.commands, label='$s_r(t)$', color='firebrick')
        # Grids and Background
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')        
        plt.xlabel('Time (t)')
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
        plt.xlabel('Time (t)')
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
        plt.xlabel('Time (t)')
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
        # plt.xlabel('Time (t)')
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

        fig, axs = plt.subplots(2, 1, sharex=True)

        axs[0].plot(Ts, X, label='X_High_Tol' , color='firebrick', linewidth=2)
        axs[0].plot(TTs, XX, label='X_Low_Tol'  , color='royalblue', linewidth=2)
        axs[0].fill_between(self.ss_original * correct_traj.ts[-1], x - self.tol_trans, x + self.tol_trans, color='firebrick', alpha=0.2)
        axs[0].fill_between(self.ss_original * smooth_traj.ts[-1], x - tol_default, x + tol_default, color='royalblue', alpha=0.2)


        velocities = np.array(correct_traj.velocities)
        times = np.array(correct_traj.ts)
        accelerations = np.gradient(velocities, times)
        jerks = np.gradient(accelerations, times)

        axs[1].plot(correct_traj.ts, np.abs(jerks), label='Jerk_High_Tol', color='firebrick')

        velocities = np.array(smooth_traj.velocities)
        times = np.array(smooth_traj.ts)
        accelerations = np.gradient(velocities, times)
        jerks = np.gradient(accelerations, times) 

        axs[1].plot(smooth_traj.ts, np.abs(jerks), label='Jerk_Low_Tol', color='royalblue')
        ellipse = Ellipse((0.13, 40), width=0.3, height=80, edgecolor='green', facecolor='none', linestyle='--')
        axs[1].add_patch(ellipse)

        axs[1].set_xlabel('Time [s]')
        axs[0].set_ylabel('X [m]')
        axs[1].set_ylabel('End-effector Jerk [$m/s^3$]')

        for ax in axs:
            ax.legend()

        plt.tight_layout()
        plt.show()


        # axs[1].plot(correct_ntraj.ts, Y, label='Y_High_Tol', color='firebrick')
        # axs[1].plot(smooth_ntraj.ts, YY, label='Y_Low_Tol', color='royalblue')
        # axs[1].fill_between(self.ss_original, y - self.tol_trans, y + self.tol_trans, color='firebrick', alpha=0.2)
        # axs[1].fill_between(self.ss_original, y - tol_default, y + tol_default, color='royalblue', alpha=0.2)
        
        # axs[2].plot(correct_ntraj.ts, Z, label='Z_High_Tol', color='firebrick')
        # axs[2].plot(smooth_ntraj.ts, ZZ, label='Z_Low_Tol', color='royalblue')
        # axs[2].fill_between(self.ss_original, z - self.tol_trans, z + self.tol_trans, color='firebrick', alpha=0.2)
        # axs[2].fill_between(self.ss_original, z - tol_default, z + tol_default, color='royalblue', alpha=0.2)
    
        # axs[1].set_ylabel('Y position')
        # axs[2].set_ylabel('Z position')

    def plot_traj_with_tols(self, correct_traj):

        self.ts_new = self.ss_new * correct_traj.ts[-1]
        x, y, z = self.demo.positions.T  
        X, Y, Z = correct_traj.positions.T

        colors = [
            "maroon",
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