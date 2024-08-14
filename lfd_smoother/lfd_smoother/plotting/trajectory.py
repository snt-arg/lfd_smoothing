
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from pydrake.all import *
from lfd_smoother.util.robot import FR3Drake, YumiDrake

from lfd_storage.smoother_storage import SmootherStorage


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.default'] = 'regular'

class TrajectoryStock:

    def __init__(self):
        self.positions = None # To be filled later by Cartesian Analysis
        self.velocities = None # To be filled later by Cartesian Analysis
        self.accelerations = None # To be filled later by Cartesian Analysis
        self.jerks = None # To be filled later by Cartesian Analysis

    def import_from_lfd_storage(self,name, t_scale=0):
        self.storage = SmootherStorage(name)
        joint_trajectory = self.storage.import_original_traj()
        self._from_joint_trajectory(joint_trajectory, t_scale)
        self.normalize_t()

    def import_from_pydrake(self,name, t_scale=0):
        self.storage = SmootherStorage(name)
        traj = self.storage.import_pydrake_traj()
        self._from_pydrake(traj,t_scale)
        self.normalize_t()

    def import_from_dmp_joint_trajectory(self, name, t_scale=0):
        self.storage = SmootherStorage(name)
        joint_trajectory = self.storage.import_dmp_traj()
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

    def plot(self, save_path='/tmp'):

        ts = self.ts
        
        data = {
            "${q}_o$ [rad]": self.ys,
            "$\dot{q}_o$ [rad/s]": self.yds,
            "$\ddot{q}_o$ [rad/s${}^2$]": self.ydds,
            "$\dddot{q}_o$ [rad/s${}^3$]": self.yddds
        }
        
        fig, axs = plt.subplots(2, 2, figsize=(12,8))
        axs = axs.ravel()

        # Set a consistent style
        sns.set(style="whitegrid")
        color_palette = sns.color_palette("tab10")

        for idx, (label, values) in enumerate(data.items()):
            axs[idx].plot(ts, np.array(values))
            # axs[idx].set_title(label)
            axs[idx].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            axs[idx].set_ylabel(label, labelpad=10, fontsize=23)
            axs[idx].grid(True)

            if idx >= 2:  # Only set x-label for the bottom subplots
                axs[idx].set_xlabel('time [s]', labelpad=10, fontsize=12)
            
        plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
        fig_filename = f'{save_path}/org.pdf'
        plt.savefig(fig_filename)
        plt.close(fig)

    def plot_individually(self, save_path='/tmp/'):
        ts = self.ts

        data = {
            "${q}_o$ [rad]": self.ys,
            "$\dot{q}_o$ [rad/s]": self.yds,
            "$\ddot{q}_o$ [rad/s${}^2$]": self.ydds,
            "$\dddot{q}_o$ [rad/s${}^3$]": self.yddds
        }

        num_joints = len(self.ys[0])  # Assuming ys, yds, ydds, yddds are lists of arrays
        
        # Set a consistent style
        sns.set(style="whitegrid")
        color_palette = sns.color_palette("tab10")

        joint_labels = [f'joint{idx + 1}' for idx in range(num_joints)]

        for joint_idx in range(num_joints):
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.ravel()

            for idx, (label, values) in enumerate(data.items()):
                axs[idx].plot(ts, np.array(values)[:, joint_idx], color=color_palette[idx], label=joint_labels[joint_idx])
                axs[idx].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                axs[idx].set_ylabel(label, labelpad=10, fontsize=12)
                # axs[idx].set_title(label, fontsize=14)
                axs[idx].grid(True)
                
                if idx >= 2:  # Only set x-label for the bottom subplots
                    axs[idx].set_xlabel('time [s]', labelpad=10, fontsize=12)

                axs[idx].legend(loc="best", fontsize=10)

            # fig.suptitle(f'Joint {joint_idx + 1} Profiles', fontsize=16)
            plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
            # plt.subplots_adjust(top=0.8)  # Adjust the top padding to fit the suptitle
            # plt.show() 
            fig_filename = f'{save_path}/joint_{joint_idx + 1}_org.pdf'
            plt.savefig(fig_filename)
            plt.close(fig)  # Close the figure to free up memory
        

    def to_joint_trajectory(self, withvelacc=True):
        joint_trajectory = JointTrajectory()
        
        joint_trajectory.joint_names = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"]
        
        for i in range(len(self.ts)):
            point = JointTrajectoryPoint()
            point.positions = self.ys[i]
            
            if withvelacc:
                if len(self.yds) > i:
                    point.velocities = self.yds[i]
                if len(self.ydds) > i:
                    point.accelerations = self.ydds[i]
            
            point.time_from_start = rospy.Duration.from_sec(self.ts[i])
            joint_trajectory.points.append(point)
        
        return joint_trajectory
    
    def add_cartesian_analysis(self, robot_type="fr3"):
        self.cartesian = CartesianAnalysis(robot_type=robot_type)
        self.cartesian.from_trajectory(self)
        return self.cartesian


class CartesianAnalysis:

    def __init__(self, robot_type="fr3") -> None:
        if robot_type == "yumi_l":
            self.robot = YumiDrake(pkg_xml="/home/abrk/catkin_ws/src/lfd/lfd_smoothing/drake/yumi_drake/package.xml",
                    urdf_path="package://yumi_drake/urdf/yumi_left.urdf")
            self.visualizer_context = self.robot.visualizer.GetMyContextFromRoot(self.robot.context)
            self.ee_body = self.robot.plant.GetBodyByName("yumi_link_7_l")
        elif robot_type == "fr3":
            self.robot = FR3Drake(pkg_xml="/home/abrk/catkin_ws/src/lfd/lfd_smoothing/drake/franka_drake/package.xml",
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

        times = np.array(traj.ts)
        self.accelerations = np.gradient(self.velocities, times)
        self.jerks = np.gradient(self.accelerations, times)
        traj.accelerations = self.accelerations
        traj.jerks = self.jerks
    
    
    def plot(self, save_path='/tmp'):
        ts = self.ts
        # positions = self.positions
        # print(positions.shape)
        # velocities = np.gradient(self.positions, ts, axis=0)
        # accelerations = np.gradient(velocities, ts, axis=0)
        # jerks = np.gradient(accelerations, ts, axis=0)


        data = {
            "Position [units]": self.positions,
            "Velocity [units/s]": self.velocities,
            "Acceleration [units/s^2]": self.accelerations,
            "Jerk [units/s^3]": self.jerks
        }

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.ravel()

        # Set a consistent style
        sns.set(style="whitegrid")
        color_palette = sns.color_palette("tab10")

        for idx, (label, values) in enumerate(data.items()):
            axs[idx].plot(ts, np.array(values), color=color_palette[idx])
            axs[idx].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs[idx].set_ylabel(label, labelpad=10, fontsize=12)
            axs[idx].grid(True)

            if idx >= 4:  # Only set x-label for the bottom subplots
                axs[idx].set_xlabel('Time [s]', labelpad=10, fontsize=12)

        plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
        fig_filename = f'{save_path}/plot.pdf'
        plt.savefig(fig_filename)
        plt.close(fig)


    def plot_3d(self):
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
