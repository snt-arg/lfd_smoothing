
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle

from lfd_interface.srv import GetDemonstration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from pydrake.all import *
from lfd_smoother.util.fr3_drake import FR3Drake


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
    
    def add_cartesian_analysis(self):
        self.cartesian = CartesianAnalysis()
        self.cartesian.from_trajectory(self)
        return self.cartesian


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
