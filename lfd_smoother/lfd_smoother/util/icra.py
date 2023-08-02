
import os
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



class TrajectoryStock:

    def __init__(self):
        pass

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

        self.tolerances = np.array(self.tolerances)
        self.tol_trans = self.tolerances[:,0]
        self.tol_rot = self.tolerances[:,1]
    
    def plot_traj_with_tols(self, traj):
        # self.func_s_tol = interpolate.interp1d(self.ss_new, self.tol_trans, kind='linear', fill_value="extrapolate")
        
        # tolerances = self.func_s_tol(traj.ss)

        fig, axs = plt.subplots(3, 1, sharex=True)

        x, y, z = self.demo.positions.T  # Transpose to get x, y, z
        X, Y, Z = traj.positions.T

        # Again, assuming your tolerances are symmetrical
        axs[0].plot(traj.ss, X, label='X')
        axs[0].plot(self.ss_new, x, label='X_Original')
        axs[0].fill_between(self.ss_new, x - self.tol_trans, x + self.tol_trans, color='gray', alpha=0.5)

        axs[1].plot(traj.ss, Y, label='Y')
        axs[1].plot(self.ss_new, y, label='Y_Original')
        axs[1].fill_between(self.ss_new, y - self.tol_trans, y + self.tol_trans, color='gray', alpha=0.5)

        axs[2].plot(traj.ss, Z, label='Z')
        axs[2].plot(self.ss_new, z, label='Z_Original')
        axs[2].fill_between(self.ss_new, z - self.tol_trans, z + self.tol_trans, color='gray', alpha=0.5)

        axs[2].set_xlabel('Normalized time')
        axs[0].set_ylabel('X position')
        axs[1].set_ylabel('Y position')
        axs[2].set_ylabel('Z position')

        for ax in axs:
            ax.legend()

        plt.tight_layout()
        plt.show()        