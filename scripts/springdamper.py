#!/usr/bin/env python3
import numpy as np
import rospy
import pickle
import os
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64, Float64MultiArray
import matplotlib.pyplot as plt

from datetime import datetime



from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem

from pydrake.all import *

def normalize_trajectory(traj):
    duration = traj.end_time() - traj.start_time()
    basis = traj.basis()
    scaled_knots = [(knot / duration) for knot in basis.knots()]
    ntraj = BsplineTrajectory(
        BsplineBasis(basis.order(), scaled_knots),
        traj.control_points())   
    return duration , ntraj 


class Particle:
    
    def __init__(self, duration):
        self.v0 = 1.0/duration
        self.duration = duration
        self.reset()
        
    def reset(self):
        self.t = 0
        self.s = 0
        self.v = self.v0
        self.reached = False

    def position(self,t,acceleration):
        self.a = acceleration
        dt = t - self.t

        if self.v < self.v0:
            self.a += 1 * self.duration * (self.v0 - self.v)
        self.v = self.a*dt + self.v
        self.constrain_v()
        ds = 0.5*self.a*(dt**2) + self.v*dt

        self.t = t
        self.s += ds
        if self.s > 1: 
            self.s = 1
            self.reached = True
        return self.s
    
    def constrain_v(self):
        if self.v < 0.2 * self.v0: 
            self.v = 0.2 * self.v0
            self.a = 0
        if self.v > self.v0:
            self.v = self.v0
            self.a = 0



class Recorder(object):
    def __init__(self, ntraj):
        self.traj = ntraj
        self.ts = []
        self.ss = []

    def update(self, t, s):
        self.ts.append(t)
        self.ss.append(s)

    def plot_trajectory(self):
        
        traj = self.traj
        ts = np.array(self.ts)
        ss = np.array(self.ss)
        
        position = []
        velocity = []
        acceleration = []
        jerk = []
        
        for s in ss:
            position.append(traj.value(s))
            velocity.append(traj.MakeDerivative().value(s))
            acceleration.append(traj.MakeDerivative(2).value(s))
            jerk.append(traj.MakeDerivative(3).value(s))
        
        # Plot position, velocity, and acceleration
        fig, axs = plt.subplots(1, 5, figsize=(15,3))
        
        # Position plot
        axs[0].plot(ts, np.array(position).squeeze(axis=2))
        axs[0].set_ylabel('Position')
        axs[0].set_xlabel('Time')
        
        # Velocity plot
        axs[1].plot(ts, np.array(velocity).squeeze(axis=2))
        axs[1].set_ylabel('Velocity')
        axs[1].set_xlabel('Time')
        
        # Acceleration plot
        axs[2].plot(ts, np.array(acceleration).squeeze(axis=2))
        axs[2].set_ylabel('Acceleration')
        axs[2].set_xlabel('Time')
        
        # Jerk plot
        axs[3].plot(ts, np.array(jerk).squeeze(axis=2))
        axs[3].set_ylabel('Jerk')
        axs[3].set_xlabel('Time')

        # Jerk plot
        axs[4].plot(ts, ss)
        axs[4].set_ylabel('s')
        axs[4].set_xlabel('Time')

        plt.tight_layout()
        current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = '/tmp/' + f'output_correction_{current_datetime}.svg'
        plt.savefig(filename, format='svg')
        plt.show() 


class AccelerationSystem(object):
    def __init__(self, in_topic, duration=20, acceleration_range = [0,-1]):
        self.in_topic = in_topic
        self.acceleration_range = acceleration_range

        self.y_attr = -1
        self.init_dynamic_system(tau=1, y_init=0, y_attr=0, alpha=20)

        rospy.Subscriber(self.in_topic, Float64, self.cb_joystick)

        self.particle = Particle(duration)
        self.duration = duration

        self.t_start = rospy.Time.now().to_sec()
        self.t = self.t_start

    def init_dynamic_system(self, tau, y_init, y_attr, alpha):
        self.spring_system = SpringDamperSystem(tau, y_init, y_attr, alpha)
        (self.x,self.xd) = self.spring_system.integrate_start()

    def cb_joystick(self, message):
        pid = 0
        # pid = abs(2 * self.duration * (self.particle.v0 - self.particle.v))
        # if pid < 0.1: pid = 0
        self.y_attr = self.map_range(message.data) + pid

    def map_range(self,value):
        input_range = [0, 1]  # input range
        output_range = self.acceleration_range  # desired output range

        # line equation parameters
        slope = (output_range[1] - output_range[0]) / (input_range[1] - input_range[0])
        intercept = output_range[0] - slope * input_range[0]

        return slope * value + intercept

    def run(self):
        dt = rospy.Time.now().to_sec() - self.t
        self.spring_system.y_attr = np.array([self.y_attr])
        (self.x,self.xd) = self.spring_system.integrate_step(dt,self.x)
        self.t = rospy.Time.now().to_sec()
        t = self.t - self.t_start
        s=self.particle.position(t,self.x[0])
        return t,s

if __name__ == '__main__':
            # Initialize the node
    rospy.init_node('republishing_node', anonymous=True)
    os.chdir(rospy.get_param("~working_dir"))

    with open("smoothpicknplaceee1.pickle", 'rb') as file:
        traj = pickle.load(file)
    duration , ntraj = normalize_trajectory(traj)

    recorder = Recorder(ntraj)
    pub_robot = rospy.Publisher("/position_joint_controller/command", Float64MultiArray , queue_size=0)

    acc_sys = AccelerationSystem('/joy_filtered', duration=5)

    s=0
    rate = rospy.Rate(50)

    while (not rospy.is_shutdown()) and s!=1:
        t,s = acc_sys.run()
        msg = Float64MultiArray()
        msg.data = ntraj.value(s)
        pub_robot.publish(msg)
        recorder.update(t,s)
        rate.sleep()
    
    recorder.plot_trajectory()
