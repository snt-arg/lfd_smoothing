#!/usr/bin/env python3
import numpy as np
import rospy
import pickle
import os
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64, Float64MultiArray
import matplotlib.pyplot as plt
from scipy import interpolate

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
    
    def __init__(self, duration, v_bounds = [0.2, 1]):
        self.v0 = 1.0/duration
        self.duration = duration
        self.v_bounds = np.array(v_bounds)
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
        if self.v < self.v_bounds[0] * self.v0: 
            self.v = self.v_bounds[0] * self.v0
            self.a = 0
        if self.v > self.v_bounds[1] * self.v0:
            self.v = self.v_bounds[1] * self.v0
            self.a = 0



class Recorder(object):
    def __init__(self, ntraj, ts_original):
        self.traj = ntraj

        self.ts_original = ts_original
        self.ss_original = ts_original / np.max(ts_original)
        self.duration = np.max(ts_original)

        self.ts = []
        self.ss = []
        self.commands = []

    def update(self, t, s, command):
        self.ts.append(t)
        self.ss.append(s)
        self.commands.append(command)
    
    def export_metadata(self):
        metadata = {}
        metadata["ts"] = self.ts
        metadata["ss"] = self.ss
        metadata["commands"] = self.commands
        return metadata
    
    def export_tolerances(self, min, max):
        inputs = self.map_range(np.array(self.commands))
        self.tols = self.tolerance(inputs,min,max)

        # Calculate the lower and upper tolerances
        lower_tolerance = -self.tols
        upper_tolerance = self.tols

        # Create the plot
        plt.fill_between(self.ts, lower_tolerance, upper_tolerance, color='grey', alpha=0.5)
        plt.xlabel('xs')
        plt.ylabel('ys')
        plt.title('Tolerance Plot')
        plt.legend()
        plt.show()        

    
    def tolerance(self,x, min, max):
        return (max-min)*(1-((1-x)**1.9)) + min


    def map_range(self,value):
        input_range = [0, -1]  # input range
        output_range = [1,0]  # desired output range

        # line equation parameters
        slope = (output_range[1] - output_range[0]) / (input_range[1] - input_range[0])
        intercept = output_range[0] - slope * input_range[0]

        return slope * value + intercept
    
    def plot_commands(self):
        plt.figure(figsize=(10,6))
        plt.plot(self.ts, self.commands, 'o-')
        plt.xlabel('ts')
        plt.ylabel('commands')
        plt.title('Commands vs ts')
        plt.grid(True)
        plt.show()        
    
    def export_new_timings(self):
        self.func_s_t = interpolate.interp1d(self.ss, self.ts, kind='linear', fill_value="extrapolate")
        self.ts_new = self.func_s_t(self.ss_original)
        self.ss_new = self.ts_new / np.max(self.ts_new)
        return self.ss_new   

    def plot_timings(self):
        plt.scatter(self.ss_original, self.ss_new)
        plt.xlabel('ss_query')
        plt.ylabel('ts_query')
        plt.title('Scatter plot of ss_query vs ts_query')
        plt.show()

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

        self.y_attr = 0
        self.init_dynamic_system(tau=0.1, y_init=0, y_attr=0, alpha=20)

        rospy.Subscriber(self.in_topic, Float64, self.cb_joystick)

        self.particle = Particle(duration)
        self.duration = duration

        self.t_start = rospy.Time.now().to_sec()
        self.t = self.t_start

    def init_dynamic_system(self, tau, y_init, y_attr, alpha):
        self.spring_system = SpringDamperSystem(tau, y_init, y_attr, alpha)
        (self.x,self.xd) = self.spring_system.integrate_start()

    def cb_joystick(self, message):
        self.y_attr = self.map_range(message.data)

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
        return t,s,self.x[0]


def import_data(demo_name):
    with open("traj/{}".format(demo_name), 'rb') as file:
        traj = pickle.load(file)
    with open("timing/{}".format(demo_name), 'rb') as file:
        timings = pickle.load(file)
    return traj, timings

def export_data(demo_name, timings_new, metadata):
    with open("timing_new/{}".format(demo_name), 'wb') as file:
       pickle.dump(timings_new,file)
    with open("metadata/{}".format(demo_name), 'wb') as file:
       pickle.dump(metadata,file)    


if __name__ == '__main__':
            # Initialize the node
    rospy.init_node('republishing_node', anonymous=True)
    os.chdir(rospy.get_param("~working_dir"))

    demo_name = rospy.get_param("~demo_name") + ".pickle"

    traj, timings = import_data(demo_name)
    duration , ntraj = normalize_trajectory(traj)

    recorder = Recorder(ntraj,timings)
    pub_robot = rospy.Publisher("/position_joint_controller/command", Float64MultiArray , queue_size=0)

    acc_sys = AccelerationSystem('/joy_filtered', duration=5)

    s=0
    rate = rospy.Rate(50)

    while (not rospy.is_shutdown()) and s!=1:
        t,s,command = acc_sys.run()
        msg = Float64MultiArray()
        msg.data = ntraj.value(s)
        pub_robot.publish(msg)
        recorder.update(t,s,command)
        rate.sleep()

    metadata = recorder.export_metadata()
    timings_new = recorder.export_new_timings()

    export_data(demo_name, timings_new, metadata)
    recorder.export_tolerances(0.01,0.05)

