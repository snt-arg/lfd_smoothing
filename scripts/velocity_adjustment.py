#!/usr/bin/env python3

import rospy
import pickle
import os
import numpy as np

from pydrake.all import *

from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64



def normalize_trajectory(traj):
    duration = traj.end_time() - traj.start_time()
    basis = traj.basis()
    scaled_knots = [(knot / duration) for knot in basis.knots()]
    ntraj = BsplineTrajectory(
        BsplineBasis(basis.order(), scaled_knots),
        traj.control_points())   
    return duration , ntraj 


class Particle:
    
    def __init__(self, v0):
        self.v0 = v0
        self.v = v0
        self.reset()
        self.sub = rospy.Subscriber('/joy_filtered', Float64, self.callback)
        self.acceleration = 0
        
    def reset(self):
        self.t = 0
        self.x = 0
        self.v = self.v0

    def callback(self, msg :  Float64):
        self.acceleration = 0.2* msg.data
        # print(self.acceleration)
         
    def acc(self, t):
        # return 0
        return 3 * np.maximum(0, np.sin(3*t) + 1)
    
    def positions(self,ts):
        self.reset()
        xs = []
        for t in ts:
            xs.append(self.position(t))
            
        return np.array(xs)

    def position(self,t):
            dt = t - self.t
            self.v = self.acceleration*dt + self.v
            dx = 0.5*self.acceleration*(dt**2) + self.v*dt
            self.t = t
            self.x +=dx
            if self.x > 1: self.x = 1
            return self.x

if __name__ == '__main__':

    rospy.init_node('velocity_adjustment_node')
    os.chdir(rospy.get_param("~working_dir"))

    with open("smoothpicknplaceee0.pickle", 'rb') as file:
        traj = pickle.load(file)

    duration , ntraj = normalize_trajectory(traj)

    pub_robot = rospy.Publisher("/position_joint_controller/command", Float64MultiArray , queue_size=0)

    timer = Particle(0.05)
    rate = rospy.Rate(50)
    start = rospy.Time().now()

    while(not rospy.is_shutdown()):
        msg = Float64MultiArray()
        t = (rospy.Time.now() - start).to_sec()
        msg.data = traj.value(timer.position(t))
        pub_robot.publish(msg)
        rate.sleep()



    
    # rospy.spin()



    