#!/usr/bin/env python3
import numpy as np
import rospy
import pickle
import os
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64, Float64MultiArray
import matplotlib.pyplot as plt
from scipy import interpolate
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from datetime import datetime
from dmpbbo.dynamicalsystems.SpringDamperSystem import SpringDamperSystem

from pydrake.all import *


from lfd_smoother.core.refinement import AccelerationSystem, Refinement

def normalize_trajectory(traj):
    duration = traj.end_time() - traj.start_time()
    basis = traj.basis()
    scaled_knots = [(knot / duration) for knot in basis.knots()]
    ntraj = BsplineTrajectory(
        BsplineBasis(basis.order(), scaled_knots),
        traj.control_points())   
    return duration , ntraj 


def import_data(demo_name):
    with open("traj/{}".format(demo_name), 'rb') as file:
        traj = pickle.load(file)
    with open("timing/{}".format(demo_name), 'rb') as file:
        timings = pickle.load(file)
    return traj, timings

def export_data(demo_name, timings_new, tolerances, metadata):
    with open("timing_new/{}".format(demo_name), 'wb') as file:
       pickle.dump(timings_new,file)
    with open("metadata/{}".format(demo_name), 'wb') as file:
       pickle.dump(metadata,file)    
    with open("tolerances/{}".format(demo_name), 'wb') as file:
       pickle.dump(tolerances,file) 


def ctrl_with_velocity(pub_vel, msg):
    for i in range(0,7):
        pub_vel[i].publish(msg.data[i])


if __name__ == '__main__':

    rospy.init_node('republishing_node', anonymous=True)
    os.chdir(rospy.get_param("~working_dir"))
    demo_name = rospy.get_param("~demo_name") + ".pickle"

    traj, timings = import_data(demo_name)
    duration , ntraj = normalize_trajectory(traj)
    ntraj_v = ntraj.MakeDerivative()

    recorder = Refinement(ntraj,timings)
    pub_pos = rospy.Publisher("/velocity_joint_controller/command", Float64MultiArray , queue_size=0)

    pub_vel=[]
    for i in range(0,7):
        pub_vel.append(rospy.Publisher('/joint%s_position_controller/command'%(i+1), Float64, queue_size=0))
    
    pub_traj = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=0)
    joint_names  = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"]

    acc_sys = AccelerationSystem('/joy_filtered', duration=5)

    s=0
    rate = rospy.Rate(50)

    while (not rospy.is_shutdown()) and s!=1:
        t,s,sd,command, cmd_raw = acc_sys.run()
        msg = Float64MultiArray()
        # msg.data = ntraj_v.value(s) * sd
        # pub_pos.publish(msg)
        # msg.data = ntraj.value(s)
        # ctrl_with_velocity(pub_vel,msg)
        
        traj = JointTrajectory()
        traj.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = ntraj.value(s)
        point.velocities = ntraj_v.value(s) * sd
        point.time_from_start = rospy.Duration(0.2)
        traj.points.append(point)
        pub_traj.publish(traj)
        recorder.update(t,s,command, cmd_raw)
        rate.sleep()
    print(t)

    metadata = recorder.export_metadata()
    timings_new = recorder.export_new_timings()
    tolerances = recorder.export_tolerances(0.01,0.05,0.1,0.3)

    export_data(demo_name, timings_new, tolerances, metadata)
    rospy.spin()
