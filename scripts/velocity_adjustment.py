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

if __name__ == '__main__':

    rospy.init_node('republishing_node', anonymous=True)
    os.chdir(rospy.get_param("~working_dir"))
    demo_name = rospy.get_param("~demo_name") + ".pickle"

    traj, timings = import_data(demo_name)
    duration , ntraj = normalize_trajectory(traj)

    recorder = Refinement(ntraj,timings)
    pub_robot = rospy.Publisher("/position_joint_controller/command", Float64MultiArray , queue_size=0)

    acc_sys = AccelerationSystem('/joy_filtered', duration=5)

    s=0
    rate = rospy.Rate(50)

    while (not rospy.is_shutdown()) and s!=1:
        t,s,command, cmd_raw = acc_sys.run()
        msg = Float64MultiArray()
        msg.data = ntraj.value(s)
        pub_robot.publish(msg)
        recorder.update(t,s,command, cmd_raw)
        rate.sleep()

    metadata = recorder.export_metadata()
    timings_new = recorder.export_new_timings()
    tolerances = recorder.export_tolerances(0.01,0.05,0.1,0.3)

    export_data(demo_name, timings_new, tolerances, metadata)
