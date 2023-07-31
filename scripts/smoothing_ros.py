#!/usr/bin/env python3

import os
import rospy
import pickle

from trajectory_msgs.msg import JointTrajectoryPoint
from lfd_interface.msg import DemonstrationMsg
from lfd_interface.srv import GetDemonstration, DemoCount

from lfd_smoother.api.trajectory_smoother import TrajectorySmoother


def export_demonstration(template : DemonstrationMsg, ts, ys, yds, ydds, is_correction=False):
    template.joint_trajectory.points.clear()
    template.pose_trajectory.points.clear()
    
    for i in range(ts.shape[0]):
        point = JointTrajectoryPoint()
        point.positions = list(ys[i,:,0])
        point.velocities = list(yds[i,:,0])
        point.accelerations = list(ydds[i,:,0])
        point.time_from_start = rospy.Duration.from_sec(ts[i])
        template.joint_trajectory.points.append(point) 
    if is_correction:
        template.name = "correct" + template.name
    else:
        template.name = "smooth" + template.name
    return template

def import_correction_data(demo_name):
    demo_name = "smooth" + demo_name
    try:
        with open("timing_new/" + demo_name +"{}".format(i) + ".pickle", 'rb') as f:
            timings = pickle.load(f)
        print("File opened and data loaded successfully.")
    except FileNotFoundError:
        print("File does not exist.")
        timings = None
    
    try:
        with open("tolerances/" + demo_name +"{}".format(i) + ".pickle", 'rb') as f:
            tolerances = pickle.load(f)
        print("File opened and data loaded successfully.")
    except FileNotFoundError:
        print("File does not exist.")
        tolerances = None

    return timings, tolerances    



if __name__ == '__main__':

    rospy.init_node('trajectory_smoother_node')
    
    os.chdir(rospy.get_param("~working_dir"))
    
    correction = rospy.get_param("~correction")


    demo_name = rospy.get_param("~demo_name")
    sc_demo_count = rospy.ServiceProxy("fetch_demo_count", DemoCount)
    resp = sc_demo_count(name=demo_name)
    demo_count = resp.count

    smoother_config = rospy.get_param("~smoother_config")
    smoother = TrajectorySmoother(smoother_config)

    rospy.wait_for_service("get_demonstration")
    sc_lfd_storage = rospy.ServiceProxy("get_demonstration", GetDemonstration)

    pub_save_demo = rospy.Publisher("save_demonstration", DemonstrationMsg , queue_size=1)
    
    for i in range(demo_count):
        resp = sc_lfd_storage(name=demo_name +"{}".format(i))
        demonstration = resp.Demonstration
        smoother.read_demo_ros(demonstration)
        if correction:
            timings,tolerances = import_correction_data(demo_name)
        else:
            timings = None
            tolerances = None
        smoother.run(timings=timings, tolerances=tolerances)
        ts, ys, yds, ydds = smoother.export_raw()
        demo_smooth = export_demonstration(demonstration, ts, ys, yds, ydds, is_correction=correction)
        pub_save_demo.publish(demo_smooth)
        traj = smoother.smoother.trajopts[0].ReconstructTrajectory(smoother.smoother.result)
        timings = smoother.timings
        with open("traj/{}.pickle".format(demo_smooth.name), 'wb') as file:
            pickle.dump(traj,file)
        with open("timing/{}.pickle".format(demo_smooth.name), 'wb') as file:
            pickle.dump(timings,file)
    
    # rospy.spin()



    