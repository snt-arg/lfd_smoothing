#!/usr/bin/env python3

import rospy

from lfd_interface.srv import GetDemonstration

from lfd_smoother.api.trajectory_smoother import TrajectorySmoother

if __name__ == '__main__':

    rospy.init_node('trajectory_smoother_node')

    smoother_config = rospy.get_param("~smoother_config")
    smoother = TrajectorySmoother(smoother_config)

    rospy.wait_for_service("get_demonstration")

    lfd_storage = rospy.ServiceProxy("get_demonstration", GetDemonstration)
    
    resp = lfd_storage(name="picknplace0")
    demonstration = resp.Demonstration

    smoother.run(demonstration)
    
    rospy.spin()



    