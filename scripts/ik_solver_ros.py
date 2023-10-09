#!/usr/bin/env python3

import numpy as np
import rospy

from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from lfd_smoothing.srv import IKService, IKServiceResponse

from lfd_smoother.util.ik_solver import IKSolver


class IKServer:
    def __init__(self):
        # self.service = rospy.Service('ik_service', IKService, self.handle_request)
        ik_solver_config = rospy.get_param("~ik_solver_config")
        self.solver = IKSolver(ik_solver_config)

    def solve_ros(self, end_effector_pose: Pose,
                    q_init: JointTrajectoryPoint):
        
        position = [end_effector_pose.position.x, 
                    end_effector_pose.position.y, end_effector_pose.position.z]
        orientation = [end_effector_pose.orientation.w, end_effector_pose.orientation.x,
                        end_effector_pose.orientation.y, end_effector_pose.orientation.z]
        q_init = q_init.positions

        q_values = self.solver.run(position, orientation, q_init)
        
        q = JointTrajectoryPoint()
        q.positions = q_values
        return q
    
    def solve(self, position, orientation, q_init = None):
        """
        position = [x,y,z]
        orientation = [w,x,y,z]
        """
        return self.solver.run(position, orientation, q_init)
    


if __name__ == '__main__':

    rospy.init_node('ik_solver')

    ik_server = IKServer()
    print(ik_server.solve([0.303, 0, 0.482], [0,-1,0,0], 
                    [0.00112954, -0.787257, -0.000108279, -2.36501, 0.00214555, 1.55991, 0.766295])
    )

    rospy.spin()








# pose = robot.create_waypoint("pose", [0.303, 0, 0.482])
# pose.set_rotation(Quaternion([0,-1,0,0]))
# q0 = np.zeros(robot.plant.num_positions())
# q0 = [0.00112954, -0.787257, -0.000108279, -2.36501, 0.00214555, 1.55991, 0.766295,1,1]
