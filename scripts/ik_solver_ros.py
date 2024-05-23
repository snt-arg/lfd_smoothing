#!/usr/bin/env python3

import numpy as np
import rospy

import actionlib
from lfd_interface.msg import PlanPoseAction, PlanPoseFeedback, PlanPoseResult, PlanPoseGoal
from lfd_interface.msg import PlanJointAction, PlanJointGoal

from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from lfd_smoothing.srv import IKService, IKServiceResponse

from lfd_smoother.util.ik_solver import IKSolver

class IKServer:
    def __init__(self, name):
        self.ik_flag = False
        self.service = rospy.Service(name, IKService, self.handle_request)
        ik_solver_config = rospy.get_param("~ik_solver_config")
        self.solver = IKSolver(ik_solver_config)
    
    def handle_request(self,req):
        self.req = req
        self.ik_flag = True
        while self.ik_flag:
            pass
        q = self.q        
        return IKServiceResponse(q_out=q)
    
    def ik_cb(self):
        self.q = self.solve_ros(self.req.end_effector_pose, self.req.q_init)
        self.ik_flag = False

    def solve_ros(self, end_effector_pose: Pose,
                    q_init: JointTrajectoryPoint):
        
        position = [end_effector_pose.position.x, 
                    end_effector_pose.position.y, end_effector_pose.position.z]
        orientation = [end_effector_pose.orientation.w, end_effector_pose.orientation.x,
                        end_effector_pose.orientation.y, end_effector_pose.orientation.z]
        # Normalize the orientation quaternion
        orientation /= np.linalg.norm(orientation)
        q_init = q_init.positions

        q_values = self.solve(position, orientation, q_init)
        
        q = JointTrajectoryPoint()
        q.positions = q_values
        return q
    
    def solve(self, position, orientation, q_init = None):
        """
        position = [x,y,z]
        orientation = [w,x,y,z]
        """
        return self.solver.run(position, orientation, q_init)


class PlanPoseActionServer:
    _feedback =  PlanPoseFeedback()
    _result = PlanPoseResult()

    def __init__(self, name, ik_server):
        self.action_name = name
        self.ik_server = ik_server
        self._as_planpose = actionlib.SimpleActionServer(self.action_name, 
                                                PlanPoseAction, execute_cb=self.execute_cb, 
                                                auto_start = False)
        self._ac_planjoint = actionlib.SimpleActionClient("/plan_joint", PlanJointAction)
        self._ac_planjoint.wait_for_server()
        
        self.ik_flag = False
        self._as_planpose.start()
    
    def ik_cb(self):
        self.q = self.ik_server.solve_ros(self.goal.pose, self.goal.q_init)
        self.ik_flag = False

    def execute_cb(self, goal : PlanPoseGoal):
        self.goal = goal
        self.ik_flag = True
        while self.ik_flag:
            pass
        q = self.q
        self._feedback.joint_position = q
        self._as_planpose.publish_feedback(self._feedback)

        goal_planjoint = PlanJointGoal(joint_position=q)
        self._ac_planjoint.send_goal(goal_planjoint)
        self._ac_planjoint.wait_for_result()

        success = self._ac_planjoint.get_result().success
        self._result.success = success
        self._as_planpose.set_succeeded(self._result)


if __name__ == '__main__':

    rospy.init_node('ik_solver')

    ik_server = IKServer('/ik_service')
    planpose = PlanPoseActionServer("/plan_pose", ik_server)

    r =rospy.Rate(50)
    while not rospy.is_shutdown():
        if planpose.ik_flag:
            planpose.ik_cb()
        if ik_server.ik_flag:
            ik_server.ik_cb()
        r.sleep()





# pose = robot.create_waypoint("pose", [0.303, 0, 0.482])
# pose.set_rotation(Quaternion([0,-1,0,0]))
# q0 = np.zeros(robot.plant.num_positions())
# q0 = [0.00112954, -0.787257, -0.000108279, -2.36501, 0.00214555, 1.55991, 0.766295,1,1]
