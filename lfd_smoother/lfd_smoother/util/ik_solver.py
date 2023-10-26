
import numpy as np

from pydrake.all import *

from lfd_smoother.util.robot import FR3Drake, YumiDrake


class IKSolver:

    def __init__(self, config):
        self.config = config
        if config["robot_type"] == "fr3":
            self.robot = FR3Drake(pkg_xml=self.config["pkg_xml"],
                            urdf_path=self.config["urdf_path"],
                            gripper_frame=self.config.get("ee_frame",None))
        elif config["robot_type"] == "yumi":
            self.robot = YumiDrake(pkg_xml=self.config["pkg_xml"],
                            urdf_path=self.config["urdf_path"],
                            gripper_frame=self.config.get("ee_frame",None))
            
        # pose = self.robot.create_waypoint("pose", [0.303, 0, 0.482])
        # pose.set_rotation(Quaternion([0,-1,0,0]))

    def run(self, position, orientation, q_init = None):
        """
        position = [x,y,z]
        orientation = [w,x,y,z]
        """

        pose = self.robot.create_waypoint("pose", position)
        pose.set_rotation(Quaternion(orientation))

        num_q = self.robot.plant.num_positions()
        q0 = np.zeros(num_q)

        if q_init is not None:
            q_init = np.array(q_init)
            num_q = q_init.size
            q0[:num_q] = q_init

        self.ik_solver = InverseKinematics(self.robot.plant, True)
        self.ik_solver.AddPositionConstraint(self.robot.plant.world_frame(),
                                             pose.translation(), self.robot.gripper_frame,
                                             [0, 0, 0], [0, 0, 0])
        
        self.ik_solver.AddOrientationConstraint(self.robot.gripper_frame, RotationMatrix(),
                                                self.robot.plant.world_frame(), pose.rotation(), 0)
        
        self.prog = self.ik_solver.get_mutable_prog()
        self.prog.AddCost((self.ik_solver.q()-q0).dot(self.ik_solver.q()-q0))


        result = Solve(self.prog)
        q = result.GetSolution(self.ik_solver.q())
        self.visualize(q)
        return q[:num_q]


    def visualize(self,q):
        self.robot.visualize(q)








