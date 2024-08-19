import numpy as np

from pydrake.all import *


class DrakeRobot:

    def __init__(self, pkg_xml, urdf_path, base_frame, gripper_frame, home_pos) -> None:
        self.urdf_path = urdf_path
        self.pkg_path = pkg_xml

        self.meshcat = StartMeshcat()
        self.meshcat.Delete()
        self.builder = DiagramBuilder()        

        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.001)
        self.robot = self.add_robot(home_pos, base_frame)

        parser = Parser(self.plant)
        self.plant.Finalize()

        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            self.meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration),
        )
        self.collision_visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder,
            self.scene_graph,
            self.meshcat,
            MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
        )
        self.meshcat.SetProperty("collision", "visible", False)

        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)

        self.num_q = self.plant.num_positions()
        self.q0 = self.plant.GetPositions(self.plant_context)
        self.gripper_frame = self.plant.GetFrameByName(gripper_frame, self.robot)
    

    def add_robot(self, home_pos, base_frame):
        parser = Parser(self.plant)
        parser.package_map().AddPackageXml(self.pkg_path)
        robot = parser.AddModelsFromUrl(self.urdf_path)[0]
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName(base_frame))

        # Set default positions:
        q0 = home_pos
        index = 0
        for joint_index in self.plant.GetJointIndices(robot):
            joint = self.plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q0[index])
                index += 1

        return robot

    def create_waypoint(self, name, position):
        X = RigidTransform(position)
        self.meshcat.SetObject(name, Sphere(0.003), rgba=Rgba(0.9, 0.1, 0.1, 1))
        self.meshcat.SetTransform(name, X)
        return X
    
    def visualize(self, q = None):
        plant_context = self.plant.GetMyContextFromRoot(self.context)
        visualizer_context = self.visualizer.GetMyContextFromRoot(self.context)
        if q is not None:
            self.plant.SetPositions(plant_context, q)
        self.visualizer.ForcedPublish(visualizer_context)




class FR3Drake(DrakeRobot):

    def __init__(self, pkg_xml, urdf_path, base_frame = "fr3_link0", gripper_frame = "fr3_link8",
                  home_pos = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78, 0.035, 0.035]):
        
        super().__init__(pkg_xml, urdf_path,base_frame, gripper_frame, home_pos)


class YumiDrake(DrakeRobot):

    def __init__(self, pkg_xml, urdf_path, left_arm = True,  base_frame = "yumi_base_link", gripper_frame = None,
                  home_pos = None):
        
        if left_arm is True:
            gripper_frame = gripper_frame or "gripper_l_tip"
            home_pos = home_pos or [0.0 , -2.26 , 2.36 , 0.52 , 0.0 , 0.7 , 0.0]            
        else:
            gripper_frame = gripper_frame or "gripper_r_tip"
            home_pos = home_pos or [0.0 , -2.26 , -2.36 , 0.52 , 0.0 , 0.7 , 0.0]
        
        super().__init__(pkg_xml, urdf_path,base_frame, gripper_frame, home_pos)