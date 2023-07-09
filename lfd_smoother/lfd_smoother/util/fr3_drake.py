import numpy as np

from pydrake.all import *


class FR3Drake:

    def __init__(self, franka_pkg_xml, urdf_path, base_frame = "fr3_link0", gripper_frame = "fr3_link8", home_pos = None):
        
        self.urdf_path = urdf_path
        self.franka_path = franka_pkg_xml
        
        self.meshcat = StartMeshcat()
        self.meshcat.Delete()
        self.builder = DiagramBuilder()

        if home_pos is None:
            home_pos = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78, 0.035, 0.035]

        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step=0.001)
        self.robot = self.add_fr3(home_pos, base_frame)

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
        
 
    
    def add_fr3(self, home_pos, base_frame):
        parser = Parser(self.plant)
        parser.package_map().AddPackageXml(self.franka_path)
        fr3 = parser.AddModelsFromUrl(self.urdf_path)[0]
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetFrameByName(base_frame))

        # Set default positions:
        q0 = home_pos
        index = 0
        for joint_index in self.plant.GetJointIndices(fr3):
            joint = self.plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q0[index])
                index += 1

        return fr3
    
    def create_waypoint(self, name, position):
        X = RigidTransform(position)
        self.meshcat.SetObject(name, Sphere(0.003), rgba=Rgba(0.9, 0.1, 0.1, 1))
        self.meshcat.SetTransform(name, X)
        return X