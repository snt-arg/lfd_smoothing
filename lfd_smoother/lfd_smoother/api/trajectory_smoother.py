import numpy as np

from pydrake.all import *

from lfd_smoother.core.trajectory_optimizer import TrajectoryOptimizer
from lfd_smoother.core.single_optimizer import SingleOptimizer
from lfd_smoother.util.demonstration import Demonstration
from lfd_smoother.util.robot import FR3Drake, YumiDrake
from lfd_smoother.util.config import FR3Config, YumiConfig



class TrajectorySmoother:

    def __init__(self, config):
        self.config = config

    def read_demo_ros(self, demonstration):
        self.demo = Demonstration()
        self.demo.read_from_ros(demonstration)
        self.demo.filter(thr_translation=self.config["demo_filter_threshold"])

    def read_demo_waypoints(self, demo_name):
        self.demo = Demonstration()
        self.demo.read_from_waypoints(demo_name)
        self.demo.filter(thr_translation=self.config["demo_filter_threshold"])

    def run(self, timings = None, tolerances = None):

        if self.config["robot_type"] == "fr3":
            self.robot = FR3Drake(pkg_xml=self.config["pkg_xml"],
                            urdf_path=self.config["urdf_path"])
            config = FR3Config(self.robot, self.demo)
        elif self.config["robot_type"] == "yumi":
            self.robot = YumiDrake(pkg_xml=self.config["pkg_xml"],
                            urdf_path=self.config["urdf_path"])

            config = YumiConfig(self.robot, self.demo)

        config.parse_from_file(self.config["config_hierarchy"][0])
        self.smoother = TrajectoryOptimizer(self.robot, config)
        self.smoother.run()
        initial_guess = self.smoother.export_cps()
        if timings is None:
            self.timings = self.smoother.export_waypoint_ts()
        else:
            self.timings = timings
            
        config.parse_from_file(self.config["config_hierarchy"][1])
        config.add_initial_guess(initial_guess,self.timings)
        self.smoother = SingleOptimizer(self.robot, config, tolerances=tolerances)
        self.result_traj = self.smoother.run()
        
    def export_raw(self):
        ts = np.linspace(0, self.result_traj.end_time(), 1000)
        
        position = []
        velocity = []
        acceleration = []
        
        for t in ts:
            position.append(self.result_traj.value(t))
            velocity.append(self.result_traj.MakeDerivative().value(t))
            acceleration.append(self.result_traj.MakeDerivative(2).value(t))

        return ts, np.array(position), np.array(velocity), np.array(acceleration)