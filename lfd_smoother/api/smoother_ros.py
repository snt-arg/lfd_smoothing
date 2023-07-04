from pydrake.all import *

from lfd_smoother.core.smoother import TrajectorySmoother
from lfd_smoother.core.single_smoother import SingleSmoother
from lfd_smoother.util.demonstration import Demonstration
from lfd_smoother.util.fr3_drake import FR3Drake
from lfd_smoother.util.config import FR3Config

import yaml


class ROSSmoother:

    def __init__(self, config_path):
        self.config = config
        # with open(config_path, "r") as file:
        #     self.config = yaml.safe_load(file)

    def run(self, demonstration):

        self.demo = Demonstration()
        self.demo.read_from_ros(demonstration)
        self.demo.filter(thr_translation=self.config["demo_filter_threshold"])

        self.robot = FR3Drake(franka_pkg_xml=self.config["franka_pkg_xml"],
                        urdf_path=self.config["urdf_path"])

        config = FR3Config(self.robot, self.demo)
        config.parse_from_file(self.config["config_hierarchy"][0])
        self.smoother = TrajectorySmoother(self.robot, config)
        self.smoother.run()
        initial_guess = self.smoother.export_cps()
        timings = self.smoother.export_waypoint_ts()
        
        config.parse_from_file("config/main.yaml")
        config.add_initial_guess(initial_guess,timings)
        config.add_solver(IpoptSolver())
        self.smoother = SingleSmoother(self.robot, config)
        self.smoother.run()