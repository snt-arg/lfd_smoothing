#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:14:08 2023

@author: abrk
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from pydrake.all import *

from manipulation import running_as_notebook
from manipulation.meshcat_utils import PublishPositionTrajectory
from manipulation.scenarios import AddIiwa, AddPlanarIiwa, AddShape, AddWsg
from manipulation.utils import ConfigureParser


def AddFrankaFr3(plant):
    urdf = "package://my_drake/manipulation/models/franka_description/urdf/fr3.urdf"

    parser = Parser(plant)
    parser.package_map().AddPackageXml("/home/abrk/catkin_ws/src/franka/franka_ros/franka_description/package.xml")
    parser.package_map().AddPackageXml("/home/abrk/thesis/drake/my_drake/package.xml")
    fr3 = parser.AddModelsFromUrl(urdf)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("fr3_link0"))

    # Set default positions:
    q0 = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78, 0.035, 0.035]
    index = 0
    for joint_index in plant.GetJointIndices(fr3):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return fr3


def create_waypoint(name, x, y, z):
    X = RigidTransform([x, y, z])
    meshcat.SetObject(name, Sphere(0.003), rgba=Rgba(0.9, 0.1, 0.1, 1))
    meshcat.SetTransform(name, X)
    return X

def create_sym_r(trajopt):
    control_points = trajopt.control_points()
    ufunc = np.vectorize(Expression)
    control_points_exp = [ufunc(control_points[:,i,np.newaxis]) for i in range(num_control_points)]
    basis_exp = BsplineBasis_[Expression](trajopt.basis().order(), trajopt.basis().knots())  
    sym_r = BsplineTrajectory_[Expression](basis_exp,control_points_exp)
    
    return sym_r

def reconstruct_trajectory(result, trajopt, t_offset):
    duration = result.GetSolution(trajopt.duration())
    basis = trajopt.basis()
    scaled_knots = [(knot * duration) + t_offset for knot in basis.knots()]
    
    return duration, BsplineTrajectory(
        BsplineBasis(basis.order(), scaled_knots),
        [result.GetSolution(trajopt.control_points())[:,i,np.newaxis] for i in range(num_control_points)])

def plot_trajectory(composite_traj):
    ts = np.linspace(0, composite_traj.end_time(), 1000)
    
    position = []
    velocity = []
    acceleration = []
    jerk = []
    
    for t in ts:
        position.append(composite_traj.value(t))
        velocity.append(composite_traj.MakeDerivative().value(t))
        acceleration.append(composite_traj.MakeDerivative(2).value(t))
        jerk.append(composite_traj.MakeDerivative(3).value(t))
    
    # Plot position, velocity, and acceleration
    fig, axs = plt.subplots(1, 4, figsize=(50,10))
    
    joint_index = 5
    
    # Position plot
    axs[0].plot(ts, np.array(position).squeeze(axis=2))
    axs[0].set_ylabel('Position')
    axs[0].set_xlabel('Time')
    
    # Velocity plot
    axs[1].plot(ts, np.array(velocity).squeeze(axis=2))
    axs[1].set_ylabel('Velocity')
    axs[1].set_xlabel('Time')
    
    # Acceleration plot
    axs[2].plot(ts, np.array(acceleration).squeeze(axis=2))
    axs[2].set_ylabel('Acceleration')
    axs[2].set_xlabel('Time')
    
    # Jerk plot
    axs[3].plot(ts, np.array(jerk).squeeze(axis=2))
    axs[3].set_ylabel('Jerk')
    axs[3].set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
    
def pose_diff(p1, q1, p2, q2):
    rb1 = RigidTransform(p1)
    rb1.set_rotation(Quaternion(q1))

    rb2 = RigidTransform(p2)
    rb2.set_rotation(Quaternion(q2))
    
    angle = np.abs(rb1.rotation().ToAngleAxis().angle() - rb2.rotation().ToAngleAxis().angle())
    translation = np.linalg.norm(rb1.translation() - rb2.translation())
    return translation, angle
    
#%%

with open('/home/abrk/catkin_ws/src/lfd/lfd_dmp/scripts/experimental/smoothing/0.pickle', 'rb') as file:
    demonstration = pickle.load(file)
  
joint_trajectory = demonstration.joint_trajectory

n_time_steps = len(joint_trajectory.points)
n_dim = len(joint_trajectory.joint_names)

ys = np.zeros([n_time_steps,n_dim])
ts = np.zeros(n_time_steps)

for (i,point) in enumerate(joint_trajectory.points):
    ys[i,:] = point.positions
    ts[i] = point.time_from_start.to_sec()


pose_trajectory = demonstration.pose_trajectory

n_time_steps = len(joint_trajectory.points)

positions = np.zeros((n_time_steps, 3))
orientations = np.zeros((n_time_steps, 4))

for (i,point) in enumerate(pose_trajectory.points):
    positions[i,:] = [point.pose.position.x, point.pose.position.y, point.pose.position.z]
    orientations[i,:] = [point.pose.orientation.w, point.pose.orientation.x, point.pose.orientation.y, point.pose.orientation.z] 
    

#%% Preparatory steps
# indices = np.arange(0,ys.shape[0]-400,1)
# ys = ys[indices]
# positions = positions[indices]
# orientations = orientations[indices]

# Add dummy gripper position (zero)
ys = np.insert(ys,[ys.shape[1], ys.shape[1]], 0, axis=1)

meshcat = StartMeshcat()

#%% Drop the points which are very close
thr_translation = 0.1
thr_angle = 0.1 

indices = [0]
for i in range(1,positions.shape[0]):  
    translation, angle = pose_diff(positions[indices[-1]], orientations[indices[-1]], positions[i], orientations[i])
    if translation > thr_translation or angle > thr_angle:
        indices.append(i)

ys = ys[indices]
ts = ts[indices]
positions = positions[indices]
orientations = orientations[indices]
#%%

meshcat.Delete()
builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
fr3 = AddFrankaFr3(plant)

waypoints = []

for i in range(0,positions.shape[0]):  
     waypoints.append(create_waypoint("wp{}".format(i), positions[i,0], positions[i,1], positions[i,2]))   
     waypoints[-1].set_rotation(Quaternion(orientations[i]))
     
parser = Parser(plant)
plant.Finalize()

visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    scene_graph,
    meshcat,
    MeshcatVisualizerParams(role=Role.kIllustration),
)
collision_visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    scene_graph,
    meshcat,
    MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
)
meshcat.SetProperty("collision", "visible", False)

diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)

num_q = plant.num_positions()
q0 = plant.GetPositions(plant_context)
gripper_frame = plant.GetFrameByName("fr3_link8", fr3)
 
#%%

num_control_points = 4
num_waypoints = len(waypoints) # Including start and goal
bspline_order = 4

trajopts = []
progs = []

for i in range(num_waypoints-1):
    trajopts.append(KinematicTrajectoryOptimization(plant.num_positions(), num_control_points, spline_order=bspline_order))

for trajopt in trajopts:
    progs.append(trajopt.get_mutable_prog())
    trajopt.AddDurationCost(10.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(
        plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    )
    trajopt.AddVelocityBounds(
        plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
    )
    trajopt.AddDurationConstraint(0.1, 5)

#%% EE constraints
# Start Constraints
start_constraint = PositionConstraint(
    plant,
    plant.world_frame(),
    waypoints[0].translation(),
    waypoints[0].translation(),
    gripper_frame,
    [0, 0, 0],
    plant_context,
)
trajopts[0].AddPathPositionConstraint(start_constraint, 0)

ori_const = OrientationConstraint(
    plant, 
    gripper_frame, 
    RotationMatrix(),
    plant.world_frame(),
    waypoints[0].rotation(),
    0.0,
    plant_context
    )

trajopts[0].AddPathPositionConstraint(ori_const, 0)

trajopts[0].AddPathVelocityConstraint(
    np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
)


# progs[0].AddQuadraticErrorCost(
#     1*np.eye(num_q), ys[0], trajopts[0].control_points()[:, 0]
# )


# Goal Constraints
goal_constraint = PositionConstraint(
    plant,
    plant.world_frame(),
    waypoints[-1].translation(),
    waypoints[-1].translation(),
    gripper_frame,
    [0, 0, 0],
    plant_context,
)
trajopts[-1].AddPathPositionConstraint(goal_constraint, 1)

ori_const = OrientationConstraint(
    plant, 
    gripper_frame, 
    RotationMatrix(),
    plant.world_frame(),
    waypoints[-1].rotation(),
    0.0,
    plant_context
    )

trajopts[-1].AddPathPositionConstraint(ori_const, 1)

trajopts[-1].AddPathVelocityConstraint(
    np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
)

# progs[-1].AddQuadraticErrorCost(
#     1*np.eye(num_q), ys[-1], trajopts[-1].control_points()[:, -1]
# )


# Middle Constraints

for i in range(1,len(waypoints)-1):
    middle_constraint = PositionConstraint(
        plant,
        plant.world_frame(),
        waypoints[i].translation(),
        waypoints[i].translation(),
        gripper_frame,
        [0, 0, 0],
        plant_context,
    )
    trajopts[i-1].AddPathPositionConstraint(middle_constraint, 1)
    trajopts[i].AddPathPositionConstraint(middle_constraint, 0)
    
    ori_const = OrientationConstraint(
        plant, 
        gripper_frame, 
        RotationMatrix(),
        plant.world_frame(),
        waypoints[i].rotation(),
        0.0,
        plant_context
        )
    
    trajopts[i-1].AddPathPositionConstraint(ori_const, 1)
    trajopts[i].AddPathPositionConstraint(ori_const, 0)
    
    # progs[i-1].AddQuadraticErrorCost(
    #     1*np.eye(num_q), ys[i], trajopts[i-1].control_points()[:, -1]
    # )
    # progs[i].AddQuadraticErrorCost(
    #     1*np.eye(num_q), ys[i], trajopts[i].control_points()[:, 0]
    # )
    
#%% Joint Constraints

# trajopts[0].AddPathPositionConstraint(ys[0], ys[0] ,0)
# trajopts[-1].AddPathPositionConstraint(ys[-1],ys[-1],1)

# trajopts[0].AddPathVelocityConstraint(
#     np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
# )

# trajopts[-1].AddPathVelocityConstraint(
#     np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
# )

# for i in range(1,len(ys)-1):
#     trajopts[i-1].AddPathPositionConstraint(ys[i], ys[i], 1)
#     trajopts[i].AddPathPositionConstraint(ys[i], ys[i], 0)
    

#%%

nprog = MathematicalProgram()

for prog in progs:
    nprog.AddDecisionVariables(prog.decision_variables())
    for const in prog.GetAllConstraints():
        nprog.AddConstraint(const.evaluator(), const.variables())
    for cost in prog.GetAllCosts():
        nprog.AddCost(cost.evaluator(), cost.variables())


sym_rs = []

for trajopt in trajopts:
    sym_rs.append(create_sym_r(trajopt))
    

for i in range(1,len(ys)-1):
    for j in range(num_q):
        nprog.AddConstraint(((sym_rs[i-1].MakeDerivative().value(1) / trajopts[i-1].duration()) 
                            - (sym_rs[i].MakeDerivative().value(0) / trajopts[i].duration()))[j][0]
                            , 0, 0)
        nprog.AddConstraint(((sym_rs[i-1].MakeDerivative(2).value(1) / trajopts[i-1].duration() / trajopts[i-1].duration()) 
                            - (sym_rs[i].MakeDerivative(2).value(0) / trajopts[i].duration() / trajopts[i].duration()))[j][0]
                            , 0, 0)
        nprog.AddLinearConstraint(sym_rs[i-1].value(1) - sym_rs[i].value(0), np.zeros((num_q, 1)), np.zeros((num_q, 1)))

#%%
    
# prog_id = 1
# sub_result = Solve(progs[prog_id])
# if not sub_result.is_success():
#     print("optimization failed")
# plot_trajectory(trajopts[prog_id].ReconstructTrajectory(sub_result))


#%%

result = Solve(nprog)

result

if not result.is_success():
    print("optimization failed")
    
#%%

for trajopt in trajopts:
    nprog.SetInitialGuess(trajopt.control_points(), result.GetSolution(trajopt.control_points()))
    
result = Solve(nprog)

if not result.is_success():
    print("optimization failed again")

#%%

output_trajs = []
t_offset = 0.0
for trajopt in trajopts:
    duration, traj = reconstruct_trajectory(result, trajopt, t_offset)
    output_trajs.append(traj)
    t_offset += duration
    
composite_traj = CompositeTrajectory(output_trajs)

PublishPositionTrajectory(
    composite_traj, context, plant, visualizer
)
collision_visualizer.ForcedPublish(
    collision_visualizer.GetMyContextFromRoot(context)
)

#%%

plot_trajectory(composite_traj)



#%%
# num_control_points = 10

# trajopt = KinematicTrajectoryOptimization(plant.num_positions(), num_control_points)
# prog = trajopt.get_mutable_prog()
# trajopt.AddDurationCost(1.0)
# trajopt.AddPathLengthCost(1.0)
# trajopt.AddPositionBounds(
#     plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
# )
# trajopt.AddVelocityBounds(
#     plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
# )

# control_points = trajopt.control_points()
# ufunc = np.vectorize(Expression)
# control_points_exp = [ufunc(control_points[:,i,np.newaxis]) for i in range(num_control_points)]

# normalized_knots = trajopt.basis().knots()
# basis_exp = BsplineBasis_[Expression](trajopt.basis().order(), normalized_knots)
# basis_normal = BsplineBasis(trajopt.basis().order(), normalized_knots)

# sym_r = BsplineTrajectory_[Expression](basis_exp,control_points_exp)
# sym_rdot = sym_r.MakeDerivative()
# sym_rddot = sym_rdot.MakeDerivative()

#%%

# trajopt.AddPathPositionConstraint(q0, q0, 0)

# qf = np.array([0.021, -0.117, 0.109, -1.99, 0.014, 1.877, 0.91, 0.035, 0.035])
# trajopt.AddPathPositionConstraint(qf, qf, 1)

# qm = np.array([-0.004, -0.469, -0.0208, -1.73, -0.001, 1.261, 0.759, 0.035, 0.035])
# tm = prog.NewContinuousVariables(1, "tm")[0]
# trajopt.AddPathPositionConstraint(qm, qm, tm)

# # start and end with zero velocity
# trajopt.AddPathVelocityConstraint(
#     np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
# )
# trajopt.AddPathVelocityConstraint(
#     np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
# )

# result = Solve(prog)
# if not result.is_success():
#     print("Trajectory optimization failed, even without collisions!")
#     print(result.get_solver_id().name())
    
# PublishPositionTrajectory(
#     trajopt.ReconstructTrajectory(result), context, plant, visualizer
# )
# collision_visualizer.ForcedPublish(
#     collision_visualizer.GetMyContextFromRoot(context)
# )