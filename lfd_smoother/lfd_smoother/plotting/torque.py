
import rospy
import matplotlib.pyplot as plt

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState



plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.default'] = 'regular'

class TorqueAnalysis:

    def __init__(self) -> None:
        
        self.torques = []
        self.velocities = []
        self.record_flag = False

        self.rate = rospy.Rate(1000)
        rospy.Subscriber("/joint_states", JointState, self.cb_record_joint_state)
        self.pub_robot = rospy.Publisher("/position_joint_controller/command", Float64MultiArray , queue_size=0)

    def cb_record_joint_state(self, data):
        if self.record_flag:
            self.torques.append(data.effort[:-2])
            self.velocities.append(data.velocity[:-2])
    
    def plot_torque(self):
        for idx, joint_torques in enumerate(zip(*self.torques)):
            plt.figure()
            plt.plot(joint_torques)
            plt.title(f'Joint {idx + 1} Torque Profile')
            plt.xlabel('Time step')
            plt.ylabel('Torque')
        plt.show()  

    def calculate_energy_consumption(self):
        energy = []
        for joint_torques, joint_velocities in zip(zip(*self.torques), zip(*self.velocities)):
            power = [abs(torque*velocity) for torque, velocity in zip(joint_torques, joint_velocities)]
            energy.append(sum(power))
        return sum(energy)
    
    def run(self, traj):
        self.record_flag = True
        t_start = rospy.Time.now().to_sec()
        t = 0  
        while (not rospy.is_shutdown()) and t < traj.ts[-2]:
            t = rospy.Time.now().to_sec() - t_start
            msg = Float64MultiArray()
            msg.data = traj.value(t)
            self.pub_robot.publish(msg)
            self.rate.sleep() 

        self.record_flag = False
        return self.calculate_energy_consumption()
    
