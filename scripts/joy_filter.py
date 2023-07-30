#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import Joy
from pykalman import KalmanFilter
import numpy as np


class KalmanSmoother:
    def __init__(self):
        # Initialize Kalman Filter
        self.kf = KalmanFilter(transition_matrices=[1],   # no change from t-1 to t
                               observation_matrices=[1],  # no scaling in observed value
                               initial_state_mean=1,     # assume initial state mean 0
                               initial_state_covariance=1,  # assume initial state covariance 1
                               transition_covariance=0.01,  # small process noise
                               observation_covariance=5) 
        self.state_estimate = 0
        self.state_covariance = 1    
        # Subscribe to the input topic
        self.sub = rospy.Subscriber('/joy_latched', Joy, self.callback)
        # Create a publisher for the smoothed values
        self.pub = rospy.Publisher('/joy_filtered', Float64, queue_size=10)

    def callback(self, msg: Joy):
        # Kalman filter step
        state_estimate, state_covariance = self.kf.filter_update(
            self.state_estimate,
            self.state_covariance,
            np.array([msg.axes[5]])
        )

        # Update state estimate and state covariance
        self.state_estimate = state_estimate
        self.state_covariance = state_covariance

        # Create a new message with the smoothed value
        smoothed_msg = Float64()
        smoothed_msg.data = self.map_range(self.state_estimate[0])

        # Publish the smoothed message
        self.pub.publish(smoothed_msg)

    def map_range(self,value):
        input_range = [1, -1]  # input range
        output_range = [0, 1]  # desired output range

        # line equation parameters
        slope = (output_range[1] - output_range[0]) / (input_range[1] - input_range[0])
        intercept = output_range[0] - slope * input_range[0]

        return slope * value + intercept

def main():
    rospy.init_node('smoother_node')

    smoother = KalmanSmoother()

    # Keep the node running until it's shut down
    rospy.spin()

if __name__ == '__main__':
    main()



    