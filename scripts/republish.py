#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy

class RepublishingNode(object):
    def __init__(self, in_topic, out_topic, rate_in_hz):
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.rate = rospy.Rate(rate_in_hz)

        self.latest_message = None



        # Subscribe to the input topic
        rospy.Subscriber(self.in_topic, Joy, self.callback)

        # Create a publisher for the output topic
        self.publisher = rospy.Publisher(self.out_topic, Joy, queue_size=10)

    def callback(self, message):
        # Store the latest received message
        self.latest_message = message

    def run(self):
        # Start the publishing loop
        while not rospy.is_shutdown():
            if self.latest_message is not None:
                self.latest_message.header.stamp = rospy.Time.now()
                self.publisher.publish(self.latest_message)
            self.rate.sleep()

if __name__ == '__main__':
            # Initialize the node
    rospy.init_node('republishing_node', anonymous=True)
    # Replace 'input_topic', 'output_topic', and 'rate_in_hz' with your actual values
    node = RepublishingNode('joy', 'joy_latched', rate_in_hz=50)
    node.run()
