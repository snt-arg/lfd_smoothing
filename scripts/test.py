#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from gui import Ui_Form

# Global variable
my_variable = 0

def variable_callback(value):
    # Update the variable when the slider changes
    global my_variable
    my_variable = value

def main():
    rospy.init_node('my_node')
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Ui_Form()
    ui.setupUi(window)
    window.show()

    rospy.spin()
    # # Create a subscriber to receive slider value updates
    # rospy.Subscriber('/slider_value', Float32, variable_callback)

    # # Your main code here
    # rate = rospy.Rate(10)  # 10 Hz
    # while not rospy.is_shutdown():
    #     # Process the updated value of my_variable
    #     rospy.loginfo('Variable value: %f', my_variable)
    #     # Your code here

    #     rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass