#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", str(data))
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
	rospy.init_node('scanListener', anonymous=True)

    #rospy.Subscriber("scan", LaserScan, callback)
	

	data = rospy.wait_for_message('scan', LaserScan, timeout=5)
	print("heard laser {0}".format(data))
	
	data = rospy.wait_for_message('odom', Odometry, timeout=5)
	print("heard odom {0}".format(data))

    # spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
	listener()