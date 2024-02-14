#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
from time import sleep

ser = serial.Serial("/dev/ttyACM0",115200)

class Subscriber_Node(Node):
    def __init__(self):
        super().__init__("Cmd_vel_subscriber")
        # self.timer_ = self.create_timer(0.5, self.send_velocity_command)
        self.get_logger().info("Subscriber has started")
        self.cmd_vel_sub = self.create_subscription(Twist,"/cmd_vel",self.cmd_callback,10)
    
    def cmd_callback(self,msg:Twist):
        self.get_logger().info(f'{msg.linear.x},{msg.angular.z}')
        ser.write(f'{msg.angular.z}'.encode('utf-8'))

def main(args=None):
    rclpy.init(args=args)
    node = Subscriber_Node()
    rclpy.spin(node)
    rclpy.shutdown()
    ser.close()