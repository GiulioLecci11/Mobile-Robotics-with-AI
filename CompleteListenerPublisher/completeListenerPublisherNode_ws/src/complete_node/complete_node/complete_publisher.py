import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
import random
import math

class LaserScanPublisher(Node):
    def __init__(self):
        super().__init__('laser_scan_publisher')
        
        # Publishers
        self.scan_publisher = self.create_publisher(LaserScan, 'scan', 10)
        self.odom_unfiltered_publisher = self.create_publisher(Odometry, 'odom/unfiltered', 10)
        
        # Subscribers
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # Timer
        self.scan_timer = self.create_timer(10.0, self.scan_timer_callback)  # Publish every second

        # Initialize position and orientation
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def scan_timer_callback(self):
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -3.1241390705108643
        scan_msg.angle_max = 3.1415927410125732
        scan_msg.angle_increment = 0.01745329238474369
        scan_msg.time_increment = 0.00034325619344599545
        scan_msg.scan_time = 0.12322897464036942
        scan_msg.range_min = 0.15000000596046448
        scan_msg.range_max = 12.0
        scan_msg.ranges = [random.uniform(0.0, 12.0) for _ in range(359)]
        scan_msg.intensities = [random.uniform(0.0, 1.0) for _ in range(359)]
        
        self.scan_publisher.publish(scan_msg)
        self.get_logger().info('Publishing laser scan data')

    def cmd_vel_callback(self, msg):
        # Handle received velocity command
        self.get_logger().info(f'Received cmd_vel: linear={msg.linear.x}, angular={msg.angular.z}')
        
        # Update position and orientation based on velocity command
        delta_time = 2.0  # Time interval for update (in seconds)
        self.x += msg.linear.x * delta_time * math.cos(self.theta)
        self.y += msg.linear.x * delta_time * math.sin(self.theta)
        self.theta += msg.angular.z * delta_time

        # Create and publish odometry unfiltered message
        odom_unfiltered_msg = Odometry()
        odom_unfiltered_msg.header.stamp = self.get_clock().now().to_msg()
        odom_unfiltered_msg.header.frame_id = 'odom_unfiltered'
        odom_unfiltered_msg.pose.pose.position.x = self.x
        odom_unfiltered_msg.pose.pose.position.y = self.y
        odom_unfiltered_msg.pose.pose.position.z = 0.0
        odom_unfiltered_msg.pose.pose.orientation.x = 0.0
        odom_unfiltered_msg.pose.pose.orientation.y = 0.0
        odom_unfiltered_msg.pose.pose.orientation.z = math.sin(self.theta / 2)
        odom_unfiltered_msg.pose.pose.orientation.w = math.cos(self.theta / 2)

        self.odom_unfiltered_publisher.publish(odom_unfiltered_msg)
        self.get_logger().info('Publishing odometry unfiltered data')

    def odom_callback(self, msg):
        # Handle received odometry data for /odom topic
        self.get_logger().info('Received odometry data for /odom')
        self.get_logger().info(f'Position: x={msg.pose.pose.position.x}, y={msg.pose.pose.position.y}')
        self.get_logger().info(f'Rotation: z={msg.pose.pose.orientation.z}')
        # Update position and orientation based on received odometry data
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        # Orientation could be updated similarly

def main(args=None):
    rclpy.init(args=args)
    node = LaserScanPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
