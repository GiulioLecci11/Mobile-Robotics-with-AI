import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
import numpy as np
import random
import threading
import torch
from torch.utils.tensorboard import SummaryWriter
from math import inf
import torch.nn as nn
import torch.nn.functional as F
import math

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.2
CYLINDER_RADIUS = 0.6
TIME_DELTA = 0.2

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu  

#*****************************Redefinition of Actor, Critic, and TD3 classes********************************
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):      
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

class td3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )

#*********************************************************************************************************

class MultiNode(Node):
    def __init__(self):
        super().__init__('multi_node')
        self.LIDAR_DIM=20 # Number of lidar data
        self.THRESHOLD=10.0 # Threshold for the lidar data
        self.x=0.0
        self.y=0.0
        self.theta=0.0
        self.previousTime=0.0
        self.lidarData=[] 
        self.actions=np.zeros(2)
        self.goal_x = random.uniform(0.0, 3.0)
        self.goal_y = random.uniform(-3.0, 3.0)
        self.stop=0
        self.distance=0.0 
        self.angle_to_goal=0.0  
        
        # Print in logger the goal position
        self.get_logger().info('Goal position: x=%f, y=%f' % (self.goal_x, self.goal_y))


        self.robot_dim = 4
        self.state_dim = self.LIDAR_DIM + self.robot_dim
        self.action_dim = 2
        self.max_action = 1
        self.expl_noise = 0.5  # Initial exploration noise starting value 
        expl_decay_steps = (
        500000  # Number of steps over which the initial exploration noise will decay over (Per ora non c'è)
        )
        expl_min = 0.1  # Exploration noise after the decay


        self.state=np.zeros(self.state_dim)

        # Create the network
        self.network = td3(self.state_dim, self.action_dim, self.max_action)

        try:
            self.get_logger().info("Will load existing model.")
            self.network.load("td3_velodyne", "./results/pytorch_models") # Original model
            #self.network.load("td3_velodyne_primoTrain", "/home/a/completeListenerPublisherNode_ws/results/pytorch_models")
            #self.network.load("td3_velodyne_terzoTrain", "/home/a/completeListenerPublisherNode_ws/results/pytorch_models")
            #self.network.load("td3_velodyne_quartoTrain", "/home/a/completeListenerPublisherNode_ws/results/pytorch_models")
            
        except:
             self.get_logger().info("Could not load the stored model parameters, initializing training with random parameters")


        #gaps initialization
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.LIDAR_DIM]]
        for m in range(self.LIDAR_DIM - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.LIDAR_DIM]
            )
        self.gaps[-1][-1] += 0.03

        # Create subscribers
        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )
        self.odom_unfiltered_subscription = self.create_subscription(
            Odometry,
            'odom/unfiltered',
            self.odom_unfiltered_callback,
            10
        )

        # Create publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_publisher = self.create_publisher(Odometry, 'odom', 10)

        # Create a timer that calls a function (usata prima per fare prova a pubblicare cmd_vel ogni 3 secondi)
        #self.timer = self.create_timer(3.0, self.timer_callback) # Publish every 3 seconds cmd_vel

        # Create a timer that calls a function every 100ms
        self.timer2 = self.create_timer(0.3, self.network_callback) # Evaluate actions from network every 0.01 seconds

    def network_callback(self):

        if self.stop==0:

            # Print di debug
            #self.get_logger().info('State: %s' % self.state)
            #self.get_logger().info('*********************************')
            # print lidarData
            #self.get_logger().info('Lidar Data: %s' % self.lidarData)
            # Print action[0] and action[1]
            self.get_logger().info('Action[0]: %f' % self.actions[0])
            self.get_logger().info('Action[1]: %f' % self.actions[1])
            # Print distance and theta
            self.get_logger().info('Distance: %f' % self.distance)
            self.get_logger().info('Angle to goal: %f' % self.angle_to_goal)
            # Print self.x self.y self.theta
            self.get_logger().info('X: %f' % self.x)
            self.get_logger().info('Y: %f' % self.y)
            self.get_logger().info('Theta: %f' % self.theta)
            # Print goal_x and goal_y
            self.get_logger().info('Goal_x: %f' % self.goal_x)
            self.get_logger().info('Goal_y: %f' % self.goal_y)
            self.get_logger().info('*********************************') 


            action=self.network.get_action(np.array(self.state))
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip( #GIULIOOOOOOOOOO QUI METTE IL RUMOREEEEEEEEE
                     -self.max_action, self.max_action
                )
            a_in = [(action[0] + 1) / 2, action[1]]
            self.actions[0]=a_in[0]
            self.actions[1]=a_in[1]
            self.state[self.LIDAR_DIM+2:]=[self.actions[0],self.actions[1]]
            cmd_vel = Twist()
            cmd_vel.linear.x = a_in[0]*0.05 # To scale to the maximum linear velocity
            cmd_vel.angular.z = a_in[1]*0.08 # To scale to the maximum angular velocity
            self.cmd_vel_publisher.publish(cmd_vel)
            #self.get_logger().info('Publishing cmd_vel: linear=%f, angular=%f' % (cmd_vel.linear.x, cmd_vel.angular.z))


    def scan_callback(self, msg):
        # Print a message containing a value of the received message
        self.lidarData=np.ones(self.LIDAR_DIM)*10.0
        min_angle = msg.angle_min
        max_angle = msg.angle_max
        angle_increment = msg.angle_increment
        angles = np.arange(min_angle, max_angle, angle_increment)  #true for the real lidar, not for the neural network one that needs a different arrangement

        network_angles = np.linspace(-np.pi/2, np.pi/2, 20)

        starting_index = int(np.floor((-np.pi/2 + np.pi) / angle_increment))
        final_index = int(np.floor((np.pi/2 + np.pi) / angle_increment))
        
        for i in range(starting_index, final_index):
            for j in range(len(self.gaps)):
                if self.gaps[j][0] <= angles[i] < self.gaps[j][1]:
                    self.lidarData[j] = min(self.lidarData[j], msg.ranges[i])
                    break

        self.state[:self.LIDAR_DIM]=self.lidarData

        cmd_vel = Twist()

        # Detect a collision from laser data
        min_laser = min(self.lidarData)
        #print min laser
        #self.get_logger().info('Min laser distance: %f' % min_laser)
        if min_laser < COLLISION_DIST:
            self.get_logger().info("Collision is detected!")
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0 # Stop the robot
            self.cmd_vel_publisher.publish(cmd_vel)
            self.stop=1
        

        #self.get_logger().info('Received scan message with %d ranges' % len(msg.ranges))
        #self.get_logger().info('Lidar Data: %s' % self.lidarData)

        # index_0=int(np.floor((0-min_angle)/angle_increment))
        # index_90=int(np.floor((np.pi/2-min_angle)/angle_increment))
        # index_180=int(np.floor((np.pi-min_angle)/angle_increment))
        # index_270=int(np.floor((-np.pi/2-min_angle)/angle_increment))

        #print ranges at the 4 indeces
        #self.get_logger().info('Lidar Data at 0 degrees DAVANTI: %f' % msg.ranges[index_0])
        #self.get_logger().info('Lidar Data at 90 degrees DESTRA: %f' % msg.ranges[index_90])
        #self.get_logger().info('Lidar Data at 180 degrees DIETRO: %f' % msg.ranges[index_180])
        #self.get_logger().info('Lidar Data at 270 degrees SINISTRA: %f' % msg.ranges[index_270])

        # threshold = 0.5
        # for i in range(len(msg.ranges)-1):
        #     if msg.ranges[i] < threshold:
        #         angle = angles[i]
        #         self.get_logger().info(f"Lidar Data below threshold at angle {angle*180/np.pi}: {msg.ranges[i]}")

        #threshold = 5.0
        #for i in range(len(self.lidarData)-1):
            # if self.lidarData[i] < threshold:
            #     angle = network_angles[i]
            #     self.get_logger().info(f"Lidar Data below threshold at angle {angle*180/np.pi}: {self.lidarData[i]}")
        
        

    def odom_unfiltered_callback(self, msg):

        #FAKE ODOMETRY
        # Calculate and publish odometry
        # self.get_logger().info('Received unfiltered odometry message')
        # odom = Odometry()
        # odom.header = Header()
        # odom.header.stamp = self.get_clock().now().to_msg()
        # odom.header.frame_id = 'odom'

        # # Randomly generate position and orientation for example purposes
        # odom.pose.pose.position.x = random.uniform(-5.0, 5.0)
        # odom.pose.pose.position.y = random.uniform(-5.0, 5.0)
        # odom.pose.pose.position.z = 0.0
        # odom.pose.pose.orientation.x = random.uniform(-1.0, 1.0)
        # odom.pose.pose.orientation.y = random.uniform(-1.0, 1.0)
        # odom.pose.pose.orientation.z = random.uniform(-1.0, 1.0)
        # odom.pose.pose.orientation.w = random.uniform(-1.0, 1.0)

        # # Publish the odometry message
        # self.odom_publisher.publish(odom)
        # self.get_logger().info('Publishing odometry message')



        #REAL ODOMETRY
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        Ts = current_time - self.previousTime
        Ts=0.0172
        self.previousTime = current_time
        linear_velocity = msg.twist.twist.linear.x
        angular_velocity = msg.twist.twist.angular.z
        # Print angular_velocity and linear_velocity
        #self.get_logger().info('Angular velocity: %f' % angular_velocity) 
        #self.get_logger().info(f'Received unfiltered odometry message: Delta t={Ts}')
        #self.get_logger().info(f'Received unfiltered odometry message: linear={linear_velocity}, angular={angular_velocity}')

        # Update position and orientation based on velocity command
        self.x += linear_velocity * Ts * np.cos(self.theta+angular_velocity*Ts/2)
        self.y += linear_velocity * Ts * np.sin(self.theta+angular_velocity*Ts/2)
        self.theta += angular_velocity * Ts

        # print self.x,y and theta
        # self.get_logger().info('X: %f' % self.x)
        # self.get_logger().info('Y: %f' % self.y)
        # self.get_logger().info('theta: %f' % self.theta)
        

        # Calculate the distance between (self.x, self.y) and a random generated position

        #distance = math.sqrt((self.x - self.goal_x)**2 + (self.y - self.goal_y)**2)
        distance = np.linalg.norm(
            [self.x - self.goal_x, self.y - self.goal_y]
        )
        self.distance=distance

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.x
        skew_y = self.goal_y - self.y
        dot = skew_x * 1 + skew_y * 0 #************************** scalar product between robot heading [1 0] (in its reference frame) and the vector which points from the robot to the goal
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2)) #***************** the angle between robot heading and goal vector obtained from dot_product=mag1*mag2*cos(angle_between)
        if skew_y < 0:
            if skew_x < 0: #****** cioè terzo quadrante
                beta = -beta 
            else: #********** cioè quarto quadrante
                beta = 0 - beta
        self.angle_to_goal = beta - self.theta #****************** AVREBBE SENSO SE ANGLE FOSSE PRESO IN SENSO ORARIO???
        if self.angle_to_goal > np.pi:
            self.angle_to_goal = np.pi - self.angle_to_goal
            self.angle_to_goal = -np.pi - self.angle_to_goal #***************PRATICAMENTE THETA=THETA-2*PI
        if self.angle_to_goal < -np.pi:
            self.angle_to_goal = -np.pi - self.angle_to_goal
            self.angle_to_goal = np.pi - self.angle_to_goal #**************** PRATICAMENTE THETA= THETA + 2*PI
        
        self.state[self.LIDAR_DIM:self.LIDAR_DIM+2]=[distance, self.angle_to_goal]

        

        # Publish the odometry message
        odom = Odometry()
        odom.header = Header()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        # Calculate the quaternion from the orientation
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = np.sin(self.theta/2)
        odom.pose.pose.orientation.w = np.cos(self.theta/2)

        # Calculate the linear and angular velocity
        odom.twist.twist.linear.x = linear_velocity*np.cos(self.theta)
        odom.twist.twist.linear.y = linear_velocity*np.sin(self.theta)
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = angular_velocity

        # publish
        #self.get_logger().info('Publishing odometry message')
        self.odom_publisher.publish(odom)

        # print distance
        #self.get_logger().info('Distance to goal: %f' % distance)
        if distance < GOAL_REACHED_DIST:
            cmd_vel=Twist()
            self.get_logger().info("Goal reached!")
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0 # Stop the robot
            self.cmd_vel_publisher.publish(cmd_vel)
            self.stop=1
            

    def timer_callback(self):
        # Publish a velocity command every 100ms
        cmd_vel = Twist()
        cmd_vel.linear.x = random.uniform(-1.0, 1.0)
        cmd_vel.angular.z = random.uniform(-1.0, 1.0)
        self.cmd_vel_publisher.publish(cmd_vel)
        self.get_logger().info('Publishing cmd_vel: linear=%f, angular=%f' % (cmd_vel.linear.x, cmd_vel.angular.z))

def main(args=None):
    rclpy.init(args=args)

    # Create the node
    node = MultiNode()

    # Use a MultiThreadedExecutor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
