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

GOAL_REACHED_DIST = 0.3                 # The distance at which the goal is considered achieved
COLLISION_DIST = 0.2                    # The distance at which a collision is detected
CYLINDER_RADIUS = 0.6                   # Radius of the cylinder shaped obstacles in the simulation environment
TIME_DELTA = 0.2                        

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu  

#*****************************Redefinition of Actor, Critic, and TD3 classes********************************

# Defining class Actor
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

# Defining class Critic
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

# Defining class Td3
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
        # Function to get the action from the actor (linear and angular velocity)
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
# The Multi Node is a Ros node that gives high level commands to the ESP32 using the TD3 network​
class MultiNode(Node):
    def __init__(self):
        super().__init__('multi_node')
        self.LIDAR_DIM=20               # The number of sectors into which the LIDAR sensor range will be divided
        self.THRESHOLD=10.0             # If an object is more than 10 meters away, it is considered to be at 10 meters.
        self.x=0.0                      # Robot position along the x axis
        self.y=0.0                      # Robot position along the y axis
        self.theta=0.0                  # Robot orientation 
        self.previousTime=self.get_clock().now()
        self.lidarData=[]               # Data coming from the LIDAR node
        self.actions=np.zeros(2)        # Actions provided by the Td3 network that are a desired angular e linear velocities
        self.goal_x = random.uniform(0.0, 3.0)    # Goal position along the x axis 
        self.goal_y = random.uniform(-3.0, 3.0)   # Goal position along the y axis
        self.stop=0                     # Flag that notify if a collision was detected
        self.distance=0.0               # Distance between the robot and the goal positions
        self.angle_to_goal=0.0          # Angle between the robot orientation and the desired orientation to reach the goal
        
        # Print in logger the goal position
        self.get_logger().info('Goal position: x=%f, y=%f' % (self.goal_x, self.goal_y))

        self.robot_dim = 4              # Distance to goal, angle to goal and the current angular and linear velocity
        self.state_dim = self.LIDAR_DIM + self.robot_dim    # Overall dimension of the state vector provided as input to Td3
        self.action_dim = 2
        self.max_action = 1             
        self.expl_noise = 0.5           # The exploration noise contribution start from a value and decrease alog the training process     # DA LEVARE????????????????????????
        expl_decay_steps = (
        500000  # Number of steps over which the initial exploration noise will decay over (Per ora non c'è)
        )
        expl_min = 0.1                  # Exploration noise after the decay


        self.state=np.zeros(self.state_dim)

        # Create the network
        self.network = td3(self.state_dim, self.action_dim, self.max_action)

        # Loading the weights of trained networks
        try:
            self.get_logger().info("Will load existing model.")
            #self.network.load("td3_velodyne", "./results/pytorch_models") # Original model
            #self.network.load("td3_velodyne_primoTrain", "/home/a/completeListenerPublisherNode_ws/results/pytorch_models")
            #self.network.load("td3_velodyne_terzoTrain", "/home/a/completeListenerPublisherNode_ws/results/pytorch_models")
            self.network.load("td3_velodyne_quartoTrain", "/home/a/completeListenerPublisherNode_ws/results/pytorch_models")
            
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

        # Create a timer that calls a function every 100ms
        self.timer_network = self.create_timer(0.1, self.network_callback) # Evaluate actions from network every 0.01 seconds

    def network_callback(self):

        if self.stop==0:

            # Print di debug                                                                    # DA LEVARE????????????????????????
            self.get_logger().info('*****************NETWORK CALLBACK*****************')
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


            # Get the action from the Td3 network
            action=self.network.get_action(np.array(self.state))
            #action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action) #GIULIOOOOOOOOOO QUI METTE IL RUMOREEEEEEEEE     # DA LEVARE????????????????????????
            a_in = [(action[0] + 1) / 2, action[1]]         # The linear velocity is bounded between 0 and 1 
            self.actions[0]=a_in[0]
            self.actions[1]=a_in[1]
            self.state[self.LIDAR_DIM+2:]=[self.actions[0],self.actions[1]]    # Assignment of actions to the state vector
            cmd_vel = Twist()
            cmd_vel.linear.x = a_in[0]*0.25                 # To scale to the maximum linear velocity
            cmd_vel.angular.z = a_in[1]*0.6                 # To scale to the maximum angular velocity
            self.cmd_vel_publisher.publish(cmd_vel)
            #self.get_logger().info('Publishing cmd_vel: linear=%f, angular=%f' % (cmd_vel.linear.x, cmd_vel.angular.z))      


    def scan_callback(self, msg):
        if self.stop==1:
            return
        self.lidarData=np.ones(self.LIDAR_DIM)*10.0
        min_angle = msg.angle_min
        max_angle = msg.angle_max
        angle_increment = msg.angle_increment
        angles = np.arange(min_angle, max_angle, angle_increment) #true for the real lidar, not for the neural network one that needs a different arrangement

        
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
        if min_laser < COLLISION_DIST:
            self.get_logger().info("Collision is detected!")
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0 # Stop the robot
            self.cmd_vel_publisher.publish(cmd_vel)
            self.stop=1
 

    def odom_unfiltered_callback(self, msg):
        if self.stop==1:
            return

        # ODOMETRY
        current_time = self.get_clock().now()
        
        delta_t = 0.0       # Elapsed time between the reception of two messages from Micro ros esp32 node
        if self.previousTime is not None and self.previousTime< current_time:
            delta_t = (current_time - self.previousTime).nanoseconds / 1e9      # Convert nanoseconds to seconds
        
        Ts = delta_t      
        #Ts=0.0172                                          # DA LEVARE???????????????
        self.previousTime = current_time
        linear_velocity = msg.twist.twist.linear.x
        angular_velocity = msg.twist.twist.angular.z
        # Print angular_velocity and linear_velocity
        #self.get_logger().info('Angular velocity: %f' % angular_velocity)                      # DA LEVARE???????????????
        # self.get_logger().info('***********ODOM UNFILTERED CALLBACK***********')
        #         # print self.x,y and theta
        # self.get_logger().info('OLD X: %f' % self.x)
        # self.get_logger().info('OLD Y: %f' % self.y)
        # self.get_logger().info('theta: %f' % self.theta)
        # self.get_logger().info(f'Received unfiltered odometry message: Delta t={Ts}')
        # self.get_logger().info(f'Received unfiltered odometry message: linear={linear_velocity}, angular={angular_velocity}')

        # Update position and orientation based on velocity command
        self.x += linear_velocity * Ts * np.cos(self.theta+angular_velocity*Ts/2)
        self.y += linear_velocity * Ts * np.sin(self.theta+angular_velocity*Ts/2)
        self.theta += angular_velocity * Ts

        # Print x, y and theta
        #self.get_logger().info('X: %f' % self.x)
        #self.get_logger().info('Y: %f' % self.y)
        #self.get_logger().info('Theta: %f' % self.theta)


        # print self.x,y and theta
        # self.get_logger().info('X: %f' % self.x)
        # self.get_logger().info('Y: %f' % self.y)
        # self.get_logger().info('theta: %f' % self.theta)
        # self.get_logger().info('*********************************')

        # Calculate the distance between the robot and the goal
        #distance = math.sqrt((self.x - self.goal_x)**2 + (self.y - self.goal_y)**2)      # DA LEVARE??????????????? Se l'istruzione sotto funzione questa non serve
        distance = np.linalg.norm(
            [self.x - self.goal_x, self.y - self.goal_y]
        )
        self.distance=distance

        # Calculate beta,: The angle between a the x axis of the reference frame and a vector connecting the robot to the goal 
        skew_x = self.goal_x - self.x           # x axis component of the vector connecting the robot to the goal
        skew_y = self.goal_y - self.y           # x axis component of the vector connecting the robot to the goal
        dot = skew_x * 1 + skew_y * 0           # Scalar product between reference frame x axis and the vector skew
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))   # Lenght of skew which is actually the distance between the robot and the goal
        # mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))      # DA LEVARE??????????????? Se l'istruzione sotto funzione questa non serve
        mag2 = 1;
        beta = math.acos(dot / (mag1 * mag2))   # Evaluating Beta
        if skew_y < 0:
            if skew_x < 0:              # Third quadrant of the Cartesian axes.
                beta = -beta 
            else:                       # Fourth quadrant of the Cartesian axes.
                beta = 0 - beta
        # Evaluating the correction angle to steer towards the target (angle_to_goal)
        self.angle_to_goal = beta - self.theta 
        if self.angle_to_goal > np.pi:
            # self.angle_to_goal = np.pi - self.angle_to_goal
            # self.angle_to_goal = -np.pi - self.angle_to_goal #***************PRATICAMENTE THETA=THETA-2*PI     # DA LEVARE???????????????????????? Se la riga sotto funziona
            self.angle_to_goal = self.angle_to_goal - 2*np.pi
        if self.angle_to_goal < -np.pi:
            # self.angle_to_goal = -np.pi - self.angle_to_goal
            # self.angle_to_goal = np.pi - self.angle_to_goal #**************** PRATICAMENTE THETA= THETA + 2*PI    # DA LEVARE????????????????????????    Se la riga sotto funziona
            self.angle_to_goal = self.angle_to_goal + 2*np.pi
        
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
