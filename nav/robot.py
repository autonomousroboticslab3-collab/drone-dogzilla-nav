"""機器人介面。換硬體只需新增一個子類別。"""
import numpy as np


class RobotBase:
    def get_pose(self):
        """回傳 (x, y, yaw)，地面座標系。"""
        raise NotImplementedError

    def send_cmd(self, v, w):
        """線速度 v (m/s)、角速度 w (rad/s)。"""
        raise NotImplementedError

    def stop(self):
        self.send_cmd(0.0, 0.0)


class SimRobot(RobotBase):
    """獨輪車模型模擬，用來在沒有硬體時驗證整條流程。"""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, dt=0.1):
        self.x, self.y, self.yaw, self.dt = x, y, yaw, dt

    def get_pose(self):
        return self.x, self.y, self.yaw

    def send_cmd(self, v, w):
        self.yaw += w * self.dt
        self.x += v * np.cos(self.yaw) * self.dt
        self.y += v * np.sin(self.yaw) * self.dt


class RosDogzilla(RobotBase):
    """Dogzilla S1 的 ROS 介面（待實機確認）。

    TODO:
      - 確認 cmd_vel topic 名稱（Yahboom 預設多為 /cmd_vel）。
      - 定位來源：初期用 /odom 加初始位姿；之後換成視覺（AprilTag）或
        空拍機 SLAM 中偵測機器人本體。
      - 需要一個「機器人 odom 座標 -> 地面座標」的轉換，見 main.py 的
        align_robot_to_map()。
    """

    def __init__(self, cmd_topic='/cmd_vel', odom_topic='/odom'):
        import rospy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        rospy.init_node('nav_bridge', anonymous=True)
        self._Twist = Twist
        self._pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self._pose = (0.0, 0.0, 0.0)
        rospy.Subscriber(odom_topic, Odometry, self._on_odom)

    def _on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        self._pose = (p.x, p.y, yaw)

    def get_pose(self):
        return self._pose

    def send_cmd(self, v, w):
        t = self._Twist()
        t.linear.x, t.angular.z = v, w
        self._pub.publish(t)
