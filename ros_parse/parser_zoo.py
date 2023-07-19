import numpy as np
import rosbag
from PIL import Image

from .base import BaseParser

__all__ = ["State", "Action", "ObservImg", "ObservRGBMap", "ObservHeightMap", "Imu", "Shock", "RPM", "Pedals"]


# /odometry/filtered_odom
class State(BaseParser):
    def __init__(self):
        super(State, self).__init__()
        self.topic = "/odometry/filtered_odom"
        self.init_pos = None

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.message.header.stamp.secs)
        self.data["nsecs"].append(value.message.header.stamp.nsecs)

        if not self.init_pos:
            self.init_pos = (value.message.pose.pose.position.x,
                             value.message.pose.pose.position.y,
                             value.message.pose.pose.position.z)

        self.data["px"].append(value.message.pose.pose.position.x)
        self.data["py"].append(value.message.pose.pose.position.y)
        self.data["pz"].append(value.message.pose.pose.position.z)
        print(value.message.header.seq, self.data["px"][-1], self.data["py"][-1], self.data["pz"][-1])

        self.data["px"][-1] -= self.init_pos[0]
        self.data["py"][-1] -= self.init_pos[1]
        self.data["pz"][-1] -= self.init_pos[2]
        print(self.data["px"][-1], self.data["py"][-1], self.data["pz"][-1])

        self.data["qx"].append(value.message.pose.pose.orientation.x)
        self.data["qy"].append(value.message.pose.pose.orientation.y)
        self.data["qz"].append(value.message.pose.pose.orientation.z)
        self.data["qw"].append(value.message.pose.pose.orientation.w)


# /cmd
class Action(BaseParser):
    def __init__(self):
        super(Action, self).__init__()
        self.topic = "/cmd"

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.message.header.stamp.secs)
        self.data["nsecs"].append(value.message.header.stamp.nsecs)

        self.data["throttle"].append(value.message.twist.linear.x)
        self.data["steer"].append(value.message.twist.angular.z)
        # self.data["vy"].append(value.message.twist.linear.y)
        # self.data["vz"].append(value.message.twist.linear.z)
        # self.data["wx"].append(value.message.twist.angular.x)
        # self.data["wy"].append(value.message.twist.angular.y)


# /multisense/left/image_rect_color
class ObservImg(BaseParser):
    def __init__(self):
        super(ObservImg, self).__init__()
        self.topic = "/multisense/left/image_rect_color"

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.message.header.stamp.secs)
        self.data["nsecs"].append(value.message.header.stamp.nsecs)

        file_name = "./img/"
        file_name += str(value.message.header.stamp.secs)
        file_name += "_" + str(value.message.header.stamp.nsecs)
        file_name += ".png"
        self.data["file_name"].append(file_name)

        arr = np.array(list(value.message.data), dtype=np.uint8)
        arr = arr.reshape((value.message.height, value.message.width, -1))
        arr = arr[:, :, 2::-1]
        Image.fromarray(arr, mode="RGB").resize((value.message.width // 2, value.message.height // 2),
                                                resample=Image.Resampling.BICUBIC).save(file_name)


# /local_rgb_map
class ObservRGBMap(BaseParser):
    def __init__(self):
        super(ObservRGBMap, self).__init__()
        self.topic = "/local_rgb_map"

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.message.header.stamp.secs)
        self.data["nsecs"].append(value.message.header.stamp.nsecs)

        file_name = "./rgb_map/"
        file_name += str(value.message.header.stamp.secs)
        file_name += "_" + str(value.message.header.stamp.nsecs)
        file_name += ".png"
        self.data["file_name"].append(file_name)

        arr = np.array(list(value.message.data), dtype=np.uint8)
        arr = arr.reshape((value.message.height, value.message.width, -1))
        arr = arr[:, :, 2::-1]
        Image.fromarray(arr, mode="RGB").resize((224, 224), resample=Image.Resampling.BICUBIC).save(file_name)


# /local_height_map
class ObservHeightMap(BaseParser):
    def __init__(self):
        super(ObservHeightMap, self).__init__()
        self.topic = '/local_height_map'

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.timestamp.secs)
        self.data["nsecs"].append(value.timestamp.nsecs)

        file_name = "./height_map/"
        file_name += str(value.timestamp.secs)
        file_name += "_" + str(value.timestamp.nsecs)
        file_name += ".npy"
        self.data["file_name"].append(file_name)

        arr = (np.array(value.message.data[0].data, dtype=np.float16).reshape((501, 501)),
               np.array(value.message.data[1].data, dtype=np.float16).reshape((501, 501)))
        arr = np.stack(arr)
        np.save(file_name, arr)


# /multisense/imu/imu_data
class Imu(BaseParser):
    def __init__(self):
        super(Imu, self).__init__()
        self.topic = '/multisense/imu/imu_data'

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.message.header.stamp.secs)
        self.data["nsecs"].append(value.message.header.stamp.nsecs)

        # self.data["orientation_x"].append(value.message.orientation.x)
        # self.data["orientation_y"].append(value.message.orientation.y)
        # self.data["orientation_z"].append(value.message.orientation.z)
        # self.data["orientation_w"].append(value.message.orientation.w)
        self.data["angular_velocity_x"].append(value.message.angular_velocity.x)
        self.data["angular_velocity_y"].append(value.message.angular_velocity.y)
        self.data["angular_velocity_z"].append(value.message.angular_velocity.z)
        self.data["linear_acceleration_x"].append(value.message.linear_acceleration.x)
        self.data["linear_acceleration_y"].append(value.message.linear_acceleration.y)
        self.data["linear_acceleration_z"].append(value.message.linear_acceleration.z)


# /shock_pos
class Shock(BaseParser):
    def __init__(self):
        super(Shock, self).__init__()
        self.topic = "/shock_pos"

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.message.header.stamp.secs)
        self.data["nsecs"].append(value.message.header.stamp.nsecs)

        self.data["front_left"].append(value.message.front_left)
        self.data["front_right"].append(value.message.front_right)
        self.data["rear_left"].append(value.message.rear_left)
        self.data["rear_right"].append(value.message.rear_right)


# /wheel_rpm
class RPM(BaseParser):
    def __init__(self):
        super(RPM, self).__init__()
        self.topic = "/wheel_rpm"

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.message.header.stamp.secs)
        self.data["nsecs"].append(value.message.header.stamp.nsecs)

        self.data["front_left"].append(value.message.front_left)
        self.data["front_right"].append(value.message.front_right)
        self.data["rear_left"].append(value.message.rear_left)
        self.data["rear_right"].append(value.message.rear_right)


# /controls
class Pedals(BaseParser):
    def __init__(self):
        super(Pedals, self).__init__()
        self.topic = "/controls"

    def parse(self, value: rosbag.bag.BagMessage):
        if value.topic != self.topic:
            return
        self.data["secs"].append(value.message.header.stamp.secs)
        self.data["nsecs"].append(value.message.header.stamp.nsecs)

        self.data["throttle"].append(value.message.throttle)
        self.data["brake"].append(value.message.brake)
        self.data["brake_invention"].append(int(value.message.brake > 50))
        self.data["throttle_invention"].append(int(value.message.throttle > 50))


if __name__ == "__main__":
    f = rosbag.Bag("/home/yangyi/wd/tartan_drive/bags/20210826_heightmaps_2/20210826_99.bag")
    cache = dict()
    parser_instance = State()
    for i, v in enumerate(f):
        parser_instance.parse(v)
        cache[v.topic] = v
    f.close()
