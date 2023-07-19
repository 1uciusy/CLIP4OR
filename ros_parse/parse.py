import multiprocessing
import os
from typing import List

import rosbag

from .base import BaseParser
from .parser_zoo import *

__all__ = ["Parser"]


class Parser(object):
    def __init__(self, path: str, save_path: str, overwrite=False, *args):  # "/home/yangyi/wd/tartan_drive"
        self.bags = self.recursive_bag_finder(path)
        self.parsers = {cls.__name__: cls for cls in args if issubclass(cls, BaseParser)}
        self.save_path = save_path
        self.overwrite = overwrite

    def recursive_bag_finder(self, path: str) -> List[str]:
        if not os.path.isdir(path):
            return [path, ] if path.endswith(".bag") else []
        res = []
        sub_folder = os.listdir(path)
        for f in sub_folder:
            if f.startswith("."):
                continue
            res += self.recursive_bag_finder(os.path.join(path, f))
        res.sort()
        return res

    def parse(self, bag):
        os.chdir(self.save_path)
        folder = bag.split("/")[-1].split(".")[0]
        try:
            os.mkdir(folder)
            os.mkdir(folder + "/rgb_map")
            os.mkdir(folder + "/height_map")
            os.mkdir(folder + "/img")
        except FileExistsError:
            pass
        files = os.listdir(folder)
        os.chdir(folder)
        if not self.overwrite and all(map(lambda k: k + ".csv" in files, self.parsers.keys())):
            return
        print(folder)

        f = rosbag.Bag(bag)
        parsers = {k: v() for k, v in self.parsers.items()}
        for i, v in enumerate(f):
            for cls_name, instance in parsers.items():
                instance.parse(v)
        for k, v in parsers.items():
            v.to_file(k + ".csv")
        f.close()

    def parallel_parse(self, size=4):
        if size == 1:
            for b in self.bags:
                self.parse(b)
            return
        with multiprocessing.Pool(size) as pool:
            pool.map(self.parse, self.bags)
        pool.close()


if __name__ == '__main__':
    parser = (State, Action, ObservImg, ObservRGBMap, ObservHeightMap, Imu, Shock, RPM, Pedals)
    # parser = (ObservHeightMap,)
    p = Parser("/home/yangyi/wd/tartan_drive/bags", "/mnt/filesystem4/yangyi/parsed_bag", False, *parser)
    # p.parse(p.bags[0])
    p.parallel_parse(size=1)
