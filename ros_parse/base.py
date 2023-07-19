import abc
from collections import deque, defaultdict

import pandas as pd
import rosbag


class BaseParser(abc.ABC):
    def __init__(self):
        super(BaseParser, self).__init__()
        self.data = defaultdict(deque)
        self.topic = ""

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def to_file(self, file_name):
        self.to_dataframe().to_csv(file_name, index=False)

    @abc.abstractmethod
    def parse(self, value: rosbag.bag.BagMessage):
        raise NotImplemented
