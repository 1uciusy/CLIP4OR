import bisect
import os
from itertools import accumulate

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize

from .dtw import DTWAlign

__all__ = ["OneBag", "IntegratedBags"]


def df_sort(df: pd.DataFrame):
    df["ts"] = df[["secs", "nsecs"]].apply(tuple, axis=1)
    df = df.sort_values(by="ts")
    del df["ts"]
    return df


img_aug = Compose([
    ToTensor(),
    Resize((224, 448))
])


class OneBag(Dataset):
    def __init__(self, file_dir, down_sample_factor=10, sparse_factor=10, n_his=160, n_future=80, img_aug=img_aug,
                 force_seg=False):
        # down_sample_factor 降采样比例，控制信号的频率高低
        # sparse_factor 控制样本间隔，样本间隔为 sparse_factor/(400/down_sample_factor)

        # print(file_dir)
        self.file_dir = file_dir
        self.sparse_factor = sparse_factor
        self.n_his = n_his
        self.n_future = n_future

        self.series_attr = ["Action", "Imu", "Pedals", "Shock", "State", "RPM"]
        # self.frame_attr = ["ObservHeightMap", "ObservImg", "ObservRGBMap"]
        self.frame_attr = ["ObservImg"]
        self.attr = self.series_attr + self.frame_attr

        self.align_helper = DTWAlign()
        self.img_helper = img_aug

        # if aligned data is not saved, use original data and align
        if "align.pkl" not in os.listdir(file_dir) or force_seg:
            data = {t: pd.read_csv(os.path.join(file_dir, t + ".csv")) for t in self.attr}

            # imu is supposed to have the highest frequency, it's used as the alignment reference
            data["Imu"] = df_sort(data["Imu"])

            for k, v in data.items():
                if k == "Imu":
                    continue
                v = df_sort(v)
                data[k] = self.align_timestamp(data["Imu"], v)

            df = pd.concat(data.values(), axis=1)
            col = [i for i, v in enumerate(df.columns) if v not in ["secs", "nsecs", "ts", "mapped_nsecs", ]]
            df = df.iloc[:, col]
            df = pd.concat((data["Imu"][["secs", "nsecs"]], df,), axis=1)
            df = df.iloc[::down_sample_factor, :]  # IMU is 400 hz, make it 400/down_sample_factor
            df.to_pickle(os.path.join(self.file_dir, "align.pkl"))
            df.mean(numeric_only=True).to_pickle(os.path.join(self.file_dir, "align_mean.pkl"))
            df.std(numeric_only=True).to_pickle(os.path.join(self.file_dir, "align_std.pkl"))
        else:
            df = pd.read_pickle(os.path.join(self.file_dir, "align.pkl"))

        df = df.loc[pd.isna(df).apply(any, axis=1) ^ 1]
        self.row = df.shape[0]
        self.length = (self.row - self.n_his - self.n_future) // self.sparse_factor

        if "pkl" not in os.listdir(self.file_dir):
            os.mkdir(os.path.join(self.file_dir, "pkl"))

        if f"{self.length - 1}.pkl" not in os.listdir(os.path.join(self.file_dir, "pkl")) or force_seg:
            for i in range(self.length):
                tmp = df.iloc[i * sparse_factor:i * sparse_factor + n_his + n_future, :]
                tmp.to_pickle(os.path.join(self.file_dir, "pkl", f"{i}.pkl"))

    def align_timestamp(self, main: pd.DataFrame, other: pd.DataFrame):
        """
        align two series to the main series on timestamps using dtw
        """
        main_ts = main[["secs", "nsecs"]].apply(tuple, axis=1)
        other_ts = other[["secs", "nsecs"]].apply(tuple, axis=1)
        align_ts = self.align_helper.align(main_ts.to_list(), other_ts.to_list())

        tmp_df = pd.DataFrame({"ts": align_ts})
        tmp_df["secs"] = tmp_df["ts"].apply(lambda x: x[0])
        tmp_df["mapped_nsecs"] = tmp_df["ts"].apply(lambda x: x[1])
        tmp_df["nsecs"] = tmp_df["ts"].apply(lambda x: x[2])

        res = pd.merge(tmp_df, other, on=["secs", "nsecs"])
        return df_sort(res)

    def __getitem__(self, idx: int):
        data = pd.read_pickle(os.path.join(self.file_dir, "pkl", f"{idx}.pkl"))
        mean = pd.read_pickle(os.path.join(self.file_dir, "align_mean.pkl"))
        std = pd.read_pickle(os.path.join(self.file_dir, "align_std.pkl"))
        res = [data.iloc[:, :29].values, mean.values.reshape((1, -1)), std.values.reshape((1, -1)), ]
        for i in range(29, data.shape[1]):
            fn = os.path.join(self.file_dir, str(data.iloc[self.n_his, i]))
            if "img" in fn:
                img = np.array(Image.open(fn), dtype=np.uint8)
                res.append(self.img_helper(img))
        return tuple(res)

    def __len__(self):
        return max(self.length, 0)


class IntegratedBags(Dataset):
    def __init__(self, file_dir, down_sample_factor=10, sparse_factor=10, n_his=160, n_future=80, img_aug=img_aug,
                 force_seg=False, train=True, split="CLIP4OR"):
        """
        Integrate all the single bag into one dataset and support index
        :param file_dir:
        :param down_sample_factor:
        :param sparse_factor:
        :param n_his:
        :param n_future:
        :param img_aug:
        :param force_seg:
        :param train:
        :param split: The way to split dataset
        """
        super(IntegratedBags, self).__init__()
        assert os.path.isdir(file_dir)
        assert split == "CLIP4OR" or split == "TartanDrive"
        self.file_dir = file_dir
        self.train = train

        bag_params = []
        for f in os.listdir(file_dir):
            if not os.path.isdir(os.path.join(self.file_dir, f)):
                continue
            p = (os.path.join(self.file_dir, f), down_sample_factor, sparse_factor, n_his, n_future, img_aug, force_seg)
            bag_params.append(p)
        bag_params.sort()

        self.data = []
        print("The first time to build dataset with current settings may take half an hour to create samples")
        for param in bag_params:
            onebag = OneBag(*param)
            if len(onebag) > 0:
                self.data.append(onebag)
        self.data_length = [0] + list(accumulate(len(data) for data in self.data))

    def __len__(self):
        return self.data_length[-1] // 4 * 3 if self.train else self.data_length[-1] - self.data_length[-1] // 4 * 3

    def __getitem__(self, idx):
        idx += 0 if self.train else self.data_length[-1] // 4 * 3
        bag_idx = bisect.bisect_right(self.data_length, idx) - 1
        idx -= self.data_length[bag_idx]
        return self.data[bag_idx][idx]

    def mode(self, train: bool):
        # better not to change it, just define a new one with train=False
        self.train = train


if __name__ == "__main__":
    # 20210903_122 # no height map or rgb map
    # 20210903_15 # no imu, img, height map or rgb map
    # 20210903_16 # no imu, img, height map or rgb map
    # 20210903_17 # no action
    # 20210903_18 # no action, imu, img, height map or rgb map
    # 20210903_19 # no action, imu, img, height map or rgb map
    # 20210903_20 # no action, imu, img, height map or rgb map
    # 20210903_21 # no action, imu, img, height map or rgb map
    # 20210903_22 # no imu, img, height map or rgb map
    # 20210910_33 # no action
    # 20210910_34 # no action
    # 20210826_97 # State error, drastically change after time (1630004901,283001759)
    test = OneBag("/mnt/filesystem4/yangyi/parsed_bag/20210826_100", force_seg=True)
    a = pd.read_pickle("/mnt/filesystem4/yangyi/parsed_bag/20210826_100/align.pkl")
    b = pd.read_pickle("/mnt/filesystem4/yangyi/parsed_bag/20210826_100/pkl/0.pkl")
    c = pd.read_csv("/mnt/filesystem4/yangyi/parsed_bag/20210826_100/State.csv")

    test2 = IntegratedBags("/mnt/filesystem4/yangyi/parsed_bag")
    arr = []
    for i in range(len(test2)):
        sensor = test2[i][0][:, 2:]
        sensor[:, 16:19] -= sensor[:1, 16:19]
        arr.append(sensor)
    arr = np.array(arr)
    arr = arr.reshape((-1, 27))
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    np.save("mean.npy", mean)
    np.save("std.npy", std)

    test3 = IntegratedBags("/mnt/filesystem4/yangyi/parsed_bag", train=False)
    arr = []
    for i in range(len(test3)):
        sensor = test3[i][0][:, 2:]
        sensor[:, 16:19] -= sensor[:1, 16:19]
        arr.append(sensor)
    arr = np.array(arr)
    arr = arr.reshape((-1, 27))
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    np.save("mean_test.npy", mean)
    np.save("std_test.npy", std)

    test4 = IntegratedBags("/mnt/filesystem4/yangyi/parsed_bag", n_his=0, n_future=240)

