import os
import sys
import time
from collections import deque

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

work_dir = "/home/yangyi/wd/paper"
parsed_bag_dir = "/mnt/filesystem4/yangyi/parsed_bag"

sys.path.append(work_dir)
from model.clip4or import CLIP4OR
from utils.dataset import IntegratedBags

os.chdir(work_dir)
nets = {
    "v": CLIP4OR(use_action=False, use_vision=True),
    "av": CLIP4OR(use_action=True, use_vision=True),
}
nets["av"].load_state_dict(torch.load("./checkpoint/av_10e_CLIP4OR_9.cpt")["model"])
nets["v"].load_state_dict(torch.load("./checkpoint/vs_10e_CLIP4OR_9.cpt")["model"])
data_set = IntegratedBags(parsed_bag_dir, train=False)
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

logit = dict()
for k, v in nets.items():
    nets[k] = v.eval().cuda(2)
    feature = deque()
    data_cache = deque()
    print(k, data_cache)
    for data in data_loader:
        data_cache.append((data[0], data[-1]))
        s = data[0][:, :, 2:].float().cuda(2)
        v = data[-1].float().cuda(2)
        b, c, h, w = v.shape
        v = v[:, :, h // 2:, :]
        std, mean = torch.std_mean(s, dim=1, keepdim=True, unbiased=True)
        s = (s - mean) / torch.max(std, torch.ones_like(std))
        # s = s[:, :, 2:]
        with torch.no_grad():
            _, vf, sf, af, vision, sensor = nets[k](v, s)
        feature.append((vf, sf, af, sensor, vision))

    sensor_feature = torch.cat(tuple(sensor for vf, sf, af, sensor, vision in feature), dim=0)
    vision_feature = torch.cat(tuple(vision for vf, sf, af, sensor, vision in feature), dim=0)

    logit[k] = sensor_feature @ vision_feature.t()


# rank-n 的命中概率
def rank(n=1, logit=logit["av"]):
    diag = torch.arange(15369).reshape((-1, 1))
    v, idx = torch.topk(logit, n, dim=1)
    idx = idx.cpu()
    match = torch.abs(idx - diag)
    # return torch.sum(torch.min(match, dim=1)[0] <= 1) / 15369 # including adjacent sample
    return torch.sum(torch.min(match, dim=1)[0] <= 0) / 15369


for k in (1, 5, 10, 50):
    print(f"w action, test: {len(data_set)}，state->vision&action, top{k}acc:{rank(k, logit['av'])}")
    print(f"w action，test: {len(data_set)}，vision&action->state, top{k}acc:{rank(k, logit['av'].t())}")
    print(f"w/o action，test: {len(data_set)}，state->vision, top{k}acc:{rank(k, logit['v'])}")
    print(f"w/o action，test: {len(data_set)}，vision->state, top{k}acc:{rank(k, logit['v'].t())}")

for idx in torch.topk(torch.diag(logit["av"]), k=11)[1]:
    print(idx)
    torch.randn()
    # analysis(int(idx))


# 找样本
def analysis(idx=8370, save_sensor=False, save_vision=False):
    b = torch.topk(logit["av"][idx], k=5)
    s = data_cache[idx][0][0, :, 7:10]
    plt.figure(figsize=(10, 5))
    plt.plot(s)
    # plt.plot(data_cache[idx][0][0, :, 2:3])
    plt.text(5, -16,
             f"linear acc var: {'%.2f' % torch.mean(torch.var(data_cache[idx][0][:, :, 7:10], dim=1, keepdim=True))}",
             fontsize=14)
    plt.text(5, -18,
             f"mean throttle: {'%.2f' % torch.mean(data_cache[idx][0][:, :, 2])}",
             fontsize=14)
    plt.ylim((-20, 10))
    if save_sensor:
        plt.savefig(f"{idx}.png", bbox_inches="tight", pad_inches=0)
    plt.show()
    for ith, v in enumerate(b[1]):
        img = (data_cache[v][1].squeeze().permute(1, 2, 0) * 255).int().numpy()
        if int(v) == idx:
            ax = plt.gca()
            ax.add_patch(plt.Rectangle((0, 0), 448, 224, color="red", fill=False, linewidth=1.5))
        plt.imshow(img)
        plt.margins(0, 0)
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        if save_vision:
            plt.savefig(f"{idx}_{v}_{ith}.png", bbox_inches="tight", pad_inches=0)
        plt.show()


torch.manual_seed(9999)
for v in torch.randint(0, len(data_set) - 1, (10,)):
    print(v)
    analysis(int(v), True, True)
    time.sleep(1)


def synthetic(idx, fake_throttle=0):
    idx = 0
    fake_throttle = 0
    data = data_set[idx]
    v = data[-1][None,...].float().cuda(2)
    s = torch.from_numpy(data[0][None, ...]).float().cuda(2)
    s[:,:,2] = fake_throttle

    return

