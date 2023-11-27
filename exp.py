import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

from Informer2020.models.model import Informer
from model.clip4or import CLIP4OR

__all__ = ["EXP"]


class GaussianNoise(nn.Module):
    def __init__(self, mean, var):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.var = var

    def forward(self, x: torch.Tensor):
        n = torch.randn_like(x)
        n *= self.var ** 0.5
        n += self.mean
        return x + n


class FeatureNormalizer(nn.Module):
    def __init__(self):
        super(FeatureNormalizer, self).__init__()

    def forward(self, x: torch.Tensor):
        return x / torch.norm(x, dim=1, keepdim=True)


class EXP(object):
    def __init__(self, rank, clip4or_cfg=None, debug=False, ):
        self.rank = rank
        self.debug = debug
        self.setting = {
            "informer": dict(),
            "CLIP4OR": dict(),
            "CLIP4OR_v": dict(),
            "CLIP4OR_s": dict(),
        }
        # self.setting["informer"]["model"] = Informer(enc_in=27, dec_in=2, c_out=7, seq_len=20, label_len=0, out_len=20,
        #                                              factor=4, d_model=128, n_heads=9, d_ff=128, dropout=0.0)
        self.setting["informer"]["model"] = Informer(enc_in=27, dec_in=2, c_out=7, seq_len=40, label_len=0, out_len=20,
                                                     factor=4, d_model=128, n_heads=9, d_ff=128, dropout=0.0)
        self.setting["CLIP4OR"]["model"] = CLIP4OR() if not clip4or_cfg else CLIP4OR(**clip4or_cfg)
        self.setting["CLIP4OR_v"]["model"] = nn.Sequential(
            CLIP4OR(imagenet_pretrained=True).vision_encoder,
            FeatureNormalizer(),
            CLIP4OR(imagenet_pretrained=True).vision_projector,
            FeatureNormalizer(),
        )
        self.setting["CLIP4OR_s"]["model"] = nn.Sequential(
            CLIP4OR(imagenet_pretrained=True).state_encoder,
            FeatureNormalizer(),
            CLIP4OR(imagenet_pretrained=True).state_projector,
            FeatureNormalizer(),
        )

        # non-parametric functions that may be used
        self.mse_loss = nn.MSELoss().cuda(self.rank)
        self.bce_loss = nn.BCELoss().cuda(self.rank)
        self.ce_loss = nn.CrossEntropyLoss().cuda(self.rank)
        self.sigmoid = nn.Sigmoid().cuda(self.rank)
        self.softmax0 = nn.Softmax(dim=0).cuda(self.rank)
        self.softmax1 = nn.Softmax(dim=1).cuda(self.rank)

        # augmentation settings for single modality experiments
        self.v_aug1 = transforms.Compose([
            transforms.GaussianBlur(kernel_size=5, ),
            transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, ),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomAutocontrast(),
            GaussianNoise(0, .001)
        ])
        self.v_aug2 = nn.Identity()
        self.s_aug1 = transforms.Compose([
            GaussianNoise(0, .001)
        ])
        self.s_aug2 = nn.Identity()

        for k, v in self.setting.items():
            if not v:
                continue
            v["model"] = v["model"].float().cuda(self.rank)
            v.update(self.setting_gen(v["model"]))

    def models_exclude(self, *args):
        """
        It's not necessary to initialize all the models this function is used to exclude redundant models
        :param args: models to exclude
        """
        for k in args:
            if k in self.setting:
                self.setting.pop(k)

    def setting_gen(self, model: nn.Module):
        if not self.debug:
            res = {"ddp": DDP(model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)}
        else:
            res = {"ddp": model}  # not actually ddp, just for single card debug use
        res["optimizer"] = torch.optim.Adam(res["ddp"].parameters())
        res["scheduler"] = LambdaLR(res["optimizer"], lambda x: (x + 1) / 10 if x < 10 else 0.99 ** (x - 10))
        return res

    # experiment global settings, such as switching model to train/eval, or taking a step in learning rate scheduler
    def train(self):
        for k, v in self.setting.items():
            if not v:
                continue
            v["model"].train()
            v["ddp"].train()

    def eval(self):
        for k, v in self.setting.items():
            if not v:
                continue
            v["model"].eval()
            v["ddp"].eval()

    def scheduler_step(self):
        for k, v in self.setting.items():
            if not v:
                continue
            v["scheduler"].step()

    # current best practice
    # 位置信息并不是平稳序列，所以需要考虑更多其他的标准化方法
    # 对位置信息进行差分+标准化
    # 对x和y同时降采样，降采样方法为平均降采样，然后差分
    def data_preprocess(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        ts = sensor[:, ::4, :2].long()  # time stamp
        s = sensor[:, :, 2:]  # all the sensor data
        b, t, n = s.shape
        # the original freq is 40 Hz, down-sample it to 10 Hz by averaging every 4 frames
        s = s.reshape((b, 4, -1, n))
        s = torch.mean(s, dim=1)
        # differentiate position x,y,z and pad the missing start value with zeros
        s[:, 1:, 16:19] = s[:, 1:, 16:19] - s[:, :-1, 16:19]
        s[:, :1, 16:19] = s[:, :1, 16:19] - s[:, :1, 16:19]
        # the first 40 frames are treated as x
        # calculate the first 40 frames' mean and std to normalize all the 60 frames
        std, mean = torch.std_mean(s[:, :40, :], dim=1, unbiased=True, keepdim=True)
        # in case small std drastically increase data to be normalized
        std = torch.max(std, torch.ones(std.shape).cuda(self.rank))

        s_train = s[:, :40, :]
        s_train = (s_train - mean) / std

        # informer input action and traj ground truth tha needed to be predicted
        action = (s[:, 39:59, :2] - mean[:, :, :2]) / std[:, :, :2]
        traj_gt = (s[:, 40:, 16:16 + 7] - mean[:, :, 16:16 + 7]) / std[:, :, 16:16 + 7]

        return s_train, action, traj_gt, ts, std, mean

    def supervised_train(self, data):
        """
        This is dynamics predictions using informer without pretrained features
        :param data:
        :return:
        """
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess(data)
        output = self.setting["informer"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                 x_mark_dec=ts[:, 39:59, :])
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer"]["optimizer"].step()
        return loss.detach()

    def supervised_eval(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess(data)
        with torch.no_grad():
            output = self.setting["informer"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                     x_mark_dec=ts[:, 39:59, :])
        output = output * std[:, :, 16:19]
        traj_gt = traj_gt * std[:, :, 16:19]

        metric = torch.mean((traj_gt - output) ** 2) ** 0.5
        metric2 = torch.mean((torch.cumsum(traj_gt, dim=1) - torch.cumsum(output, dim=1)) ** 2) ** 0.5
        return metric, metric2

    # 预训练第二版，instance级的自监督，（图像+动作）&传感器
    def pretrain(self, data, finetune=False):
        s = data[0][:, :, 2:].float().cuda(self.rank)
        v = data[-1].float().cuda(self.rank)
        b, c, h, w = v.shape
        v = v[:, :, h // 2:, :]
        std, mean = torch.std_mean(s, dim=1, keepdim=True, unbiased=True)
        s = (s - mean) / torch.max(std, torch.ones_like(std))
        logit, vf, sf, af, v_emb, s_emb = self.setting["CLIP4OR"]["ddp"](v, s)
        if finetune:
            return vf, sf, af

        b = s.shape[0]
        gt = torch.eye(b).cuda(self.rank)

        loss = self.ce_loss(logit, gt) / 2 + self.ce_loss(logit.t(), gt) / 2
        self.setting["CLIP4OR"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["CLIP4OR"]["optimizer"].step()

        return loss.detach(), \
            torch.sum(torch.diag(self.softmax0(logit))) / b, \
            torch.sum(torch.diag(self.softmax1(logit))) / b

    def finetune_train(self, data):
        # use data_preprocess4
        self.setting["CLIP4OR"]["ddp"].eval()
        with torch.no_grad():
            vf, sf, af = self.pretrain(data, finetune=True)

        s_train, action, traj_gt, ts, std, mean = self.data_preprocess(data)
        output = self.setting["informer"]["ddp"](x_enc=s_train[:, 20:40, :], x_mark_enc=ts[:, 20:40, :],
                                                 x_dec=action, x_mark_dec=ts[:, 39:59, :],
                                                 pretrain_vision=vf, pretrain_sensor=sf)
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer"]["optimizer"].step()
        return loss.detach()

    def finetune_eval(self, data):
        self.setting["CLIP4OR"]["ddp"].eval()
        s_eval, action, traj_gt, ts, std, mean = self.data_preprocess(data)
        with torch.no_grad():
            vf, sf, af = self.pretrain(data, finetune=True)
            output = self.setting["informer"]["ddp"](x_enc=s_eval[:, 20:40, :], x_mark_enc=ts[:, 20:40, :],
                                                     x_dec=action, x_mark_dec=ts[:, 39:59, :],
                                                     pretrain_vision=vf, pretrain_sensor=sf)
        output = output * std[:, :, 16:16 + 7]
        quaternion = traj_gt[:, :, 3:] * std[:, :, 16 + 3:16 + 3 + 4]
        traj_gt = traj_gt[:, :, :3] * std[:, :, 16:16 + 3]

        metric = torch.mean((traj_gt - output[:, :, :3]) ** 2) ** 0.5
        metric_traj = (torch.cumsum(traj_gt, dim=1) - torch.cumsum(output[:, :, :3], dim=1)) ** 2
        metric_quat = (quaternion - output[:, :, 3:]) ** 2
        metric2 = torch.concat((metric_traj, metric_quat), dim=-1)
        metric2 = torch.mean(metric2) ** 0.5

        return metric, metric2

    def pretrain_single_modality(self, data):
        """
        train vision and sensor single modality pretrain model at the same time
        :param data:
        :return: loss of two pretraining model and the diagonal sum of similarity matrix
        """
        s = data[0][:, :, 2:].float().cuda(self.rank)
        v = data[-1].float().cuda(self.rank)
        b, c, h, w = v.shape
        v = v[:, :, h // 2:, :]
        std, mean = torch.std_mean(s, dim=1, keepdim=True, unbiased=True)
        s = (s - mean) / torch.max(std, torch.ones_like(std))

        v1 = self.v_aug1(v)
        v2 = self.v_aug2(v)
        v_1 = self.setting["CLIP4OR_v"]["ddp"](v1)
        v_2 = self.setting["CLIP4OR_v"]["ddp"](v2)
        logit_v = v_1 @ v_2.t()

        s1 = self.s_aug1(s)
        s2 = self.s_aug2(s)
        s_1 = self.setting["CLIP4OR_s"]["ddp"](s1)
        s_2 = self.setting["CLIP4OR_s"]["ddp"](s2)
        logit_s = s_1 @ s_2.t()

        gt = torch.eye(b).cuda(self.rank)
        loss = .0
        loss += self.ce_loss(logit_s, gt)
        loss += self.ce_loss(logit_v, gt)
        return loss, \
            torch.sum(torch.diag(self.softmax0(logit_v))) / b, \
            torch.sum(torch.diag(self.softmax0(logit_s))) / b


if __name__ == '__main__':
    pass
