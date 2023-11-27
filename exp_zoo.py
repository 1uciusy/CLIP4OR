import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

from Informer2020.models.model import Informer
from model.clip4or import CLIP4OR
from utils.ts_similarity import batch_sim

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


class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()

    def forward(self, x: torch.Tensor):
        return x / torch.norm(x, dim=1, keepdim=True)


class PatchTSTConfig(object):
    def __init__(self):  # (b, 160+80, 27)
        self.enc_in = 27
        self.seq_len = 160
        self.pred_len = 80
        self.e_layers = 6
        self.n_heads = 16
        self.d_model = 128
        self.d_ff = 128
        self.dropout = 0.1
        self.fc_dropout = 0.1
        self.head_dropout = 0.1
        self.individual = 0  # default
        self.patch_len = 10
        self.stride = 5
        self.padding_patch = "end"  # default
        self.revin = 1  # default
        self.affine = 0  # default
        self.subtract_last = 0  # default
        self.decomposition = 0  # default
        self.kernel_size = 25  # default, won't work cuz decomposition is set to 0


class EXP(object):
    def __init__(self, rank, debug=False):
        self.rank = rank
        self.debug = debug
        self.setting = {
            "informer_best": dict(),
            "informer_test1": dict(),
            "informer_test2": dict(),
            "informer_test3": dict(),
            "informer_test4": dict(),
            "informer_test4clip": dict(),
            "informer_test4s": dict(),
            "informer_test5": dict(),
            "PatchTST_test1": dict(),
            "PatchTST_test2": dict(),
            "informer_test6": dict(),
            "PatchTST_test3": dict(),
            "PatchTST_test4": dict(),
            "Clip": dict(),
            "Clip_240": dict(),
            "Clip_240_v_as": dict(),
            "Clip_240_v": dict(),
            "Clip_240_scratch": dict(),
            "Clip_240_scratch_v": dict(),
            "Clip_240_scratch_s": dict(),
        }
        # self.setting["informer_best"]["model"] = Informer(enc_in=27, dec_in=2, c_out=3, seq_len=160, label_len=0,
        #                                                   out_len=80, factor=4, d_model=128, n_heads=9, d_ff=128,
        #                                                   dropout=0.0)
        # self.setting["informer_test1"]["model"] = Informer(enc_in=27, dec_in=2, c_out=3, seq_len=160, label_len=0,
        #                                                    out_len=80, factor=4, d_model=128, n_heads=9, d_ff=128,
        #                                                    dropout=0.0)
        # self.setting["informer_test2"]["model"] = Informer(enc_in=27, dec_in=2, c_out=3, seq_len=160, label_len=0,
        #                                                    out_len=80, factor=4, d_model=128, n_heads=9, d_ff=128,
        #                                                    dropout=0.0)
        # self.setting["informer_test3"]["model"] = Informer(enc_in=27, dec_in=2, c_out=3, seq_len=160, label_len=0,
        #                                                    out_len=80, factor=4, d_model=128, n_heads=9, d_ff=128,
        #                                                    dropout=0.0)
        self.setting["informer_test4"]["model"] = Informer(enc_in=27, dec_in=2, c_out=7, seq_len=40, label_len=0,
                                                           out_len=20, factor=4, d_model=128, n_heads=9, d_ff=128,
                                                           dropout=0.0)
        self.setting["informer_test4clip"]["model"] = Informer(enc_in=27, dec_in=2, c_out=7, seq_len=40, label_len=0,
                                                               out_len=20, factor=4, d_model=128, n_heads=9, d_ff=128,
                                                               dropout=0.0)
        self.setting["informer_test4s"]["model"] = Informer(enc_in=27, dec_in=2, c_out=7, seq_len=40, label_len=0,
                                                            out_len=20, factor=4, d_model=128, n_heads=9, d_ff=128,
                                                            dropout=0.0)
        # self.setting["informer_test5"]["model"] = Informer(enc_in=27, dec_in=2, c_out=3, seq_len=160, label_len=0,
        #                                                    out_len=20, factor=4, d_model=128, n_heads=9, d_ff=128,
        #                                                    dropout=0.0)
        # self.setting["informer_test6"]["model"] = Informer(enc_in=27, dec_in=2, c_out=3, seq_len=40, label_len=0,
        #                                                    out_len=20, factor=4, d_model=128, n_heads=9, d_ff=128,
        #                                                    dropout=0.0)
        #
        # self.patch_tst_config = PatchTSTConfig()
        # self.setting["PatchTST_test1"]["model"] = PatchTSTModel(copy.deepcopy(self.patch_tst_config))
        # self.setting["PatchTST_test3"]["model"] = PatchTSTModel(copy.deepcopy(self.patch_tst_config))
        # self.patch_tst_config.pred_len = 20
        # self.patch_tst_config.seq_len = 40
        # self.setting["PatchTST_test2"]["model"] = PatchTSTModel(copy.deepcopy(self.patch_tst_config))
        # self.setting["PatchTST_test4"]["model"] = PatchTSTModel(copy.deepcopy(self.patch_tst_config))

        self.setting["Clip"]["model"] = CLIP4OR(scale=False)
        self.setting["Clip_240"]["model"] = CLIP4OR(imagenet_pretrained=True)
        # self.setting["Clip_240"]["model"].load_state_dict(torch.load("./checkpoint/Clip_240_9.cpt")["model"])
        self.setting["Clip_240_v_as"]["model"] = CLIP4OR(imagenet_pretrained=False)


        self.setting["Clip_240_v"]["model"] = nn.Sequential(
            CLIP4OR(imagenet_pretrained=True).vision_encoder,
            Normalizer(),
            CLIP4OR(imagenet_pretrained=True).vision_projector,
            Normalizer(),
        )
        self.setting["Clip_240_scratch"]["model"] = CLIP4OR(imagenet_pretrained=False)
        self.setting["Clip_240_scratch_v"]["model"] = nn.Sequential(
            CLIP4OR(imagenet_pretrained=False).vision_encoder,
            Normalizer(),
            CLIP4OR(imagenet_pretrained=False).vision_projector,
            Normalizer(),
        )

        self.setting["Clip_240_scratch_s"]["model"] = nn.Sequential(
            CLIP4OR().state_encoder,
            Normalizer(),
            CLIP4OR().state_projector,
            Normalizer(),
        )
        # self.setting["Clip_240_scratch_s"]["model"].load_state_dict(
        #     torch.load("./checkpoint/Clip_240_scratch_s_9.cpt")["model"])

        self.mse_loss = nn.MSELoss().cuda(self.rank)
        self.bce_loss = nn.BCELoss().cuda(self.rank)
        self.ce_loss = nn.CrossEntropyLoss().cuda(self.rank)
        self.sigmoid = nn.Sigmoid().cuda(self.rank)
        self.softmax0 = nn.Softmax(dim=0).cuda(self.rank)
        self.softmax1 = nn.Softmax(dim=1).cuda(self.rank)

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

    @staticmethod
    def stage(epoch):
        # 判断这个epoch应该pretrain还是精调
        if epoch < 10:
            return True, False
        else:
            return False, True

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
    # 滑动窗口标准化，x和y在一起求均值方差
    def data_preprocess(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        ts = sensor[:, :, :2].long()
        s = sensor[:, :, 2:]
        std, mean = torch.std_mean(s, dim=1, unbiased=True, keepdim=True)
        std += 1e-9

        s_train = (s[:, :160, :] - mean) / std
        action = (s[:, 159:239, :2] - mean[:, :, :2]) / std[:, :, :2]
        traj_gt = (s[:, 160:, 16:19] - mean[:, :, 16:19]) / std[:, :, 16:19]

        return s_train, action, traj_gt, ts, std, mean

    def train_step(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess(data)
        output = self.setting["informer_best"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                      x_mark_dec=ts[:, 159:239, :])
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_best"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_best"]["optimizer"].step()
        return loss

    def eval_step(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess(data)
        with torch.no_grad():
            output = self.setting["informer_best"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                          x_mark_dec=ts[:, 159:239, :])
        output = output * std[:, :, 16:19] + mean[:, :, 16:19]
        traj_gt = traj_gt * std[:, :, 16:19] + mean[:, :, 16:19]

        metric = torch.mean((traj_gt - output) ** 2) ** 0.5
        return metric

    # 控制变量，滑动窗口标准化时，x和y仅位置信息一起标准化，其他信息只在x标准化
    def data_preprocess1(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        ts = sensor[:, :, :2].long()
        s = sensor[:, :, 2:]
        std, mean = torch.std_mean(s[:, :160, :], dim=1, unbiased=True, keepdim=True)
        std2, mean2 = torch.std_mean(s[:, :, 16:19], dim=1, unbiased=True, keepdim=True)
        std += 1e-9
        std2 += 1e-9

        s_train = s[:, :160, :]
        s_train[:, :, :16] = (s_train[:, :, :16] - mean[:, :, :16]) / std[:, :, :16]
        s_train[:, :, 19:] = (s_train[:, :, 19:] - mean[:, :, 19:]) / std[:, :, 19:]
        s_train[:, :, 16:19] = (s_train[:, :, 16:19] - mean2) / std2

        action = (s[:, 159:239, :2] - mean[:, :, :2]) / std[:, :, :2]
        traj_gt = (s[:, 160:, 16:19] - mean2) / std2

        return s_train, action, traj_gt, ts, std, mean, std2, mean2

    def train_step1(self, data):
        s_train, action, traj_gt, ts, std, mean, std2, mean2 = self.data_preprocess1(data)
        output = self.setting["informer_test1"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                       x_mark_dec=ts[:, 159:239, :])
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test1"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test1"]["optimizer"].step()
        return loss

    def eval_step1(self, data):
        s_train, action, traj_gt, ts, std, mean, std2, mean2 = self.data_preprocess1(data)
        with torch.no_grad():
            output = self.setting["informer_test1"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                           x_mark_dec=ts[:, 159:239, :])
        output = output * std2 + mean2
        traj_gt = traj_gt * std2 + mean2

        metric = torch.mean((traj_gt - output) ** 2) ** 0.5
        return metric

    # 滑动窗口标准化时，y使用x的均值方差
    def data_preprocess2(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        ts = sensor[:, :, :2].long()
        s = sensor[:, :, 2:]
        std, mean = torch.std_mean(s[:, :160, :], dim=1, unbiased=True, keepdim=True)
        std += 1e-9

        s_train = s[:, :160, :]
        s_train = (s_train - mean) / std

        action = (s[:, 159:239, :2] - mean[:, :, :2]) / std[:, :, :2]
        traj_gt = (s[:, 160:, 16:19] - mean[:, :, 16:19]) / std[:, :, 16:19]

        return s_train, action, traj_gt, ts, std, mean

    def train_step2(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess2(data)
        output = self.setting["informer_test2"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                       x_mark_dec=ts[:, 159:239, :])
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test2"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test2"]["optimizer"].step()
        return loss

    def eval_step2(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess2(data)
        with torch.no_grad():
            output = self.setting["informer_test2"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                           x_mark_dec=ts[:, 159:239, :])
        output = output * std[:, :, 16:19]
        traj_gt = traj_gt * std[:, :, 16:19]

        metric = torch.mean((traj_gt - output) ** 2) ** 0.5
        return metric

    # 位置信息并不是平稳序列，所以需要考虑更多其他的标准化方法
    # 对位置信息进行差分标准化
    def data_preprocess3(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        ts = sensor[:, :, :2].long()
        s = sensor[:, :, 2:]
        s[:, 1:, 16:19] = s[:, 1:, 16:19] - s[:, :-1, 16:19]
        s[:, :1, 16:19] = torch.zeros(s[:, :1, 16:19].shape)
        std, mean = torch.std_mean(s[:, :160, :], dim=1, unbiased=True, keepdim=True)
        # in case some std is too small
        std = torch.max(std, torch.ones(std.shape).cuda(self.rank))

        s_train = s[:, :160, :]
        s_train = (s_train - mean) / std

        action = (s[:, 159:239, :2] - mean[:, :, :2]) / std[:, :, :2]
        traj_gt = (s[:, 160:, 16:19] - mean[:, :, 16:19]) / std[:, :, 16:19]

        return s_train, action, traj_gt, ts, std, mean

    def train_step3(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess3(data)
        output = self.setting["informer_test3"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                       x_mark_dec=ts[:, 159:239, :])
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test3"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test3"]["optimizer"].step()
        return loss

    def eval_step3(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess3(data)
        with torch.no_grad():
            output = self.setting["informer_test3"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                           x_mark_dec=ts[:, 159:239, :])
        output = output * std[:, :, 16:19]
        traj_gt = traj_gt * std[:, :, 16:19]

        metric = torch.mean((traj_gt - output) ** 2) ** 0.5
        metric2 = torch.mean((torch.cumsum(traj_gt, dim=1) - torch.cumsum(output, dim=1)) ** 2) ** 0.5
        return metric, metric2

    # 差分标准化结果可以接受，大概在0.4左右徘徊，但是收敛仍较慢，考虑到步长现在是40hz,降采样减少预测点可能会降低累积误差
    # 第一种降采样方案：对x和y同时降采样，降采样方法为平均降采样，然后差分
    def data_preprocess4(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        ts = sensor[:, ::4, :2].long()
        s = sensor[:, :, 2:]
        b, t, n = s.shape
        s = s.reshape((b, 4, -1, n))
        s = torch.mean(s, dim=1)
        s[:, 1:, 16:19] = s[:, 1:, 16:19] - s[:, :-1, 16:19]
        s[:, :1, 16:19] = s[:, :1, 16:19] - s[:, :1, 16:19]
        std, mean = torch.std_mean(s[:, :40, :], dim=1, unbiased=True, keepdim=True)
        # in case some std is too small
        std = torch.max(std, torch.ones(std.shape).cuda(self.rank))

        s_train = s[:, :40, :]
        s_train = (s_train - mean) / std

        action = (s[:, 39:59, :2] - mean[:, :, :2]) / std[:, :, :2]
        traj_gt = (s[:, 40:, 16:16 + 7] - mean[:, :, 16:16 + 7]) / std[:, :, 16:16 + 7]

        return s_train, action, traj_gt, ts, std, mean

    def train_step4(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess4(data)
        output = self.setting["informer_test4"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                       x_mark_dec=ts[:, 39:59, :])
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test4"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test4"]["optimizer"].step()
        return loss

    def eval_step4(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess4(data)
        with torch.no_grad():
            output = self.setting["informer_test4"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                           x_mark_dec=ts[:, 39:59, :])
        output = output * std[:, :, 16:16 + 7]
        quaternion = traj_gt[:, :, 3:] * std[:, :, 16 + 3:16 + 3 + 4]
        traj_gt = traj_gt[:, :, :3] * std[:, :, 16:16 + 3]

        metric = torch.mean((traj_gt - output[:, :, :3]) ** 2) ** 0.5
        metric_traj = (torch.cumsum(traj_gt, dim=1) - torch.cumsum(output[:, :, :3], dim=1)) ** 2
        metric_quat = (quaternion - output[:, :, 3:]) ** 2
        metric2 = torch.concat((metric_traj, metric_quat), dim=-1)
        metric2 = torch.mean(metric2) ** 0.5
        return metric, metric2

    # 第二种降采样方案：只对y进行降采样，此时x和y步长间隔不同，对x按照y的频率进行采样，取多个均值方差然后平均
    def data_preprocess5(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        ts = sensor[:, :, :2].long()
        s = sensor[:, :, 2:]
        b, t, n = s.shape
        s = s.reshape((b, 4, -1, n))  # b, 4, 60, n

        s[:, :, 1:, 16:19] = s[:, :, 1:, 16:19] - s[:, :, :-1, 16:19]
        s[:, :, :1, 16:19] = s[:, :, :1, 16:19] - s[:, :, :1, 16:19]
        std, mean = torch.std_mean(s[:, :, :40, :], dim=2, unbiased=True, keepdim=True)
        std = torch.mean(std, dim=1)
        mean = torch.mean(mean, dim=1)
        # in case some std is too small
        std = torch.max(std, torch.ones(std.shape).cuda(self.rank))

        traj_gt = torch.mean(s[:, :, 40:, 16:19], dim=1)
        traj_gt = (traj_gt - mean[:, :, 16:19]) / std[:, :, 16:19]
        action = torch.mean(s[:, :, 39:59, :2], dim=1)
        action = (action - mean[:, :, :2]) / std[:, :, :2]

        s = s.reshape((b, t, n))
        std2, mean2 = torch.std_mean(s[:, :160, :], dim=1, unbiased=True, keepdim=True)
        std2 += 1e-9
        s_train = s[:, :160, :]
        s_train = (s_train - mean2) / std2

        return s_train, action, traj_gt, ts, std, mean

    def train_step5(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess5(data)
        output = self.setting["informer_test5"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :160, :], x_dec=action,
                                                       x_mark_dec=ts[:, 159:239, :])
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test5"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test5"]["optimizer"].step()
        return loss

    def eval_step5(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess5(data)
        with torch.no_grad():
            output = self.setting["informer_test5"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                           x_mark_dec=ts[:, 39:59, :])
        output = output * std[:, :, 16:19]
        traj_gt = traj_gt * std[:, :, 16:19]

        metric = torch.mean((traj_gt - output) ** 2) ** 0.5
        metric2 = torch.mean((torch.cumsum(traj_gt, dim=1) - torch.cumsum(output, dim=1)) ** 2) ** 0.5
        return metric, metric2

    # PatchTST 实验
    # 首先沿用Informer最佳性能的数据准备代码，但是因为PatchTST没有decoder，所以action也是预测结果
    def data_preprocess6(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        s = sensor[:, :, 2:]
        std, mean = torch.std_mean(s, dim=1, unbiased=True, keepdim=True)
        std += 1e-9
        s_norm = (s - mean) / std
        return s, s_norm, std, mean

    def train_step6(self, data):
        s, s_norm, std, mean = self.data_preprocess6(data)
        output = self.setting["PatchTST_test1"]["ddp"](s_norm[:, :160, :])
        # 动作和轨迹两个损失，后续实验可以只做一个观察情况
        loss = self.mse_loss(output[:, :, 16:19], s_norm[:, 160:, 16:19])
        loss += self.mse_loss(output[:, :, :2], s_norm[:, 160:, :2])
        self.setting["PatchTST_test1"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["PatchTST_test1"]["optimizer"].step()
        return loss

    def eval_step6(self, data):
        s, s_norm, std, mean = self.data_preprocess6(data)
        with torch.no_grad():
            output = self.setting["PatchTST_test1"]["ddp"](s[:, :160, :])

        output = output * std + mean
        gt = s_norm * std + mean

        traj_metric = torch.mean((gt[:, 160:, 16:19] - output[:, :, 16:19]) ** 2) ** 0.5
        action_metric = torch.mean((gt[:, 160:, :2] - output[:, :, :2]) ** 2) ** 0.5
        return traj_metric, action_metric

    # 使用平均降采样后10hz数据
    def data_preprocess7(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        s = sensor[:, :, 2:]
        b, t, n = s.shape
        s = s.reshape((b, 4, -1, n))
        s = torch.mean(s, dim=1)
        std, mean = torch.std_mean(s, dim=1, unbiased=True, keepdim=True)
        std += 1e-9
        s_norm = (s - mean) / std
        return s, s_norm, std, mean

    def train_step7(self, data):
        s, s_norm, std, mean = self.data_preprocess7(data)
        output = self.setting["PatchTST_test2"]["ddp"](s_norm[:, :40, :])

        loss = self.mse_loss(output[:, :, 16:19], s_norm[:, 40:, 16:19])
        loss += self.mse_loss(output[:, :, :2], s_norm[:, 40:, :2])
        self.setting["PatchTST_test2"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["PatchTST_test2"]["optimizer"].step()
        return loss

    def eval_step7(self, data):
        s, s_norm, std, mean = self.data_preprocess7(data)
        with torch.no_grad():
            output = self.setting["PatchTST_test2"]["ddp"](s[:, :40, :])

        output = output * std + mean
        gt = s_norm * std + mean

        traj_metric = torch.mean((gt[:, 40:, 16:19] - output[:, :, 16:19]) ** 2) ** 0.5
        action_metric = torch.mean((gt[:, 40:, :2] - output[:, :, :2]) ** 2) ** 0.5
        return traj_metric, action_metric

    # 使用均值降采样，不差分，局部标准化策略，Informer，其实这个实验不是很有意义
    def data_preprocess8(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        ts = sensor[:, ::4, :2].long()
        s = sensor[:, :, 2:]
        b, t, n = s.shape
        s = s.reshape((b, 4, -1, n))
        s = torch.mean(s, dim=1)
        std, mean = torch.std_mean(s, dim=1, unbiased=True, keepdim=True)
        std += 1e-9

        s_train = (s[:, :40, :] - mean) / std
        action = (s[:, 39:59, :2] - mean[:, :, :2]) / std[:, :, :2]
        traj_gt = (s[:, 40:, 16:19] - mean[:, :, 16:19]) / std[:, :, 16:19]

        return s_train, action, traj_gt, ts, std, mean

    def train_step8(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess8(data)
        output = self.setting["informer_test6"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                       x_mark_dec=ts[:, 39:59, :])
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test6"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test6"]["optimizer"].step()
        return loss

    def eval_step8(self, data):
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess8(data)
        with torch.no_grad():
            output = self.setting["informer_test6"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                           x_mark_dec=ts[:, 39:59, :])
        output = output * std[:, :, 16:19] + mean[:, :, 16:19]
        traj_gt = traj_gt * std[:, :, 16:19] + mean[:, :, 16:19]

        metric = torch.mean((traj_gt - output) ** 2) ** 0.5
        return metric

    # PatchTST 位置信息差分实验
    # 原始数据差分，输入标准化
    def data_preprocess9(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        s = sensor[:, :, 2:]
        s[:, 1:, 16:19] = s[:, 1:, 16:19] - s[:, :-1, 16:19]
        s[:, :1, 16:19] = s[:, :1, 16:19] - s[:, :1, 16:19]
        std, mean = torch.std_mean(s[:, :160, :], dim=1, unbiased=True, keepdim=True)
        std = torch.max(std, torch.ones(std.shape).cuda(self.rank))
        s_norm = (s - mean) / std
        return s, s_norm, std, mean

    def train_step9(self, data):
        s, s_norm, std, mean = self.data_preprocess9(data)
        output = self.setting["PatchTST_test3"]["ddp"](s_norm[:, :160, :])

        loss = self.mse_loss(output[:, :, 16:19], s_norm[:, 160:, 16:19])
        loss += self.mse_loss(output[:, :, :2], s_norm[:, 160:, :2])
        self.setting["PatchTST_test3"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["PatchTST_test3"]["optimizer"].step()
        return loss

    def eval_step9(self, data):
        s, s_norm, std, mean = self.data_preprocess9(data)
        with torch.no_grad():
            output = self.setting["PatchTST_test3"]["ddp"](s[:, :160, :])

        output = output * std + mean
        gt = s_norm * std + mean

        traj_metric = torch.mean((gt[:, 160:, 16:19] - output[:, :, 16:19]) ** 2) ** 0.5
        action_metric = torch.mean((gt[:, 160:, :2] - output[:, :, :2]) ** 2) ** 0.5
        return traj_metric, action_metric

    # 平均降采样数据差分
    def data_preprocess10(self, data: tuple):
        sensor = data[0].float().cuda(self.rank)
        s = sensor[:, :, 2:]
        b, t, n = s.shape
        s = s.reshape((b, 4, -1, n))
        s = torch.mean(s, dim=1)
        s[:, 1:, 16:19] = s[:, 1:, 16:19] - s[:, :-1, 16:19]
        s[:, :1, 16:19] = s[:, :1, 16:19] - s[:, :1, 16:19]
        std, mean = torch.std_mean(s[:, :40, :], dim=1, unbiased=True, keepdim=True)
        std = torch.max(std, torch.ones(std.shape).cuda(self.rank))
        s_norm = (s - mean) / std
        return s, s_norm, std, mean

    def train_step10(self, data):
        s, s_norm, std, mean = self.data_preprocess10(data)
        output = self.setting["PatchTST_test4"]["ddp"](s_norm[:, :40, :])

        loss = self.mse_loss(output[:, :, 16:19], s_norm[:, 40:, 16:19])
        loss += self.mse_loss(output[:, :, :2], s_norm[:, 40:, :2])
        self.setting["PatchTST_test4"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["PatchTST_test4"]["optimizer"].step()
        return loss

    def eval_step10(self, data):
        s, s_norm, std, mean = self.data_preprocess10(data)
        with torch.no_grad():
            output = self.setting["PatchTST_test4"]["ddp"](s[:, :40, :])

        output = output * std + mean
        gt = s_norm * std + mean

        traj_metric = torch.mean((gt[:, 40:, 16:19] - output[:, :, 16:19]) ** 2) ** 0.5
        action_metric = torch.mean((gt[:, 40:, :2] - output[:, :, :2]) ** 2) ** 0.5
        return traj_metric, action_metric

    # 预训练第一版，主要是BCE+图像传感器拉近，实施时进行过调整，比如图像与传感器位置
    # 实施时冻结了图像编码器的encoder
    def pretrain_step(self, data, finetune=False):
        s = data[0].float().cuda(self.rank)
        v = data[-1].float().cuda(self.rank)
        s_past = s[:, :160, 2:]
        s_fut = s[:, 160:, 2:]
        std, mean = torch.std_mean(s_past, dim=1, keepdim=True, unbiased=True)
        s_past = (s_past - mean) / torch.max(std, torch.ones_like(std))

        logit, vision, sensor = self.setting["Clip"]["ddp"](v, s_past)
        if finetune:
            return vision, sensor
        logit = self.sigmoid(logit)
        gt = batch_sim(s_fut[:, :, :2], s_fut[:, :, :2]).cuda(self.rank)

        loss = self.bce_loss(logit, gt) / 2 + self.bce_loss(logit.t(), gt) / 2
        self.setting["Clip"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["Clip"]["optimizer"].step()
        return loss, torch.sum(torch.diag(logit)) / logit.shape[0]

    def finetune_train_step(self, data):
        # use data_preprocess4
        self.setting["Clip"]["ddp"].eval()
        with torch.no_grad():
            vision, sensor = self.pretrain_step(data, finetune=True)

        s_train, action, traj_gt, ts, std, mean = self.data_preprocess4(data)
        output = self.setting["informer_test4clip"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                           x_mark_dec=ts[:, 39:59, :], pretrain_vision=vision,
                                                           pretrain_sensor=sensor)
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test4clip"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test4clip"]["optimizer"].step()
        return loss

    def finetune_eval_step(self, data):
        self.setting["Clip"]["ddp"].eval()
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess4(data)
        with torch.no_grad():
            vision, sensor = self.pretrain_step(data, finetune=True)
            output = self.setting["informer_test4clip"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                               x_mark_dec=ts[:, 39:59, :], pretrain_vision=vision,
                                                               pretrain_sensor=sensor)
        output = output * std[:, :, 16:19]
        traj_gt = traj_gt * std[:, :, 16:19]

        metric = torch.mean((traj_gt - output) ** 2) ** 0.5
        metric2 = torch.mean((torch.cumsum(traj_gt, dim=1) - torch.cumsum(output, dim=1)) ** 2) ** 0.5
        return metric, metric2

    # 预训练第二版，instance级的自监督，（图像+动作）&传感器，替换编码器选择是否imagenet pretrained
    def pretrain_step2(self, data, finetune=False):
        s = data[0][:, :, 2:].float().cuda(self.rank)
        v = data[-1].float().cuda(self.rank)
        b, c, h, w = v.shape
        v = v[:, :, h // 2:, :]
        std, mean = torch.std_mean(s, dim=1, keepdim=True, unbiased=True)
        s = (s - mean) / torch.max(std, torch.ones_like(std))
        logit, vf, sf, af = self.setting["Clip_240"]["ddp"](v, s)
        if finetune:
            return vf, sf, af

        b = s.shape[0]
        gt = torch.eye(b).cuda(self.rank)

        loss = self.ce_loss(logit, gt) / 2 + self.ce_loss(logit.t(), gt) / 2
        self.setting["Clip_240"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["Clip_240"]["optimizer"].step()
        return loss, torch.sum(torch.diag(self.softmax0(logit))) / b, torch.sum(torch.diag(self.softmax1(logit))) / b

    def finetune_train_step2(self, data):
        # use data_preprocess4
        self.setting["Clip_240"]["ddp"].eval()
        with torch.no_grad():
            vf, sf, af = self.pretrain_step2(data, finetune=True)

        s_train, action, traj_gt, ts, std, mean = self.data_preprocess4(data)
        output = self.setting["informer_test4clip"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                           x_mark_dec=ts[:, 39:59, :], pretrain_vision=vf,
                                                           pretrain_sensor=sf)
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test4clip"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test4clip"]["optimizer"].step()
        return loss

    def finetune_eval_step2(self, data):
        self.setting["Clip_240"]["ddp"].eval()
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess4(data)
        with torch.no_grad():
            vf, sf, af = self.pretrain_step2(data, finetune=True)
            output = self.setting["informer_test4clip"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                               x_mark_dec=ts[:, 39:59, :], pretrain_vision=vf,
                                                               pretrain_sensor=sf)
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
        s = data[0][:, :, 2:].float().cuda(self.rank)
        v = data[-1].float().cuda(self.rank)
        b, c, h, w = v.shape
        v = v[:, :, h // 2:, :]
        std, mean = torch.std_mean(s, dim=1, keepdim=True, unbiased=True)

        v1 = self.v_aug1(v)
        v2 = self.v_aug2(v)
        v_1 = self.setting["Clip_240_v"]["ddp"](v1)
        v_2 = self.setting["Clip_240_v"]["ddp"](v2)
        logit_v = v_1 @ v_2.t()

        gt = torch.eye(b).cuda(self.rank)
        loss = .0
        loss += self.ce_loss(logit_v, gt)
        return loss, 0, torch.sum(torch.diag(self.softmax0(logit_v))) / b

    def pretrain_single_modality_scratch(self, data):
        s = data[0][:, :, 2:].float().cuda(self.rank)
        v = data[-1].float().cuda(self.rank)
        b, c, h, w = v.shape
        v = v[:, :, h // 2:, :]
        std, mean = torch.std_mean(s, dim=1, keepdim=True, unbiased=True)
        s = (s - mean) / torch.max(std, torch.ones_like(std))

        v1 = self.v_aug1(v)
        v2 = self.v_aug2(v)
        v_1 = self.setting["Clip_240_scratch_v"]["ddp"](v1)
        v_2 = self.setting["Clip_240_scratch_v"]["ddp"](v2)
        logit_v = v_1 @ v_2.t()

        gt = torch.eye(b).cuda(self.rank)
        loss = .0
        loss += self.ce_loss(logit_v, gt)
        return loss, 0, torch.sum(torch.diag(self.softmax0(logit_v))) / b

    def pretrain_single_modality_s(self, data, finetune=False):
        # 排除前置序列时间信息
        s = data[0][:, :, 2:].float().cuda(self.rank)
        v = data[-1].float().cuda(self.rank)
        b, c, h, w = v.shape
        std, mean = torch.std_mean(s, dim=1, keepdim=True, unbiased=True)
        s = (s - mean) / torch.max(std, torch.ones_like(std))

        # 排除动作信息
        s = s[:, :, 2:]
        s1 = self.s_aug1(s)
        s2 = self.s_aug2(s)
        s_1 = self.setting["Clip_240_scratch_s"]["ddp"](s1)
        s_2 = self.setting["Clip_240_scratch_s"]["ddp"](s2)
        logit_v = s_1 @ s_2.t()
        if finetune:
            return s_2

        gt = torch.eye(b).cuda(self.rank)
        loss = self.ce_loss(logit_v, gt)
        return loss, 0, torch.sum(torch.diag(self.softmax0(logit_v))) / b

    def finetune_train_step3(self, data):
        # use data_preprocess4
        self.setting["Clip_240_scratch_s"]["ddp"].eval()
        with torch.no_grad():
            sf = self.pretrain_single_modality_s(data, finetune=True)

        s_train, action, traj_gt, ts, std, mean = self.data_preprocess4(data)
        output = self.setting["informer_test4s"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                        x_mark_dec=ts[:, 39:59, :], pretrain_sensor=sf)
        loss = self.mse_loss(output, traj_gt)
        self.setting["informer_test4s"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["informer_test4s"]["optimizer"].step()
        return loss

    def finetune_eval_step3(self, data):
        self.setting["Clip_240_scratch_s"]["ddp"].eval()
        s_train, action, traj_gt, ts, std, mean = self.data_preprocess4(data)
        with torch.no_grad():
            sf = self.pretrain_single_modality_s(data, finetune=True)
            output = self.setting["informer_test4s"]["ddp"](x_enc=s_train, x_mark_enc=ts[:, :40, :], x_dec=action,
                                                            x_mark_dec=ts[:, 39:59, :], pretrain_sensor=sf)
        output = output * std[:, :, 16:16 + 7]
        quaternion = traj_gt[:, :, 3:] * std[:, :, 16 + 3:16 + 3 + 4]
        traj_gt = traj_gt[:, :, :3] * std[:, :, 16:16 + 3]

        metric = torch.mean((traj_gt - output[:, :, :3]) ** 2) ** 0.5
        metric_traj = (torch.cumsum(traj_gt, dim=1) - torch.cumsum(output[:, :, :3], dim=1)) ** 2
        metric_quat = (quaternion - output[:, :, 3:]) ** 2
        metric2 = torch.concat((metric_traj, metric_quat), dim=-1)
        metric2 = torch.mean(metric2) ** 0.5

        return metric, metric2

    def pretrain_step4(self, data, finetune=False):
        s = data[0][:, :, 2:].float().cuda(self.rank)
        v = data[-1].float().cuda(self.rank)
        b, c, h, w = v.shape
        v = v[:, :, h // 2:, :]
        std, mean = torch.std_mean(s, dim=1, keepdim=True, unbiased=True)
        s = (s - mean) / torch.max(std, torch.ones_like(std))
        logit, vf, sf, af = self.setting["Clip_240_v_as"]["ddp"](v, s)
        if finetune:
            return vf, sf, af

        b = s.shape[0]
        gt = torch.eye(b).cuda(self.rank)

        loss = self.ce_loss(logit, gt) / 2 + self.ce_loss(logit.t(), gt) / 2
        self.setting["Clip_240_v_as"]["optimizer"].zero_grad()
        loss.backward()
        self.setting["Clip_240_v_as"]["optimizer"].step()
        return loss, torch.sum(torch.diag(self.softmax0(logit))) / b, torch.sum(torch.diag(self.softmax1(logit))) / b

    # if more experiment settings required, append below


if __name__ == '__main__':
    pass
