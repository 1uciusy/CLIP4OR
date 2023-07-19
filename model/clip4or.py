import numpy as np
import torch
from torch import nn
from torchvision import models

__all__ = ["freeze", "get_proj", "get_swin_b", "get_swin_t", "get_swin_s", "CLIP4OR"]


def freeze(net: nn.Module):
    for layer in net.parameters():
        if hasattr(layer, "requires_grad"):
            layer.requires_grad = False


def get_swin_b(pretrain=True, param_freeze=False):
    if pretrain:
        swin = models.swin_b(weights=models.Swin_B_Weights(models.Swin_B_Weights.DEFAULT))
    else:
        swin = models.swin_b()
    swin.head = nn.Identity()
    if param_freeze:
        freeze(swin)
    return swin


def get_swin_s(pretrain=True, param_freeze=False):
    if pretrain:
        swin = models.swin_s(weights=models.Swin_S_Weights(models.Swin_S_Weights.DEFAULT))
    else:
        swin = models.swin_s()
    swin.head = nn.Identity()
    if param_freeze:
        freeze(swin)
    return swin


def get_swin_t(pretrain=True, param_freeze=False):
    if pretrain:
        swin = models.swin_t(weights=models.Swin_T_Weights(models.Swin_T_Weights.DEFAULT))
    else:
        swin = models.swin_t()
    swin.head = nn.Identity()
    if param_freeze:
        freeze(swin)
    return swin


def get_proj(in_features=128, out_features=128):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.GELU(),
        nn.Linear(out_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.GELU(),
        nn.Linear(out_features, out_features),
    )


# class SensorConvStem(nn.Module):
#     def __init__(self, enc_in=27, max_seq_length=240, d_model=128):
#         super(SensorConvStem, self).__init__()
#         self.max_seq_length = max_seq_length
#         self.sensor_conv_stem = nn.Conv1d(in_channels=enc_in, out_channels=128, kernel_size=11, padding=5,
#                                           padding_mode="replicate")
#         self.position_embedding = nn.Parameter(torch.randn((max_seq_length, d_model)) / d_model ** 0.5)
#
#     def forward(self, sensor: torch.Tensor):  # batch_size, input_length, channel
#         # 如果长度不够，则右padding
#         b, l, c = sensor.shape
#         sensor = nn.functional.pad(sensor, (0, 0, self.max_seq_length - l, 0), mode="replicate")
#         sensor = sensor.permute(0, 2, 1)  # batch_size, channel, input_length
#         sensor = self.sensor_conv_stem(sensor)
#         sensor = sensor.permute(0, 2, 1)  # batch_size, input_length, channel
#         sensor = sensor + self.position_embedding
#
#         return sensor


# class SensorNaiveTFEncoder(nn.Module):
#     def __init__(self, enc_in=27, d_model=128, n_head=128, max_seq_length=160, num_layers=6, dim_feedforward=128):
#         """
#         :param enc_in: 输入特征维度
#         :param d_model: tokenization后的维度
#         :param n_head:
#         :param max_seq_length: 最大序列长，position_embedding的序列长，输入不足这个长度会padding
#         :param num_layers: encoder block个数
#         :param dim_feedforward: 前向传播维度
#         """
#         super(SensorNaiveTFEncoder, self).__init__()
#
#         self.conv_stem = SensorConvStem(enc_in=enc_in, max_seq_length=max_seq_length, d_model=d_model)
#         self.contrastive_token = nn.Parameter(torch.randn((d_model,)) / d_model ** 0.5)
#
#         sensor_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
#                                                   activation=nn.GELU(), batch_first=True, )
#         self.sensor_encoder = nn.TransformerEncoder(sensor_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
#
#     def forward(self, sensor: torch.Tensor):
#         b, t, n = sensor.shape
#         sensor = self.conv_stem(sensor)
#         token = self.contrastive_token.repeat((b, 1, 1))
#         sensor = torch.concat((token, sensor), dim=1)
#         sensor = self.sensor_encoder(sensor)
#         # return torch.flatten(sensor[:, :1, :], start_dim=1)
#         return sensor


class Sensor1DCNN(nn.Module):
    def __init__(self, enc_in=27, max_seq_length=240, d_model=128):
        super(Sensor1DCNN, self).__init__()
        self.max_seq_length = max_seq_length
        self.gn1 = nn.GroupNorm(4, max_seq_length)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=enc_in, out_channels=d_model, kernel_size=11, padding=5, padding_mode="replicate"),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=5)
        )
        self.gn2 = nn.GroupNorm(4, max_seq_length // 5)
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=4)
        )
        self.gn3 = nn.GroupNorm(4, max_seq_length // 5 // 4)
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=4)
        )
        self.gn4 = nn.GroupNorm(1, max_seq_length // 5 // 4 // 4)
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3),
            nn.GELU()
        )

    def forward(self, x):  # batch, seq len, enc_in
        b, l, c = x.shape
        x = nn.functional.pad(x, (0, 0, self.max_seq_length - l, 0), mode="replicate")
        x = self.gn1(x)
        x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.gn2(x)
        x = x.permute(0, 2, 1)
        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.gn3(x)
        x = x.permute(0, 2, 1)
        x = self.block3(x)
        x = x.permute(0, 2, 1)
        x = self.gn4(x)
        x = x.permute(0, 2, 1)
        x = self.block4(x)
        x = x.squeeze(-1)
        return x


class CLIP4OR(nn.Module):
    def __init__(self, enc_in=27, d_model=128, n_head=128, max_seq_length=240, out_features=128, scale=True,
                 action_vision=False, vision_encoder="swin_t", imagenet_pretrained=False):
        """
        :param enc_in: input sensor dimension
        :param d_model: Attention dimension
        :param n_head: Attention head
        :param max_seq_length: maximum sensor length supported
        :param scale: learnable temperature, True for enable
        :param action_vision: fuse action and vision before aligned to state, True for yes
        """
        assert d_model % n_head == 0
        super(CLIP4OR, self).__init__()
        self.action_vision = action_vision
        self.action_vision_cmb = nn.Identity()
        if action_vision:
            # self.action_encoder = SensorNaiveTFEncoder(enc_in=2, d_model=d_model, n_head=n_head,
            #                                            max_seq_length=max_seq_length, num_layers=num_layers,
            #                                            dim_feedforward=dim_feedforward)
            self.action_encoder = Sensor1DCNN(enc_in=2, d_model=d_model, max_seq_length=max_seq_length)
            self.action_projector = get_proj(in_features=d_model, out_features=out_features)
            self.action_vision_cmb = nn.Linear(in_features=out_features * 2, out_features=out_features)
            enc_in -= 2

        if vision_encoder.lower() == "swin_t":
            self.vision_encoder = nn.Sequential(
                get_swin_t(pretrain=imagenet_pretrained),
                nn.Linear(in_features=768, out_features=128)
            )
        elif vision_encoder.lower() == "swin_b":
            self.vision_encoder = nn.Sequential(
                get_swin_b(pretrain=imagenet_pretrained),
                nn.Linear(in_features=1024, out_features=128)
            )
        self.vision_projector = get_proj(in_features=out_features, out_features=out_features)

        # self.sensor_encoder = SensorNaiveTFEncoder(enc_in, d_model=d_model, n_head=n_head,
        #                                            max_seq_length=max_seq_length,
        #                                            num_layers=num_layers, dim_feedforward=dim_feedforward)
        self.sensor_encoder = Sensor1DCNN(enc_in=enc_in, max_seq_length=max_seq_length, d_model=d_model)
        self.sensor_projector = get_proj(in_features=d_model, out_features=out_features)
        self.scale = scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_vision(self, vision: torch.Tensor, sensor: torch.Tensor):
        vision_feature = self.vision_encoder(vision)
        vision_feature = vision_feature / torch.norm(vision_feature, dim=1, keepdim=True)

        vision = self.vision_projector(vision_feature)
        vision = vision / torch.norm(vision, dim=1, keepdim=True)

        action_feature = None
        if self.action_vision:
            action = sensor[:, :, :2]
            action_feature = self.action_encoder(action)
            action_feature = action_feature / torch.norm(action_feature, dim=1, keepdim=True)
            action_output = self.action_projector(action_feature)
            action_output = action_output / torch.norm(action_output, dim=1, keepdim=True)
            vision = torch.cat((vision, action_output), dim=1)

        vision = self.action_vision_cmb(vision)
        vision = vision / torch.norm(vision, dim=1, keepdim=True)
        return vision, vision_feature, action_feature

    def encode_sensor(self, sensor: torch.Tensor):
        if self.action_vision:
            sensor = sensor[:, :, 2:]
        sensor_feature = self.sensor_encoder(sensor)
        sensor_feature = sensor_feature / torch.norm(sensor_feature, dim=1, keepdim=True)

        sensor = self.sensor_projector(sensor_feature)
        sensor = sensor / torch.norm(sensor, dim=1, keepdim=True)
        return sensor, sensor_feature

    def forward(self, vision: torch.Tensor, sensor: torch.Tensor):
        vision_emb, vf, af = self.encode_vision(vision, sensor)
        sensor_emb, sf = self.encode_sensor(sensor)

        logit_scale = torch.exp(self.logit_scale)
        logit = vision_emb @ sensor_emb.t()
        if self.scale:
            logit = logit_scale * logit

        return logit, vf, sf, af, vision_emb, sensor_emb  # logit[a][b], dot product of vision[a] & sensor[b]


if __name__ == '__main__':
    enc_in = 27
    d_model = 128
    n_head = 128
    max_seq_length = 240
    seq_length = 160
    num_layers = 6
    dim_feedforward = 128

    clip = CLIP4OR()
    test_data = torch.randn((2, seq_length, enc_in))
    test_output = clip.encode_sensor(test_data)
    test_img = torch.randn((2, 3, 128, 128))
    test_output_img = clip.encode_vision(test_img, test_data)
    what = clip(test_img, test_data)

    clip = CLIP4OR(action_vision=True)
    test_data = torch.randn((2, seq_length, enc_in))
    test_output = clip.encode_sensor(test_data)
    test_img = torch.randn((2, 3, 128, 128))
    test_output_img = clip.encode_vision(test_img, test_data)
    what = clip(test_img, test_data)
