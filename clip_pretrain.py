import os
from collections import Counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from Informer2020.models.model import Informer
from model.clip4or import CLIP
from utils.dataset import IntegratedBags
from utils.ts_similarity import batch_sim

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
n_epoch = 500
batch_size = 64


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Logger(SummaryWriter):
    def __init__(self, log_dir):
        super(Logger, self).__init__(log_dir)
        self.counter = Counter()

    def log_scalar(self, tag, value):
        self.add_scalar(tag, value, self.counter[tag])
        self.counter[tag] += 1


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


def main_worker(rank, world_size, cpt=None):
    setup(rank, world_size)

    train_dataset = IntegratedBags("/mnt/filesystem4/yangyi/parsed_bag", train=True)
    train_sampler = distributed.DistributedSampler(train_dataset)
    train_loader = DataLoaderX(train_dataset, batch_size=batch_size, sampler=train_sampler)

    eval_dataset = IntegratedBags("/mnt/filesystem4/yangyi/parsed_bag", train=False)
    eval_sampler = distributed.DistributedSampler(eval_dataset)
    eval_loader = DataLoaderX(eval_dataset, batch_size=batch_size, sampler=eval_sampler)

    # training part
    model = CLIP().train().to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    # loss_fn = nn.CrossEntropyLoss().to(rank)
    loss_fn = nn.BCELoss().to(rank)
    optimizer = torch.optim.AdamW(ddp_model.parameters())
    eval_loss_fn = nn.MSELoss().to(rank)

    if cpt:
        ddp_model.load_state_dict(torch.load(cpt, map_location=f"cuda:{rank}"))
        optimizer.load_state_dict(torch.load(cpt, map_location=f"cuda:{rank}"))

    epoch_iter = trange(n_epoch) if rank == 0 else range(n_epoch)
    logger = Logger("init_exp")

    for epoch in epoch_iter:
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        train(rank, ddp_model, train_loader, optimizer, loss_fn, logger)
        dist.barrier()

        # evaluate/finetune from scratch every epoch
        if epoch % 10 == 9:
            if rank == 0:
                checkpoint = {
                    "model": ddp_model.module.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                torch.save(checkpoint, f"checkpoint_{epoch}.cpt")
            ddp_model.eval()
            eval_model = Informer(enc_in=27, dec_in=2, c_out=3, seq_len=160, label_len=0, out_len=80,
                                  factor=4, d_model=128, n_heads=9, d_ff=128, dropout=0.1)
            eval_model = eval_model.float().cuda(rank)
            eval_ddp = DDP(eval_model, device_ids=[rank], output_device=rank)
            eval_optim = torch.optim.Adam(eval_ddp.parameters())
            eval_sampler.set_epoch(epoch)
            evaluate(rank, ddp_model, train_loader, eval_ddp, eval_optim, eval_loss_fn, eval_loader, logger)

    cleanup()


def train(rank, model: CLIP, dataloader: DataLoaderX, optimizer: torch.optim.AdamW, loss_fn, logger: Logger):
    # train one epoch
    dataloader = tqdm(dataloader) if rank == 0 else dataloader
    for i, data in enumerate(dataloader):
        s = data[0].float().cuda(rank)
        v = data[1].float().cuda(rank)

        s_past = s[:, :160, :]
        s_fut = s[:, 160:, :]
        std, mean = torch.std_mean(s_past, dim=1, keepdim=True, unbiased=True)
        s_past = (s_past - mean) / (std + 1e-9)

        logit = model(v, s_past)
        sigmoid = nn.Sigmoid()
        logit = sigmoid(logit)

        gt = batch_sim(s_fut[:, :, :2], s_fut[:, :, :2]).cuda(rank)
        # gt = gt / torch.norm(gt, dim=1, p=1, keepdim=True)
        # gt = gt - torch.mean(gt, dim=1) + 1. / gt.shape[1]

        loss = loss_fn(logit, gt) / 2 + loss_fn(logit.t(), gt) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if rank == 0:
            logger.log_scalar("loss", loss)


def evaluate(rank, encoder: DDP, train_dataloader: DataLoaderX, eval_model: DDP, eval_optim, eval_loss_fn,
             test_dataloader, logger: Logger, finetune_epoch=5):
    # evaluate/fine tune after one epoch is trained
    for i in range(finetune_epoch):
        eval_model.train()
        for i, data in enumerate(train_dataloader):
            s = data[0].float().cuda(rank)
            v = data[1].float().cuda(rank)

            s_past = s[:, :160, :]
            std, mean = torch.std_mean(s_past, dim=1, keepdim=True, unbiased=True)
            s_past = (s_past - mean) / (std + 1e-9)

            with torch.no_grad():
                v_feature = encoder.module.vision_encoder(v)
                s_feature = encoder.module.sensor_encoder(s_past)

            curr_point = s[:, 159:160, 16:19]
            traj_prev = s[:, 159:239, 16:19] - curr_point
            traj_gt = s[:, 160:, 16:19] - curr_point
            output, hn = eval_model(s_feature, v_feature, traj_prev)
            loss = eval_loss_fn(traj_gt, output)
            eval_optim.zero_grad()
            loss.backward()
            eval_optim.step()

    eval_model.eval()
    for i, data in enumerate(test_dataloader):
        s = data[0].float().cuda(rank)
        v = data[1].float().cuda(rank)

        s_past = s[:, :160, :]
        std, mean = torch.std_mean(s_past, dim=1, keepdim=True, unbiased=True)
        s_past = (s_past - mean) / (std + 1e-9)

        with torch.no_grad():
            v_feature = encoder.module.vision_encoder(v)
            v_feature = v_feature / torch.norm(v_feature, dim=1, keepdim=True)
            s_feature = encoder.module.sensor_encoder(s_past)
            s_feature = s_feature / torch.norm(s_feature, dim=1, keepdim=True)

        curr_point = s[:, 159:160, 16:19]
        traj_prev = s[:, 159:239, 16:19] - curr_point
        traj_gt = s[:, 160:, 16:19] - curr_point
        output, hn = eval_model(s_feature, v_feature, traj_prev)
        logger.log_scalar("metric", torch.mean((output - traj_gt) ** 2) ** 0.5)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    main(main_worker, world_size)
