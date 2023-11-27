"""
This is the code used for ablation study and various tryouts
"""
import os
import warnings
from collections import Counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, distributed
from tqdm import trange

from exp_zoo import EXP
from utils.dataset import IntegratedBags

warnings.warn("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
n_epoch = 30
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
    os.environ['MASTER_PORT'] = '12357'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


def main_worker(rank, world_size, ):
    setup(rank, world_size)
    pretrain_dataset = IntegratedBags("/mnt/filesystem4/yangyi/parsed_bag", train=True, n_his=0, n_future=240)
    pretrain_sampler = distributed.DistributedSampler(pretrain_dataset)
    pretrain_loader = DataLoaderX(pretrain_dataset, batch_size=batch_size, sampler=pretrain_sampler)

    train_dataset = IntegratedBags("/mnt/filesystem4/yangyi/parsed_bag", train=True)
    train_sampler = distributed.DistributedSampler(train_dataset)
    train_loader = DataLoaderX(train_dataset, batch_size=batch_size, sampler=train_sampler)

    eval_dataset = IntegratedBags("/mnt/filesystem4/yangyi/parsed_bag", train=False)
    eval_sampler = distributed.DistributedSampler(eval_dataset)  # 不必要set_epoch
    eval_loader = DataLoaderX(eval_dataset, batch_size=batch_size, sampler=eval_sampler)

    epoch_iter = trange(n_epoch) if rank == 0 else range(n_epoch)
    logger = Logger("baseline_end2end")

    exp = EXP(rank)

    for epoch in epoch_iter:
        # 判断这个epoch需要执行什么任务
        # pretrain, finetune = exp.stage(epoch)
        pretrain, finetune = False, True
        pretrain, finetune = True, False
        exp.train()

        pretrain_sampler.set_epoch(epoch)
        train_sampler.set_epoch(epoch)

        if pretrain:
            for i, data in enumerate(pretrain_loader):
                loss_pretrain, logit_diag_sum1, logit_diag_sum2 = exp.pretrain_step4(data)
                if rank == 0:
                    logger.log_scalar("loss_pretrain_v_aa", loss_pretrain)
                    logger.log_scalar("logit_diag_sum_v_as", (logit_diag_sum1 + logit_diag_sum2) / 2)

                loss_pretrain, logit_diag_sum1, logit_diag_sum2 = exp.pretrain_step2(data)
                if rank == 0:
                    logger.log_scalar("loss_pretrain_norm", loss_pretrain)
                    logger.log_scalar("logit_diag_sum_norm", (logit_diag_sum1 + logit_diag_sum2) / 2)


        if finetune:
            for i, data in enumerate(train_loader):
                loss = exp.finetune_train_step2(data)
                if rank == 0:
                    logger.log_scalar("loss_finetune2", loss)

                loss = exp.train_step4(data)
                if rank == 0:
                    logger.log_scalar("train_loss4", loss)

                loss = exp.finetune_train_step3(data)
                if rank == 0:
                    logger.log_scalar("loss_finetune3", loss)

            exp.eval()
            metric_ft_step4 = metric_ft_acc4 = 0
            metric_ft_step2 = metric_ft_acc2 = 0
            metric_ft_steps = metric_ft_accs = 0
            for i, data in enumerate(eval_loader):
                a, b = exp.eval_step4(data)
                metric_ft_step4 += a
                metric_ft_acc4 += b
                a, b = exp.finetune_eval_step2(data)
                metric_ft_step2 += a
                metric_ft_acc2 += b
                a, b = exp.finetune_eval_step3(data)
                metric_ft_steps += a
                metric_ft_accs += b

            if rank == 0:
                logger.log_scalar("eval_metric4_step", metric_ft_step4 / len(eval_loader))
                logger.log_scalar("eval_metric4_acc", metric_ft_acc4 / len(eval_loader))

                logger.log_scalar("eval_finetune2_step", metric_ft_step2 / len(eval_loader))
                logger.log_scalar("eval_finetune2_acc", metric_ft_acc2 / len(eval_loader))

                logger.log_scalar("eval_finetune_s_step", metric_ft_steps / len(eval_loader))
                logger.log_scalar("eval_finetune_s_acc", metric_ft_accs / len(eval_loader))


        exp.scheduler_step()

        if rank == 0:
            for k, d in exp.setting.items():
                if k != "Clip_240_v_as" or k != "Clip_240":
                    continue
                print(k, d.keys())
                state = {
                    "epoch": epoch,
                    "model": d["ddp"].module.state_dict(),
                    "optimizer": d["optimizer"].state_dict(),
                    "scheduler": d["scheduler"].state_dict(),
                }
                torch.save(state, f"./checkpoint/{k}_{epoch}.cpt")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    main(main_worker, world_size)
