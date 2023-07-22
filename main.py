import argparse
import os
from collections import Counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
from torch.utils.data import distributed, DataLoader
from tqdm import trange

from exp import EXP
from ros_parse.parse import Parser
from ros_parse.parser_zoo import *
from utils.dataset import IntegratedBags

parser = argparse.ArgumentParser(prog="CLIP4OR", description="", )
parser.add_argument("--num_gpu", default=2, type=int)
# extract data from bags
parser.add_argument("--parse_bag", default=False, type=bool)
parser.add_argument("--bag_path", default="./bags", type=str)  # directory the bag is stored
parser.add_argument("--parse_path", default="./parsed_bags", type=str)  # directory the parsed data is stored
parser.add_argument("--n_parallel_parse", default=1, type=int)  # number of workers to parse, not the more, the better

# pretrain
parser.add_argument("--pretrain", default=False, type=bool)
parser.add_argument("--pretrain_epochs", default=10, type=int)
parser.add_argument("--pretrain_batch", default=64, type=int)
# finetune
parser.add_argument("--finetune", default=False, type=bool)
parser.add_argument("--finetune_epochs", default=240, type=int)
parser.add_argument("--finetune_batch", default=64, type=int)
# resume
parser.add_argument("--resume_epoch", default=None, type=int)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Logger(SummaryWriter):
    def __init__(self, log_dir=None):
        super(Logger, self).__init__(log_dir)
        self.counter = Counter()

    def log_scalar(self, tag, value):
        self.add_scalar(tag, value, self.counter[tag])
        self.counter[tag] += 1


def main(world_size, command=parser, ):
    args = command.parse_args()
    print(args)
    if args.parse_bag:
        parse(args)

    mp.spawn(main_worker, args=(world_size, args,), nprocs=world_size, join=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def parse(args):
    parser = (State, Action, ObservImg, ObservRGBMap, ObservHeightMap, Imu, Shock, RPM, Pedals)
    p = Parser(args.bag_path, args.parse_path, False, *parser)
    print(
        f"The TartanDrive ROS bag saved in {args.bag_path} will be parsed into {args.parse_path} with {args.n_parallel_parse} workers", )
    p.parallel_parse(size=args.n_parallel_parse)


def main_worker(rank, world_size, args, ):
    setup(rank, world_size)
    model = EXP(rank)
    model.models_exclude("CLIP4OR_v", "CLIP4OR_s", "informer")
    # multiprocessing may cause problem when logger is defined, `mkdir runs` first would be fine.
    if rank == 0:
        logger = Logger()
    # it's suggested to build dataset alone outside the multiprocessing first
    dataset = IntegratedBags(args.parse_path)
    sampler = distributed.DistributedSampler(dataset)
    dataloader = DataLoaderX(dataset, batch_size=args.pretrain_batch, sampler=sampler)

    eval_dataset = IntegratedBags(args.parse_path, train=False)
    eval_sampler = distributed.DistributedSampler(eval_dataset)
    eval_dataloader = DataLoaderX(eval_dataset, batch_size=args.finetune_batch, sampler=eval_sampler)

    # resume from certain checkpoint
    if args.resume_epoch is not None:
        checkpoints = os.listdir("./checkpoint")
        for k, d in model.setting.items():
            if f"{args.resume_epoch}_{k}.cpt" in checkpoints:
                state = torch.load(f"./checkpoint/{args.resume_epoch}_{k}.cpt", map_location=f"cuda:{rank}")
                model.setting[k]["model"].load_state_dict(state["model"])
                model.setting[k]["optimizer"].load_state_dict(state["optimizer"])
                model.setting[k]["scheduler"].load_state_dict(state["scheduler"])

    if args.pretrain:
        model.train()
        epoch_iter = trange(args.pretrain_epochs) if rank == 0 else range(args.pretrain_epochs)

        for epoch in epoch_iter:
            sampler.set_epoch(epoch)
            for i, data in enumerate(dataloader):
                loss, sim_v2s, sim_s2v = model.pretrain(data)
                if rank == 0:
                    logger.log_scalar("pretrain_loss", loss)
                    logger.log_scalar("batch_similarity_v2s", sim_v2s)
                    logger.log_scalar("batch_similarity_s2v", sim_s2v)
            if rank == 0:
                for k, d in model.setting.items():
                    state = {
                        "epoch": epoch,
                        "model": d["ddp"].module.state_dict(),
                        "optimizer": d["optimizer"].state_dict(),
                        "scheduler": d["scheduler"].state_dict(),
                    }
                    torch.save(state, f"./checkpoint/{epoch}_{k}.cpt")
            dist.barrier()
        model.scheduler_step()

    dist.barrier()
    if args.finetune:
        epoch_iter = trange(args.finetune_epochs) if rank == 0 else range(args.finetune_epochs)
        for epoch in epoch_iter:
            model.train()
            sampler.set_epoch(epoch)
            for i, data in enumerate(dataloader):
                loss = model.finetune_train(data)
                if rank == 0:
                    logger.log_scalar("finetune_loss", loss)

            model.eval()
            eval_sampler.set_epoch(epoch)
            m1 = m2 = 0
            for i, data in enumerate(eval_dataloader):
                metric_step, metric_acc = model.finetune_eval(data)
                m1 += metric_step
                m2 += metric_acc
            if rank == 0:
                logger.log_scalar("mse_step", m1 / len(eval_dataloader))
                logger.log_scalar("mse_accumulated", m2 / len(eval_dataloader))
    cleanup()


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    args = parser.parse_args()
    world_size = min(n_gpus, args.num_gpu)
    main(world_size=world_size)
