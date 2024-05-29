import argparse
import torch
from torch import distributed, nn

from efficientnet import Config
from efficientnet.datasets import DatasetFactory
from efficientnet.models import ModelFactory
from efficientnet.optim import OptimFactory, SchedulerFactory
from efficientnet.trainer import Trainer
from efficientnet.utils import distributed_is_initialized
from swin_transformer1_swin import SwinTransformer
CUDA_LAUNCH_BLOCKING=1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/mnist.yaml')
    parser.add_argument('-r', '--root', type=str, help='Path to dataset.')
    parser.add_argument('--resume', type=str, default=None)

    # distributed
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)

    return parser.parse_args()


def init_process(backend, init_method, world_size, rank):
    distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )


def load_config():
    args = parse_args()
    config = Config.from_yaml(args.config)

    if args.root:
        config.dataset.root = args.root

    config.update(vars(args))

    return config


def main():
    torch.backends.cudnn.benchmark = True

    config = load_config()
    print(config)

    if config.world_size > 1:
        init_process(config.backend, config.init_method, config.world_size, config.rank)

    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

#    model = ModelFactory.create(**config.model)
    model = SwinTransformer(#img_size=224,
                         #       patch_size=4,
                                in_chans=3,
                                num_classes=1000,
                        #        embed_dim=96,
                          #      depths=[2, 2, 6, 2],
                           #     num_heads=[ 3, 6, 12, 24 ],
                                window_size=7,
                                layers=(2, 2, 6, 2),
                                hidden_dim=96,
                                heads=(3, 6, 12, 24),
                             #   mlp_ratio=4.,
                         #       qkv_bias=True,
                        #        qk_scale=True,
                                drop_rate=0.5,
                                drop_path_rate=0.5,
                          #      ape=False,
                            #    patch_norm=True,
                            #    use_checkpoint=True
                        )
    model.load_state_dict(torch.load('./swin_tiny_patch4_window7_224.pth')['model'],strict=False)
    model.head = nn.Linear(model.head.in_features, 13)
    print(model)
    if distributed_is_initialized():
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        if config.data_parallel:
            model = nn.DataParallel(model)
        model.to(device)

    optimizer = OptimFactory.create(model.parameters(), **config.optimizer)
    scheduler = SchedulerFactory.create(optimizer, **config.scheduler)

    train_loader, valid_loader = DatasetFactory.create(**config.dataset)

    trainer = Trainer(model, optimizer, train_loader, valid_loader, scheduler, device, config.output_dir)

    if config.resume is not None:
        trainer.resume(config.resume)

    trainer.fit(config.num_epochs)


if __name__ == "__main__":
    main()
