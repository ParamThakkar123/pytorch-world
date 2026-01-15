import torch
from torchvision.datasets import CIFAR10
from logging import getLogger

logger = getLogger()


def make_cifar10(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    drop_last=True,
    train=True,
    download=False,  # new
):
    dataset = CIFAR10(
        root=root_path,
        train=train,
        download=download,
        transform=transform,
    )
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    logger.info("CIFAR10 data loader created")
    return dataset, data_loader, dist_sampler
