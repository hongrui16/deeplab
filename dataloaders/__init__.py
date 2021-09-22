from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, basicDataset
from torch.utils.data import DataLoader
import torch
import sys

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'basicDataset':
        # print(f'calling {__file__}, {sys._getframe().f_lineno}')
        train_set = basicDataset.BasicDataset(args, split="train")
        val_set = basicDataset.BasicDataset(args, split="val")
        test_set = basicDataset.BasicDataset(args, split="test")
        num_class = args.n_classes

        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=args.world_size, rank=args.rank)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=args.world_size, rank=args.rank)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), drop_last=False, sampler=train_sampler, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=(val_sampler is None), drop_last=False, sampler=val_sampler, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=(test_sampler is None), drop_last=False, sampler=test_sampler, **kwargs)
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

