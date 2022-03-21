from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, \
sbd, basicDataset, custom_pot, GC10_DET, custom_pot_seg

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
        # test_set = basicDataset.BasicDataset(args, split="train")
        num_class = args.n_classes

        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), drop_last=True, sampler=train_sampler, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=(val_sampler is None), drop_last=False, sampler=val_sampler, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=(test_sampler is None), drop_last=False, sampler=test_sampler, **kwargs)
        
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'CustomPot':
        # print(f'calling {__file__}, {sys._getframe().f_lineno}')
        '''
        label_names = ['pot', 'LaSi_rect', 'TuQi', 'ZhouBian', 'HuaHen_rect', 'HuaHen']
                        0      1            2       3           4              5
        '''    
        train_set = custom_pot.CustomPot(args, split="train")
        val_set = custom_pot.CustomPot(args, split="val")
        test_set = custom_pot.CustomPot(args, split="test")
        # test_set = basicDataset.BasicDataset(args, split="train")
        if args.pot_train_mode == 1: #不区分类别
            num_class = 2
        elif args.pot_train_mode == 2:#区分类别，不训练划痕
            num_class = 4
        elif args.pot_train_mode == 3:#区分类别，不训练划痕，皱边
            num_class = 3
        elif args.pot_train_mode == 3:#区分类别，不训练划痕，皱边，凸起
            num_class = 2
        elif args.pot_train_mode == 4:#区分类别，不训练划痕，皱边，拉丝
            num_class = 2
        else:
            num_class = args.n_classes

        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), drop_last=True, sampler=train_sampler, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=(val_sampler is None), drop_last=False, sampler=val_sampler, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=(test_sampler is None), drop_last=False, sampler=test_sampler, **kwargs)
        
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'CustomPotSeg':
        # print(f'calling {__file__}, {sys._getframe().f_lineno}')
        '''
            label_dict = {'lasi_heavy': 11, 'lasi_medium':12, 'lasi_slight':13,
        'gengshang_heavy':21, 'gengshang_medium':22, 'gengshang_slight':23,  
        'gengshi_heavy':31, 'gengshi_medium':32, 'gengshi_slight':33,
        'shayan_heavy':41, 'shayan_medium':42, 'shayan_medium':43,
        'huahen_heavy':51, 'huahen_medium':52, 'huahen_medium':53,
        'zhoubian_heavy':61, 'zhoubian_medium':62, 'zhoubian_medium':63,
        'bowen_heavy':71, 'bowen_medium':72, 'bowen_medium':73,
        'youwu_heavy':81, 'youwu_medium':82, 'youwu_medium':83,
        }
        '''    
        train_set = custom_pot_seg.CustomPotSeg(args, split="train")
        val_set = custom_pot_seg.CustomPotSeg(args, split="val")
        test_set = custom_pot_seg.CustomPotSeg(args, split="test")
        # test_set = basicDataset.BasicDataset(args, split="train")
        if args.pot_train_mode == 1: #不区分类别
            num_class = 2
        elif args.pot_train_mode == 2: #不区分类别,只处理前三类
            num_class = 2
        elif args.pot_train_mode == 3: #不区分类别,只处理前三类
            num_class = 2
        elif args.pot_train_mode == 4: #不区分类别,只处理前三类
            num_class = 3
        elif args.pot_train_mode == 5: #将heavy, medium马赛克, 只处理slight一类
            num_class = 2
        else:
            num_class = args.n_classes

        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), drop_last=True, sampler=train_sampler, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=(val_sampler is None), drop_last=False, sampler=val_sampler, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=(test_sampler is None), drop_last=False, sampler=test_sampler, **kwargs)
        
        return train_loader, val_loader, test_loader, num_class


    elif args.dataset == 'GC10_DET':
        # print(f'calling {__file__}, {sys._getframe().f_lineno}')
        '''
        label_names = ['bg',  '1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban', '6_siban', '7_yiwu', '8_yahen', '9_zhehen', '10_yaozhe']
        '''    
        train_set = GC10_DET.GC10_DET(args, split="train")
        val_set = GC10_DET.GC10_DET(args, split="val")
        test_set = GC10_DET.GC10_DET(args, split="test")
        # test_set = basicDataset.BasicDataset(args, split="train")
        if args.pot_train_mode == 1: #不区分类别
            num_class = 2
        elif args.pot_train_mode == 2:#区分类别
            num_class = 11
        elif args.pot_train_mode == 3:#只训练1_chongkong和2_hanfeng，并区分类别
            num_class = 3
        else:
            num_class = args.n_classes

        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), drop_last=True, sampler=train_sampler, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=(val_sampler is None), drop_last=False, sampler=val_sampler, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=(test_sampler is None), drop_last=False, sampler=test_sampler, **kwargs)
        
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

