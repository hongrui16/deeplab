class Path(object):
    @staticmethod
    def db_root_dir(dataset, args = None):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/data1/cv_public_dataset/COCO2017'
        elif dataset == 'basicDataset':
            # return '/home/hongrui/project/metro_pro/dataset/youtubeHandcraft'
            # return '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'
            return '/comp_robot/hongrui/metro_pro/dataset/1st_5000/'
            if args:
                return args.dataset_dir
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
