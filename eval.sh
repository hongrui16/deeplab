# CUDA_VISIBLE_DEVICES=0,1,2 python train_main.py --testValTrain 1  --resume run/workspace/TASK_2021-11-29_12-26-11/run/basicDataset/deeplab-resnet/experiment_3/model_best.pth.tar
# CUDA_VISIBLE_DEVICES=0,1,2 python train_main.py --testValTrain 1  --resume run/workspace/TASK_2021-11-09_20-28-58/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar
CUDA_VISIBLE_DEVICES=0 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-12-22_14-52-00/run/basicDataset/deeplab-resnet/experiment_1/model_best.pth.tar
CUDA_VISIBLE_DEVICES=0 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-12-22_18-53-46/run/basicDataset/deeplab-resnet/experiment_1/model_best.pth.tar
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# srun -N1 --cpus-per-task 4 --gres gpu:4 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --distinguish_left_right_semantic
