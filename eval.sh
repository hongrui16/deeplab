# CUDA_VISIBLE_DEVICES=1,2,3 python train_main.py --testValTrain 1 --n_classes 3 --resume run/workspace/TASK_2021-11-19_19-15-18/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar --only_eval_main_rails
# CUDA_VISIBLE_DEVICES=1,2,3 python train_main.py --testValTrain 1 --n_classes 2 --resume run/workspace/TASK_2021-11-19_19-19-08/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar --only_eval_main_rails
# CUDA_VISIBLE_DEVICES=1,2,3 python train_main.py --testValTrain 1 --n_classes 3 --resume run/workspace/TASK_2021-11-09_20-28-58/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar --only_eval_main_rails
CUDA_VISIBLE_DEVICES=0,1,2 python train_main.py --testValTrain 1 --n_classes 3 --resume run/workspace/TASK_2021-11-09-19-21-43/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar --only_eval_main_rails