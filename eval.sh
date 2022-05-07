# CUDA_VISIBLE_DEVICES=4,5,6,7 python train_main.py --testValTrain 1  --resume run/workspace/TASK_2021-12-24_18-31-15/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar --dataset_dir /comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/
# CUDA_VISIBLE_DEVICES=0,1 python train_main.py --testValTrain 1  --resume run/workspace/TASK_2021-12-23_11-51-00/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar  --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted
# CUDA_VISIBLE_DEVICES=0,1 python train_main.py --testValTrain 1  --resume run/workspace/TASK_2021-12-24_18-09-20/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar  --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted
# CUDA_VISIBLE_DEVICES=0,1 python train_main.py --testValTrain 1  --resume run/workspace/TASK_2021-12-24_18-31-15/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar  --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted
# CUDA_VISIBLE_DEVICES=0,1,2 python train_main.py --testValTrain 1  --resume run/workspace/TASK_2021-11-09_20-28-58/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar
# CUDA_VISIBLE_DEVICES=3 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-11-09_20-28-58/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted
# CUDA_VISIBLE_DEVICES=3 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-11-29_12-26-11/run/basicDataset/deeplab-resnet/experiment_3/model_best.pth.tar --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted
# CUDA_VISIBLE_DEVICES=0 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-12-22_14-52-00/run/basicDataset/deeplab-resnet/experiment_1/model_best.pth.tar
# CUDA_VISIBLE_DEVICES=0 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-12-22_18-53-46/run/basicDataset/deeplab-resnet/experiment_1/model_best.pth.tar
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# CUDA_VISIBLE_DEVICES=4 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-12-24_18-09-20/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar  --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted
# CUDA_VISIBLE_DEVICES=4 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-12-24_18-31-15/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar  --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted
# CUDA_VISIBLE_DEVICES=4 python train_main.py --testValTrain 1 --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-12-23_11-51-00/run/basicDataset/deeplab-resnet/experiment_0/model_best.pth.tar  --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --distinguish_left_right_semantic
# CUDA_VISIBLE_DEVICES=1 python train_main.py --testValTrain 1  --dump_image_for_cal_chamferDist --resume run/workspace/TASK_2021-12-28_16-27-13/run/basicDataset/deeplab-resnet/experiment_1/model_best.pth.tar  --dataset_dir /comp_robot/hongrui/metro_pro/dataset/twoRail/sorted

# CUDA_VISIBLE_DEVICES=6 python train_main.py --batch_size 16 --testValTrain 1  --base_size 480 \
# --crop_size 480 --dataset CustomPotSeg  --use_txtfile \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/  \
# --pot_train_mode 1 --backbone drn --ramdom_cut_postives \
# --resume run/CustomPotSeg/deeplab-drn/experiment_23/model_best.pth.tar

CUDA_VISIBLE_DEVICES=7 python train_main.py --batch_size 24 --test_batch_size 24 --testValTrain 1  --base_size 480 \
--crop_size 480 --dataset CustomPotSeg  --use_txtfile \
--dataset_dir /home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/  \
--pot_train_mode 1 --backbone drn --ramdom_cut_postives \
--resume run/CustomPotSeg/deeplab-drn/experiment_26/model_best.pth.tar