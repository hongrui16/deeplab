# CUDA_VISIBLE_DEVICES=5,6,7 python train_main.py --testValTrain 4 --use_txtfile
# CUDA_VISIBLE_DEVICES=5,6,7 python train_main.py --testValTrain 4 --use_txtfile --base_size 720 --crop_size 720
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# srun -N1 --cpus-per-task 4 --gres gpu:4 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --distinguish_left_right_semantic

# CUDA_VISIBLE_DEVICES=1 python train_main.py --testValTrain 4 --use_txtfile --base_size 640 --crop_size 640 --dataset CustomPot --n_classes 2 \
# --dataset_dir /home/hongrui/project/metro_pro/dataset --rotate_degree 30 --epochs 500 \
# --resume run/CustomPot/deeplab-resnet/experiment_0/model_best.pth.tar 

# CUDA_VISIBLE_DEVICES=1 python train_main.py --testValTrain 4 --use_txtfile --base_size 640 --crop_size 640 --dataset CustomPot --n_classes 4 \
# --dataset_dir /home/hongrui/project/metro_pro/dataset --rotate_degree 30 --epochs 500 \
# --diff_all_classes --ignore_huahen

# CUDA_VISIBLE_DEVICES=2 python train_main.py --testValTrain 4 --use_txtfile --base_size 640 --crop_size 640 --dataset CustomPot --n_classes 3 \
# --dataset_dir /home/hongrui/project/metro_pro/dataset --rotate_degree 30 --epochs 500 \
# --diff_all_classes --ignore_huahen --ignore_zhoubian

# CUDA_VISIBLE_DEVICES=1 python train_main.py --testValTrain 4 --use_txtfile --base_size 640 --crop_size 640 --dataset CustomPot --n_classes 2 \
# --dataset_dir /home/hongrui/project/metro_pro/dataset --rotate_degree 30 --epochs 500 \
# --pot_train_mode 3
CUDA_VISIBLE_DEVICES=2 python train_main.py --testValTrain 4 --use_txtfile --base_size 640 --crop_size 640 --dataset CustomPot --n_classes 2 \
--dataset_dir /home/hongrui/project/metro_pro/dataset --rotate_degree 30 --epochs 500 \
--pot_train_mode 4