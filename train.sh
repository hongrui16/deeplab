CUDA_VISIBLE_DEVICES=0,1,2 python train_main.py --testValTrain 4 --n_classes 3  --distinguish_left_right_semantic
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# srun -N1 --cpus-per-task 4 --gres gpu:4 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --distinguish_left_right_semantic
