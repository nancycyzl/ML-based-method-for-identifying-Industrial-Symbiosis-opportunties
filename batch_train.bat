@echo off

python train.py --dataset_mode produce --epochs 40 --output_dir output_new/train_produce_v5

python train.py --dataset_mode need --epochs 50 --output_dir output_new/train_need_v3
