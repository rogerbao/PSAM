#!/usr/bin/env sh
python3 main.py --mode='evaluate' --dataset='Makeup10k_Eastern' --c_dim=5 --c2_dim=5 --image_size=256 --batch_size=1 \
		--test_iters=400 --dataset2_image_path='../data/Makeup10k_Eastern' \
		--test_traverse=True --task_name='PSAM-Makeup10k_Eastern-200epoch' \
		--pretrained_model=''
