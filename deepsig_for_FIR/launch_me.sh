#!/bin/bash
gpu=$1

# dgx-2
# --h5_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \

# dgx-1
# --h5_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
​
python2 ./DeepSig.py \
        --train_cnn \
        --h5_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --data_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --batch_size 32 \
        --epochs 25 \
        --fir_size 10 \
        --max_steps 1000 \
        --num_ex_mod 106496 \
        --num_classes 24 \
        --id_gpu $gpu \
        --patience 3 \
	> /home/salvo/deepsig_res/out.log \
	2> /home/salvo/deepsig_res/err.log