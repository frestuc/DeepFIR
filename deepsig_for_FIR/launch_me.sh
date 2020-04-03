#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
gpu=$1

# dgx-2
# --h5_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \

# dgx-1
# --h5_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
​
python2 ./DeepSig.py \
        --train_fir_perdev \
	--load_indexes \
        --h5_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --data_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --batch_size 128 \
        --epochs 50 \
        --fir_size 100 \
        --num_ex_mod 106496 \
        --num_classes 24 \
        --id_gpu $gpu \
        --patience 10 \
#	> /home/salvo/deepsig_res/out.log \
#	2> /home/salvo/deepsig_res/err.log
