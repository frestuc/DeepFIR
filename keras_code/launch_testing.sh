#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
gpu=$1
fir_n=$2
batch=$3
constr=$4

# dgx-2
# --h5_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \

# dgx-1
# --h5_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \

# baseline
python2 ./DeepSigTesting.py \
        --test_single_model \
        --data_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --model_name /home/salvo/deepsig_res/modulation_model.hdf5 \
        --batch_size $batch \
        --max_steps 0 \
        --num_classes 24 \
        --save_file_name baseline_accuracy \
        --id_gpu $gpu \
        --save_path /home/salvo/deepsig_res/results/baseline/batch_$batch

# FIR
# python2 ./DeepSigTesting.py \
#         --test_perdev_model \
#         --data_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
#         --model_name /home/salvo/deepsig_res/modulation_model.hdf5 \
#         --batch_size $batch \
#         --max_steps 0 \
#         --num_classes 24 \
#         --save_file_name fir_accuracy \
#         --id_gpu $gpu \
#         --save_path /home/salvo/deepsig_res/results/FIR/constrained_$constr/batch_$batch/$fir_n \
#         --models_path /home/salvo/deepsig_res/res_${batch}_con$constr/per_dev_$fir_n/per_dev \
