#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
gpu=$1

# dgx-2
# --h5_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \

# dgx-1
# --h5_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/WDMyBook2/salvo/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \

python2 ./DeepSig.py \
        --test_perdev_model \
        --data_path C:\Users\totix\Desktop\darpa\bin_files\2018.01\GOLD_XYZ_OSC.0001_1024.hdf5 \
        --model_name C:\Users\totix\Desktop\darpa\deepsig\modulation_model.hdf5 \
        --models_path C:\Users\totix\Desktop\darpa\deepsig\per_dev \
        --batch_size 32 \
        --max_steps 0 \
        --num_classes 24 \
        --save_file_name per_dev_fir_results \
        --save_path C:\Users\totix\Desktop\darpa\deepsig
