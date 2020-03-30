gpu=$1

# dgx-2
# --h5_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5 \

# dgx-1
# --h5_path /mnt/WDMyBook/bruno/deepsig/GOLD_XYZ_OSC.0001_1024.hdf5 \
# --data_path /mnt/WDMyBook/bruno/deepsig/GOLD_XYZ_OSC.0001_1024.hdf5 \
​
python ./DeepSig.py \
        --train_cnn True \
        --train_fir False \
        --h5_path /mnt/WDMyBook/bruno/deepsig/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --data_path /mnt/WDMyBook/bruno/deepsig/GOLD_XYZ_OSC.0001_1024.hdf5 \
        --batch_size 32 \
        --epochs 25 \
        --fir_size 10 \
        --num_ex_mod 106496 \
        --num_classes 24 \
        --load_indexes False \
        --id_gpu $gpu \
        --patience 3 \
	> ./out.log \
	2> ./err.log