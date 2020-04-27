import h5py
import numpy as np
import scipy.io as sio

f = h5py.File('GOLD_XYZ_OSC.0001_1024.hdf5', 'r')

X = f['X']
Y = f['Y']
Z = f['Z']

n_per_class = 106496
snr_levels_per_class = 26
n_per_classl_per_snr = 4096
n_extract_samples = snr_levels_per_class*24

index_extract = np.array(range(n_extract_samples))
index_extract *= n_per_classl_per_snr
index_extract += np.random.randint(n_per_classl_per_snr-1, size=n_extract_samples)

sio.savemat('dataset_extract.mat', {'X':X[index_extract], 'Y':Y[index_extract], 'Z':Z[index_extract]})