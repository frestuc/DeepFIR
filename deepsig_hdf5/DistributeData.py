import os
import h5py
import pickle

from tqdm import tqdm

class DistributeData():
    '''Class to distribute large file containing examples into
    multiple smaller files with one example.'''

    def __init__(self, data_path='/mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
                 save_path='/mnt/nas/bruno/deepsig/data/'):
        '''Initializer.'''
        self.data_path = data_path
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.distribute()


    def distribute(self):
        '''Distribute file into sub-files.'''
        f = h5py.File(self.data_path)
        X = f['X']
        Y = f['Y']
        for i in tqdm(range(len(X)), total=len(X)):
            data = {'X': X[i], 'Y': Y[i]}
            with open(os.path.join(self.save_path, str(i)+'.pkl'), 'wb') as f:
                pickle.dump(data, f)
       
if __name__ == '__main__':
    DistributeData() 
