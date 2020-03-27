import scipy.io as spio

class Utils():

    def read_file_mat(file, keep_shape):
        '''
        Read data saved in .mat file
        Input: file path
        Output:
            -ex_data: complex signals
            -samples_in_example: example length
        '''
        mat_data = spio.loadmat(file)
        if mat_data.has_key('complexSignal'):
            complex_data = mat_data['complexSignal']  # try to use views here also
        elif mat_data.has_key('f_sig'):
            complex_data = mat_data['f_sig']
        if not keep_shape:
            real_data = np.reshape(complex_data.real, (complex_data.shape[1], 1))
            imag_data = np.reshape(complex_data.imag, (complex_data.shape[1], 1))
            complex_data = np.concatenate((real_data, imag_data), axis=1)
        samples_in_example = complex_data.shape[0]
        return complex_data, samples_in_example