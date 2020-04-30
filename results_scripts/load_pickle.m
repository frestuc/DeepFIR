function a = load_pickle(filename)
  if ~exist(filename,'file')
    error('%s is not a file',filename);
  end
  outname = [tempname() '.mat'];
  pyscript = ['import pickle;import sys;import scipy.io;file=open("' filename '", "rb");dat=pickle.load(file);file.close();scipy.io.savemat("' outname '.dat")'];
system(['python -c "' pyscript '"']);
a = load(outname);
end