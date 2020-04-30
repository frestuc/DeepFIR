function fir_taps = getTapsFromHDF5(con,NUM_TAPS,batch_num)

dataset_name = '/phi:0';
base_path = '/home/doc/Desktop/results_fir/res_';

add_str1 = '/per_dev_';
add_str2 = '/per_dev/FIR_model_';
end_str = '.hdf5';

fir_taps = struct('n_taps',[],'taps',[]);

for t = 1 : numel(NUM_TAPS)
    n_taps = NUM_TAPS(t);
    
    fir_taps(t).taps = zeros(2,n_taps,24);
    
    for c = 1 : 24
        
        id_class = c-1;
        
        file_name = strcat(base_path,num2str(batch_num),con,add_str1,num2str(n_taps),add_str2,num2str(id_class),end_str);
        info = h5info(file_name);
        pre_dataset_path = info.Groups(1).Groups(10).Groups(1).Name;
        
        data = squeeze(h5read(file_name,strcat(pre_dataset_path,dataset_name)));
        fir_taps(t).n_taps = n_taps;
        fir_taps(t).taps(:,:,c) = data;
    end
end

return