close all
clc
clear all

load('dataset_extract.mat')

NUM_TAPS = [10 25 50 100 200];
batch_num = 32;
sig_length = 1024;

fir_taps_01 = getTapsFromHDF5('_con01',NUM_TAPS,batch_num);
fir_taps_05 = getTapsFromHDF5('_con05',NUM_TAPS,batch_num);
fir_taps_1 = getTapsFromHDF5('_con1',NUM_TAPS,batch_num);
fir_taps_2 = getTapsFromHDF5('_con2',NUM_TAPS,batch_num);
fir_taps_u = getTapsFromHDF5('_unconstr',NUM_TAPS,batch_num);

%%

tags = {'OOK' '4ASK' '8ASK' 'BPSK' 'QPSK' '8PSK' '16PSK' '32PSK' '16APSK' ...
    '32APSK' '64APSK' '128APSK' '16QAM' '32QAM' '64QAM' '128QAM' '256QAM' ...
    'AM-SSB-WC' 'AM-SSB-SC' 'AM-DSB-WC' 'AM-DSB-SC' 'FM' 'GMSK' 'OQPSK'};

ind_max = 26;
index_snr = 26;

CLASS_INTEREST = [4 5 15];

for class_idx = 1 : numel(CLASS_INTEREST)
    
    class_id = CLASS_INTEREST(class_idx);
    
    index = index_snr + ind_max*(class_id-1);
    
    for fir_taps_num_conf = 1 : 1
        
        taps_n = fir_taps_01(fir_taps_num_conf).taps(:,:,class_id);
        
        %         taps_n = zeros(2,NUM_TAPS(fir_taps_num_conf));
        %         taps_n(1,1) = 1;
        
        filtered_signal = conv((taps_n(1,:) + 1i.*taps_n(2,:)),(X(index,:,1) + 1i.*X(index,:,2)));
        filtered_signal = filtered_signal(1:sig_length);
        %         original_signal = X(index,:,1) + 1i.*X(index,:,2);
        
        figure(class_idx)
        subplot(1,3,1);
        plot(X(index,:,1),'-','LineWidth',1.5)
        hold on
        plot(X(index,:,2),'--','LineWidth',1.5)
        plot(real(filtered_signal),':','LineWidth',1.5)
        plot(imag(filtered_signal),'-.','LineWidth',1.5)
        grid on
        xlim([0 1024])
        legend('Before, Real','Before, Imaginary','After, Real','After, Imaginary')
        %         title(tags{class_id})
        set(gca,'FontSize',14)
        subplot(1,3,2);
        h_1 = cdfplot(abs(X(index,:,1) + 1i.*X(index,:,2)))
        hold on
        h_2 = cdfplot(abs(filtered_signal))
        set( h_1, 'LineStyle', '-','LineWidth',1.5)
        set( h_2, 'LineStyle', ':','LineWidth',1.5)
        ylabel('CDF, Amplitude')
        legend('Before','After')
        title(tags{class_id})
        set(gca,'FontSize',14)
        subplot(1,3,3);
        h_1 = cdfplot(angle(X(index,:,1) + 1i.*X(index,:,2)))
        hold on
        h_2 = cdfplot(angle(filtered_signal))
        title('')
        ylabel('CDF, Phase')
        legend('Before','After')
        set(gca,'FontSize',14)
        set( h_1, 'LineStyle', '-','LineWidth',1.5)
        set( h_2, 'LineStyle', ':','LineWidth',1.5)
        
        %         figure(class_id)
        %         subplot(numel(CLASS_INTEREST),3,(fir_taps_num_conf-1)*3 + 1);
        %         plot(X(index,:,1),'k-')
        %         hold on
        %         plot(X(index,:,2),'r-')
        %         plot(real(filtered_signal),'k--')
        %         plot(imag(filtered_signal),'r--')
        %         xlim([0 1024])
        %         legend('Before, R','Before, I','After, R','After, I')
        %         title(tags{class_id})
        %         subplot(2,3,(fir_taps_num_conf-1)*3 + 2);
        %         cdfplot(abs(X(index,:,1) + 1i.*X(index,:,2)))
        %         hold on
        %         cdfplot(abs(filtered_signal))
        %         ylabel('CDF Amplitude')
        %         legend('Before','After')
        %         subplot(2,3,(fir_taps_num_conf-1)*3 + 3);
        %         cdfplot(angle(X(index,:,1) + 1i.*X(index,:,2)))
        %         hold on
        %         cdfplot(angle(filtered_signal))
        %         ylabel('CDF Phase')
        %         legend('Before','After')
        
    end
end

%% QPSK

% decimation_factor = 8;
% class_id = 5;
% index_snr = 26;
%
% start_in = 1;
%
% index = index_snr + ind_max*(class_id-1);
%
% original_sig = X(index,:,1) + 1i.*X(index,:,2);
%
% taps_class = fir_taps_01(1).taps(:,:,class_id);
% filtered_signal = conv((taps_class(1,:) + 1i.*taps_class(2,:)),original_sig);
% filtered_signal = filtered_signal(1:sig_length);
%
% figure
% stem(real(original_sig(start_in:decimation_factor:end)),imag(original_sig(start_in:decimation_factor:end)),'LineStyle','none')
% hold on
% stem(real(filtered_signal(start_in:decimation_factor:end)),imag(filtered_signal(start_in:decimation_factor:end)),'LineStyle','none')
% legend('Before','After')
%
% %% BPSK
%
% decimation_factor = 16;
% class_id = 13;
% index_snr = 26;
%
% start_in = 1;
%
% index = index_snr + ind_max*(class_id-1);
%
% original_sig = X(index,:,1) + 1i.*X(index,:,2);
%
% taps_class = fir_taps_01(1).taps(:,:,class_id);
% filtered_signal = conv((taps_class(1,:) + 1i.*taps_class(2,:)),original_sig);
% filtered_signal = filtered_signal(1:sig_length);
%
% figure
% stem(real(original_sig(start_in:decimation_factor:end)),imag(original_sig(start_in:decimation_factor:end)),'LineStyle','none')
% hold on
% stem(real(filtered_signal(start_in:decimation_factor:end)),imag(filtered_signal(start_in:decimation_factor:end)),'LineStyle','none')
% legend('Before','After')
%
