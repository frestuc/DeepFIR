close all
clc
clear all

set(0,'DefaultFigureColormap',feval('hot'));

x_bas = load('/home/doc/Desktop/results_fir/results/mat_results/bas_32.mat');

x_c_01_10 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_01_10.mat');
x_c_01_25 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_01_25.mat');
x_c_01_50 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_01_50.mat');
x_c_01_100 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_01_100.mat');
x_c_01_200 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_01_200.mat');

x_c_05_10 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_05_10.mat');
x_c_05_25 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_05_25.mat');
x_c_05_50 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_05_50.mat');
x_c_05_100 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_05_100.mat');
x_c_05_200 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_05_200.mat');

x_c_1_10 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_1_10.mat');
x_c_1_25 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_1_25.mat');
x_c_1_50 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_1_50.mat');
x_c_1_100 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_1_100.mat');
x_c_1_200 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_1_200.mat');

x_c_2_10 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_2_10.mat');
x_c_2_25 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_2_25.mat');
x_c_2_50 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_2_50.mat');
x_c_2_100 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_2_100.mat');
x_c_2_200 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_c_2_200.mat');

x_u_10 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_u_10.mat');
x_u_25 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_u_25.mat');
x_u_50 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_u_50.mat');
x_u_100 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_u_100.mat');
x_u_200 = load('/home/doc/Desktop/results_fir/results/mat_results/fir_32_u_200.mat');

clim = [0 1];

batch_accuracy_baseline = x_bas.pickle_data.batch_accuracy.*ones(5,1);

batch_accuracy_c_01 = [x_c_01_10.pickle_data.batch_accuracy;
    x_c_01_25.pickle_data.batch_accuracy;
    x_c_01_50.pickle_data.batch_accuracy;
    x_c_01_100.pickle_data.batch_accuracy;
    x_c_01_200.pickle_data.batch_accuracy];

batch_accuracy_c_05 = [x_c_05_10.pickle_data.batch_accuracy;
    x_c_05_25.pickle_data.batch_accuracy;
    x_c_05_50.pickle_data.batch_accuracy;
    x_c_05_100.pickle_data.batch_accuracy;
    x_c_05_200.pickle_data.batch_accuracy];

batch_accuracy_c_1 = [x_c_1_10.pickle_data.batch_accuracy;
    x_c_1_25.pickle_data.batch_accuracy;
    x_c_1_50.pickle_data.batch_accuracy;
    x_c_1_100.pickle_data.batch_accuracy;
    x_c_1_200.pickle_data.batch_accuracy];

batch_accuracy_c_2 = [x_c_2_10.pickle_data.batch_accuracy;
    x_c_2_25.pickle_data.batch_accuracy;
    x_c_2_50.pickle_data.batch_accuracy;
    x_c_2_100.pickle_data.batch_accuracy;
    x_c_2_200.pickle_data.batch_accuracy];

batch_accuracy_u = [x_u_10.pickle_data.batch_accuracy;
    x_u_25.pickle_data.batch_accuracy;
    x_u_50.pickle_data.batch_accuracy;
    x_u_100.pickle_data.batch_accuracy;
    x_u_200.pickle_data.batch_accuracy];

batch_accuracy = [batch_accuracy_baseline, batch_accuracy_c_01, batch_accuracy_c_05, batch_accuracy_c_1, batch_accuracy_c_2, batch_accuracy_u];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



example_accuracy_baseline = x_bas.pickle_data.example_accuracy.*ones(5,1);

example_accuracy_c_01 = [x_c_01_10.pickle_data.example_accuracy;
    x_c_01_25.pickle_data.example_accuracy;
    x_c_01_50.pickle_data.example_accuracy;
    x_c_01_100.pickle_data.example_accuracy;
    x_c_01_200.pickle_data.example_accuracy];

example_accuracy_c_05 = [x_c_05_10.pickle_data.example_accuracy;
    x_c_05_25.pickle_data.example_accuracy;
    x_c_05_50.pickle_data.example_accuracy;
    x_c_05_100.pickle_data.example_accuracy;
    x_c_05_200.pickle_data.example_accuracy];

example_accuracy_c_1 = [x_c_1_10.pickle_data.example_accuracy;
    x_c_1_25.pickle_data.example_accuracy;
    x_c_1_50.pickle_data.example_accuracy;
    x_c_1_100.pickle_data.example_accuracy;
    x_c_1_200.pickle_data.example_accuracy];

example_accuracy_c_2 = [x_c_2_10.pickle_data.example_accuracy;
    x_c_2_25.pickle_data.example_accuracy;
    x_c_2_50.pickle_data.example_accuracy;
    x_c_2_100.pickle_data.example_accuracy;
    x_c_2_200.pickle_data.example_accuracy];

example_accuracy_u = [x_u_10.pickle_data.example_accuracy;
    x_u_25.pickle_data.example_accuracy;
    x_u_50.pickle_data.example_accuracy;
    x_u_100.pickle_data.example_accuracy;
    x_u_200.pickle_data.example_accuracy];

example_accuracy = [example_accuracy_baseline, example_accuracy_c_01, example_accuracy_c_05, example_accuracy_c_1, example_accuracy_c_2, example_accuracy_u];


%%

%%%%%%%%%%%%%%%%%%%%%%%%%% figure
% subplot(1,5,1);
% rst_data_plot([reshape(diff_01_10,1,[])' reshape(diff_05_10,1,[])' reshape(diff_1_10,1,[])' reshape(diff_2_10,1,[])' reshape(diff_u_10,1,[])'])
% subplot(1,5,2);
% rst_data_plot([reshape(diff_01_25,1,[])' reshape(diff_05_25,1,[])' reshape(diff_1_25,1,[])' reshape(diff_2_25,1,[])' reshape(diff_u_25,1,[])'])
% subplot(1,5,3);
% rst_data_plot([reshape(diff_01_50,1,[])' reshape(diff_05_50,1,[])' reshape(diff_1_50,1,[])' reshape(diff_2_50,1,[])' reshape(diff_u_50,1,[])'])
% subplot(1,5,4);
% rst_data_plot([reshape(diff_01_100,1,[])' reshape(diff_05_100,1,[])' reshape(diff_1_100,1,[])' reshape(diff_2_100,1,[])' reshape(diff_u_100,1,[])'])
% subplot(1,5,5);
% rst_data_plot([reshape(diff_01_200,1,[])' reshape(diff_05_200,1,[])' reshape(diff_1_200,1,[])' reshape(diff_2_200,1,[])' reshape(diff_u_200,1,[])'])
%%%%%%%%


NUM_TAPS = [10 25 50 100 200];
batch_num = 32;

fir_taps_01 = getTapsFromHDF5('_con01',NUM_TAPS,batch_num);
fir_taps_05 = getTapsFromHDF5('_con05',NUM_TAPS,batch_num);
fir_taps_1 = getTapsFromHDF5('_con1',NUM_TAPS,batch_num);
fir_taps_2 = getTapsFromHDF5('_con2',NUM_TAPS,batch_num);
fir_taps_u = getTapsFromHDF5('_unconstr',NUM_TAPS,batch_num);

%%

% figure(1)
% subplot(1,5,1);
% cdfplot(abs(reshape(fir_taps_01(1).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_05(1).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_1(1).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_2(1).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_u(1).taps,1,[]))); % Use 256 bins.
% xlabel('Taps magnitude, abs()')
% set(gca,'XScale','log')
% subplot(1,5,2);
% cdfplot(abs(reshape(fir_taps_01(2).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_05(2).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_1(2).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_2(2).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_u(2).taps,1,[]))); % Use 256 bins.
% xlabel('Taps magnitude, abs()')
% set(gca,'XScale','log')
% subplot(1,5,3);
% cdfplot(abs(reshape(fir_taps_01(3).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_05(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_1(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_2(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_u(3).taps,1,[]))); % Use 256 bins.
% xlabel('Taps magnitude, abs()')
% set(gca,'XScale','log')
% subplot(1,5,4);
% cdfplot(abs(reshape(fir_taps_01(4).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_05(4).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_1(4).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_2(4).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_u(4).taps,1,[]))); % Use 256 bins.
% xlabel('Taps magnitude, abs()')
% set(gca,'XScale','log')
% subplot(1,5,5);
% cdfplot(abs(reshape(fir_taps_01(5).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_05(5).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_1(5).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_2(5).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_u(5).taps,1,[]))); % Use 256 bins.
% xlabel('Taps magnitude, abs()')
% set(gca,'XScale','log')
% legend('\epsilon=0.1','\epsilon=0.5','\epsilon=1','\epsilon=2','\epsilon=\infty')

%%

% figure(11)
% subplot(1,5,1);
% cdfplot(abs(reshape(fir_taps_01(1).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_01(2).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_01(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_01(4).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_01(5).taps,1,[]))); % Use 256 bins.
% % set(gca,'XScale','log')
% xlabel('Taps magnitude, abs()')
% subplot(1,5,2);
% cdfplot(abs(reshape(fir_taps_05(1).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_05(2).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_05(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_05(4).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_05(5).taps,1,[]))); % Use 256 bins.
% % set(gca,'XScale','log')
% xlabel('Taps magnitude, abs()')
% subplot(1,5,3);
% cdfplot(abs(reshape(fir_taps_1(1).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_1(2).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_1(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_1(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_1(4).taps,1,[]))); % Use 256 bins.
% xlabel('Taps magnitude, abs()')
% % set(gca,'XScale','log')
% subplot(1,5,4);
% cdfplot(abs(reshape(fir_taps_2(1).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_2(2).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_2(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_2(4).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_2(5).taps,1,[]))); % Use 256 bins.
% xlabel('Taps magnitude, abs()')
% % set(gca,'XScale','log')
% subplot(1,5,5);
% cdfplot(abs(reshape(fir_taps_u(1).taps,1,[]))); % Use 256 bins.
% hold on
% cdfplot(abs(reshape(fir_taps_u(2).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_u(3).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_u(4).taps,1,[]))); % Use 256 bins.
% cdfplot(abs(reshape(fir_taps_u(5).taps,1,[]))); % Use 256 bins.
% xlabel('Taps magnitude, abs()')
% % set(gca,'XScale','log')
% legend('M=10','M=25','M=50','M=100','M=200')
% 

%%

fir_unit_10 = zeros(2,10);
fir_unit_10(1,1) = 1;
fir_unit_10 = repmat(fir_unit_10,[1 1 24]);
fir_unit_25 = zeros(2,25);
fir_unit_25(1,1) = 1;
fir_unit_25 = repmat(fir_unit_25,[1 1 24]);
fir_unit_50 = zeros(2,50);
fir_unit_50(1,1) = 1;
fir_unit_50 = repmat(fir_unit_50,[1 1 24]);
fir_unit_100 = zeros(2,100);
fir_unit_100(1,1) = 1;
fir_unit_100 = repmat(fir_unit_100,[1 1 24]);
fir_unit_200 = zeros(2,200);
fir_unit_200(1,1) = 1;
fir_unit_200 = repmat(fir_unit_200,[1 1 24]);

diff_01_10 = fir_taps_01(1).taps - fir_unit_10;
diff_01_25 = fir_taps_01(2).taps - fir_unit_25;
diff_01_50 = fir_taps_01(3).taps - fir_unit_50;
diff_01_100 = fir_taps_01(4).taps - fir_unit_100;
diff_01_200 = fir_taps_01(5).taps - fir_unit_200;

diff_05_10 = fir_taps_05(1).taps - fir_unit_10;
diff_05_25 = fir_taps_05(2).taps - fir_unit_25;
diff_05_50 = fir_taps_05(3).taps - fir_unit_50;
diff_05_100 = fir_taps_05(4).taps - fir_unit_100;
diff_05_200 = fir_taps_05(5).taps - fir_unit_200;

diff_1_10 = fir_taps_1(1).taps - fir_unit_10;
diff_1_25 = fir_taps_1(2).taps - fir_unit_25;
diff_1_50 = fir_taps_1(3).taps - fir_unit_50;
diff_1_100 = fir_taps_1(4).taps - fir_unit_100;
diff_1_200 = fir_taps_1(5).taps - fir_unit_200;

diff_2_10 = fir_taps_2(1).taps - fir_unit_10;
diff_2_25 = fir_taps_2(2).taps - fir_unit_25;
diff_2_50 = fir_taps_2(3).taps - fir_unit_50;
diff_2_100 = fir_taps_2(4).taps - fir_unit_100;
diff_2_200 = fir_taps_2(5).taps - fir_unit_200;

diff_u_10 = fir_taps_u(1).taps - fir_unit_10;
diff_u_25 = fir_taps_u(2).taps - fir_unit_25;
diff_u_50 = fir_taps_u(3).taps - fir_unit_50;
diff_u_100 = fir_taps_u(4).taps - fir_unit_100;
diff_u_200 = fir_taps_u(5).taps - fir_unit_200;

conf_per = 0.05;

[avg_01_10, CI_01_10] = compute_confidence(reshape(diff_01_10,1,[]),conf_per);
[avg_01_25, CI_01_25] = compute_confidence(reshape(diff_01_25,1,[]),conf_per);
[avg_01_50, CI_01_50] = compute_confidence(reshape(diff_01_50,1,[]),conf_per);
[avg_01_100, CI_01_100] = compute_confidence(reshape(diff_01_100,1,[]),conf_per);
[avg_01_200, CI_01_200] = compute_confidence(reshape(diff_01_200,1,[]),conf_per);

[avg_05_10, CI_05_10] = compute_confidence(reshape(diff_05_10,1,[]),conf_per);
[avg_05_25, CI_05_25] = compute_confidence(reshape(diff_05_25,1,[]),conf_per);
[avg_05_50, CI_05_50] = compute_confidence(reshape(diff_05_50,1,[]),conf_per);
[avg_05_100, CI_05_100] = compute_confidence(reshape(diff_05_100,1,[]),conf_per);
[avg_05_200, CI_05_200] = compute_confidence(reshape(diff_05_200,1,[]),conf_per);

[avg_1_10, CI_1_10] = compute_confidence(reshape(diff_1_10,1,[]),conf_per);
[avg_1_25, CI_1_25] = compute_confidence(reshape(diff_1_25,1,[]),conf_per);
[avg_1_50, CI_1_50] = compute_confidence(reshape(diff_1_50,1,[]),conf_per);
[avg_1_100, CI_1_100] = compute_confidence(reshape(diff_1_100,1,[]),conf_per);
[avg_1_200, CI_1_200] = compute_confidence(reshape(diff_1_200,1,[]),conf_per);

[avg_2_10, CI_2_10] = compute_confidence(reshape(diff_2_10,1,[]),conf_per);
[avg_2_25, CI_2_25] = compute_confidence(reshape(diff_2_25,1,[]),conf_per);
[avg_2_50, CI_2_50] = compute_confidence(reshape(diff_2_50,1,[]),conf_per);
[avg_2_100, CI_2_100] = compute_confidence(reshape(diff_2_100,1,[]),conf_per);
[avg_2_200, CI_2_200] = compute_confidence(reshape(diff_2_200,1,[]),conf_per);

[avg_u_10, CI_u_10] = compute_confidence(reshape(diff_u_10,1,[]),conf_per);
[avg_u_25, CI_u_25] = compute_confidence(reshape(diff_u_25,1,[]),conf_per);
[avg_u_50, CI_u_50] = compute_confidence(reshape(diff_u_50,1,[]),conf_per);
[avg_u_100, CI_u_100] = compute_confidence(reshape(diff_u_100,1,[]),conf_per);
[avg_u_200, CI_u_200] = compute_confidence(reshape(diff_u_200,1,[]),conf_per);

% figure(12)
% subplot(1,5,1);
% cdfplot(abs(reshape(diff_01_10,1,[])))
% hold on
% cdfplot(abs(reshape(diff_01_25,1,[])))
% cdfplot(abs(reshape(diff_01_50,1,[])))
% cdfplot(abs(reshape(diff_01_100,1,[])))
% cdfplot(abs(reshape(diff_01_200,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')
% subplot(1,5,2);
% cdfplot(abs(reshape(diff_05_10,1,[])))
% hold on
% cdfplot(abs(reshape(diff_05_25,1,[])))
% cdfplot(abs(reshape(diff_05_50,1,[])))
% cdfplot(abs(reshape(diff_05_100,1,[])))
% cdfplot(abs(reshape(diff_05_200,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')
% subplot(1,5,3);
% cdfplot(abs(reshape(diff_1_10,1,[])))
% hold on
% cdfplot(abs(reshape(diff_1_25,1,[])))
% cdfplot(abs(reshape(diff_1_50,1,[])))
% cdfplot(abs(reshape(diff_1_100,1,[])))
% cdfplot(abs(reshape(diff_1_200,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')
% subplot(1,5,4);
% cdfplot(abs(reshape(diff_2_10,1,[])))
% hold on
% cdfplot(abs(reshape(diff_2_25,1,[])))
% cdfplot(abs(reshape(diff_2_50,1,[])))
% cdfplot(abs(reshape(diff_2_100,1,[])))
% cdfplot(abs(reshape(diff_2_200,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')
% subplot(1,5,5);
% cdfplot(abs(reshape(diff_u_10,1,[])))
% hold on
% cdfplot(abs(reshape(diff_u_25,1,[])))
% cdfplot(abs(reshape(diff_u_50,1,[])))
% cdfplot(abs(reshape(diff_u_100,1,[])))
% cdfplot(abs(reshape(diff_u_200,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')

%%

% figure(13)
% subplot(1,5,1);
% cdfplot(abs(reshape(diff_01_10,1,[])))
% hold on
% cdfplot(abs(reshape(diff_05_10,1,[])))
% cdfplot(abs(reshape(diff_1_10,1,[])))
% cdfplot(abs(reshape(diff_2_10,1,[])))
% cdfplot(abs(reshape(diff_u_10,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')
% subplot(1,5,2);
% cdfplot(abs(reshape(diff_01_25,1,[])))
% hold on
% cdfplot(abs(reshape(diff_05_25,1,[])))
% cdfplot(abs(reshape(diff_1_25,1,[])))
% cdfplot(abs(reshape(diff_2_25,1,[])))
% cdfplot(abs(reshape(diff_u_25,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')
% subplot(1,5,3);
% cdfplot(abs(reshape(diff_01_50,1,[])))
% hold on
% cdfplot(abs(reshape(diff_05_50,1,[])))
% cdfplot(abs(reshape(diff_1_50,1,[])))
% cdfplot(abs(reshape(diff_2_50,1,[])))
% cdfplot(abs(reshape(diff_u_50,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')
% subplot(1,5,4);
% cdfplot(abs(reshape(diff_01_100,1,[])))
% hold on
% cdfplot(abs(reshape(diff_05_100,1,[])))
% cdfplot(abs(reshape(diff_1_100,1,[])))
% cdfplot(abs(reshape(diff_2_100,1,[])))
% cdfplot(abs(reshape(diff_u_100,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')
% subplot(1,5,5);
% cdfplot(abs(reshape(diff_01_200,1,[])))
% hold on
% cdfplot(abs(reshape(diff_05_200,1,[])))
% cdfplot(abs(reshape(diff_1_200,1,[])))
% cdfplot(abs(reshape(diff_2_200,1,[])))
% cdfplot(abs(reshape(diff_u_200,1,[])))
% legend('M=10','M=25','M=50','M=100','M=200')

%%

% figure(2)
% subplot(1,5,1);
% T = bplot(abs([reshape(diff_01_10,1,[])' reshape(diff_05_10,1,[])' reshape(diff_1_10,1,[])' reshape(diff_2_10,1,[])' reshape(diff_u_10,1,[])']),'whisker',5);
% legend(T)
% title('M=10')
% ylabel('Distance between ideal FIR')
% set(gca,'Xtick',1:5)
% set(gca, 'XTickLabel', {'\epsilon=0.1' '\epsilon=0.5' '\epsilon=1' '\epsilon=2' '\epsilon=\infty'})
% subplot(1,5,2);
% bplot(abs([reshape(diff_01_25,1,[])' reshape(diff_05_25,1,[])' reshape(diff_1_25,1,[])' reshape(diff_2_25,1,[])' reshape(diff_u_25,1,[])']),'whisker',5)
% title('M=25')
% ylabel('Distance between ideal FIR')
% set(gca,'Xtick',1:5)
% set(gca, 'XTickLabel', {'\epsilon=0.1' '\epsilon=0.5' '\epsilon=1' '\epsilon=2' '\epsilon=\infty'})
% subplot(1,5,3);
% bplot(abs([reshape(diff_01_50,1,[])' reshape(diff_05_50,1,[])' reshape(diff_1_50,1,[])' reshape(diff_2_50,1,[])' reshape(diff_u_50,1,[])']),'whisker',5)
% title('M=50')
% ylabel('Distance between ideal FIR')
% set(gca,'Xtick',1:5)
% set(gca, 'XTickLabel', {'\epsilon=0.1' '\epsilon=0.5' '\epsilon=1' '\epsilon=2' '\epsilon=\infty'})
% subplot(1,5,4);
% bplot(abs([reshape(diff_01_100,1,[])' reshape(diff_05_100,1,[])' reshape(diff_1_100,1,[])' reshape(diff_2_100,1,[])' reshape(diff_u_100,1,[])']),'whisker',5)
% title('M=100')
% ylabel('Distance between ideal FIR')
% set(gca,'Xtick',1:5)
% set(gca, 'XTickLabel', {'\epsilon=0.1' '\epsilon=0.5' '\epsilon=1' '\epsilon=2' '\epsilon=\infty'})
% subplot(1,5,5);
% bplot(abs([reshape(diff_01_200,1,[])' reshape(diff_05_200,1,[])' reshape(diff_1_200,1,[])' reshape(diff_2_200,1,[])' reshape(diff_u_200,1,[])']),'whisker',5)
% title('M=200')
% ylabel('Distance between ideal FIR')
% set(gca,'Xtick',1:5)
% set(gca, 'XTickLabel', {'\epsilon=0.1' '\epsilon=0.5' '\epsilon=1' '\epsilon=2' '\epsilon=\infty'})

% figure
% subplot(1,5,1);
% rst_data_plot([reshape(diff_01_10,1,[])' reshape(diff_05_10,1,[])' reshape(diff_1_10,1,[])' reshape(diff_2_10,1,[])' reshape(diff_u_10,1,[])'])
% subplot(1,5,2);
% rst_data_plot([reshape(diff_01_25,1,[])' reshape(diff_05_25,1,[])' reshape(diff_1_25,1,[])' reshape(diff_2_25,1,[])' reshape(diff_u_25,1,[])'])
% subplot(1,5,3);
% rst_data_plot([reshape(diff_01_50,1,[])' reshape(diff_05_50,1,[])' reshape(diff_1_50,1,[])' reshape(diff_2_50,1,[])' reshape(diff_u_50,1,[])'])
% subplot(1,5,4);
% rst_data_plot([reshape(diff_01_100,1,[])' reshape(diff_05_100,1,[])' reshape(diff_1_100,1,[])' reshape(diff_2_100,1,[])' reshape(diff_u_100,1,[])'])
% subplot(1,5,5);
% rst_data_plot([reshape(diff_01_200,1,[])' reshape(diff_05_200,1,[])' reshape(diff_1_200,1,[])' reshape(diff_2_200,1,[])' reshape(diff_u_200,1,[])'])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(3)
subplot(1,2,1);
bar(batch_accuracy)
grid on
set(gca, 'XTickLabel', {'M=10' 'M=25' 'M=50' 'M=100' 'M=200'})
legend('Baseline','\epsilon=0.1','\epsilon=0.5','\epsilon=1','\epsilon=2')
xlabel('Number of taps, M')
ylabel('Per-Batch Accuracy (PBA)')
subplot(1,2,2);
bar(example_accuracy)
grid on
set(gca, 'XTickLabel', {'M=10' 'M=25' 'M=50' 'M=100' 'M=200'})
legend('Baseline','\epsilon=0.1','\epsilon=0.5','\epsilon=1','\epsilon=2','\epsilon=\infty')
xlabel('Number of taps, M')
ylabel('Per-Slice Accuracy (PSA)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%


figure(5)
subplot(5,6,1);
imagesc(x_bas.pickle_data.confusion_matrix,clim)
title('Baseline')
subplot(5,6,2);
imagesc(x_c_01_10.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.1, M=10')
subplot(5,6,3);
imagesc(x_c_01_25.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.1, M=25')
subplot(5,6,4);
imagesc(x_c_01_50.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.1, M=50')
subplot(5,6,5);
imagesc(x_c_01_100.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.1, M=100')
subplot(5,6,6);
imagesc(x_c_01_200.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.1, M=200')

subplot(5,6,1+6);
imagesc(x_bas.pickle_data.confusion_matrix,clim)
title('Baseline')
subplot(5,6,2+6);
imagesc(x_c_05_10.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.5, M=10')
subplot(5,6,3+6);
imagesc(x_c_05_25.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.5, M=25')
subplot(5,6,4+6);
imagesc(x_c_05_50.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.5, M=50')
subplot(5,6,5+6);
imagesc(x_c_05_100.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.5, M=100')
subplot(5,6,6+6);
imagesc(x_c_05_200.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=0.5, M=200')

subplot(5,6,1+12);
imagesc(x_bas.pickle_data.confusion_matrix,clim)
title('Baseline')
subplot(5,6,2+12);
imagesc(x_c_1_10.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=1, M=10')
subplot(5,6,3+12);
imagesc(x_c_1_25.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=1, M=25')
subplot(5,6,4+12);
imagesc(x_c_1_50.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=1, M=50')
subplot(5,6,5+12);
imagesc(x_c_1_100.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=1, M=100')
subplot(5,6,6+12);
imagesc(x_c_1_200.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=1, M=200')

subplot(5,6,1+18);
imagesc(x_bas.pickle_data.confusion_matrix,clim)
title('Baseline')
subplot(5,6,2+18);
imagesc(x_c_2_10.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=2, M=10')
subplot(5,6,3+18);
imagesc(x_c_2_25.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=2, M=25')
subplot(5,6,4+18);
imagesc(x_c_2_50.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=2, M=50')
subplot(5,6,5+18);
imagesc(x_c_2_100.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=2, M=100')
subplot(5,6,6+18);
imagesc(x_c_2_200.pickle_data.confusion_matrix,clim)
title('Constrained, \epsilon=2, M=200')

subplot(5,6,1+24);
imagesc(x_bas.pickle_data.confusion_matrix,clim)
title('Baseline')
subplot(5,6,2+24);
imagesc(x_u_10.pickle_data.confusion_matrix,clim)
title('Unconstrained, M=10')
subplot(5,6,3+24);
imagesc(x_u_25.pickle_data.confusion_matrix,clim)
title('Unconstrained, M=25')
subplot(5,6,4+24);
imagesc(x_u_50.pickle_data.confusion_matrix,clim)
title('Unconstrained, M=50')
subplot(5,6,5+24);
imagesc(x_u_100.pickle_data.confusion_matrix,clim)
title('Unconstrained, M=100')
subplot(5,6,6+24);
imagesc(x_u_200.pickle_data.confusion_matrix,clim)
title('Unconstrained, M=200')

%%

figure(6)
subplot(3,3,1);
imagesc(x_bas.pickle_data.confusion_matrix,clim)
set(gca,'FontSize',14)
title('Baseline')
subplot(3,3,2);
imagesc(x_c_01_10.pickle_data.confusion_matrix,clim)
set(gca,'FontSize',14)
title('\epsilon=0.1, M=10')
subplot(3,3,3);
imagesc(x_c_01_200.pickle_data.confusion_matrix,clim)
set(gca,'FontSize',14)
title('\epsilon=0.1, M=200')
subplot(3,3,4);
imagesc(x_bas.pickle_data.confusion_matrix,clim)
set(gca,'FontSize',14)
title('Baseline')
subplot(3,3,5);
imagesc(x_c_1_10.pickle_data.confusion_matrix,clim)
title('\epsilon=1, M=10')
set(gca,'FontSize',14)
subplot(3,3,6);
imagesc(x_c_1_200.pickle_data.confusion_matrix,clim)
title('\epsilon=1, M=200')
set(gca,'FontSize',14)
subplot(3,3,7);
imagesc(x_bas.pickle_data.confusion_matrix,clim)
title('Baseline')
set(gca,'FontSize',14)
subplot(3,3,8);
imagesc(x_u_10.pickle_data.confusion_matrix,clim)
title('\epsilon=\infty, M=10')
set(gca,'FontSize',14)
subplot(3,3,9);
imagesc(x_u_200.pickle_data.confusion_matrix,clim)
set(gca,'FontSize',14)
title('\epsilon=\infty, M=200')