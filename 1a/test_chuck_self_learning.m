% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo
%
%    Last Modified: 2020_08_05
%             raw 신호를 이용,
%               calib은 train, validation set으로
%                   eval은 test set으로 mimic real task
%                      but data 뽑는 위치 랜덤하지 않다.
% ----------------------------------------------------------------------- %
%%
clc
close all
clear all

% Rereferencing method
ref_method = [0]; % Non(0), CAR(1), LAP(2)

% Filter order
filt_ord = [10];

% Reference electrode number
ref = 33;

% Input parameters
% data_labels = ['a' 'b' 'f' 'g'];
% data_labels = ['a'];
% for data_label = data_labels
data_label = 'f';

%%
% FILENAME = strcat('D:\바탕화면\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_eval_ds1',data_label,'.mat');
FILENAME = strcat('D:\바탕화면\Verilog RNN\my_git_folder\1a\eval_',data_label,'.mat');
load(FILENAME);

cnt = ALLEEG(3).data;
% cnt= 0.1*double(cnt);
% cnt = cnt';

% cnt_c = cnt([27 29 31 44 46 50 52 54],:);
 cnt_c = cnt([27 29 31 44-1 46-1 50-1 52-1 54-1],:);

clear cnt


%%
FILENAME = strcat('D:\바탕화면\Motor Imagery EEG data\true_labels\BCICIV_eval_ds1',data_label,'_1000Hz_true_y.mat');
load(FILENAME);

true_y = downsample(true_y,10);
%%
load("D:\바탕화면\Verilog RNN\my_git_folder\1a\net.mat");

step = 1;
lastsize = 0;
for i = 1:step:size(cnt_c,2)-100
    if mod(i,100) < 5
        fprintf(repmat('\b',1,lastsize));
        lastsize = fprintf("%d",i);
    end
    test_x{i,1} = cnt_c(:,i:i+99); 

end

  ypred_net = classify(net,test_x,'SequenceLength','longest');

%%

total = 0;
good = 0;
for i = 1:size(cnt_c,2)-100
    if true_y(i) == -1
        total = total + 1;
        if ypred_net(i) == categorical(-1)
            good = good + 1;
        end
    elseif true_y(i) == 0
        total = total + 1;
        if ypred_net(i) == categorical(0)
            good = good + 1;
        end
    elseif true_y(i) == 1
        total = total + 1;
        if ypred_net(i) == categorical(1)
            good = good + 1;
        end
    end
end
fprintf("\nAcc: %.4f\n",good / total);


% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %

%%
function [a_next, c_next, G_u, G_f, G_o, c_tmp] = lstm_forward(xt, a_prev, c_prev,W_ux,W_fx,W_ox,W_cx,W_ua,W_fa,W_oa,W_ca, b_u, b_f, b_o, b_c)
G_u = my_sigmoid(W_ux*xt + W_ua*a_prev + b_u);
G_f = my_sigmoid(W_fx*xt + W_fa*a_prev + b_f);
G_o = my_sigmoid(W_ox*xt + W_oa*a_prev + b_o);
c_tmp = tanh(W_cx*xt + W_ca*a_prev + b_c);


c_next = G_u.*c_tmp + G_f.*c_prev;
a_next = G_o.*tanh(c_next);

end

function output = my_sigmoid(a)
output = 1./(1+exp(-a));
end

function [da_prev, dc_prev, dW_ux, dW_fx, dW_cx, dW_ox, dW_ua, dW_fa, dW_ca, dW_oa, db_u, db_f, db_c, db_o] = lstm_cell_back(da_next, dc_next, lstm_unit, xt, W_ux,W_fx,W_ox,W_cx,W_ua,W_fa,W_oa,W_ca)
a_next = lstm_unit.a_next;
c_next = lstm_unit.c_next;
a_prev = lstm_unit.a_prev;
c_prev = lstm_unit.c_prev;
G_u = lstm_unit.G_u;
G_f = lstm_unit.G_f;
G_o = lstm_unit.G_o;
c_tmp = lstm_unit.c_tmp;

dpeter = (1-tanh(c_next).^2).*G_o.*da_next;

dut = (dc_next.*c_tmp + c_tmp.*dpeter).*G_u .*(1 - G_u);
dft = (dc_next.*c_prev + c_prev.*dpeter).*G_f.*(1 - G_f);
dct = (dc_next.*G_u + G_u.*dpeter).*(1 - c_tmp.^2);
dot = da_next.*tanh(c_next).*G_o.*(1 - G_o);



db_u = (dut);
db_f = (dft);
db_c = (dct);
db_o = (dot);

dW_ux = dut * xt';
dW_fx = dft * xt';
dW_cx = dct * xt';
dW_ox = dot * xt';

dW_ua = dut * a_prev';
dW_fa = dft * a_prev';
dW_ca = dct * a_prev';
dW_oa = dot * a_prev';

da_prev = W_ua'*dut + W_fa'*dft + W_ca'*dct + W_oa'*dot;

dc_prev = dc_next.*dft + dot.*(1-tanh(c_next).^2).*dft.*da_next;
end
