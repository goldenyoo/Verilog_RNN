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
data_label = 'b';

%%
FILENAME = strcat('D:\바탕화면\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_eval_ds1',data_label,'.mat');
% FILENAME = strcat('D:\바탕화면\Verilog RNN\my_git_folder\1a\eval_',data_label,'.mat');
load(FILENAME);

% cnt = ALLEEG(4).data;
cnt= 0.1*double(cnt);
cnt = cnt';

cnt_c = cnt([27 29 31 44 46 50 52 54],:);
% cnt_c = cnt([27 29 31 44-1 46-1 50-1 52-1 54-1],:);

clear cnt


%%
FILENAME = strcat('D:\바탕화면\Motor Imagery EEG data\true_labels\BCICIV_eval_ds1',data_label,'_1000Hz_true_y.mat');
load(FILENAME);

true_y = downsample(true_y,10);
%%
load("D:\바탕화면\Verilog RNN\my_git_folder\1a\net.mat");

W_ux = net.Layers(2).InputWeights(1:length(net.Layers(2).InputWeights)/4,:);
W_fx = net.Layers(2).InputWeights(length(net.Layers(2).InputWeights)/4+1:length(net.Layers(2).InputWeights)*2/4,:);
W_cx = net.Layers(2).InputWeights(length(net.Layers(2).InputWeights)*2/4+1:length(net.Layers(2).InputWeights)*3/4,:);
W_ox = net.Layers(2).InputWeights(length(net.Layers(2).InputWeights)*3/4+1:length(net.Layers(2).InputWeights)*4/4,:);
W_ua = net.Layers(2).RecurrentWeights(1:length(net.Layers(2).RecurrentWeights)/4,:);
W_fa = net.Layers(2).RecurrentWeights(length(net.Layers(2).RecurrentWeights)/4+1:length(net.Layers(2).RecurrentWeights)*2/4,:);
W_ca = net.Layers(2).RecurrentWeights(length(net.Layers(2).RecurrentWeights)*2/4+1:length(net.Layers(2).RecurrentWeights)*3/4,:);
W_oa = net.Layers(2).RecurrentWeights(length(net.Layers(2).RecurrentWeights)*3/4+1:length(net.Layers(2).RecurrentWeights)*4/4,:);
b_u  = net.Layers(2).Bias(1:length(net.Layers(2).Bias)/4,:);
b_f  = net.Layers(2).Bias(length(net.Layers(2).Bias)/4+1:length(net.Layers(2).Bias)*2/4,:);
b_c  = net.Layers(2).Bias(length(net.Layers(2).Bias)*2/4+1:length(net.Layers(2).Bias)*3/4,:);
b_o  = net.Layers(2).Bias(length(net.Layers(2).Bias)*3/4+1:length(net.Layers(2).Bias)*4/4,:);

W1 = net.Layers(3).Weights;
b1 = net.Layers(3).Bias;

learning_rate = 0.01;

step = 1;
lastsize = 0;
for i = 1:step:size(cnt_c,2)-100
    if mod(i,100) < 5
        fprintf(repmat('\b',1,lastsize));
        lastsize = fprintf("%d",i);
    end
    test_x = cnt_c(:,i:i+99);
    
    a_next = zeros(100 ,1);
    c_next = zeros(100 ,1);
    
    
    for t = 1:100
        xt = test_x(:,t);
        a_prev = a_next;
        c_prev = c_next;
        
        lstm_unit.a_prev = a_prev;
        lstm_unit.c_prev = c_prev;
        
        [a_next, c_next, G_u, G_f, G_o, c_tmp] = lstm_forward(xt, a_prev, c_prev,W_ux,W_fx,W_ox,W_cx,W_ua,W_fa,W_oa,W_ca, b_u, b_f, b_o, b_c);
        lstm_unit.a_next = a_next;
        lstm_unit.c_next = c_next;
        lstm_unit.G_u = G_u;
        lstm_unit.G_f = G_f;
        lstm_unit.G_o = G_o;
        lstm_unit.c_tmp = c_tmp;
        
        lstm_units{t,1} = lstm_unit;
    end
    
    Z1 = W1 * a_next + b1;
    
    A1 = (1/(exp(Z1(1))+ exp(Z1(2)) + exp(Z1(3)))) *[exp(Z1(1)); exp(Z1(2)); exp(Z1(3))];
    
    
    [tmp, ypred_tmp] = max(Z1);
    
    if (ypred_tmp == 1)
        ypred(i,1) = -1;
    elseif (ypred_tmp == 2)
        ypred(i,1) = 0;
    else
        ypred(i,1) = 1;
    end
    
    if i < 0
        if (true_y(i,1) == 1)&(true_y(i+99,1) == 1) | (true_y(i,1) == 0)&(true_y(i+99,1) == 0) | (true_y(i,1) == -1)&(true_y(i+99,1) == -1)
            
%             learn_cnt = learn_cnt + 1;
            
            if  ypred(i) == -1
                y = [1;0;0];
            elseif ypred(i) == 0
                y = [0;1;0];
            else
                y = [0;0;1];
            end
            
            if  true_y(i,1) == -1
                t = [1;0;0];
            elseif true_y(i,1) == 0
                t = [0;1;0];
            else
                t = [0;0;1];
            end
            
            % First layer
            
            back = (A1-t);
            
            dW1 = back * a_next';
            db1 = back;
            
            back2 = W1' * back;
            
            
            % LSTM backpropagation
            dc_next = zeros(100,1);
            da_next = back2;
            
            dW_ux = 0;
            dW_fx = 0;
            dW_cx = 0;
            dW_ox = 0;
            
            dW_ua = 0;
            dW_fa = 0;
            dW_ca = 0;
            dW_oa = 0;
            
            db_u = 0;
            db_f = 0;
            db_c = 0;
            db_o = 0;
            
            
            for back_itr = 100:-1:1
                xt = test_x(:,back_itr);
                [da_prev, dc_prev, ddW_ux, ddW_fx, ddW_cx, ddW_ox, ddW_ua, ddW_fa, ddW_ca, ddW_oa, ddb_u, ddb_f, ddb_c, ddb_o] = lstm_cell_back(da_next, dc_next, lstm_units{back_itr,1}, xt,W_ux,W_fx,W_ox,W_cx,W_ua,W_fa,W_oa,W_ca);
                da_next = da_prev;
                dc_next = dc_prev;
                
                dW_ux = dW_ux + ddW_ux;
                dW_fx = dW_fx + ddW_fx;
                dW_cx = dW_cx + ddW_cx;
                dW_ox = dW_ox + ddW_ox;
                
                dW_ua = dW_ua + ddW_ua;
                dW_fa = dW_fa + ddW_fa;
                dW_ca = dW_ca + ddW_ca;
                dW_oa = dW_oa + ddW_oa;
                
                db_u = db_u + ddb_u;
                db_f = db_f + ddb_f;
                db_c = db_c + ddb_c;
                db_o = db_o + ddb_o;
            end
            
            % Update
            %Update
            W1 = W1 - learning_rate * dW1;
            b1 = b1 - learning_rate * db1;
            
            %Update
            W_ux = W_ux - learning_rate * dW_ux;
            W_fx = W_fx - learning_rate * dW_fx;
            W_cx = W_cx - learning_rate * dW_cx;
            W_ox = W_ox - learning_rate * dW_ox;
            
            W_ua = W_ua - learning_rate * dW_ua;
            W_fa = W_fa - learning_rate * dW_fa;
            W_ca = W_ca - learning_rate * dW_ca;
            W_oa = W_oa - learning_rate * dW_oa;
            
            b_u = b_u - learning_rate * db_u;
            b_f = b_f - learning_rate * db_f;
            b_c = b_c - learning_rate * db_c;
            b_o = b_o - learning_rate * db_o;
            
        end
    end
    clear lstm_units
end

%%
total = 0;
good = 0;
for i = 100000:size(cnt_c,2)-100
    if true_y(i) == -1
        total = total + 1;
        if ypred(i) == -1
            good = good + 1;
        end
    elseif true_y(i) == 0
        total = total + 1;
        if ypred(i) == 0
            good = good + 1;
        end
    elseif true_y(i) == 1
        total = total + 1;
        if ypred(i) == 1
            good = good + 1;
        end
    end
end
fprintf("\nAcc: %.4f\n",good / total);
%     clear ypred true_y
% end

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
