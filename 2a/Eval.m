% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo
%
%    Last Modified: 2020_03_30
%
% ----------------------------------------------------------------------- %
clear all
% clc

% data_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9'];
data_labels = ['2'];

c1 = 0;
c2 = 0;

for data_label = data_labels
    
    FILENAME = strcat('D:\바탕화면\Verilog RNN\my_git_folder\2a\ICA_A0',data_label,'E.mat');
    load(FILENAME);

    s = ALLEEG(2).data';
    
    
    FILENAME = strcat('D:\바탕화면\BCIIV_2a_mat\A0',data_label,'E_mat');
    load(FILENAME, 'h');
    
    FILENAME1 = strcat('D:\바탕화면\BCIIV_2a_mat\true_labels\A0',data_label,'E.mat');
    load(FILENAME1);
    
    % NaN 제거, 바로 이전 data point로 대체
    tmp = ~isfinite(s);
    for i = 1:size(tmp,1)
        for j = 1:size(tmp,2)
            if tmp(i,j) == 1
                s(i,j) = s(i-1,j);
            end
        end
    end
    
    %%
    
    Class_unknown = [];
    Class_1 = [];
    Class_2 = [];
    Class_3 = [];
    Class_4 = [];
    
    
    i=1;
    while i <=length(h.EVENT.TYP)
        if h.EVENT.TYP(i)== 783
            Class_unknown = [Class_unknown h.EVENT.POS(i)];
            i = i + 1;
        else
            i = i + 1;
        end
    end
    
    for i = 1:length(Class_unknown)
        if classlabel(i) == 1
            Class_1 = [Class_1 Class_unknown(i)];
        elseif classlabel(i) == 2
            Class_2 = [Class_2 Class_unknown(i)];
        elseif classlabel(i) == 3
            Class_3 = [Class_3 Class_unknown(i)];
        else
            Class_4 = [Class_4 Class_unknown(i)];
        end
        
    end
    
    %%
    Class_mix = [Class_1 Class_2; ones(1,length(Class_1)) ones(1,length(Class_2))*2];
    [B, I] = sort(Class_mix(1,:),2);
    label = Class_mix(2,I);
    
    for i = 1:length(B)
        XTest{i+c1,1} = my_normalization(s(B(i):B(i)+313,1:22))';
        YTest(i+c1,1) = label(i);
    end
    
    c1 = c1 + length(B);
end
YTest = categorical(YTest);


%%

FILENAME = 'D:\바탕화면\Verilog RNN\my_git_folder\2a\net.mat';
load(FILENAME);

W_ux = (net.Layers(2).InputWeights(1:length(net.Layers(2).InputWeights)/4,:)) ;
W_fx = (net.Layers(2).InputWeights(length(net.Layers(2).InputWeights)/4+1:length(net.Layers(2).InputWeights)*2/4,:));
W_cx = (net.Layers(2).InputWeights(length(net.Layers(2).InputWeights)*2/4+1:length(net.Layers(2).InputWeights)*3/4,:));
W_ox = (net.Layers(2).InputWeights(length(net.Layers(2).InputWeights)*3/4+1:length(net.Layers(2).InputWeights)*4/4,:));
W_ua = (net.Layers(2).RecurrentWeights(1:length(net.Layers(2).RecurrentWeights)/4,:));
W_fa = (net.Layers(2).RecurrentWeights(length(net.Layers(2).RecurrentWeights)/4+1:length(net.Layers(2).RecurrentWeights)*2/4,:));
W_ca = (net.Layers(2).RecurrentWeights(length(net.Layers(2).RecurrentWeights)*2/4+1:length(net.Layers(2).RecurrentWeights)*3/4,:));
W_oa = (net.Layers(2).RecurrentWeights(length(net.Layers(2).RecurrentWeights)*3/4+1:length(net.Layers(2).RecurrentWeights)*4/4,:));
b_u  = (net.Layers(2).Bias(1:length(net.Layers(2).Bias)/4,:));
b_f  = (net.Layers(2).Bias(length(net.Layers(2).Bias)/4+1:length(net.Layers(2).Bias)*2/4,:));
b_c  = (net.Layers(2).Bias(length(net.Layers(2).Bias)*2/4+1:length(net.Layers(2).Bias)*3/4,:));
b_o  = (net.Layers(2).Bias(length(net.Layers(2).Bias)*3/4+1:length(net.Layers(2).Bias)*4/4,:));

W1 = (net.Layers(3).Weights);
b1 = (net.Layers(3).Bias);

W2 = (net.Layers(6).Weights);
b2 = (net.Layers(6).Bias);

W3 = (net.Layers(9).Weights);
b3 = (net.Layers(9).Bias);


back_cnt = 0;

dW3 = 0; db3 = 0;
dW2 = 0; db2 = 0;
dW1 = 0; db1 = 0;
dW_ux = 0; dW_fx = 0; dW_cx = 0; dW_ox = 0;
dW_ua = 0; dW_fa = 0; dW_ca = 0; dW_oa = 0;
db_u = 0; db_f = 0; db_c = 0; db_o = 0;

for i = 1:length(XTest)
    test_x = XTest{i,1};
    a_next = ones(314 ,1);
    c_next = ones(314 ,1);
    
    for t = 1:314
        xt =  test_x(:,t);
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
        clear lstm_unit
    end
    
    Z1 = W1 * a_next + b1;
    for k = 1:length(Z1)
        if Z1(k) <= 0
            A1(k,1) = 0;
        else
            A1(k,1) = Z1(k);
        end
    end
    
    Z2 = W2 * A1 + b2;
    for k = 1:length(Z2)
        if Z2(k) <= 0
            A2(k,1) = 0;
        else
            A2(k,1) = Z2(k);
        end
    end
    
    Z3 = W3 * A2 + b3;
    
    A3 =  (1/(exp(Z3(1))+ exp(Z3(2)) )) *[exp(Z3(1)); exp(Z3(2))];
    
    [tmp, ypred_tmp] = max(A3);
    
    if (ypred_tmp == 1)
        ypred(i,1) = 1;
    else
        ypred(i,1) = 2;
    end
    
    % Backpropagation
    if (i < 50) %&& ( YTest(i,1) ~= categorical(ypred(i,1)))
        back_cnt = back_cnt + 1;
        
        if YTest(i,1) == categorical(1)
            t = [1;0];
        else
            t = [0;1];
        end
        
        % softmax
        back1 = (A3 - t);
        
        % fc
        ddW3 = back1 * A2';
        ddb3 = back1;
        back2 = W3' * back1;
        
        dW3 = dW3 + ddW3;
        db3 = db3 + ddb3; 
        
        % ReLu
        back2(find(A2)) = 0;
        
        % fc
        ddW2 = back2 * A1';
        ddb2 = back2;
        back3 = W2' * back2;
        
        dW2 = dW2 + ddW2;
        db2 = db2 + ddb2;
        
        % ReLu
        back3(find(A1)) = 0;
        
        % fc
        ddW1 = back3 * a_next';
        ddb1 = back3;
        
        dW1 = dW1 + ddW1;
        db1 = db1 + ddb1;
        
        % LSTM backpropagation
        dc_next = zeros(314,1);
        da_next = W1' * back3;       
        
        for back_itr = 314:-1:1
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
        
    elseif i < 51
        learning_rate = 1;
        % Update
        W3 = W3 - 0.1* (dW3/back_cnt); %0.1
        b3 = b3 - 0.1* (db3/back_cnt); %0.1
        
        W2 = W2 - 0.1* (dW2/back_cnt); % 1 or 0.1
        b2 = b2 - 0.1* (db2/back_cnt); % 1 or 0.1
        
        W1 = W1 - 0.1* (dW1/back_cnt); % 1 or 0.1 ?
        b1 = b1 - 0.1* (db1/back_cnt); % 1 or 0.1
        
        %Update
        W_ux = W_ux - 1* (dW_ux/back_cnt); % 1 or 10?
        W_fx = W_fx - 1* (dW_fx/back_cnt);
        W_cx = W_cx - 1* (dW_cx/back_cnt);
        W_ox = W_ox - 1* (dW_ox/back_cnt);
        
        W_ua = W_ua - 1* (dW_ua/back_cnt); % 1 or 10?
        W_fa = W_fa - 1* (dW_fa/back_cnt);
        W_ca = W_ca - 1* (dW_ca/back_cnt);
        W_oa = W_oa - 1* (dW_oa/back_cnt);
        
        b_u = b_u - 1* (db_u/back_cnt);
        b_f = b_f - 1* (db_f/back_cnt);
        b_c = b_c - 1* (db_c/back_cnt);
        b_o = b_o - 1* (db_o/back_cnt);
    end

    
end

YPred = categorical(ypred);
%%

YPred_net = classify(net,XTest,'SequenceLength','longest');


acc_net = sum(YPred_net == YTest)./numel(YTest);
acc = sum(YPred == YTest)./numel(YTest);

fprintf('data label: %s\n',data_label);
fprintf('Total data \n');
fprintf('original net: %f  \n',acc_net);
fprintf('backpropagation: %f  \n',acc);



acc_net = sum(YPred_net(50:end) == YTest(50:end))./numel(YTest(50:end));
acc = sum(YPred(50:end) == YTest(50:end))./numel(YTest(50:end));

fprintf('Part of data \n');
fprintf('original net: %f  \n',acc_net);
fprintf('backpropagation: %f  \n',acc);

%%
function n_signal = my_normalization(s)
assert(isempty(find(isnan(s))));

Mean = mean(s);
Std = std(s);

n_signal = (s - Mean)./Std;
end

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

% dut = (c_tmp.*dpeter).*G_u .*(1 - G_u);
% dft = ( c_prev.*dpeter).*G_f.*(1 - G_f);
% dct = (G_u.*dpeter).*(1 - c_tmp.^2);
% dot = da_next.*tanh(c_next).*G_o.*(1 - G_o);

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