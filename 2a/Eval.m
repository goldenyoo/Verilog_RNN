% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_03_30                           
%                                                            
 % ----------------------------------------------------------------------- %
clear all
clc

data_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9'];
% data_labels = ['1'];

c1 = 0;
c2 = 0;

for data_label = data_labels

FILENAME = strcat('D:\바탕화면\BCIIV_2a_mat\A0',data_label,'E_mat');
load(FILENAME);

FILENAME1 = strcat('D:\바탕화면\BCIIV_2a_mat\true_labels\A0',data_label,'E.mat');
load(FILENAME1);

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
m = 16;
for i = 1:length(Class_1)
   [XTest_1{i+c1,1}, XTest_2{i+c1,1}] = my_SAX(s(Class_1(i):Class_1(i)+313,1:22),m);
   YTest(i+c1,1) = 1;
end

for i = 1:length(Class_2)
   [XTest_1{i+length(Class_1)+c1,1}, XTest_2{i+length(Class_1)+c1,1}] = my_SAX(s(Class_2(i):Class_2(i)+313,1:22),m);
   YTest(i+length(Class_1)+c1,1) = 2;
end

c1 = c1 + length(Class_1) + length(Class_2);
end
YTest = categorical(YTest);


%% 

FILENAME = 'D:\바탕화면\Verilog RNN\my_git_folder\2a\net.mat';
load(FILENAME);



YPred_1 = classify(net_1,XTest_1,'SequenceLength','longest');
YPred_2 = classify(net_2,XTest_2,'SequenceLength','longest');


for k = 1:length(YPred_1)
    L = 0;
    R = 0;
    if YPred_1(k) == categorical(1)
       L = L + 1; 
    elseif YPred_1(k) == categorical(2)
        R = R + 1;
    end
    if YPred_2(k) == categorical(1)
       L = L + 1; 
    elseif YPred_2(k) == categorical(2)
        R = R + 1;
    end
    
    if L > R
        YPred(k,1) = 1;
    else
        YPred(k,1) = 2;
    end
end

YPred = categorical(YPred);

acc = sum(YPred == YTest)./numel(YTest);
disp(sprintf('Score: %f  ',acc));
    
%% 
function n_signal = my_normalization(s)
    Mean = mean(s);
    Std = std(s);
    
    n_signal = (s - Mean)./Std;
end

function [A,M,C] = my_hjorth(s,m)
    q = floor(length(s)/m);
    A = [];
    M = [];
    C = [];
    O = [];
    for k = 1:m
       tmp_s = s((k-1)*q+1:k*q,:);
       a = var(tmp_s);
       dif_y = diff(tmp_s);
       m = (var(dif_y)./a).^(0.5);
       dif_yy = diff(tmp_s,2);
       c = (var(dif_yy)./a).^(0.5);
       
       A = [A; std(tmp_s)];
       M = [M; m];
       C = [C; c];
       
    end
    
    A = A';
    M = M';
    C = C';
    O = [A M C];
    
end
function [K, A] = my_SAX(s,m)
n_signal = my_normalization(s);
q = floor(length(n_signal)/m);
K = [];
A = [];
for j = 1:m
   t = q*(j-1):q*j;
   v = n_signal(t+1,:);
   v_bar = mean(v);
   t_bar = mean(t);
   
   k_son = (t - t_bar)*(v - v_bar);
   k_mom = sum((t - t_bar).^2);
   
   k = k_son / k_mom;
   b = v_bar - k*t_bar;
   
   a = k*t_bar + b;
   
   K = [K k'];
   A = [A a'];
end

end