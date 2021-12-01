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
for i = 1:length(Class_1)
   XTest{i+c1,1} = my_hjorth(s(Class_1(i):Class_1(i)+313,1:22),16)';
   YTest(i+c1,1) = 1;
end

for i = 1:length(Class_2)
   XTest{i+length(Class_1)+c1,1} = my_hjorth(s(Class_2(i):Class_2(i)+313,1:22),16)';
   YTest(i+length(Class_1)+c1,1) = 2;
end

c1 = c1 + length(Class_1) + length(Class_2);
end
YTest = categorical(YTest);


%% 

FILENAME = 'D:\바탕화면\Verilog RNN\my_git_folder\2a\net.mat';
load(FILENAME);



YPred = classify(net,XTest, ...
    'SequenceLength','longest');

acc = sum(YPred == YTest)./numel(YTest);
disp(sprintf('Score: %f  ',acc));
    
%% 
function n_signal = my_normalization(s)
    Mean = mean(s);
    Std = std(s);
    
    n_signal = (s - Mean)./Std;
end

function [O] = my_hjorth(s,m)
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
    
    O = [A M C];
    
end
