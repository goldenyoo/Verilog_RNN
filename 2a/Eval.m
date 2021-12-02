% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_03_30                           
%                                                            
 % ----------------------------------------------------------------------- %
clear all
clc

% data_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']; 
data_labels = ['9']; 

c1 = 0;
c2 = 0;

for data_label = data_labels

FILENAME = strcat('D:\바탕화면\BCIIV_2a_mat\A0',data_label,'E_mat');
load(FILENAME);

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



YPred = classify(net,XTest, ...
    'SequenceLength','longest');

acc = sum(YPred == YTest)./numel(YTest);
disp(sprintf('Score: %f  ',acc));
    
%% 
function n_signal = my_normalization(s)
    assert(isempty(find(isnan(s))));
    
    Mean = mean(s);
    Std = std(s);
    
    n_signal = (s - Mean)./Std;
end
