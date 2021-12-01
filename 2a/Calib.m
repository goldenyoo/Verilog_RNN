% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_03_30                           
%                                                            
 % ----------------------------------------------------------------------- %
clear all
clc

data_labels = ['1','2', '3', '4', '5', '6','7', '8', '9'];

c1 = 0;
c2 = 0;
c3 = 0;
c4 = 0;


for data_label = data_labels
    
    FILENAME = strcat('D:\바탕화면\BCIIV_2a_mat\A0',data_label,'T_2_mat');
    load(FILENAME);


    %% Calculate covariance for all Classes
    Class_1 = [];
    Class_2 = [];
    Class_3 = [];
    Class_4 = [];
    reject = [];

    i=1;
    while i <=length(h.EVENT.TYP)
        if h.EVENT.TYP(i)== 1023
            reject = [reject h.EVENT.POS(i)];
            i = i + 1;
        elseif h.EVENT.TYP(i)== 769
            Class_1 = [Class_1 h.EVENT.POS(i)];
            i = i + 1;
        elseif h.EVENT.TYP(i)== 770
            Class_2 = [Class_2 h.EVENT.POS(i)];
            i = i + 1;
        elseif h.EVENT.TYP(i)== 771
            Class_3 = [Class_3 h.EVENT.POS(i)];
            i = i + 1;
        elseif h.EVENT.TYP(i)== 772
            Class_4 = [Class_4 h.EVENT.POS(i)];
            i = i + 1;
        else
            i = i + 1;
        end    
    end

    for i = 1:length(reject)
        k = reject(1,i);
        cc1 = find(Class_1 == k+500);
        cc2 = find(Class_2 == k+500);
        cc3 = find(Class_3 == k+500);
        cc4 = find(Class_4 == k+500);


        Class_1(cc1) = []; 
        Class_2(cc2) = [];
        Class_3(cc3) = [];
        Class_4(cc4) = [];
    end
    %% 
    m = 30;
    for i = 1:length(Class_1)
       X1(:,:,i+c1) = my_hjorth(s(Class_1(i):Class_1(i)+313,1:22),m);
       Y1(i+c1,1) = 1;
    end

    for i = 1:length(Class_2)
       X2(:,:,i+c2) = my_hjorth(s(Class_2(i):Class_2(i)+313,1:22),m);
       Y2(i+c2,1) = 2;
    end

    for i = 1:length(Class_3)
       X3(:,:,i+c3) = my_hjorth(s(Class_3(i):Class_3(i)+313,1:22),m);
       Y3(i+c3,1) = 3;
    end

    for i = 1:length(Class_4)
       X4(:,:,i+c4) = my_hjorth(s(Class_4(i):Class_4(i)+313,1:22),m);
       Y4(i+c4,1) = 4;
    end
    c1 = c1 + length(Class_1);
    c2 = c2 + length(Class_2);
    c3 = c3 + length(Class_3);
    c4 = c4 + length(Class_4);
end
%% 
save("D:\바탕화면\Verilog RNN\my_git_folder\2a\Calib_data_a.mat",'X1','X2','X3','X4','Y1','Y2', 'Y3', 'Y4');

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