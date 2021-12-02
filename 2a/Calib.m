% ----------------------------------------------------------------------- %
%    File_name: Calib.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_03_30                           
%                                                            
 % ----------------------------------------------------------------------- %
clear all
clc

% data_labels = ['1','2', '3', '4', '5', '6','7', '8', '9'];
data_labels = ['8'];

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
    
    %% Augmentation
    
    for j = length(Class_1):-1:1
       Class_1 = [Class_1 Class_1(j)-5 Class_1(j)-4 Class_1(j)-3 Class_1(j)-2 Class_1(j)-1 Class_1(j)+1 Class_1(j)+2 Class_1(j)+3 Class_1(j)+4 Class_1(j)+5]; 
    end
    
    for j = length(Class_2):-1:1
       Class_2 = [Class_2 Class_2(j)-5 Class_2(j)-4 Class_2(j)-3 Class_2(j)-2 Class_2(j)-1 Class_2(j)+1 Class_2(j)+2 Class_2(j)+3 Class_2(j)+4 Class_2(j)+5]; 
    end
    
    for j = length(Class_3):-1:1
       Class_3 = [Class_3 Class_3(j)-5 Class_3(j)-4 Class_3(j)-3 Class_3(j)-2 Class_3(j)-1 Class_3(j)+1 Class_3(j)+2 Class_3(j)+3 Class_3(j)+4 Class_3(j)+5]; 
    end
    
    for j = length(Class_4):-1:1
       Class_4 = [Class_4 Class_4(j)-5 Class_4(j)-4 Class_4(j)-3 Class_4(j)-2 Class_4(j)-1 Class_4(j)+1 Class_4(j)+2 Class_4(j)+3 Class_4(j)+4 Class_4(j)+5]; 
    end
    
    
    
    %% 

    for i = 1:length(Class_1)
       X1(:,:,i+c1) = my_normalization(s(Class_1(i):Class_1(i)+313,1:22));
       Y1(i+c1,1) = 1;
    end

    for i = 1:length(Class_2)
       X2(:,:,i+c2) = my_normalization(s(Class_2(i):Class_2(i)+313,1:22));
       Y2(i+c2,1) = 2;
    end

    for i = 1:length(Class_3)
       X3(:,:,i+c3) = my_normalization(s(Class_3(i):Class_3(i)+313,1:22));
       Y3(i+c3,1) = 3;
    end

    for i = 1:length(Class_4)
       X4(:,:,i+c4) = my_normalization(s(Class_4(i):Class_4(i)+313,1:22));
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
    assert(isempty(find(isnan(s))))
    Mean = mean(s);
    Std = std(s);
    
    n_signal = (s - Mean)./Std;
end
