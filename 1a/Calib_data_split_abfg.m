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

%%%%%%%%%%%%%%%%%%%%%%%%%%%
step = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%

Fs = 100;

X=[];
% Training only for training data set
it = 1;
itt = 1;

chunk = 100-1;
fs = 100;

% data_labels = ['a' 'b' 'f' 'g'];
data_labels = ['b'];

for data_label = data_labels

    FILENAME = strcat('D:\바탕화면\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_calib_ds1',data_label,'.mat');
    load(FILENAME);
%     clear cnt nfo 
%     FILENAME = strcat('D:\바탕화면\Verilog RNN\my_git_folder\1a\eeglab_ICA_data_',data_label,'.mat');
%     load(FILENAME);
    
    % Data rescale
%     cnt = ALLEEG(4).data;

    cnt= 0.1*double(cnt);
    cnt = cnt';
   
%     cnt_c = cnt([27 29 31 44-1 46-1 50-1 52-1 54-1],:); 
    cnt_c = cnt([27 29 31 44 46 50 52 54],:);

    clear cnt  
        
    
    for i = 1:length(mrk.pos)

        for k = 0:step:300       
           
            E = cnt_c(:,mrk.pos(1,i)+(k):mrk.pos(1,i)+chunk+k);
            X(:,:,it) = E;

            % According to its class, divide calculated covariance
            if mrk.y(1,i) == 1
                Y(it) = 1;
            else
                Y(it) = -1;
            end
            it = it + 1;
        end
        for k = 0:step:300
             E = cnt_c(:,mrk.pos(1,i)+400+k:mrk.pos(1,i)+400+chunk+k);
         
             X(:,:,it) = E;
             Y(it) = 0;
             it = it + 1;
        end
    end

    clear mrk
    clear cnt_c
end

%% 

save("D:\바탕화면\Verilog RNN\my_git_folder\1a\BCIIV_1a_rawdata.mat",'X','Y','-v7.3');
% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
