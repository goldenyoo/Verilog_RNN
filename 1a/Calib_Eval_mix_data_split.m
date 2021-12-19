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

data_labels = ['a' 'b' 'f' 'g'];
% data_labels = ['g'];

for data_label = data_labels
    
    FILENAME = strcat('D:\바탕화면\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_calib_ds1',data_label,'.mat');
    load(FILENAME);
    
    % Data rescale
    cnt= double(cnt);
    cnt = cnt';
    
    cnt_c = cnt([27 29 31 44 46 50 52 54],:);
    
    clear cnt
    
    for i = 1:length(mrk.pos)
        %         fprintf("pos: %d\n",i);
        for k = 0:10:300
            
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
        %         for k = 0:300
        %              E = cnt_c(:,mrk.pos(1,i)+400+k:mrk.pos(1,i)+400+chunk+k);
        %
        %              X(:,:,it) = E;
        %              Y(it) = 0;
        %              it = it + 1;
        %         end
    end
    
    clear mrk
    clear cnt_c
    
    % Eval data
    FILENAME = strcat('D:\바탕화면\Motor Imagery EEG data\BCICIV_1_mat\BCICIV_eval_ds1',data_label,'.mat');
    load(FILENAME);
    
    cnt= double(cnt);
    cnt = cnt';
    
    cnt_c = cnt([27 29 31 44 46 50 52 54],:);
    
    clear cnt
    
    FILENAME = strcat('D:\바탕화면\Motor Imagery EEG data\true_labels\BCICIV_eval_ds1',data_label,'_1000Hz_true_y.mat');
    load(FILENAME);
    
    true_y = downsample(true_y,10);
    
    j = 1;
    while j < size(cnt_c,2) - 99
        if (true_y(j,1) == 1)&(true_y(j+99,1) == 1)
            E = cnt_c(:,j:j+99);
            
            X(:,:,it) = E;
            Y(it) = 1;
            
            it = it + 1;
        elseif (true_y(j,1) == -1)&(true_y(j+99,1) == -1)
            E = cnt_c(:,j:j+99);
            
            X(:,:,it) = E;
            Y(it) = -1;
            
            it = it + 1;
        end
        j = j + 10;
    end
    
end

clear cnt_c E nfo true_y step chunk fs Fs i it itt h k
%%

idx = randperm(size(X,3));
it_1 = 1;
it_2 = 1;
it_3 = 1;
it_4 = 1;
it_5 = 1;
it_6 = 1;
it_7 = 1;
it_8 = 1;
it_9 = 1;
it_10 = 1;

for k = 1: size(X,3)
    if k < size(X,3)*0.1
        X1{it_1,1} = X(:,:,idx(k));
        Y1(it_1,1) = Y(idx(k));
        it_1 = it_1 + 1;
    elseif k < size(X,3)*0.2
        X2{it_2,1} = X(:,:,idx(k));
        Y2(it_2,1) = Y(idx(k));
        it_2 = it_2 + 1;
    elseif k < size(X,3)*0.3
        X3{it_3,1} = X(:,:,idx(k));
        Y3(it_3,1) = Y(idx(k));
        it_3 = it_3 + 1;
    elseif k < size(X,3)*0.4
        X4{it_4,1} = X(:,:,idx(k));
        Y4(it_4,1) = Y(idx(k));
        it_4 = it_4 + 1;
    elseif k < size(X,3)*0.5
        X5{it_5,1} = X(:,:,idx(k));
        Y5(it_5,1) = Y(idx(k));
        it_5 = it_5 + 1;
    elseif k < size(X,3)*0.6
        X6{it_6,1} = X(:,:,idx(k));
        Y6(it_6,1) = Y(idx(k));
        it_6 = it_6 + 1;
    elseif k < size(X,3)*0.7
        X7{it_7,1} = X(:,:,idx(k));
        Y7(it_7,1) = Y(idx(k));
        it_7 = it_7 + 1;
    elseif k < size(X,3)*0.8
        X8{it_8,1} = X(:,:,idx(k));
        Y8(it_8,1) = Y(idx(k));
        it_8 = it_8 + 1;
    elseif k < size(X,3)*0.9
        X9{it_9,1} = X(:,:,idx(k));
        Y9(it_9,1) = Y(idx(k));
        it_9 = it_9 + 1;
    else
        X10{it_10,1} = X(:,:,idx(k));
        Y10(it_10,1) = Y(idx(k));
        it_10 = it_10 + 1;
    end
end
Y1 = categorical(Y1);
Y2 = categorical(Y2);
Y3 = categorical(Y3);
Y4 = categorical(Y4);
Y5 = categorical(Y5);
Y6 = categorical(Y6);
Y7 = categorical(Y7);
Y8 = categorical(Y8);
Y9 = categorical(Y9);
Y10 = categorical(Y10);

XTrain{1,1} = [X2;X3;X4;X5;X6;X7;X8;X9;X10];
XTrain{2,1} = [X1;X3;X4;X5;X6;X7;X8;X9;X10];
XTrain{3,1} = [X1;X2;X4;X5;X6;X7;X8;X9;X10];
XTrain{4,1} = [X1;X2;X3;X5;X6;X7;X8;X9;X10];
XTrain{5,1} = [X1;X2;X3;X4;X6;X7;X8;X9;X10];
XTrain{6,1} = [X1;X2;X3;X4;X5;X7;X8;X9;X10];
XTrain{7,1} = [X1;X2;X3;X4;X5;X6;X8;X9;X10];
XTrain{8,1} = [X1;X2;X3;X4;X5;X6;X7;X9;X10];
XTrain{9,1} = [X1;X2;X3;X4;X5;X6;X7;X8;X10];
XTrain{10,1} = [X1;X2;X3;X4;X5;X6;X7;X8;X9];

YTrain{1,1} = [Y2;Y3;Y4;Y5;Y6;Y7;Y8;Y9;Y10];
YTrain{2,1} = [Y1;Y3;Y4;Y5;Y6;Y7;Y8;Y9;Y10];
YTrain{3,1} = [Y1;Y2;Y4;Y5;Y6;Y7;Y8;Y9;Y10];
YTrain{4,1} = [Y1;Y2;Y3;Y5;Y6;Y7;Y8;Y9;Y10];
YTrain{5,1} = [Y1;Y2;Y3;Y4;Y6;Y7;Y8;Y9;Y10];
YTrain{6,1} = [Y1;Y2;Y3;Y4;Y5;Y7;Y8;Y9;Y10];
YTrain{7,1} = [Y1;Y2;Y3;Y4;Y5;Y6;Y8;Y9;Y10];
YTrain{8,1} = [Y1;Y2;Y3;Y4;Y5;Y6;Y7;Y9;Y10];
YTrain{9,1} = [Y1;Y2;Y3;Y4;Y5;Y6;Y7;Y8;Y10];
YTrain{10,1} = [Y1;Y2;Y3;Y4;Y5;Y6;Y7;Y8;Y9];

XTest{1,1} = X1;
XTest{2,1} = X2;
XTest{3,1} = X3;
XTest{4,1} = X4;
XTest{5,1} = X5;
XTest{6,1} = X6;
XTest{7,1} = X7;
XTest{8,1} = X8;
XTest{9,1} = X9;
XTest{10,1} = X10;

YTest{1,1} = Y1;
YTest{2,1} = Y2;
YTest{3,1} = Y3;
YTest{4,1} = Y4;
YTest{5,1} = Y5;
YTest{6,1} = Y6;
YTest{7,1} = Y7;
YTest{8,1} = Y8;
YTest{9,1} = Y9;
YTest{10,1} = Y10;

%% 

layers = [
    sequenceInputLayer(8,"Name","sequence")
    lstmLayer(100,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule','piecewise', ...
    'MaxEpochs',30, ...
    'Shuffle','never', ...   % "once', 'never', 'every-epoch'
    'Verbose',1, ...
    'Plots','training-progress')

for h = 1:10
   
    net = trainNetwork(XTrain{h,1},YTrain{h,1},layers,options);
    YPred = classify(net,XTest{h,1},'SequenceLength','longest');
    
    acc(h,1) = sum(YPred == YTest{h,1})./numel(YTest{h,1});
    
    FILENAME = strcat('D:\바탕화면\Verilog RNN\my_git_folder\1a\Acc_',int2str(h),'.mat');
    save(FILENAME,'acc');
end


% ----------------------------------------------------------------------- %
%                               EOF
% ----------------------------------------------------------------------- %
