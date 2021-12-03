clc
close all
clear all

FILENAME = 'D:\바탕화면\Verilog RNN\my_git_folder\2a\Calib_data_a.mat';
load(FILENAME);

%% Using X1, X2 (Left hand vs Right hand)
tmp_X_1 = cat(3,X1_1,X2_1);
tmp_X_2 = cat(3,X1_2,X2_2);


tmp_Y = [Y1;Y2];

idx = randperm(size(tmp_X_1,3));
train_it = 1;
dev_it = 1;

for k = 1: size(tmp_X_1,3)
    if k < size(tmp_X_1,3)*0.8
        XTrain_1{train_it,1} = tmp_X_1(:,:,idx(k));
        XTrain_2{train_it,1} = tmp_X_2(:,:,idx(k));

        YTrain(train_it,1) = tmp_Y(idx(k));
        train_it = train_it + 1;
    else 
        XValidation_1{dev_it,1} = tmp_X_1(:,:,idx(k));
        XValidation_2{dev_it,1} = tmp_X_2(:,:,idx(k));

        YValidation(dev_it,1) = tmp_Y(idx(k));
        dev_it = dev_it + 1;
 
    end
end
YTrain = categorical(YTrain);
YValidation = categorical(YValidation);

%% 
m = 16;
layers = [
    sequenceInputLayer(10,"Name","sequence")
    lstmLayer(16,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

% layers(2).CellState = ones(m,1);
% layers(2).HiddenState= ones(m,1);


options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'MiniBatchSize',64, ...
    'MaxEpochs',120, ...
    'ValidationData',{XValidation_1,YValidation}, ...
    'Shuffle','every-epoch', ...   % "once', 'never', 'every-epoch'
    'Verbose',1, ...
    'Plots','training-progress')

net_1 = trainNetwork(XTrain_1,YTrain,layers,options);

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'MiniBatchSize',64, ...
    'MaxEpochs',120, ...
    'ValidationData',{XValidation_2,YValidation}, ...
    'Shuffle','every-epoch', ...   % "once', 'never', 'every-epoch'
    'Verbose',1, ...
    'Plots','training-progress')

net_2 = trainNetwork(XTrain_2,YTrain,layers,options);


%% 
save("D:\바탕화면\Verilog RNN\my_git_folder\2a\net.mat",'net_1','net_2');