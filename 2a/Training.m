clc
close all
clear all

FILENAME = 'D:\바탕화면\Verilog RNN\my_git_folder\2a\Calib_data_a.mat';
load(FILENAME);

%% Using X1, X2 (Left hand vs Right hand)
tmp_X = cat(3,X1,X2);
tmp_Y = [Y1;Y2];

idx = randperm(size(tmp_X,3));
train_it = 1;
dev_it = 1;

for k = 1: size(tmp_X,3)
    if k < size(tmp_X,3)*0.9
        XTrain{train_it,1} = tmp_X(:,:,idx(k))';
        YTrain(train_it,1) = tmp_Y(idx(k));
        train_it = train_it + 1;
    else 
        XValidation{dev_it,1} = tmp_X(:,:,idx(k))';
        YValidation(dev_it,1) = tmp_Y(idx(k));
        dev_it = dev_it + 1;
 
    end
end
YTrain = categorical(YTrain);
YValidation = categorical(YValidation);

%% 
layers = [
    sequenceInputLayer(22,"Name","sequence")
    lstmLayer(314,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(240,"Name","fc_2")
    reluLayer("Name","relu_1")
    dropoutLayer(0.5,"Name","dropout_1")
    fullyConnectedLayer(50,"Name","fc_3")
    reluLayer("Name","relu_2")
    dropoutLayer(0.5,"Name","dropout_2")
    fullyConnectedLayer(2,"Name","fc_1")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

layers(2).CellState = ones(314,1);
layers(2).HiddenState= ones(314,1);

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10, ...
    'MiniBatchSize',64, ...
    'MaxEpochs',60, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Shuffle','every-epoch', ...   % "once', 'never', 'every-epoch'
    'Verbose',1, ...
    'Plots','training-progress')

net = trainNetwork(XTrain,YTrain,layers,options);
%% 
save("D:\바탕화면\Verilog RNN\my_git_folder\2a\net.mat",'net');
