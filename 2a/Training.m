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
    if k < size(tmp_X,3)*0.8
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
m = 30;

layers = [
    sequenceInputLayer(66,"Name","sequence")
    lstmLayer(m,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(2,"Name","fc_1")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

% layers(2).CellState = ones(m,1);
% layers(2).HiddenState= ones(m,1);

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...
    'MaxEpochs',150, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Shuffle','every-epoch', ...   % "once', 'never', 'every-epoch'
    'Verbose',1, ...
    'Plots','training-progress')

net = trainNetwork(XTrain,YTrain,layers,options);
%% 
save("D:\바탕화면\Verilog RNN\my_git_folder\2a\net.mat",'net');
