% ----------------------------------------------------------------------- %
%    File_name: trying_rnn_idle_v2.m
%    Programmer: Seungjae Yoo                             
%                                           
%    Last Modified: 2020_08_05                           
%                  calib은 train, validation set으로 
%                   eval은 test set으로 mimic real task
 % ----------------------------------------------------------------------- %

clc
close all
clear all

FILENAME = 'D:\바탕화면\Verilog RNN\new\11_30\eeglab\BCIIV_1a_rawdata.mat';
load(FILENAME);

%% Train, Validation

idx = randperm(size(X,3));
train_it = 1;
dev_it = 1;

for k = 1: size(X,3)
    if k < size(X,3)*0.9
        XTrain{train_it,1} = X(:,:,idx(k));
        YTrain(train_it,1) = Y(idx(k));
        train_it = train_it + 1;
    else 
        XValidation{dev_it,1} = X(:,:,idx(k));
        YValidation(dev_it,1) = Y(idx(k));
        dev_it = dev_it + 1;
 
    end
end
YTrain = categorical(YTrain);
YValidation = categorical(YValidation);

%% 

layers = [
    sequenceInputLayer(7,"Name","sequence")
    lstmLayer(100,"Name","lstm","OutputMode","last")
    fullyConnectedLayer(3,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];



options = trainingOptions('rmsprop', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule','piecewise', ...
    'MaxEpochs',30, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Shuffle','every-epoch', ...   % "once', 'never', 'every-epoch'
    'Verbose',1, ...
    'Plots','training-progress')

net = trainNetwork(XTrain,YTrain,layers,options);


 %% 
save("D:\바탕화면\Verilog RNN\new\11_30\eeglab\net.mat",'net');