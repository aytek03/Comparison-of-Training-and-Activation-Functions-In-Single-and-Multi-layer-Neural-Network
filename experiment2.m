function [MSE,alloutputs] = experiment2(neuron1,neuron2,train_algorithm,activation_func1,activation_func2)

% AYTEK�N YILDIZHAN
% N18147923
% CMP684 Neural Networks Project 1b

%----------------------------------------------------------------
% This script is a multi-layer feed-forward network, 
% has two hidden layers, and takes 5 parameters, which is 

% a number of neuron of first hidden layer,
% a number of neuron of second hidden layer,
% training algorithm,
% activation function of first hidden layer and
% activation function of second hidden layer.

% It returns final MSE value and all MSE outputs.
% You can see also training, validation and test values
% and the plots are at the bottom of the code. 

%----------------------------------------------------------------


%trainlm was the best training fuction so it was used in this experiment

% #NEURON = 2 x 2


%[MSE,alloutputs] = experiment2(2,2,'trainlm','tansig','tansig');
%[MSE,alloutputs] = experiment2(2,2,'trainlm','logsig','logsig');
%[MSE,alloutputs] = experiment2(2,2,'trainlm','purelin','purelin');
%[MSE,alloutputs] = experiment2(2,2,'trainlm','hardlim','hardlim');

%--------------------------------------------------------------------

% #NEURON = 4 x 4

%[MSE,alloutputs] = experiment2(4,4,'trainlm','tansig','tansig');
%[MSE,alloutputs] = experiment2(4,4,'trainlm','logsig','logsig');
%[MSE,alloutputs] = experiment2(4,4,'trainlm','purelin','purelin');
%[MSE,alloutputs] = experiment2(4,4,'trainlm','hardlim','hardlim');

%--------------------------------------------------------------------

% #NEURON = 8 x 8

%[MSE,alloutputs] = experiment2(8,8,'trainlm','tansig','tansig');
%[MSE,alloutputs] = experiment2(8,8,'trainlm','logsig','logsig');
%[MSE,alloutputs] = experiment2(8,8,'trainlm','purelin','purelin');
%[MSE,alloutputs] = experiment2(8,8,'trainlm','hardlim','hardlim');

%--------------------------------------------------------------------

% #NEURON = 16 x 16

%[MSE,alloutputs] = experiment2(16,16,'trainlm','tansig','tansig');
%[MSE,alloutputs] = experiment2(16,16,'trainlm','logsig','logsig');
%[MSE,alloutputs] = experiment2(16,16,'trainlm','purelin','purelin');
%[MSE,alloutputs] = experiment2(16,16,'trainlm','hardlim','hardlim');

%--------------------------------------------------------------------

% #NEURON = 32 x 32

%[MSE,alloutputs] = experiment2(32,32,'trainlm','tansig','tansig');
%[MSE,alloutputs] = experiment2(32,32,'trainlm','logsig','logsig');
%[MSE,alloutputs] = experiment2(32,32,'trainlm','purelin','purelin');
%[MSE,alloutputs] = experiment2(32,32,'trainlm','hardlim','hardlim');

%--------------------------------------------------------------------

% #NEURON = 64 x 64

%[MSE,alloutputs] = experiment2(64,64,'trainlm','tansig','tansig');
%[MSE,alloutputs] = experiment2(64,64,'trainlm','logsig','logsig');
%[MSE,alloutputs] = experiment2(64,64,'trainlm','purelin','purelin');
%[MSE,alloutputs] = experiment2(64,64,'trainlm','hardlim','hardlim');

%--------------------------------------------------------------------

% #NEURON = 128 x 128

%[MSE,alloutputs] = experiment2(128,128,'trainlm','tansig','tansig');
%[MSE,alloutputs] = experiment2(128,128,'trainlm','logsig','logsig');
%[MSE,alloutputs] = experiment2(128,128,'trainlm','purelin','purelin');
%[MSE,alloutputs] = experiment2(128,128,'trainlm','hardlim','hardlim');

%--------------------------------------------------------------------


[input_values,target_values] = house_dataset; %from help nndatasets

net = feedforwardnet([neuron1,neuron2]); % two hidden layer, 2*2 4*4,8*8,16*16,32*32,64*64,128*128
%You can use fitnet([neuron1,neuron2],trainFcn) instead of feedforwardnet

net.trainFcn = train_algorithm; 

%You can use fitnet([neuron],trainFcn) instead of feedforwardnet


% trainlm, Levenberg-Marquardt back-propagation 
% trainscg, Scaled conjugate gradient back-propagation
% traingdm, Gradient descent with momentum
% traingda, Gradient descent with adaptive learning rate backpropagation
% traingdx, Gradient descent with momentum and adaptive learning rate backpropagation
% traingd, Gradient descent backpropagation

net.performFcn = 'mse'; %Mean Squared Error

net.layers{1}.transferFcn = activation_func1;
net.layers{2}.transferFcn = activation_func2;
%tansig logsig hardlim purelin

net.layers{3}.transferFcn = 'purelin'; %output neuron is linear

net.divideFcn= 'divideblock'; % divide the data manually
net.divideParam.trainRatio = 0.60; % training data
net.divideParam.valRatio= 0.20; % validation data 
net.divideParam.testRatio=  0.20;  % testing data


net.trainParam.epochs = 5000; %default epoc number
net.trainParam.show=1; %show after every 1 epoch iteration
net.trainParam.max_fail = 10; %Maximum validation failures

%train the network
[net,tr] = train(net,input_values,target_values);

%plot of functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};

%test the network and network response after training
final_output = net(input_values); 
e = gsubtract(target_values,final_output);
MSE = perform(net, target_values, final_output);

%simulating the outputs 
alloutputs = sim(net,input_values);

% Training, Validation and Test Performance
% trainTargets = target_values .* tr.trainMask{1};
% valTargets = target_values  .* tr.valMask{1};
% testTargets = target_values  .* tr.testMask{1};
% 
% MSE OUTPUTS of Training, Validation and Test Performance
% trainP = perform(net,trainTargets,final_output);
% valP = perform(net,valTargets,final_output);
% testP = perform(net,testTargets,final_output);

%Plots

% figure, plotperform(tr);
% figure, plottrainstate(tr);
% figure, plotfit(net, input_values, target_values);
% figure, plotregression(target_values, final_output);
% figure, ploterrhist(e);

end
%---------------------------------