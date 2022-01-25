function [MSE,alloutputs] = experiment1(neuron,train_algorithms,activation_func)

% AYTEKÝN YILDIZHAN
% N18147923
% CMP684 Neural Networks Project 1a
%nnstart
%----------------------------------------------------------------
% This script is a single layer feed-forward network and
% takes 3 parameters, which is 
% a number of neuron, training algorihm and activation function.

% It returns final MSE value and all MSE outputs.
% You can see also training, validation and test values
% and the plots are at the bottom of the code. 

%----------------------------------------------------------------
% #NEURON = 2

%[MSE,alloutputs] = experiment1(2,'trainlm','tansig');
%[MSE,alloutputs] = experiment1(2,'trainscg','tansig');
%[MSE,alloutputs] = experiment1(2,'traingdm','tansig');
%[MSE,alloutputs] = experiment1(2,'traingda','tansig');
%[MSE,alloutputs] = experiment1(2,'traingdx','tansig');
%[MSE,alloutputs] = experiment1(2,'traingd','tansig');

%[MSE,alloutputs] = experiment1(2,'trainlm','logsig');
%[MSE,alloutputs] = experiment1(2,'trainscg','logsig');
%[MSE,alloutputs] = experiment1(2,'traingdm','logsig');
%[MSE,alloutputs] = experiment1(2,'traingda','logsig');
%[MSE,alloutputs] = experiment1(2,'traingdx','logsig');
%[MSE,alloutputs] = experiment1(2,'traingd','logsig');

%[MSE,alloutputs] = experiment1(2,'trainlm','purelin');
%[MSE,alloutputs] = experiment1(2,'trainscg','purelin');
%[MSE,alloutputs] = experiment1(2,'traingdm','purelin');
%[MSE,alloutputs] = experiment1(2,'traingda','purelin');
%[MSE,alloutputs] = experiment1(2,'traingdx','purelin');
%[MSE,alloutputs] = experiment1(2,'traingd','purelin');

%[MSE,alloutputs] = experiment1(2,'trainlm','hardlim');
%[MSE,alloutputs] = experiment1(2,'trainscg','hardlim');
%[MSE,alloutputs] = experiment1(2,'traingdm','hardlim');
%[MSE,alloutputs] = experiment1(2,'traingda','hardlim');
%[MSE,alloutputs] = experiment1(2,'traingdx','hardlim');
%[MSE,alloutputs] = experiment1(2,'traingd','hardlim');

%--------------------------------------------------------------------

% #NEURON = 4

%[MSE,alloutputs] = experiment1(4,'trainlm','tansig');
%[MSE,alloutputs] = experiment1(4,'trainscg','tansig');
%[MSE,alloutputs] = experiment1(4,'traingdm','tansig');
%[MSE,alloutputs] = experiment1(4,'traingda','tansig');
%[MSE,alloutputs] = experiment1(4,'traingdx','tansig');
%[MSE,alloutputs] = experiment1(4,'traingd','tansig');

%[MSE,alloutputs] = experiment1(4,'trainlm','logsig');
%[MSE,alloutputs] = experiment1(4,'trainscg','logsig');
%[MSE,alloutputs] = experiment1(4,'traingdm','logsig');
%[MSE,alloutputs] = experiment1(4,'traingda','logsig');
%[MSE,alloutputs] = experiment1(4,'traingdx','logsig');
%[MSE,alloutputs] = experiment1(4,'traingd','logsig');

%[MSE,alloutputs] = experiment1(4,'trainlm','purelin');
%[MSE,alloutputs] = experiment1(4,'trainscg','purelin');
%[MSE,alloutputs] = experiment1(4,'traingdm','purelin');
%[MSE,alloutputs] = experiment1(4,'traingda','purelin');
%[MSE,alloutputs] = experiment1(4,'traingdx','purelin');
%[MSE,alloutputs] = experiment1(4,'traingd','purelin');

%[MSE,alloutputs] = experiment1(4,'trainlm','hardlim');
%[MSE,alloutputs] = experiment1(4,'trainscg','hardlim');
%[MSE,alloutputs] = experiment1(4,'traingdm','hardlim');
%[MSE,alloutputs] = experiment1(4,'traingda','hardlim');
%[MSE,alloutputs] = experiment1(4,'traingdx','hardlim');
%[MSE,alloutputs] = experiment1(4,'traingd','hardlim');

%--------------------------------------------------------------------

% #NEURON = 8

%[MSE,alloutputs] = experiment1(8,'trainlm','tansig');
%[MSE,alloutputs] = experiment1(8,'trainscg','tansig');
%[MSE,alloutputs] = experiment1(8,'traingdm','tansig');
%[MSE,alloutputs] = experiment1(8,'traingda','tansig');
%[MSE,alloutputs] = experiment1(8,'traingdx','tansig');
%[MSE,alloutputs] = experiment1(8,'traingd','tansig');

%[MSE,alloutputs] = experiment1(8,'trainlm','logsig');
%[MSE,alloutputs] = experiment1(8,'trainscg','logsig');
%[MSE,alloutputs] = experiment1(8,'traingdm','logsig');
%[MSE,alloutputs] = experiment1(8,'traingda','logsig');
%[MSE,alloutputs] = experiment1(8,'traingdx','logsig');
%[MSE,alloutputs] = experiment1(8,'traingd','logsig');

%[MSE,alloutputs1] = experiment1(8,'trainlm','purelin');
%[MSE,alloutputs2] = experiment1(8,'trainscg','purelin');
%[MSE,alloutputs3] = experiment1(8,'traingdm','purelin');
%[MSE,alloutputs4] = experiment1(8,'traingda','purelin');
%[MSE,alloutputs5] = experiment1(8,'traingdx','purelin');
%[MSE,alloutputs6] = experiment1(8,'traingd','purelin');

%[MSE,alloutputs] = experiment1(8,'trainlm','hardlim');
%[MSE,alloutputs] = experiment1(8,'trainscg','hardlim');
%[MSE,alloutputs] = experiment1(8,'traingdm','hardlim');
%[MSE,alloutputs] = experiment1(8,'traingda','hardlim');
%[MSE,alloutputs] = experiment1(8,'traingdx','hardlim');
%[MSE,alloutputs] = experiment1(8,'traingd','hardlim');

%--------------------------------------------------------------------

% #NEURON = 16

%[MSE,alloutputs] = experiment1(16,'trainlm','tansig');
%[MSE,alloutputs] = experiment1(16,'trainscg','tansig');
%[MSE,alloutputs] = experiment1(16,'traingdm','tansig');
%[MSE,alloutputs] = experiment1(16,'traingda','tansig');
%[MSE,alloutputs] = experiment1(16,'traingdx','tansig');
%[MSE,alloutputs] = experiment1(16,'traingd','tansig');

%[MSE,alloutputs] = experiment1(16,'trainlm','logsig');
%[MSE,alloutputs] = experiment1(16,'trainscg','logsig');
%[MSE,alloutputs] = experiment1(16,'traingdm','logsig');
%[MSE,alloutputs] = experiment1(16,'traingda','logsig');
%[MSE,alloutputs] = experiment1(16,'traingdx','logsig');
%[MSE,alloutputs] = experiment1(16,'traingd','logsig');

%[MSE,alloutputs] = experiment1(16,'trainlm','purelin');
%[MSE,alloutputs] = experiment1(16,'trainscg','purelin');
%[MSE,alloutputs] = experiment1(16,'traingdm','purelin');
%[MSE,alloutputs] = experiment1(16,'traingda','purelin');
%[MSE,alloutputs] = experiment1(16,'traingdx','purelin');
%[MSE,alloutputs] = experiment1(16,'traingd','purelin');

%[MSE,alloutputs] = experiment1(16,'trainlm','hardlim');
%[MSE,alloutputs] = experiment1(16,'trainscg','hardlim');
%[MSE,alloutputs] = experiment1(16,'traingdm','hardlim');
%[MSE,alloutputs] = experiment1(16,'traingda','hardlim');
%[MSE,alloutputs] = experiment1(16,'traingdx','hardlim');
%[MSE,alloutputs] = experiment1(16,'traingd','hardlim');

%--------------------------------------------------------------------

% #NEURON = 32

%[MSE,alloutputs] = experiment1(32,'trainlm','tansig');
%[MSE,alloutputs] = experiment1(32,'trainscg','tansig');
%[MSE,alloutputs] = experiment1(32,'traingdm','tansig');
%[MSE,alloutputs] = experiment1(32,'traingda','tansig');
%[MSE,alloutputs] = experiment1(32,'traingdx','tansig');
%[MSE,alloutputs] = experiment1(32,'traingd','tansig');

%[MSE,alloutputs] = experiment1(32,'trainlm','logsig');
%[MSE,alloutputs] = experiment1(32,'trainscg','logsig');
%[MSE,alloutputs] = experiment1(32,'traingdm','logsig');
%[MSE,alloutputs] = experiment1(32,'traingda','logsig');
%[MSE,alloutputs] = experiment1(32,'traingdx','logsig');
%[MSE,alloutputs] = experiment1(32,'traingd','logsig');

%[MSE,alloutputs] = experiment1(32,'trainlm','purelin');
%[MSE,alloutputs] = experiment1(32,'trainscg','purelin');
%[MSE,alloutputs] = experiment1(32,'traingdm','purelin');
%[MSE,alloutputs] = experiment1(32,'traingda','purelin');
%[MSE,alloutputs] = experiment1(32,'traingdx','purelin');
%[MSE,alloutputs] = experiment1(32,'traingd','purelin');

%[MSE,alloutputs] = experiment1(32,'trainlm','hardlim');
%[MSE,alloutputs] = experiment1(32,'trainscg','hardlim');
%[MSE,alloutputs] = experiment1(32,'traingdm','hardlim');
%[MSE,alloutputs] = experiment1(32,'traingda','hardlim');
%[MSE,alloutputs] = experiment1(32,'traingdx','hardlim');
%[MSE,alloutputs] = experiment1(32,'traingd','hardlim');

%--------------------------------------------------------------------

% #NEURON = 64

%[MSE,alloutputs] = experiment1(64,'trainlm','tansig');
%[MSE,alloutputs] = experiment1(64,'trainscg','tansig');
%[MSE,alloutputs] = experiment1(64,'traingdm','tansig');
%[MSE,alloutputs] = experiment1(64,'traingda','tansig');
%[MSE,alloutputs] = experiment1(64,'traingdx','tansig');
%[MSE,alloutputs] = experiment1(64,'traingd','tansig');

%[MSE,alloutputs] = experiment1(64,'trainlm','logsig');
%[MSE,alloutputs] = experiment1(64,'trainscg','logsig');
%[MSE,alloutputs] = experiment1(64,'traingdm','logsig');
%[MSE,alloutputs] = experiment1(64,'traingda','logsig');
%[MSE,alloutputs] = experiment1(64,'traingdx','logsig');
%[MSE,alloutputs] = experiment1(64,'traingd','logsig');

%[MSE,alloutputs] = experiment1(64,'trainlm','purelin');
%[MSE,alloutputs] = experiment1(64,'trainscg','purelin');
%[MSE,alloutputs] = experiment1(64,'traingdm','purelin');
%[MSE,alloutputs] = experiment1(64,'traingda','purelin');
%[MSE,alloutputs] = experiment1(64,'traingdx','purelin');
%[MSE,alloutputs] = experiment1(64,'traingd','purelin');

%[MSE,alloutputs] = experiment1(64,'trainlm','hardlim');
%[MSE,alloutputs] = experiment1(64,'trainscg','hardlim');
%[MSE,alloutputs] = experiment1(64,'traingdm','hardlim');
%[MSE,alloutputs] = experiment1(64,'traingda','hardlim');
%[MSE,alloutputs] = experiment1(64,'traingdx','hardlim');
%[MSE,alloutputs] = experiment1(64,'traingd','hardlim');

%--------------------------------------------------------------------

% #NEURON = 128

%[MSE,alloutputs] = experiment1(128,'trainlm','tansig');
%[MSE,alloutputs] = experiment1(128,'trainscg','tansig');
%[MSE,alloutputs] = experiment1(128,'traingdm','tansig');
%[MSE,alloutputs] = experiment1(128,'traingda','tansig');
%[MSE,alloutputs] = experiment1(128,'traingdx','tansig');
%[MSE,alloutputs] = experiment1(128,'traingd','tansig');

%[MSE,alloutputs] = experiment1(128,'trainlm','logsig');
%[MSE,alloutputs] = experiment1(128,'trainscg','logsig');
%[MSE,alloutputs] = experiment1(128,'traingdm','logsig');
%[MSE,alloutputs] = experiment1(128,'traingda','logsig');
%[MSE,alloutputs] = experiment1(128,'traingdx','logsig');
%[MSE,alloutputs] = experiment1(128,'traingd','logsig');

%[MSE,alloutputs] = experiment1(128,'trainlm','purelin');
%[MSE,alloutputs] = experiment1(128,'trainscg','purelin');
%[MSE,alloutputs] = experiment1(128,'traingdm','purelin');
%[MSE,alloutputs] = experiment1(128,'traingda','purelin');
%[MSE,alloutputs] = experiment1(128,'traingdx','purelin');
%[MSE,alloutputs] = experiment1(128,'traingd','purelin');

%[MSE,alloutputs] = experiment1(128,'trainlm','hardlim');
%[MSE,alloutputs] = experiment1(128,'trainscg','hardlim');
%[MSE,alloutputs] = experiment1(128,'traingdm','hardlim');
%[MSE,alloutputs] = experiment1(128,'traingda','hardlim');
%[MSE,alloutputs] = experiment1(128,'traingdx','hardlim');
%[MSE,alloutputs] = experiment1(128,'traingd','hardlim');

%--------------------------------------------------------------------


[input_values,target_values] = house_dataset; %from help nndatasets

net = feedforwardnet(neuron); % one hidden layer, 2,4,8,16,32,64,128
%You can use fitnet([neuron],trainFcn) instead of feedforwardnet

net.trainFcn = train_algorithms; 

%You can use fitnet([neuron],trainFcn) instead of feedforwardnet

% trainlm, Levenberg-Marquardt back-propagation 
% trainscg, Scaled conjugate gradient back-propagation
% traingdm, Gradient descent with momentum
% traingda, Gradient descent with adaptive learning rate backpropagation
% traingdx, Gradient descent with momentum and adaptive learning rate backpropagation
% traingd, Gradient descent backpropagation

net.performFcn = 'mse'; %Mean Squared Error

net.layers{1}.transferFcn = activation_func;
%tansig logsig purelin hardlim

net.layers{2}.transferFcn = 'purelin'; %output neuron is linear

% net.divideParam.trainRatio = 0.60; % training set [%]
% net.divideParam.valRatio = 0.20; % validation set [%]
% net.divideParam.testRatio = 0.20; % test set [%]

net.divideFcn= 'divideblock'; % divide the data manually
net.divideParam.trainRatio = 0.60; % training data
net.divideParam.valRatio= 0.20; % validation data 
net.divideParam.testRatio=  0.20;  % testing data


net.trainParam.epochs = 5000; %default epoc number
net.trainParam.show=1; %show after every iteration
net.trainParam.max_fail = 10; %Maximum validation failures

%train the network
[net,tr] = train(net,input_values,target_values);

%plot of functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};

%test the network and network response after training
final_output = net(input_values); 
%e = gsubtract(target_values,final_output);
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