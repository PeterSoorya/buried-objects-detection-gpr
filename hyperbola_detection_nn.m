clc
clear all
close all

cifar10Data = pwd;

%% Load Training and Testing data
[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);
% convert to grayscale
trainingImages = trainingImages(:,:,1,:) * 0.299 + trainingImages(:,:,2,:) * 0.587 + trainingImages(:,:,3,:) * 0.114;
testImages = testImages(:,:,1,:) * 0.299 + testImages(:,:,2,:) * 0.587 + testImages(:,:,3,:) * 0.114;

figure
thumbnails = trainingImages(:,:,:,1:100);
montage(thumbnails)

%% Create a CNN

% Create the image input layer for 32x32x1 CIFAR-10 images.
[height,width,numChannels, ~] = size(trainingImages);
numImageCategories = 10;
imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize)

% Convolutional layer parameters
filterSize = [5 5];
numFilters_1 = 16;
numFilters_2 = 32;
numFilters_3 = 64;

middleLayers = [

convolution2dLayer(filterSize,numFilters_1,'Padding',2)
reluLayer()
maxPooling2dLayer(3,'Stride',2)

convolution2dLayer(filterSize,numFilters_2,'Padding',2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)

convolution2dLayer(filterSize,numFilters_3,'Padding',2)
reluLayer()
maxPooling2dLayer(3,'Stride',2)

]


finalLayers = [
    
fullyConnectedLayer(64)
reluLayer
fullyConnectedLayer(numImageCategories)
softmaxLayer
classificationLayer
]


layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters_1]);

% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);

doTraining = true;

if doTraining    
    % Train a network.
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
else
    % Load pre-trained detector for the example if available.       
end

% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;

% rescale the weights to the range [0, 1] for better visualization
w = rescale(w);

figure
montage(w)

% Run the network on the test set.
YTest = classify(cifar10Net, testImages);

% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels)

%% Transfer Learning
%% Load the ground truth data

data = load('gtruth_3.mat'); % load the labelled data(radargram images with bounding box coordinates for the hyperbolas)

I = imread(data.gTruth.imageFilename{1});
I = insertObjectAnnotation(I,'Rectangle',data.gTruth.hyperbola{1},'Hyperbola','LineWidth',8);

figure
imshow(I)

  
    % Set training options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 1e-3, ...
        'MiniBatchSize', 128, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 20, ...
        'Verbose', true);
    
    % Train an R-CNN object detector.  
    rcnn = trainRCNNObjectDetector(data.gTruth, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])

% Read test image
testImage = imread('hypb_1.jpg');

% Detect stop signs
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128)

% Display the detection results
[score, idx] = max(score);

bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);

figure
imshow(outputImage)