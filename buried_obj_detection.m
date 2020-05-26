clc
clear all
close all

%% read the radargram data

[rh d] = readgssi('400MHz-Limestone_2_rev.dzt');

%% View radargram and save the image for further processing

figure, imagesc(d), colormap gray
saveas(gcf,'hypb_1.jpg')

%% clutter supression

a           = mean(d,2);
d_clutter_sup       = d-a;

%% filter out high frequency system noise 

fpass = 800e6;
fs = 6.5e9;
y = lowpass(d_clutter_sup,fpass,fs);

%% data after pre processing

figure, imagesc(y), colormap gray

figure;
subplot(121)
imagesc(d), colormap gray
title('Before Pre-processing')
xlabel('Traces')
ylabel('Samples')
subplot(122)
imagesc(y), colormap gray
title('After Pre-processing')
xlabel('Traces')
ylabel('Samples')

%% Plot one of the the data segment
figure; plot(d(:,1))
title('Scan Segment')
xlabel('Samples')
ylabel('Amplitude')

%% calcuate the welch PSD

I = d(:,610);
segmentLength = 16;
noverlap = 8;
nfft = 64
pxx = pwelch(d,segmentLength,noverlap,nfft);
figure;
imagesc(pxx)
title('welch PSD')
xlabel('Scan Axis')
ylabel('Spectral Values')

%take only first 12 coeff of PSD
pxx1        = pxx(1:12,:);

%% neural network classification
%include the code for nn for image region detection here

%% Get the required regions in the image

% for demo purpose the image regions are taken manually here. Take the regions with hyperbola
I1 = y(140:225,1020:1180);
figure;
subplot(2,2,1)
imshow(I1,[])
I2 = y(200:325,900:1100);
subplot(2,2,2)
imshow(I2,[])
I3 = y(260:385,700:1000);
subplot(2,2,3)
imshow(I3,[])
I4 = y(400:500,500:800);
subplot(2,2,4)
imshow(I4,[])

%% edge detection

figure;
subplot(241)
imshow(I1,[]);
title('Image Region with Hyperbola')
% I = rgb2gray(I);
E1 = edge(I1, 'Canny',0.5);
subplot(242)
imshow(E1,[]);
title('After Edge Detection')
subplot(243)
imshow(I2,[]);
title('Image Region with Hyperbola')
% I = rgb2gray(I);
E2 = edge(I2, 'Canny',0.5);
subplot(244)
imshow(E2,[]);
title('After Edge Detection')
subplot(245)
imshow(I3,[]);
title('Image Region with Hyperbola')
% I = rgb2gray(I);
E3 = edge(I3, 'Canny',0.5);
subplot(246)
imshow(E3,[]);
title('After Edge Detection')
subplot(247)
imshow(I4,[]);
title('Image Region with Hyperbola')
% I = rgb2gray(I);
E4 = edge(I4, 'Canny',0.5);
subplot(248)
imshow(E4,[]);
title('After Edge Detection')

%% pattern recognition - To find the peak of each hyperbola

% choose parabola sizes to try
C = 0.01:0.001:0.02;
c_length = numel(C);
[M,N] = size(I1);
% Define accumulator array H(N,M,C) and initialize with zeros
H = zeros(M,N,c_length);
% vote to fill H
[y_edge, x_edge] = find(E1); % get edge points
for i = 1:length(x_edge) % for all edge points
    for c_idx=1:c_length % for all c
        for a = 1:N
            b = round(y_edge(i)-C(c_idx)*(x_edge(i)-a)^2);
            if(b < M && b >= 1) H(b,a,c_idx)=H(b,a,c_idx)+1; end
        end
    end
end

%find the peaks 

[r c p] = size(H);

for i = 1:p
PEAKS_{i} = houghpeaks(H(:,:,i));
end

% Cluster the peaks to get only the required peaks
A = [];
B = [];

for i = 1:p
A  = [A;PEAKS_{i}(1,1)];
B  = [B;PEAKS_{i}(1,2)];
end

% Create a table with the data and variable names
T = table(A, B, 'VariableNames', { 'X', 'Y'} )
% Write data to text file
writetable(T, 'MyFile.txt')

inputfile = 'MyFile.txt';
maxdist = 30
minClusterSize = 1
method = 'point'
mergeflag = 'merge'

[clustersCentroids,clustersGeoMedians,clustersXY] = clusterXYpoints(inputfile,maxdist,minClusterSize,method,mergeflag);

figure;
imshow(I1,[]);
title('Apex of hyperbola detected using Hough Transform')
axis on
hold on;
plot(clustersCentroids(1,2),clustersCentroids(1,1), 'r+', 'MarkerSize', 30, 'LineWidth', 2);
hold on
plot(clustersCentroids(2,2),clustersCentroids(2,1), 'r+', 'MarkerSize', 30, 'LineWidth', 2);





