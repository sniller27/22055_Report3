%% Read images
clear; clc; close all; % Clear workspace and figures
% setup
folder_path = 'Melanoma\';
% henter alle billeder
images = dir(fullfile(folder_path, '*.jpg'));
% numel = er antal af billeder
images_count = numel(images);

x_seg = cell(1,images_count);

for i=1:images_count
    img = imread(fullfile(images(i).folder, images(i).name)); % read image
    img_grayscale = rgb2gray(img);
    x_seg(i) = {img_grayscale};
end

%% show all images as grayscales
rows = 5;
columns = 5;

figure;
sgtitle('Melanoma');
for i=1:images_count
    subplot(columns,rows,i);
    imshow(cell2mat(x_seg(i)))
    title(i);
end

%% Image histograms (imhist seems to lag have an annoying scale bar)
figure;
subplot(2,1,1);
hold on;
title('Image histograms');
for i=1:images_count
gray_image = cell2mat(x_seg(i));
[H,W] = size(gray_image);
M = H*W;
%%tt = double(reshape(cell2mat(x_seg(i)), M, 1));
histogram(double(reshape(gray_image, M, 1)));
end
hold off;

subplot(2,1,2);
hold on;
title('Image histograms equalized');
for i=1:images_count
gray_image = histeq(cell2mat(x_seg(i)));
[H,W] = size(gray_image);
M = H*W;
%%tt = double(reshape(cell2mat(x_seg(i)), M, 1));
histogram(double(reshape(gray_image, M, 1)));
end
hold off;

%% Thresholding
rows = 5;
columns = 5;

T1 = 0; % Lower limit (cancer melanoma)
T2 = 90; % Upper limit (cancer melanoma)

figure;
sgtitle('Images thresholded');
hold on;
for i=1:images_count
    
    gray_image = cell2mat(x_seg(i));
    binI = (gray_image > T1) & (gray_image < T2); % thresholding
    subplot(columns,rows,i);
    imshow(binI);
    
end
hold off;

T1 = 0; % Lower limit (cancer melanoma)
T2 = 90; % Upper limit (cancer melanoma)

figure;
sgtitle('Histogram equalized images thresholded');
hold on;
for i=1:images_count
    
    gray_image = histeq(cell2mat(x_seg(i)));
    binI = (gray_image > T1) & (gray_image < T2); % thresholding
    subplot(columns,rows,i);
    imshow(binI);
    
end
hold off;