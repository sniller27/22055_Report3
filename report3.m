%% clear
clear; clc; close all; % Clear workspace and figures

%% 1.b)
Im = imread('test22.jpg');

% doesn't seem to be the right len_y, but still looks good?
[len_x, len_y] = size(Im);
P = len_x*len_y;
% P = numel(I); % same

% P = size(Im,1)*size(Im,2); % I WOULD EXPECT IT TO BE THIS ONE, BUT IT DOESN'T LOOK GOOD.

%% histograms (normalized and non-normalized)
% [counts, binLocations] = imhist(I);
% 
% figure;
% subplot(2,1,1);
% imhist(I);
% title('Non-normalized histogram');
% 
% % normalized histogram (by using stem plot)
% subplot(2,1,2);
% stem(binLocations, counts./P, 'Marker', 'none'); % normalized
% title('Normalized histogram');
% xlim([0 255]);
% %ylim([0 max(counts)]);
% %axis([min(J) max(J) 0 255]);

%% 1.b) Manual histogram equalization
[counts, binLocations] = imhist(Im);
cdf = cumsum(counts./P);

% never used
% min_cdf = min(cdf);
% depth = 1;
% result = round((cdf-min_cdf)/(P-min_cdf)*(depth-1));

L = 256;
g = floor((L-1)*cdf); % 1. way
g1 = ceil(L*cdf)-1; % 2. way
[J,T] = histeq(Im); % 3. way

% Transformation curves
figure;
sgtitle('Transformation curves');
subplot(4,1,1);
%stem(binLocations, cdf, 'Marker', 'none'); % normalized
plot(binLocations, cdf);
title('normalized');
subplot(4,1,2);
%stem(g, 'Marker', 'none');
plot(g);
title('non-normalized 1');
subplot(4,1,3);
%stem(g1, 'Marker', 'none');
plot(g1);
title('non-normalized 2');
subplot(4,1,4);
%stem((0:255)/255,T, 'Marker', 'none');
plot((0:255)/255, T);
title('via histeq function');

Im2 = Im+1;
g_u = g(Im2);
im_equalized = uint8(g_u);

% Images
figure;
sgtitle('Image enhancements');
subplot(3,1,1);
imshow(Im);
title('Original image');
subplot(3,1,2);
imshow(im_equalized);
title('histogram equalization (manual)');
subplot(3,1,3);
imshow(histeq(Im));
title('histogram equalization (histeq function)');

% Histograms
figure;
sgtitle({['Image histograms'], ['']}); % overlaps with subplot titles due to imhist() functions
subplot(3,1,1);
imhist(Im);
title('Original image');
subplot(3,1,2);
imhist(im_equalized);
title('histogram equalization (manual)');
subplot(3,1,3);
imhist(histeq(Im));
title('histogram equalization (histeq function)');

%% 1.b) Histogram equalization (Even more manual...but slower) (https://www.imageeprocessing.com/2011/04/matlab-code-histogram-equalization.html)
Im = imread('test22.jpg');

numofpixels=size(Im,1)*size(Im,2);

% declare vectors
im_matrix = uint8(zeros(size(Im,1),size(Im,2)));
freq=zeros(256,1);
probf=zeros(256,1);
probc=zeros(256,1);
cum=zeros(256,1);
output=zeros(256,1);

%freq counts the occurrence of each pixel value.
%The probability of each occurrence is calculated by probf.
for i=1:size(Im,1)
    for j=1:size(Im,2)

        value=Im(i,j);

        freq(value+1)=freq(value+1)+1; % non-normalized
        probf(value+1)=freq(value+1)/numofpixels; % normalized

    end
end

sum=0;
no_bins=255;

%The cumulative distribution probability is calculated. 
for i=1:size(probf)

   sum=sum+freq(i);

   cum(i)=sum; % non-normalized
   probc(i)=cum(i)/numofpixels; % normalized
   output(i)=round(probc(i)*no_bins); % according to book

end

% overfører alle bits med +1 (?)
for i=1:size(Im,1)
    for j=1:size(Im,2)

        im_matrix(i,j)=output(Im(i,j)+1);

    end
end

% Transformation curves
figure;
sgtitle('Transformation curves');
subplot(4,1,1);
%stem(binLocations, cdf, 'Marker', 'none'); % normalized
plot(binLocations, cdf);
title('Normalized method 1');
subplot(4,1,2);
%stem(cum, 'Marker', 'none');
plot(cum);
title('Non-normalized method 2');
subplot(4,1,3);
%stem(probc, 'Marker', 'none');
plot(probc);
title('Normalized method 2');
subplot(4,1,4);
%stem(output, 'Marker', 'none');
plot(output);
title('Corrected method 2');

% Images
figure;
sgtitle('Image enhancements');
subplot(2,1,1);
imshow(im_equalized);
title('Equalized image 1');
subplot(2,1,2);
imshow(im_matrix);
title('Equalized image 2');

% Histograms
figure;
sgtitle('Histograms');
subplot(2,1,1);
imhist(im_equalized);
title('Equalized image 1');
subplot(2,1,2);
imhist(im_matrix);
title('Equalized image 2');

%% 1.c)

% setup
folder_path = 'biomedical_images_cp4\';
% henter alle billeder
images = dir(fullfile(folder_path, '*.tif'));
% numel = er antal af billeder
images_count = numel(images);

% % read first image (1) via path
% img = imread(fullfile(images(1).folder, images(1).name));
% % kan testes via imshow()
% imshow(img);

rows = 2;

figure;
sgtitle('Thresholding of biomedical images');
for i=1:images_count
    
    img = imread(fullfile(images(i).folder, images(i).name)); % read image
    
    if (i == 1)
        subplot(images_count,rows,1);
    else
        subplot(images_count,rows,i+i-1);
    end
    
    imshow(img); % plot original image
    title('Original')
    
    subplot(images_count,rows,i+i);
    Im2 = im_equalization(img);
    imshow(Im2); % plot histogram equalized images
    title('Histogram equalized')
end

%% 1.c)
% t = Tiff('biomedical_images_cp4\chest_image.tiff', 'r');
% imageData = read(t);
% imshow(imageData);

% setup
folder_path = 'images_cp3\';
% henter alle billeder
images = dir(fullfile(folder_path, '*.tif'));
% numel = er antal af billeder
images_count = numel(images);

% read first image (1) via path
%img = imread(fullfile(images(1).folder, images(1).name));
% kan testes via imshow()
%imshow(img);
rows = 2;

figure;
sgtitle('Thresholding of fractured human spine and rice images');
for i=1:images_count
    
    img = imread(fullfile(images(i).folder, images(i).name)); % read image
    
    if (i == 1)
        subplot(images_count,rows,1);
    else
        subplot(images_count,rows,i+i-1);
    end
    
    imshow(img); % plot original image
    title('Original')
    
    subplot(images_count,rows,i+i);
    Im2 = im_equalization(img);
    imshow(Im2); % plot histogram equalized images
    title('Histogram equalized')
end

%% 2 a.)
% setup
folder_path = 'Melanoma\';
% henter alle billeder
images = dir(fullfile(folder_path, '*.jpg'));
% numel = er antal af billeder
images_count = numel(images);

rows = 5;

figure;
sgtitle('Melanoma');
for i=1:images_count
    
    img = imread(fullfile(images(i).folder, images(i).name)); % read image

    subplot(5,rows,i);
    img_grayscale = rgb2gray(img);
    imshow(img_grayscale); % plot original image

end

%%
% setup
folder_path = 'Melanoma\';
% henter alle billeder
images = dir(fullfile(folder_path, '*.jpg'));

img = imread(fullfile(images(1).folder, images(20).name));
img_grayscale = rgb2gray(img);
%imshow(img_grayscale);

% imhist => viser 118-181

%T1 = 30; % Lower limit (normal melanoma)
%T2 = 110; % Upper limit (normal melanoma)

T1 = 0; % Lower limit (cancer melanoma)
T2 = 40; % Upper limit (cancer melanoma)

binI = (img_grayscale > T1) & (img_grayscale < T2); % thresholding
imshow(binI);

% thresholding? => morphological operations?
% histogram equalization?
% circularity?

%% histogram equalization function
function [im_equalized] = im_equalization(Im)
    
    [len_x, len_y] = size(Im);
    P = len_x*len_y;
    
    [counts, binLocations] = imhist(Im);
    cdf = cumsum(counts./P);

    L = 256;
    g = floor((L-1)*cdf);

    Im2 = Im+1;
    g_u = g(Im2);
    im_equalized = uint8(g_u);
    
end
