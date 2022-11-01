%% Github: https://github.com/sniller27/22055_Report3
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

rows = 4;
columns = 4;

figure;
sgtitle('Thresholding of biomedical images');
for i=1:images_count
    
    img = imread(fullfile(images(i).folder, images(i).name)); % read image
    
    if (i == 1)
        subplot(rows,columns,1);
    else
        subplot(rows,columns,i+i-1);
    end
    
    imshow(img); % plot original image
    title('Original')
    
    subplot(rows,columns,i+i);
    Im2 = im_equalization(img);
    imshow(Im2); % plot histogram equalized images
    title('Histogram equalized')
end

% TRANSFORMATION CURVES
L = 256;

rows = 4;
columns = 2;

figure;
sgtitle('Transformation curves of biomedical images');
for i=1:images_count
    
    img = imread(fullfile(images(i).folder, images(i).name)); % read image
    [counts, binLocations] = imhist(img);
    
    [len_x, len_y] = size(Im);
    P = len_x*len_y;
    
    cdf = cumsum(counts./P);
    
    g = floor((L-1)*cdf); % 1. way

    subplot(rows,columns,i);
    Im2 = im_equalization(img);
    plot(g); % plot histogram equalized images
    title('Transformation curve')
    xlim([0 256]);
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
rows = 5;
columns = 2;

figure;
sgtitle('Thresholding of fractured human spine and rice images');
for i=1:images_count
    
    img = imread(fullfile(images(i).folder, images(i).name)); % read image
    
    if (i == 1)
        subplot(rows,columns,1);
    else
        subplot(rows,columns,i+i-1);
    end
    
    imshow(img); % plot original image
    title('Original')
    
    subplot(rows,columns,i+i);
    Im2 = im_equalization(img);
    imshow(Im2); % plot histogram equalized images
    title('Histogram equalized');
end

% TRANSFORMATION CURVES
L = 256;

rows = 5;
columns = 1;

figure;
sgtitle('Transformation curves of fractured human spine and rice images');
for i=1:images_count
    
    img = imread(fullfile(images(i).folder, images(i).name)); % read image
    [counts, binLocations] = imhist(img);
    
    [len_x, len_y] = size(Im);
    P = len_x*len_y;
    
    cdf = cumsum(counts./P);
    
    g = floor((L-1)*cdf); % 1. way

    subplot(rows,columns,i);
    Im2 = im_equalization(img);
    plot(g); % plot histogram equalized images
    title('Transformation curve');
    xlim([0 256]);
end

%% 2 a.) read melanoma images (takes some time)
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

%% all grayscales
rows = 5;
columns = 5;

figure;
sgtitle('Melanoma');
for i=1:images_count
    subplot(columns,rows,i);
    imshow(cell2mat(x_seg(i)))
    title(i);
end

%% all thresholded
rows = 5;
columns = 5;

T1 = 0; % Lower limit (cancer melanoma)
T2 = 90; % Upper limit (cancer melanoma)

figure;
sgtitle('Melanoma');
for i=1:images_count
    gray_image = cell2mat(x_seg(i));
    % gray_image = histeq(cell2mat(x_seg(i)));
    binI = (gray_image > T1) & (gray_image < T2); % thresholding
    
    se = strel('disk',10);        
    
    BWe = imdilate(binI,se);

    L = bwlabel(BWe,8);
    L1 = label2rgb(L);

    imgStats = regionprops(L, 'All');

    circularity = [imgStats.Circularity];
    area = [imgStats.Area];

    idx = find(area > 47000 & circularity > 0.29);
    %idx = find([imgStats.Circularity] > 0.6 & [imgStats.Circularity] < 0.62);

    BW2 = ismember(L,idx);
    
    subplot(columns,rows,i);
    %imshow(binI);
    imagesc(BW2); 
    axis image;
    
    if isempty(idx)~=1
        title(circularity(idx(1)));
    end
end

%%
no = 1; % 6, 7

%gray_image = cell2mat(x_seg(no));
gray_image = histeq(cell2mat(x_seg(no)));
binI = (gray_image > 0) & (gray_image < 20); % thresholding (smaller ones)
%binI = (gray_image > 0) & (gray_image < 130); % thresholding (big ones)
figure;
imshow(binI)

%%
se = strel('disk',10);        
BWe = imdilate(binI,se);

L = bwlabel(BWe,8);
L1 = label2rgb(L);

imgStats = regionprops(L, 'All');

circularity = [imgStats.Circularity];
area = [imgStats.Area];

idx = find(area > 47000 & circularity > 0.29);
%idx = find([imgStats.Circularity] > 0.6 & [imgStats.Circularity] < 0.62);

BW2 = ismember(L,idx);
figure, imagesc(BW2); 
axis image;

if isempty(idx)~=1
        title(circularity(idx(1)));
end

%% max-values in histograms
for i=1:images_count
    [counts,x] = imhist(cell2mat(x_seg(i)));
    %find(x == max(counts))
    max(counts)
end

%% histograms
figure;
%hist(double(cell2mat(x_seg(1))), length(double(cell2mat(x_seg(1))))); % takes one minute!!!!!!
hold on;
imhist(cell2mat(x_seg(1)));
imhist(cell2mat(x_seg(4)));
imhist(cell2mat(x_seg(5)));
imhist(cell2mat(x_seg(6)));
imhist(cell2mat(x_seg(9)));
imhist(cell2mat(x_seg(12)));
imhist(cell2mat(x_seg(10)));
ylim([0 500000]);
%hold off;

%% histograms equalized
figure;
%hist(double(cell2mat(x_seg(1))), length(double(cell2mat(x_seg(1))))); % takes one minute!!!!!!
hold on;
imhist(histeq(cell2mat(x_seg(1))));
imhist(histeq(cell2mat(x_seg(4))));
imhist(histeq(cell2mat(x_seg(5))));
imhist(histeq(cell2mat(x_seg(6))));
imhist(histeq(cell2mat(x_seg(9))));
imhist(histeq(cell2mat(x_seg(12))));
imhist(histeq(cell2mat(x_seg(10))));
ylim([0 500000]);
%hold off;

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

%T1 = 0; % Lower limit (cancer melanoma)
%T2 = 40; % Upper limit (cancer melanoma)


% histogram equalized
% img_grayscale = histeq(img_grayscale);

T1 = 0; % Lower limit (cancer melanoma)
T2 = 90; % Upper limit (cancer melanoma)
%210-255 er lineal

% MARK EDGES
% [~,threshold] = edge(img_grayscale,'sobel');
% fudgeFactor = 0.5;
% img_grayscale = edge(img_grayscale,'sobel',threshold * fudgeFactor);

% figure;
% imshow(img_grayscale);

binI = (img_grayscale > T1) & (img_grayscale < T2); % thresholding

figure;
imshow(binI);

%binI = imclearborder(binI,4);

%seD = strel('diamond',1);
%binI = imerode(binI,seD);

se = strel('disk',3);        
BWe = imopen(binI,se);

se = strel('disk',10);        
BWe = imdilate(BWe,se);

[~,threshold] = edge(BWe,'sobel');
fudgeFactor = 0.5;
BWe = edge(BWe,'sobel',threshold * fudgeFactor);

% figure;
% imshow(BWe);

% labels
L = bwlabel(BWe,8);
L1 = label2rgb(L);

imgStats = regionprops(L, 'All');

cellPerimeter = [imgStats.Perimeter];
cellArea = [imgStats.Area];

%figure, plot(cellPerimeter, cellArea, '.');  xlabel('Perimeter'); ylabel('Area');

% figure, hist([imgStats.Area]); title('Cell Area Histogram');

% idx = find([imgStats.Area] > 1500);
% %idx = find([imgStats.Area] > 1500 & [imgStats.Area] < 4000);
% 
% 
% BW2 = ismember(L,idx);
% figure, imagesc(BW2); axis image; title('Object with area > 200');
% 

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
