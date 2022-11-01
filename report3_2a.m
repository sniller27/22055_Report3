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
rows = 5;
columns = 5;

figure;
hold on;
title('Image histograms');
thresholds = zeros(1,images_count);
for i=1:images_count
subplot(columns,rows,i);
gray_image = cell2mat(x_seg(i));
[H,W] = size(gray_image);
M = H*W;
%%tt = double(reshape(cell2mat(x_seg(i)), M, 1));
hout = histogram(double(reshape(gray_image, M, 1)));
hold on;

cutoff = 190; % was originally 'end' (improves thresholding of multithresh())
histLine = hout.Values(1:2:cutoff);
histLineX = hout.BinEdges(2:2:cutoff);

% stuff
[val,troughloc] = findpeaks(-histLine,histLineX,'MinPeakProminence',1000);
plot(histLineX,histLine);
%plot(troughloc,-val,'*');
xlim([0 255]);

bimodal_threshold = double(multithresh(gray_image));
[ difference, index ] = min( abs( histLineX-bimodal_threshold ) );

plot(histLineX(index),histLine(index),'*');

thresholds(i) = (histLineX(index));

%bimodal_min = find(histLine==multithresh(gray_image));
%plot(troughloc,-val,'*');

%[pks,locs] = sort(findpeaks(histLine), 'desc');

% max1_index = find(histLine==pks(1));
% max2_index = find(histLine==pks(2));
% 
% max1 = histLineX(max1_index);
% max2 = histLineX(max2_index);
% 
% xline(max1,'r');
% xline(max2,'r');

% figure;
% plot(hout.Data);

% figure;
% plot(histLineX(max2_index:max1_index),histLine(max2_index:max1_index))
% xlim([0 255]);
% xline(max1,'r');
% xline(max2,'r');

%min_index = find(histLine==min(histLine(max2_index:max1_index)));
%xline(histLineX(min_index),'b');
%plot(histLineX(min_index),min(histLine(max2_index:max1_index)),'*');

%findpeaks(histLine(max1:max2))

%[val,troughloc] = findpeaks(-histLine,histLineX,'MinPeakProminence',1);

%counter = 1;

% while(length(val)>1)
%     [val,troughloc2] = findpeaks(-histLine,histLineX,'MinPeakProminence',1+counter);
%     counter = counter+1;
% end

% thresholds(i) = (troughloc2);
% plot(histLineX,histLine);
% plot(troughloc2,-val,'*');
end

hold off;

% subplot(2,1,2);
% hold on;
% title('Image histograms equalized');
% for i=1:images_count
% gray_image = histeq(cell2mat(x_seg(i)));
% [H,W] = size(gray_image);
% M = H*W;
% %%tt = double(reshape(cell2mat(x_seg(i)), M, 1));
% histogram(double(reshape(gray_image, M, 1)));
% end
% hold off;

%% Thresholding
rows = 5;
columns = 5;

T1 = 0; % Lower limit (cancer melanoma)
T2 = 100; % Upper limit (cancer melanoma)

figure;
sgtitle('Images thresholded');
hold on;
for i=1:images_count
    
    gray_image = cell2mat(x_seg(i));
    binI = (gray_image > 0) & (gray_image < thresholds(i)); % thresholding
    
    subplot(columns,rows,i);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     se = strel('disk',5);            
%     binI = imdilate(binI,se);

    L = bwlabel(binI,8);

    imgStats = regionprops(L, 'All');

    circularity = [imgStats.Circularity];
    
     area = [imgStats.Area];
     
     idx = find(area > 22000); % shows all melanomas (but with rulers)
     idx = find(area > 22000 & circularity > 0.025); % good for removing of rulers (but also 2 melanomas)
 
    binI = ismember(L,idx);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    binI = imfill(binI,'holes');
    
    imshow(binI);
    %imshow(labeloverlay(gray_image,binI));
    
end
hold off;

% T1 = 50; % Lower limit (cancer melanoma)
% T2 = 200; % Upper limit (cancer melanoma)
% 
% figure;
% sgtitle('Histogram equalized images thresholded');
% hold on;
% for i=1:images_count
%     
%     gray_image = histeq(cell2mat(x_seg(i)));
%     binI = (gray_image > T1) & (gray_image < T2); % thresholding
%     subplot(columns,rows,i);
%     imshow(binI);
%     
% end
% hold off;

%%
no = 10;

%gray_image = cell2mat(x_seg(no));
gray_image = cell2mat(x_seg(no));
%binI = (gray_image > 0) & (gray_image < 20); % thresholding (smaller ones)
binI = (gray_image > 0) & (gray_image < thresholds(no)); % thresholding
%binI = (gray_image > 0) & (gray_image < 130); % thresholding (big ones)
% se = strel('disk',10);            
% BWe = imdilate(binI,se);

    L = bwlabel(binI,8);

    imgStats = regionprops(L, 'All');

     circularity = [imgStats.Circularity];
    
     area = [imgStats.Area];
%     
     idx = find(area > 22000 & circularity > 0.024 & area < 3290250);
     binI = ismember(L,idx);

figure;
imshow(binI)


