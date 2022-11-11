%% 2.a) Read images
clear; clc; close all; % Clear workspace and figures

folder_path = 'Melanoma\'; % images folder
images = dir(fullfile(folder_path, '*.jpg')); % jpg-files in the images folder
images_count = numel(images); % number of images

%% 2.a) Display color images
rows = 5; columns = 5;
melanoma_image = cell(1,images_count); % create cell-array for grayscale images

figure;
sgtitle('Melanomas');
for i=1:images_count
    % read image and store in cell-array
    img = imread(fullfile(images(i).folder, images(i).name));
    melanoma_image(i) = {img};
    
    % show image
    subplot(columns,rows,i);
    imshow(cell2mat(melanoma_image(i)))
    title(i);
end

%% 2.a) Display grayscale images
rows = 5; columns = 5;
melanoma_image_grayscale = cell(1,images_count); % create cell-array for grayscale images

figure;
sgtitle('Melanomas (grayscaled)');
%sgtitle('GREEN-channel melanomas');
for i=1:images_count
    img = cell2mat(melanoma_image(i)); % read image
    melanoma_image_grayscale(i) = {rgb2gray(img)}; % convert to grayscale
    %melanoma_image_grayscale(i) = {img(:,:,1)}; % RED-channel (brighter?)
    %melanoma_image_grayscale(i) = {img(:,:,2)}; % GREEN-channel (darker?)
    %melanoma_image_grayscale(i) = {img(:,:,3)}; % BLUE-channel
    
    % show image
    subplot(columns,rows,i);
    imshow(cell2mat(melanoma_image_grayscale(i)))
    title(i);
end

%% 2.b) Histograms of grayscaled melanomas (imhist seems to lag have an annoying scale bar)
rows = 5; columns = 5;
thresholds = zeros(1,images_count);
cutoff = 195; % was originally 'end' (improves thresholding of multithresh())

figure;
sgtitle('Histograms of grayscaled melanomas');
for i=1:images_count
    
    gray_image = cell2mat(melanoma_image_grayscale(i)); % get grayscale image
    gray_image = gray_image(gray_image < cutoff); % helps moving threshold to the left
    [H,W] = size(gray_image); % image size
    M = H*W; % number of pixels
    
    subplot(columns,rows,i);
    title(i);
    hold on;
    
    hout = histogram(double(reshape(gray_image, M, 1)));
    
    histLine = hout.Values(1:2:end);
    histLineX = hout.BinEdges(2:2:end);
    
    bimodal_threshold = double(multithresh(gray_image)); % threshold pixel values
    [ difference, index ] = min( abs( histLineX-bimodal_threshold ) ); % finding closest value to threshold
    
    thresholds(i) = (histLineX(index));
    
    % plot histogram and threshold
    %plot(histLineX,histLine);
    plot(histLineX(index),histLine(index),'*');
    xlim([0 255]);

end

hold off;

%% 2.b) Thresholding
thresholded_images = cell(1,images_count);

rows = 5;
columns = 5;

T1 = 0; % Lower limit (cancer melanoma)
T2 = 100; % Upper limit (cancer melanoma)

figure;
sgtitle('Images thresholded');
hold on;
for i=1:images_count
    
    gray_image = cell2mat(melanoma_image_grayscale(i));
    %gray_image = histeq(cell2mat(melanoma_image_grayscale(i)));
    %gray_image = adapthisteq(cell2mat(melanoma_image_grayscale(i)));
    
    % MULTIPLE THRESHOLDING DONE MANUALLY (more precise due analyzed and changed histograms points)
    binI = (gray_image > 0) & (gray_image < thresholds(i)); % thresholding ... 4 sec
    
    % MULTIPLE THRESHOLDING VIA MULTITHRESH
    % (last peak from ruler should be removed first)
%     level1 = multithresh(gray_image); % REMEMBER SECOND PARAM ON IMSHOW(XXX,[]); ... 37 SEC ... returns normalized threshold in same range as image
%     binI = imquantize(gray_image,level1); % thresholds multithresh

    % MULTIPLE THRESHOLDING VIA GRAYTRESH
    % (last peak from ruler should be removed first)
%     level2 = graythresh(gray_image); % FASTER ... 8 SEC ... returns normalized threshold [0,1]
%     binI = imbinarize(gray_image,level2); % thresholds graytresh

    % MULTIPLE THRESHOLDING VIA OTSUTRESH
%     [counts,x] = imhist(gray_image,8);
%      T = otsuthresh(counts);
%      binI = imbinarize(gray_image,T);

    % MULTIPLE THRESHOLDING VIA ADAPTTRESH
    % (last peak from ruler should be removed first)
%     T = adaptthresh(gray_image,0.64);
%     binI = imbinarize(gray_image,T);
    
    subplot(columns,rows,i);
    title(i);
    hold on;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     se = strel('disk',5);            
%     binI = imdilate(binI,se);

    L = bwlabel(binI,8);

    imgStats = regionprops(L, 'All');

    circularity = [imgStats.Circularity];
    
     area = [imgStats.Area];
     
     %idx = find(area > 22000); % shows all melanomas (but with rulers)
     idx = find(area > 22000 & circularity > 0.025); % good for removing of rulers (but also 2 melanomas)
 
    binI = ismember(L,idx);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    binI = imfill(binI,'holes');
    
    thresholded_images(i) = {binI};
    
    imshow(binI);
    %imshow(binI,[]) % for multihresh
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
%     gray_image = histeq(cell2mat(melanoma_image_grayscale(i)));
%     binI = (gray_image > T1) & (gray_image < T2); % thresholding
%     subplot(columns,rows,i);
%     imshow(binI);
%     
% end
% hold off;

%% 2.b) Thresholding with bwperim (only perimeters)

figure;
sgtitle('Perimeters of melanomas (bwperim)');
hold on;
for i=1:images_count
    
    thresholded_image = cell2mat(thresholded_images(i));
    
    subplot(columns,rows,i);
    title(i);
    hold on;
    
    imshow(bwperim(thresholded_image,8));
    
end
hold off;

%% 2.b) Detection: only select melanomas from segments (masks) ... based on histograms pixel means (maligns are darker)
benign = cell(1,14);
malign = cell(1,10);
detection_result = cell(1,25);

min_benign = 0;
max_malign = 0;

figure;
sgtitle('Melanoma masks');
hold on;
for i=1:images_count
    
    % MAKING IMAGE MASKS
    thresholded_image = cell2mat(thresholded_images(i));
    gray_image = cell2mat(melanoma_image_grayscale(i));
    
    mask = cast(thresholded_image, class(gray_image));
    maskedImage = bsxfun(@times,gray_image,cast(mask,class(gray_image)));
    
    nonZeroIndexes = maskedImage ~= 0;
    oki = maskedImage(nonZeroIndexes);
    % maskedRgbImage2 = (maskedRgbImage > 0);
    
    if i >= 15
        malign(i-14) = {oki};
    else
       benign(i) = {oki};
    end
    
    %subplot(columns,rows,i);
    cutoff_malign = 44; 
    oki = oki(oki < cutoff_malign);
    %hout = histogram(oki);

    %mean(oki) % 56-61 (benign)
    %mean(oki) % 43-55 (malign)
    %median(oki)
    %mean(oki)
%     
%     if i < 15
%         if i == 1
%             min_benign = mean(oki);
%         end
%         
%         if min_benign > mean(oki)
%             min_benign = mean(oki);
%         end
%     else
%         if i == 15
%             max_malign = mean(oki);
%         end
%         
%         if max_malign < mean(oki)
%             max_malign = mean(oki);
%         end
%     end
    
    % DETECT BASED ON MEAN PIXEL VALUE
    if mean(oki) < 37.5
        detection_result(i) = {'Malign'};
        fprintf('malign \n');
    else
        detection_result(i) = {'Benign'};
        fprintf('benign \n');
    end
    
%     subplot(columns,rows,i);
%     imshow(maskedImage);
%     title(i);
    
end
hold off;

if min_benign < max_malign
        fprintf('No separation between benign and malign \n');
else
        fprintf('Maybe there is a separation between benign and malign \n');
end

%% b.) DETECTION: print detections
rows = 5;
columns = 5;

figure;
sgtitle('Melanoma classification');
for i=1:images_count
    subplot(columns,rows,i);
    imshow(cell2mat(melanoma_image_grayscale(i)))
%     if i == 13
%         title(cell2mat(detection_result(i)),'Color','red');
%     else
         title(cell2mat(detection_result(i)),'Color','green');
%     end
    
end

%% 2.c) 
edged_images = cell(1,images_count);
rows = 5; columns = 5;
% fudge_increment = 0.4;
%figure;
% sgtitle('Images thresholded');
% hold on;

%% 2.c) Segmentation based on edge-detection and morphology (1. part of the algorithm)
rows = 5; columns = 5;
edge_segmented = cell(1,25);

fudgeFactor = 0.5; % 0.4-0.6
%increase = 0.02;

figure;
sgtitle('Segmentation based on edge detection and morphology');
hold on;
for i=1:25
    
    % fudgeFactor = fudgeFactor + increase;
    
    % DIFFERENT KINDS OF IMAGE ENHANCEMENTS
    % link: https://www.mathworks.com/help/images/contrast-enhancement-techniques.html
    %gray_image = cell2mat(melanoma_image_grayscale(9)); % fudge=0.5.
    gray_image = histeq(cell2mat(melanoma_image_grayscale(i))); % fudge=0.5.
    %gray_image = adapthisteq(cell2mat(melanoma_image_grayscale(9)));
    %gray_image = imadjust(cell2mat(melanoma_image_grayscale(i))); % fudge=0.5.
    %gray_image = imsharpen(gray_image);
    %gray_image = imgaussfilt(gray_image,1);
    
    % EDGE THRESHOLDING
    [~,threshold] = edge(gray_image,'sobel');
    
    BWs = edge(gray_image,'sobel',threshold * fudgeFactor); % edge detection based on threshold

%     fudgeFactor = 1.5; % 0.4-0.6
%     BWs = edge(gray_image,'Canny',threshold * fudgeFactor);
    
%     fudgeFactor = 0.5; % 0.4-0.6
%     BWs = edge(gray_image,'prewitt',threshold * fudgeFactor);
    
    BWs = logical(255) - BWs; % inverting the image
    
    SE = strel('disk',6);     
    BWs = imerode(BWs,SE);
    
    SE = strel('disk',12);     
    BWs = imdilate(BWs,SE);
    
    %BWs = imfill(BWs,'holes');
        
    SE = strel('disk',18);     
    BWs = imerode(BWs,SE);
    
    BWs = imfill(BWs);
    
    %%%%%%%%%%%%%%%%%%%%%
    L = bwlabel(BWs,8);

    imgStats = regionprops(L, 'All');

    circularity = [imgStats.Circularity];
    
    area = [imgStats.Area];
     
    idx = find(area > 85000 & circularity > 0.22); % shows all melanomas (but with rulers)
    %idx = find(circularity > 0.025); % shows all melanomas (but with rulers)
    %idx = find(area > 22000 & circularity > 0.025); % good for removing of rulers (but also 2 melanomas)
    
    [qty,data] = size(idx);
    
    if data >= 1
        edge_segmented(i) = {1};
    else
        edge_segmented(i) = {0};
    end
     
    BWs = ismember(L,idx);
    BWs = imclearborder(BWs,4);
    %%%%%%%%%%%%%%%%%%%%%
    
    edged_images(i) = {BWs};
    
%     figure;
%     imshow(BWs);

     subplot(columns,rows,i);
     imshow(BWs);
    
    %imshow(labeloverlay(gray_image,binI));
    title(fudgeFactor);
    %hold on;
    
end
hold off;


%% 2.c) Segmentation based on edge-detection and morphology (2. part of the algorithm)
rows = 5; columns = 5;
fudgeFactor = 0.7; % 0.4-0.6

figure;
sgtitle('Melanoma');
for i=1:25
    
    segmented = cell2mat(edge_segmented(i)); % fudge=0.5.
    if segmented == 0
        
        gray_image = histeq(cell2mat(melanoma_image_grayscale(i)));

        [~,threshold] = edge(gray_image,'sobel');
        
        %BWs = logical(255) - BWs; % inverting the image

        BWs = edge(gray_image,'sobel',threshold * fudgeFactor);
        
        
        SE = strel('disk',6);     
        BWs = imdilate(BWs,SE);
        
        SE = strel('disk',5);     
        BWs = imerode(BWs,SE);
        
%         SE = strel('disk',10);     
%         BWs = imdilate(BWs,SE);
        
        BWs = imfill(BWs,'holes');
        
        %%%%%%%%%%%%%%%%%%%%%
        L = bwlabel(BWs,8);

        imgStats = regionprops(L, 'All');

        circularity = [imgStats.Circularity];

        area = [imgStats.Area];

        idx = find(area > 1200000 & circularity > 0.05); % shows all melanomas (but with rulers)
        %idx = find(circularity > 0.025); % shows all melanomas (but with rulers)
        %idx = find(area > 22000 & circularity > 0.025); % good for removing of rulers (but also 2 melanomas)

        BWs = ismember(L,idx);
        
        %%%%%%%%%%%%%%%%%%%%%
        
        
        subplot(columns,rows,i);
        imshow(BWs);
        
    end
    
end






