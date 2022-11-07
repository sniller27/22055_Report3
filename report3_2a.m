%% Read images
clear; clc; close all; % Clear workspace and figures

folder_path = 'Melanoma\'; % images folder
images = dir(fullfile(folder_path, '*.jpg')); % jpg-files in the images folder
images_count = numel(images); % number of images

%% Display color images
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

%% Display grayscale images
rows = 5; columns = 5;
melanoma_image_grayscale = cell(1,images_count); % create cell-array for grayscale images

figure;
sgtitle('Melanomas (grayscaled)');
for i=1:images_count
    img = cell2mat(melanoma_image(i)); % read image
    melanoma_image_grayscale(i) = {rgb2gray(img)}; % convert to grayscale
    
    % show image
    subplot(columns,rows,i);
    imshow(cell2mat(melanoma_image_grayscale(i)))
    title(i);
end

%% Histograms of grayscaled melanomas (imhist seems to lag have an annoying scale bar)
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

%% Thresholding
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
    binI = (gray_image > 0) & (gray_image < thresholds(i)); % thresholding
    
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

%%
no = 1;

%gray_image = cell2mat(melanoma_image_grayscale(no));
gray_image = cell2mat(melanoma_image_grayscale(no));
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
     idx = find(area > 22000 & circularity > 0.025); % good for removing of rulers (but also 2 melanomas)
%     idx = find(area > 22000 & circularity > 0.024 & area < 3290250);
     binI = ismember(L,idx);

figure;
imshow(binI);

%%
mask = cast(binI, class(gray_image));
maskedRgbImage = bsxfun(@times,gray_image,cast(mask,class(gray_image)));

% figure;
% imshow(maskedRgbImage);

nonZeroIndexes = maskedRgbImage ~= 0;
oki = maskedRgbImage(nonZeroIndexes);
% maskedRgbImage2 = (maskedRgbImage > 0);

figure;
histogram(oki);
%imshow(imfuse(gray_image,binI));

%%
benign = cell(1,14);
malign = cell(1,10);
detection_result = cell(1,25);

figure;
sgtitle('histogram');
hold on;
for i=1:images_count
    
    thresholded_image = cell2mat(thresholded_images(i));
    gray_image = cell2mat(melanoma_image_grayscale(i));
    
    mask = cast(thresholded_image, class(gray_image));
    maskedImage = bsxfun(@times,gray_image,cast(mask,class(gray_image)));

    % figure;
    % imshow(maskedRgbImage);

    nonZeroIndexes = maskedImage ~= 0;
    oki = maskedImage(nonZeroIndexes);
    % maskedRgbImage2 = (maskedRgbImage > 0);
    
    if i >= 15
        malign(i-14) = {oki};
    else
       benign(i) = {oki};
    end
    
    %subplot(columns,rows,i);
    cutoff_malign = 65;
    oki = oki(oki < cutoff_malign);
    hout = histogram(oki);

    %mean(oki) % 56-61 (benign)
    %mean(oki) % 43-55 (malign)
    %median(oki)
    
    % DETECT
    if mean(oki) < 55.5
        detection_result(i) = {'Malign'};
        %fprintf('malign \n');
    else
        detection_result(i) = {'Benign'};
        %fprintf('benign \n');
    end
    
end
hold off;
%% print detections

rows = 5;
columns = 5;

figure;
sgtitle('Melanoma');
for i=1:images_count
    subplot(columns,rows,i);
    imshow(cell2mat(melanoma_image_grayscale(i)))
    title(cell2mat(detection_result(i)));
end



%% 3.)
%clear; clc; close all; % Clear workspace and figures

% SHAPE: DOT (similar to creating optical mask with small circular aperature (dot))
n = 2^10;                 % size of mask
M = zeros(n);
I = 1:n; 
x = I-n/2;                % mask x-coordinates 
y = n/2-I;                % mask y-coordinates
[X,Y] = meshgrid(x,y);    % create 2-D mask grid
R = 10;                   % aperture radius
A = (X.^2 + Y.^2 <= R^2); % circular aperture of radius R
M(A) = 1;                 % set mask elements inside aperture to 1
X = M;

figure;
sgtitle('fft2 and ifft2 on optical mask with aperature (circle/dot)');

% original matrix
subplot(3,3,1);
imagesc(X);
title('Original 2D-matrix');

% fft2
Y = fft2(X);
subplot(3,3,2);
imagesc(abs(Y));
title('fft2');

% fft2 via fft (same)
Y1 = fft(fft(X).').';
subplot(3,3,3);
imagesc(abs(Y1));
title('fft2 via fft');

% fftn (same)
Y_fftn = fftn(X);
subplot(3,3,4);
imagesc(abs(Y_fftn));
title('fft2 via fftn');

% fft2 + shifting (zero-frequency components centered)
Y_shift = fftshift(Y);
subplot(3,3,5);
imagesc(abs(Y_shift));
title('fft2 with shift');

% fft2 + truncate (resize matrix before fft2)
truncate_no = 506;
Y_truncate = fft2(X,truncate_no,truncate_no);
subplot(3,3,6);
imagesc(abs(Y_truncate));
title('fft2 truncated (506x506 matrix)');

% fft2 + shifting + log (more enhanced view)
Y_shift_log = fftshift(Y);
subplot(3,3,7);
imagesc(abs(log(Y_shift_log)));
title('fft2 with shift and log');

% ifft2 (reverse)
X_new = ifft2(Y);
subplot(3,3,8);
imagesc(abs(X_new));
title('ifft2');

% ifft2 + shifts (same)
X_new2 = ifft2(fftshift(Y));
subplot(3,3,9);
imagesc(abs(X_new2));
title('ifft2 with shift');

%% SHAPE: SQUARE
X_rec = zeros(30,30);
X_rec(11:20,11:20) = 1;

% X_rec = zeros(30,30);
% X_rec(16:25,11:20) = 1;

% X_rec = zeros(30,30);
% X_rec(8:23,8:23) = 1;

figure;
sgtitle('fft2 square image');

% original matrix
subplot(2,3,1);
imagesc(X_rec);
title('Rectangle');

% fft2
Y_rec = fft2(X_rec);
subplot(2,3,2);
imagesc(abs(Y_rec));
title('fft2');

% fft2 + shifting (zero-frequency components centered)
Y_rec_shift = fftshift(Y_rec);
subplot(2,3,3);
imagesc(abs(Y_rec_shift));
title('fft2 with shift');

% fft2 + log (more enhanced view)
subplot(2,3,4);
imagesc(abs(log(Y_rec)));
title('fft2 with shift and log');

% fft2 + shifting + log (more enhanced view)
Y_rec_shift_log = fftshift(Y_rec);
subplot(2,3,5);
imagesc(abs(log(Y_rec_shift_log)));
title('fft2 with shift and log');

%% SHAPE: RECTANGLE
X_rec = zeros(30,30);
X_rec(5:24,13:17) = 1;

figure;
sgtitle('fft2 rectangle image');

% original matrix
subplot(2,3,1);
imagesc(X_rec);
title('Rectangle');

% fft2
Y_rec = fft2(X_rec);
subplot(2,3,2);
imagesc(abs(Y_rec));
title('fft2');

% fft2 + shifting (zero-frequency components centered)
Y_rec_shift = fftshift(Y_rec);
subplot(2,3,3);
imagesc(abs(Y_rec_shift));
title('fft2 with shift');

% fft2 + log (more enhanced view)
subplot(2,3,4);
imagesc(abs(log(Y_rec)));
title('fft2 with shift and log');

% fft2 + shifting + log (more enhanced view)
Y_rec_shift_log = fftshift(Y_rec);
subplot(2,3,5);
imagesc(abs(log(Y_rec_shift_log)));
title('fft2 with shift and log');


%%
% clear; clc; close all; % Clear workspace and figures
% f = zeros(30,30);
% f(5:24,13:17) = 1;
% 
% figure;
% imshow(f,'InitialMagnification','fit')
% 
% F = fft2(f);
% F2 = log(abs(F));
% imshow(F2,[-1 5],'InitialMagnification','fit');
% colormap(jet); colorbar
% 
% F = fft2(f,256,256);
% 
% figure;
% imshow(log(abs(F)),[-1 5],'InitialMagnification','fit'); colormap(jet); colorbar
% 
% F = fft2(f,256,256);
% F2 = fftshift(F);
% 
% figure;
% imshow(log(abs(F2)),[-1 5],'InitialMagnification','fit'); 
% colormap(jet); 
% colorbar



%% test with real images (not working)
% image_test = cell2mat(melanoma_image(1));
% fft2_image = fft2(image_test);
% 
% figure;
% imagesc(abs(image_test));
% 
% figure;
% imagesc(abs(fft2_image));
% 
% figure;
% imagesc(abs(fftshift(fft2_image)));
% 
% % figure;
% % imagesc(ifft2(fft2_image));