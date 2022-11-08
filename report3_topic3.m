%% 3.) fft2 and ifft2 
clear; clc; close all; % Clear workspace and figures

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

%% 3.) fwind2
% clear; clc; close all; % Clear workspace and figures

% Design of BP-filter via 2D-window method
[f1,f2] = freqspace(21,'meshgrid'); % creating frequency range vectors f1 and f2 (each with length 21) 
% (it's suggested to use freqspace due to higher accuracy)
r = sqrt(f1.^2 + f2.^2); % distance of each position from the center frequency
Hd = ones(21); % desired frequency response
Hd((r<0.1)|(r>0.5)) = 0; % desired passband between 0.1 and 0.5

win = fspecial('gaussian',21,2); % creating Gaussian window
win = win ./ max(win(:)); % normalizing window

% plot of desired frequency response
figure;
sgtitle('Window method in 2D (fwind2)');

subplot(3,1,1);
mesh(f1,f2,Hd);
title('Desired 2D frequency response');

subplot(3,1,2);
mesh(win);
title('Gaussian window');

h = fwind2(Hd,win); % creating 2D FIR-filter based on desired frequency response and window.

% plot of 2D FIR-filter frequency response (similar to the desired frequency response)(mesh plot)
subplot(3,1,3);
freqz2(h);
title('Computed 2D FIR-filter frequency response');

%% filter2 + fwind2 (for applying on images)
square_img = X_rec;
melanoma_img = cell2mat(melanoma_image_grayscale(1));

square_img_filtered = filter2(h, square_img);
melanoma_img_filtered = filter2(h, melanoma_img);
% melanoma_img_filtered = filter2(fspecial('average',3), melanoma_img);

figure;
sgtitle('2D-FIR filtering');

% original matrix
subplot(2,2,1);
imagesc(square_img);
title('Original square');

subplot(2,2,2);
imagesc(square_img_filtered);
title('Square filtered by 2D-FIR filter');

subplot(2,2,3);
imagesc(melanoma_img);
title('Melanoma');

subplot(2,2,4);
imagesc(melanoma_img_filtered);
title('Melanoma filtered by 2D-FIR filter');
