%% clear all
clear
clc

%% read image

% input image
img_rgb = imread('test.tif');

% convert to yuv color space
img_yuv = rgb2ycbcr(img_rgb);

% get the bright channal and its size and total elements
img_y = img_yuv(:,:,1);
[row, col] = size(img_y);
total = row * col;

%% generate two set of coordinate

ai = 1:2:total;
bi = 2:2:total;

%% modify the bright

degree = 1;
img_y_m = img_y;
img_y_m(ai) = img_y(ai) + degree;
img_y_m(bi) = img_y(bi) - degree;

%% output image

% original image
subplot(1,2,1);
imshow(img_rgb);
title('original image');

% embeded image
subplot(1,2,2);
img_yuv(:,:,1) = img_y_m;
img_rgb = ycbcr2rgb(img_yuv);
imshow(img_rgb);
title('embeded image       d = 1');
imwrite(img_rgb,'embeded image.tif');