%% clear all
clear
clc

%% read image

% input image
img_rgb = imread('embeded image.tif');

% convert to yuv color space
img_yuv = rgb2ycbcr(img_rgb);

% get the bright channal and its size and total elements
img_y = img_yuv(:,:,1);
[row, col] = size(img_y);
total = row * col;
%% detect

% generate two set of coordinate
ai = 1:2:total;
bi = 2:2:total;

% algin the dimension
if(length(ai) > length(bi))
    ai(end) = [];
elseif(length(ai) < length(bi))
    bi(end) = [];
end

% detect value
flag = sum( img_y(ai) - img_y(bi) ) / total;

%% output

detect_degree = 0.4;
if(flag > detect_degree)
    disp 'YES have watermark';
else
    disp 'NO watermark';
end