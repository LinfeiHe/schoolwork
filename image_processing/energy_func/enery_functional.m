%% clear all
clear
clc

%% load
img = imread('test.tif');
img = rgb2gray(img);
img = imnoise(img,'salt & pepper', 0.02);  % add salt noise

[M, N] = size(img);
imshow(img);
%% total variation method
lambda = 0.8;
alpha = 0.25;
temp = zeros(M+2, N+2);
temp(2:M+1, 2:N+1) = double(img);

for i = 2 : M+1
    temp(i,1) = temp(i,2);
    temp(i,N+2) = temp(i,N+1);
end
for j = 2 : N+1
    temp(1,j) = temp(2,j);
    temp(M+2,j) = temp(M+1,j);
end
I = temp;

detu = zeros(M+2,N+2);
for k = 1 : 50
    for i = 2 : M+1
        for j = 2 : N+1
            detu(i, j) = temp(i+1, j) + temp(i-1, j) + temp(i, j+1) + temp(i, j-1) - 4*temp(i, j);
        end
    end
    temp = temp + alpha * (lambda * detu - temp + I);
    imshow(uint8(temp(2:M+1,2:N+1)))
    title(['带保真项K=',num2str(k)]);
    pause(0.03);
end