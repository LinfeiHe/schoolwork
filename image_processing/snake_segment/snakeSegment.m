clear
clc

%%
img = imread('test.jpg');
I = double(rgb2gray(img));
I = imresize(I,[200,200],'bicubic');
[width, height] = size(I);
sigma = 0.01;
H = fspecial('gaussian',[3,3],sigma);
G = imfilter(I, H, 'replicate');
[dGx, dGy] = gradient(G);
g = 1 ./ (1 + dGx.^2 + dGy.^2);

% parameter configuration
N = 100; % number of points
M = 2000000; % iterative times
R = 80;
x = zeros(N,1);
y = zeros(N,1);
gg = zeros(N,1);

[x0, y0] = deal(width/2, height/2);

dgx = zeros(N,1);
dgy = zeros(N,1);
% initialize the curve
for i = 1:N
    x(i) = round(R*cos(2*pi/N*i) + x0);
    y(i) = round(R*sin(2*pi/N*i) + y0);
end
% start
figure(1);
imshow(uint8(I));

for k = 1 : M
    dx = diff([x(N);x]);
    ddx = diff([dx;dx(1)]);
    dy = diff([y(N);y]);
    ddy = diff([dy;dy(1)]);
    Np = [-dy ./ sqrt(dx .^2 + dy .^2), dx./ sqrt(dx .^2 + dy .^2)];
    K = (dx .* ddy - dy .* ddx) ./ power(dx.^2 + dy .^2, 1.5);
    for i = 1 : N
        x1 = round(x(i));
        y1 = round(y(i));
        gg(i) = g(y1, x1);
        dgx(i) = (g(x1+1, y1) - g(x1-1, y1))/2;
        dgy(i) = (g(x1, y1+1) - g(x1, y1-1))/2;
    end
    x = x + gg .* K .* Np(:,1) + sum([dgx,dgy].*Np,2).* Np(:,1);
    y = y + gg .* K .* Np(:,2) + sum([dgx,dgy].*Np,2).* Np(:,2);
    
%     imshow(uint8(I));
%     hold on
%     plot(y, x, 'b.');
%     title(['迭代了K=',num2str(k)]);
%     pause(0.00001);
%     hold off;
    
    if (k==800000)% 设定停止代数并输出图像
        imshow(uint8(I));
        hold on
        plot(y, x, 'b.');
        title(k);
        pause(0.001);
        hold off;
    end
end