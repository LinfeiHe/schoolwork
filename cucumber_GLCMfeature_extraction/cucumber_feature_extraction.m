clear;clc;
%% initial
SOURCE = 'leaf1.png';
GLCM_NUMLEVEL = 32;                         % set GLCM gray level
MIN_REGION_ELEMENT = 80;                    % set minimum elements for compute

% output dir
[~, FILE_NAME, ~]= fileparts(SOURCE);
mkdir(FILE_NAME);

%% segment to subimage
img = imread(SOURCE);
img_gray = rgb2gray(img);
imhist(img_gray)
bw = im2bw(img_gray,graythresh(img_gray));
bw = imfill(~bw,'holes');
[L, NUM] = bwlabel(bw);                     % label the regions
stats = regionprops(L,'BoundingBox');       % segmentation

% get features
file_location = sprintf('./%s/log.txt', FILE_NAME);
file_log = fopen(file_location, 'w');

color_moment = zeros(size(stats,1), 9);
glcm_contrast = zeros(size(stats,1), 4);
glcm_corr = zeros(size(stats,1), 4);
glcm_energy = zeros(size(stats,1), 4);
glcm_entropy = zeros(size(stats,1), 4);
for k = 1:size(stats,1)
    bb = stats(k).BoundingBox;
    piece = imcrop(img_gray,bb);
    piece_rgb = imcrop(img,bb);
    [m, n] = size(piece);
    if m * n > MIN_REGION_ELEMENT
        % save subimage
        file_location = sprintf('./%s/%03d.jpg', FILE_NAME, k);
        imwrite(piece, file_location);
        % GLCM
        r = 256 / GLCM_NUMLEVEL;
        temp = piece ./ r;
        offsets = [0 1;-1 1;-1 0;-1 -1];    % 0, 45, 90, 135 degree
        GLCMS = graycomatrix(temp, 'Of', offsets, 'NumLevels', GLCM_NUMLEVEL);
        for l = 1:4                         % four direction
            ux = mean(GLCMS(:, :, l));
            uy = mean(GLCMS(:, :, l), 2);
            sigmax = var(GLCMS(:, :, l));
            sigmay = var(GLCMS(:, :, l)');
            P = double(GLCMS(:, :, l)) ./ sum(sum(GLCMS(:, :, l)));
                % Contrast
                for i = 1:GLCM_NUMLEVEL
                    for j = 1:GLCM_NUMLEVEL
                        glcm_contrast(k, l) = glcm_contrast(k, l) + abs(i - j) * P(i, j);
                    end
                end
                % Correlation
                for i = 1:GLCM_NUMLEVEL
                    for j = 1:GLCM_NUMLEVEL
                        glcm_corr(k, l) = glcm_corr(k, l) + ((i * j) * P(i ,j) - ux * uy);
                    end
                end
                glcm_corr(k, l) = glcm_corr(k, l) ./ (sigmax * sigmay');
                % Energy
                for i = 1:GLCM_NUMLEVEL
                    for j = 1:GLCM_NUMLEVEL
                        glcm_energy(k, l) = glcm_energy(k, l) +  power(P(i, j), 2);
                    end
                end
                % Entropy
                for i = 1:GLCM_NUMLEVEL
                    for j = 1:GLCM_NUMLEVEL
                        if P(i, j) == 0
                            continue;
                        end
                        glcm_entropy(k, l) = glcm_entropy(k, l) -  P(i, j) * log(P(i, j));
                    end
                end
        end
        % Color Moment based on RGB
        [count_r, ~] = imhist(piece_rgb(:,:,1));
        [count_g, ~] = imhist(piece_rgb(:,:,2));
        [count_b, ~] = imhist(piece_rgb(:,:,3));
        count_rgb = [count_r, count_g, count_b];
        color1 = mean(count_rgb);
        color2 = std(count_rgb) / power(m * n, 1 / 2);
        color3 = std(count_rgb) / power(m * n, 1 / 3);
        color_moment(k,:) = [color1, color2, color3];
    else
        string = sprintf('label %03d with too small size(%d, %d), has been dropped\n', k, m, n);
        fprintf(file_log, string);
    end
end
fclose(file_log);

%% drawing and save
% label original
f1 = figure(1);
imshow(img);
for k = 1:size(stats,1)
    bb = stats(k).BoundingBox;
    string = int2str(k);
    text(bb(1), bb(2), string, 'Color', 'blue');  
    hold on
end
file_location = sprintf('./%s/original_labelled.jpg', FILE_NAME);
saveas(f1, file_location);

% save color moment and GLCM features
file_location = sprintf('./%s/ColorMoment.mat', FILE_NAME);
save(file_location, 'color_moment');
file_location = sprintf('./%s/GLCM_Contrast.mat', FILE_NAME);
save(file_location, 'glcm_contrast');
file_location = sprintf('./%s/GLCM_Corr.mat', FILE_NAME);
save(file_location, 'glcm_corr');
file_location = sprintf('./%s/GLCM_Energy.mat', FILE_NAME);
save(file_location, 'glcm_energy');
file_location = sprintf('./%s/GLCM_Entropy.mat', FILE_NAME);
save(file_location, 'glcm_entropy');

