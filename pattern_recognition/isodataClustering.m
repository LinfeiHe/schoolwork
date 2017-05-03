clear,clc;
% ISODATA clustering algorithm apply for iris_data classfication

load iris_data.mat;
x = features;
%x = [0 0;0 1;4 4;4 5;5 4;5 5;1 0];

K = 3;          % target numbers of classes
minNum = 40;     % minimum numbers of elements
sigmaT = 1;     % threshold of division
minDis = 1;     % minimum Euclidean distance between two classes
I = 100;       % iterative times

% step1: set the initial numbers of classes, clustering center
c=5;   
z = x(1:c, :);
% start
k = 0;
for k = 1 : I
	Z = cell(1,c);
    % setp2: get the Euclidean distance
    for i = 1 : length(x)
        d = [];
        for j = 1 : c
        d(j) = sqrt(sum((x(i,:) - z(j,:)).^2));
        end
        [~, index] = min(d);
        Z{index} = [Z{index}; x(i,:)];
    end
    
    % step3: if the elements of the class is less than minNum,go to step1
    n = [];
    for i = 1 : c
        n(i) = size(Z{i}, 1);   % number of elements in per class
    end
    flag = sum(n < minNum);
    if ( flag )
        c = c - flag;
        z = x(1:c, :);
        continue;
    end
    
    % step4: modify the clustering center
    temp2 = 0;
    for i = 1 : c
        z(i,:) = mean(Z{i});
        temp1 = 0;
        for j = 1 : n(i)
            temp1 = temp1 + sqrt(sum((Z{i}(j,:) - z(i,:)).^2));
        end
        Dc(i) = temp1 / n(i);           % class average diatance
        temp2 = temp2 + Dc(i) * n(i);
    end
    D = temp2 / size(x,1);              % total average distance
    
    % step5: division
    if ( (k < I && c <= K/2) || (k < I && mod(k,2) == 1 && c < 2*K && c > K/2))
		sigma = [];
        for i = 1 : c
			temp = 0;
			for j = 1 : n(i)
				temp = temp + (Z{i}(j,:) - z(i,:)).^2; 
			end
			sigma(i,:) = (temp ./ n(i)).^0.5;
		end
        sigmaMax = max(sigma,2);
		for i = 1 : c
			if (sigmaMax(i) > sigmaT)
				if ((Dc(i) > D & n(i) > 2*(minNum + 1)) || c < K/2)
					r = 0.5 * sigmaMax(i);
					z(i,:) = z(i,:) + r;
					z(c+1, :) = z(i,:) - r;
                    c = c + 1;
				end
			end
        end
        continue;
    % step6: assimulation          
    elseif (k == I || ( k < I && mod(k,2) == 0 || c >= 2*K && c <= K/2))
        if ( k == I)
            minDis = 0;
        end    
		% the distance of arbitrary two classes
        kk = 1;
        for i = 1 : c-1
            for j = i+1 : c
                Dij(kk,1) = sqrt(sum((z(i,:)-z(j,:)).^2));
                Dij(kk,2) = i;
                Dij(kk,3) = j;
                kk = kk + 1;
            end
        end
        temp3 = sortrows(Dij(find(Dij(:,1) < minDis),:), 1);
        if (size(temp3,1) >= 2)
            for i = 1 : size(temp3,1)
                z(c + 1,:) = n(temp3(i, 2)) * z(temp3(i, 2),:) + n(temp3(i, 3)) * z(temp3(i, 3),:) / n(temp3(i, 2)) + n(temp3(i, 3));
                z(temp3(i, 2),:) = [];
                z(temp3(i, 3),:) = [];
                c = c - 1;
            end
        end
        continue;
    end
end
Z{:}

accuracy = (sum((ceil(find(ismember(x,Z{1},'rows'))/50)==3)) + sum((ceil(find(ismember(x,Z{2},'rows'))/50)==1)) + sum((ceil(find(ismember(x,Z{3},'rows'))/50)==2))) / 150
