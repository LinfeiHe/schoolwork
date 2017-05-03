clear,clc;
% K-mean clustering algorithm apply for iris_data classfication
load iris_data.mat;
x = features;
%x = [0 0;0 1;4 4;4 5;5 4;5 5;1 0];

z1 = x(1,:);
z2 = x(2,:);
z3 = x(3,:);
Z1 = [];
Z2 = [];
Z3 = [];
for k = 1 : 100
    temp1 = z1;
    temp2 = z2;
    temp3 = z3;
    for i = 1 : length(x)
        d1 = sqrt(sum((x(i,:) - z1).^2));
        d2 = sqrt(sum((x(i,:) - z2).^2));
        d3 = sqrt(sum((x(i,:) - z3).^2));
        if(d1 == min([d1,d2,d3]))
            Z1 = [Z1; x(i,:)];
        elseif(d2 == min([d1,d2,d3]))
            Z2 = [Z2; x(i,:)];
        else
            Z3 = [Z3; x(i,:)];
        end
    end
    z1 = mean(Z1);
    z2 = mean(Z2);
    z3 = mean(Z3);
    if (temp1 == z1 & temp2 == z2 & temp3 == z3)
        break;
    end
    Z1 = [];
    Z2 = [];
    Z3 = [];
end
k
accuracy = (sum((ceil(find(ismember(x,Z1,'rows'))/50)==3)) + sum((ceil(find(ismember(x,Z2,'rows'))/50)==2)) + sum((ceil(find(ismember(x,Z3,'rows'))/50)==1))) / 150