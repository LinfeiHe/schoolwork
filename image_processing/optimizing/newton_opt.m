%% clear all
clear
clc

%% newton method for minimizing function

% step1: configuration
X0 = [2,3];      % initial point
E = 0.00001;        % error coefficient
N = 1000;          % iterative times
syms x1 x2;
X = [x1, x2];       % variables
y = x1 + x2 / ((x1 * x2) - 1);
G = jacobian(y, X);
H = hessian(y, X);

% step2: start iteration
yX = zeros(N, 1);
yx = zeros(N, 1);
for k = 1 : N
    g = double(subs(G, X, X0));
    h = double(subs(H, X, X0));
    if(det(h)==0)
        h = eye(2);
    else
        h = inv(h);
    end
    x(k,:) = X0 - (h * (g'))';
    yX= double(subs(y, X, X0));
    yx(k) = double(subs(y, X, x(k,:)));
    if((yX - yx(k)) <= E)
        break;
    else
        X0 = x(k,:);
    end
end

%% draw
k
double(x(k,:))
double(subs(y, X, x(k,:)))

