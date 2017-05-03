%% clear all
clear
clc

%% gradient method
% step1: configuration
X0 = [1,1];         % initial point
E = 0.000001;        % error coefficient
syms x y;
X = [x, y];         % variables
f = cos(x) * cos(y);
G = jacobian(f, X);

% step2: start iteration
k = 0;
syms m;
while(1)
    g = double(subs(G, X, X0));
    x = X0 - m .* g;
    h = diff(subs(f, X, x),'m');
    a = double(solve(h,'m'));
    x = X0 - a .* g;
    yX = double(subs(f, X, X0));
    yx = double(subs(f, X, x));
    if(abs(yX - yx) <= E)
        break;
    else
        X0 = x;
    end
    k = k + 1;
end
k
double(x)
double(subs(f, X, x))