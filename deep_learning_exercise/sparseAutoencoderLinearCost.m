function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

[~, m] = size(data);
z2 = bsxfun(@plus, W1 * data, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2 * a2, b2);
h = z3;
p = sum(a2, 2) ./ m;

sparsity = sum(sparsityParam.*log(sparsityParam./p)+(1-sparsityParam).*log((1-sparsityParam)./(1-p)));
cost = 0.5 / m * sum(sum((h-data).^2)) + 0.5 * lambda * (sum(sum(W1.^2))+sum(sum(W2.^2))) + beta .* sparsity; 

delta3 = (h - data);
temp = beta .* (-sparsityParam./p + (1-sparsityParam)./(1-p));
delta2 = bsxfun(@plus, W2' * delta3, temp);
delta2 = delta2 .* a2 .* (1 - a2);
W2grad = delta3 * a2'./ m + lambda.*W2;
b2grad = sum(delta3, 2)./ m;
W1grad = delta2 * data'./ m + lambda.*W1;
b1grad = sum(delta2, 2)./ m;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end