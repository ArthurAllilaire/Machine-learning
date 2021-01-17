function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%Theta = n * 1
% X = m * n
% p = m * 1
% For every value in X find the 
% The easier calculation is this: m*n * n*1 = m*1
% Theta' * X' = 1*n * n*m = 1 *m matrix (once transposed = p)
p = sigmoid(X * theta);
for i = 1:length(p),
  if p(i) >= 0.5,
    p(i) = 1;
  else
    p(i) = 0;
  endif
end



% =========================================================================


end