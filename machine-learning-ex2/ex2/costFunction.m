function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
% y = m * 1 matrix
% X = m * n matrix
% Theta = n * 1 matrix

% List of the operations undertaken and in order by computer
%Transpose of y = (1 * m) matrix
% predictions = Transpose of Theta * X' = 1*n .* n*m = 1*m
% log(predictions') = m*1 matrix
%The easier calculation is: X * theta = m*n * n*1 = m*1
% -y * predictions = 1*m * m*1
% y' (1 * m) .* h(X) (m *n) = (1 * n)
% scalar (1 * 1 matrix) * 
predictions = sigmoid(X*theta);
J = (1/m) * (((-y')*(log(predictions)))-(1-(y')) * log( 1 - predictions));

%h(theta)-y = 1 * n matrix
% sigmoid(theta' * x) = 1 * n * m *n matrix =  - y (m *1 matrix) 
% (predictions(m*1) - y(m*1)) * X = 1*m * m*n = 1 * n
% X'(n*m) * prediction_difference(m*1) = n*1
grad = 1/m .* (X' * (predictions - y));






% =============================================================

end
