function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% theta = 2-dimensional vector
% x = m * 2 matrix
% y = m * 1 matrix
% Calculating predictions
predictions = X * theta;
% Calculating difference between predictions and real answer
prediction_offset = predictions - y;
% Calculating the square of the difference 
pred_cost = prediction_offset.^2;
% Calculating the sum of the differences (takin it from matrix to scalar
J = sum(pred_cost);
% Getting J 
theta(1) = 0;
J = J/(2*m);
J = J + (lambda*sum(theta.^2))/(2*m);

grad = X' * (prediction_offset/m);
grad = grad + ((lambda*theta)/m);









% =========================================================================

grad = grad(:);

end
