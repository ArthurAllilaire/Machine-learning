function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% theta = 2-dimensional vector
% x = m * 2 matrix
% y = m * 1 matrix
% Calculating predictions
predictions = X * theta;
% Calculating difference between predictions and real answer
prediction_offset = predictions - y;
% Calculating the square of the difference 
prediction_offset = prediction_offset.^2;
% Calculating the sum of the differences (takin it from matrix to scalar
J = sum(prediction_offset);
% Getting J 
J = 1/(2*m) * J;




% =========================================================================

end
