function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % theta = 2(n - number of features)-dimensional vector (2 x 1 matrix)
    % x = m * 2(n) matrix
    % y = m * 1 matrix
    % Need to calculate the new theta 1
    % Involves Getting the old theta 
    % Subtracting alpha * 1/m * difference in predictions
    % Calculating difference in predictions (m * 2(n) * 2(n) * 1 = m * 1)
    predictions = X * theta;
    % Calculating difference between predictions and real answer
    %  (m vector - m vector = m vector)
    prediction_offset = predictions - y;
    % Multiplying by the corresponding column in the X matrix 
    %(but do both thetas at the same time - will save each theta as a value in the vector created)
    % transposed to be able to multiply them together (1 * m * m * 1 = 1 * 1) (:,1)
    prediction_difference = X' * prediction_offset;
    % To get sum of all elements first have to turn prediction_difference into a column vector
    % difference = sum(prediction_difference(:))
    % Subract alpha * 1/m
    learning_rate = alpha / m;
    difference_to_subtract = prediction_difference * learning_rate;
    %Subtract difference to corresponding theta
    theta = theta - difference_to_subtract;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
