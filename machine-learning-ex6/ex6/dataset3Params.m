function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
possible_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
predict_accuracy = zeros(length(possible_values));
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval)) - double is a way of storing numbers
%Makes it more presice - doesn't impact anything - get the mean of all of the
%times that we predicted it wrong - High mean would mean many of them were one
%therefore lots of them were wrong - so when x100 is percentage that it got wrong.
for i = 1:length(possible_values)
  C_temp = possible_values(i);
  for j=1:length(possible_values)
    sigma_temp = possible_values(j);
    model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    predictions = svmPredict(model, Xval);
    predict_accuracy(i,j) = mean(double(predictions ~= yval));
  endfor
  
end
predict_accuracy
%Get the minimum value and set the sigma value corresponding to the row it was found at
[accuracy, locationS] = min(min(predict_accuracy));%By default goes minimum value of columns
sigma = possible_values(locationS);

%Get the minimum value and set the C value corresponding to the column it was found at
[accuracy, locationC] = min(min(predict_accuracy, [], 2));
C = possible_values(locationC);
 






% =========================================================================

end
