function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)
%LEARNINGCURVERANDOM Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
% this is the bit where you increase m and see how it affects the cost
%Need to get the average of 50 randomly selected samples for each value of i
for i = 1:m
  %Create two vectors to store all of the values for the iteration of 50 samples
    random_train = zeros(50,1)
    random_val = zeros(50,1)
  %outside the loop so that we can record the values
    for j = 1:50    
    %Create the random sets for train and cross_val then use them to get the corresponding examples
    %Get a vector of random digits that is in m and that is i units long
    random_num_train = int32(rand(i,1).*m)
    X_temp = X(random_num_train,:);
    y_temp = y(random_num_train);
    random_num_val = int32(rand(i,1).*m)
    Xval_temp = Xval(random_num_train,:);
    yval_temp = yval(random_num_train);
    %Train theta on the random set
    theta = trainLinearReg(X_temp,y_temp,lambda);
    %Use that to get cost for the random training set - no regularisation 
    [J, grad] = linearRegCostFunction(X_temp, y_temp, theta, 0);
    random_train(i) = J;
    %Use that to get cost for the random cross val set - no regularisation 
    [J_val, grad] = linearRegCostFunction(Xval_temp, yval_temp, theta, 0);
    random_val(i) = J_val;
  endfor
  %Average out the values registered
  error_train(i) = sum(random_train)/50
  error_val(i) = sum(random_val)/50
  
  
  
  
  
end


end
