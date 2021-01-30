function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%Link to tutorial: https://www.coursera.org/learn/machine-learning/discussions/all/threads/a8Kce_WxEeS16yIACyoj1Q
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% y = 5000 *1
% X = 500*401 - Once I have added the ones
% Theta1 = 25*401
% Theta2 = 10*26
% Add ones to the X's
X = [ones(size(X,1),1), X];
% first need to calculate all the nodes values for X
%z2 = 5000*401 * 401*25 = 5000*25
z2 = X*Theta1';
%add the bias unit
a2 = [ones(size(z2,1),1), sigmoid(z2)];
% calculate predictions
%z3 = 10*26 * 26*5000 = 10*5000
%z3 = 5000*26 * 26*10 = 5000*10
z3 = a2 * Theta2';
%Last hypothesis = h(X) a3=5000*10 - each row is the probability of each one
a3 = sigmoid(z3);
%J1 = 
%Need to compare the 1*10 of a3 with the 1*1 number of Y, penalise any value
% that is 
% I need to change the number into a vector with that index with a 1 nad all other
%with 0's
real_y = zeros(size(a3));
for i = 1:length(y)
  real_y(i,y(i)) = 1;
end
%Instead of doing the above without a for loop is
%Create an example of each different labels vector (using eye)
%eye_matrix = eye(num_labels)
%For every y value add it to a new matrix as the corresponding row in the y vector
%This works becuase when you index a vector they go through all the vector elements (just like the for loop above)
%y_matrix = eye_matrix(y,:)
%real_y = 5000*10
%a3 = 5000*10
%the end goal should be 1*1 - right now its 10*10
%First matrix calculation = 10*5000(real_y') * 5000*10 = 10*10
%Second matrix calculation = 
%Don't need to transpose as element wise multiplication
J = (1/m) * (((-real_y).*(log(a3)))-(1-(real_y)) .* log( 1 - a3));
J = sum(sum(J));
%What its sayng is calculate the cost difference for every activation node
%for every training example and give us the sum
%We only need the main diagonal as the other multiplications don't match the y vector
%and corresponding a3 - they are offset by some amount and so are useless.
% -------------------------------------------------------------
%Now regularisation
% This was the old regularisation J = J + ((lambda/(2*m)) * sum(theta.^2));
% =========================================================================
% Have to sum up all of the Theta1 and Theta2 squared, without bias units
% Theta1 = 25*401
% Theta2 = 10*26
%Make bias units 0
Theta1_temp = Theta1;
Theta2_temp = Theta2;
Theta1_temp(:,1) = 0;
Theta2_temp(:,1) = 0;
%Square everything
Theta1_temp=Theta1_temp.^2;
Theta2_temp = Theta2_temp.^2;
%Sum all the elements of theta
theta_value = sum(sum(Theta1_temp)) + sum(sum(Theta2_temp));
%Scale it lamda/2m
theta_value = (lambda*theta_value)/(2*m);
%Add it to the cost - That's regularisation!
J = J + theta_value;

%Backpropogation
%Sizes:
%z2 = 5000*25
%a2 = 5000*26
%a3 = 5000 *10
%d3 = 5000*10
%d2 = 5000*25
%delta 3 calculated m*k - m*k
d3 = a3 - real_y;
d3*Theta2(:,2:end);
%Sigmoid gradient of (z2(m*h))
SGz2 = sigmoidGradient(z2);
%Calculating d2 m*k * k*h = m*h
d2 = (d3 * Theta2(:,2:end)).* SGz2;
%Calculateing Delta1 (capital) h*m * m*n = h*n
Delta1 = d2' * X;
%Calculating Delta2 r*m * m*h+1 = r*h+1
Delta2 = d3' * a2;
%Delta1 = 25*401
%Delta2 = 10*26
Theta1_grad = Delta1 /m ;
Theta2_grad = Delta2 / m;

%Regularisation
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = Theta1_grad+((lambda.*Theta1)./m);
Theta2_grad = Theta2_grad+((lambda.*Theta2)./m);



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
