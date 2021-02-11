function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%Get the squared prediction difference - includes non-rated items (set to 0)
  %num_movies*num_features * num_features*num_users = num*movies*num_users = y
pred_dif = (X*Theta' - Y).^2;
%Get only the values that have a movie rating from user i
pred_dif = pred_dif.*R;
%Get the sum and * 1/2
J = sum(sum(pred_dif)) / 2;
%Regularisation
%Theta regularization
Theta_cost = sum(sum(Theta.^2)) * (lambda/2);

%Theta regularization
X_cost = sum(sum(X.^2)) * (lambda/2);
J = J + Theta_cost + X_cost;
%Gradient 

%Same as above - get prediction difference then get only values we are interested in
pred_dif = X*Theta' - Y;
%movies that have been rated
pred_dif = pred_dif.*R;
%Multiply by theta to get derivative for X
  %nm*nu * nu*nf = nm*nf
  %we only want to multiply pred_dif by theta values (users) that have rated the films
  %this is done already as when multiplying the two matrices any movies
  %that don't have ratings from user i will be multiplied by 0
X_grad = pred_dif*Theta;

%same for Theta derivative
  %end goal is a nu*nf
  %done through: nu*nm * nm*nf
Theta_grad = pred_dif'*X;

%Regularisation of Gradient
X_grad = X_grad + (lambda.*X);
Theta_grad = Theta_grad + (lambda.*Theta);










% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
