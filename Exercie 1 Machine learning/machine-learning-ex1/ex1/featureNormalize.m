function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
% size(X,2) is returning a scalar of number of elements in 2nd dimentsion (columns) 
mu = zeros(1, size(X, 2));
% creates a matrix of all zeros that is row n (columns of x)-dimensional vector
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
% To normalise X I need to get the standard deviation and mean of every column
for n_feature = 1:length(mu)
  % Store both of those in horizontal vector (with n values) - created above
  mu(n_feature) = mean(X(:,n_feature))
  sigma(n_feature) = std(X(:,n_feature))
  % apply them (subtract mean) then divid by sigma to X_norm to make X_norm
  X_norm(:,n_feature) = (X_norm(:,n_feature) - mu(n_feature)) / sigma(n_feature)
end






% ============================================================

end
