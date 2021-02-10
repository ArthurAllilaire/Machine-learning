function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
%For 1 to k (every centroid)
for i = 1:K
  %Find all the affiliated x values
    %idx == 1 turns all values that aren't the K cluster to zeros
    %find then returns the index of all the 1 values
  place = find(idx==i);
  %compute the sum of all affiliated X values 
    %Get only affiliated X values by multiplying each X example by corresponding idx == 1 example
    %sum all of them (leaving a 1*n vector) this is done by the below matrix multiplication
    %X = m*n, place = m*1    n*m * m*1 = n*1
    
  centroid = X'*(idx==i);
  %then divide by number of examples assigned to the cluster (average)
  %length(nonzeros(place)) returns a vector of 1's Ck*1 
  %where Ck is number of X affiliated to the corresponding centroid
  centroids(i,:) = centroid/length(place);
%End result should be a 1*n vector and is stored in the centroid matrix
endfor






% =============================================================


end

