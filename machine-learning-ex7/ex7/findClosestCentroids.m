function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
%By default assign to cluster 1 (for later on in code)
idx = ones(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%X = m*n
%Initialising cost for later
cost = 0;
% loop over all training examples
for i = 1:length(idx)
  X_example = X(i,:);
  %For each training examples loop over all possible centroids
  for j = 1:K
    %Calculate the cost of particular centroid
    cost_temp = sum((X_example - centroids(j,:)) .^ 2);
    %Don't have anything to compare the first example to so automatically assign it 
    %as the lowest cost value, cluster 1 default (see above formation of idx)
    if j == 1
      cost = cost_temp;
    else
      % see if cost_temp is smaller than previous smallest cost
      if cost_temp<cost
        %Set cost to new lower cost
        cost = cost_temp;
        %Assign X_example to new clusteroid
        idx(i) = j;
      endif
    endif
  endfor





% =============================================================

end

