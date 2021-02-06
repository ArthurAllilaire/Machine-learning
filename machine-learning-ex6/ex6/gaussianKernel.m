function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%both x's have dimensions n*1
%Distance between x1 and x2
distance = x1-x2;
%Get the sum of the distance squared
total_distance = sum(distance.^2);
%Divide the total_distance by 2sigma squared
f1 = -(total_distance/(2*sigma^2));
%Get e to the power of f1
sim = e^f1;





% =============================================================
    
end
