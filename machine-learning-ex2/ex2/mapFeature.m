function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
% out = X1(column 1) vector of 1's 
out = ones(size(X1(:,1)));
for i = 1:degree
  % Every time you repeat the 6 times do below once more (i +1 times)
    for j = 0:i
      %Adding one more element to out = 
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end