function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


mu =  (1/m) * sum(X);

% Option 1 : Vectorized implementation 
% we need to subtract vector of 1x2 from matrix of 307 x2 
% i.e subtract mu vector from each row of matrix
sigma2 = 1/m * sum((bsxfun(@minus,X,mu)) .^2);		

% Option 2: using for loop
%sigmaSum = zeros(1,n);
%for i=1:m;
%	x = X(i,:);
%	dif = (x - mu).^2;
%	sigmaSum = sigmaSum + dif;
%end;

%sigma2 = (1/m) .* sigmaSum';







% =============================================================


end
