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
%

% now J_All = all the squared errors in a matrix of shape: num_movies x num_users
% we need to sum up all costs only for the corresponding R(i,j) = 1
J_all = 1/2 * (X*Theta' - Y).^2;

%lets remove/zero-out all the costs where R(i,j) = 0
% shape of new J_all : num_movies x num_users
J_all = J_all .* R;

%now sum up all the remaining costs[this is just the ones for which R(i,j)=1, rest are zeroed-out]
J = sum(sum(J_all));

% gradient w.r.to X
% size: num_movies x num_users   X  num_users  x num_features  = num_movies  x num_features
% zeroed-out with .* R
X_grad = ((X*Theta' - Y) .* R) * Theta;


% gradient w.r.t Theta
% zeroed-out with .* R ; final shape: num_users x num_features
Theta_grad = ((X*Theta' - Y) .* R)' * X;



% Add regularization for X and Theta, to un-regularized J
X_reg = (lambda/2.0) * sumsqr(X);
Theta_reg = (lambda/2.0) * sumsqr(Theta);
J = J + X_reg + Theta_reg;


% Add regularization for gradients, X_grad and Theta_grad
X_grad_reg = lambda .* X ;
X_grad = X_grad + X_grad_reg;

Theta_grad_reg = lambda .* Theta;
Theta_grad = Theta_grad + Theta_grad_reg;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
