function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Option 1, write out everything
h_theta = sigmoid(X * theta)

% new J = non-regularized J + regularization
theta(1) = 0
J = 1/m * (-y' * log(h_theta) - ( 1-y') * log(1-h_theta)) + ( lambda/(2*m) * theta' * theta )

% new grad = old grad + regularization for theta_1 onwards i.e. make theta_0=0 and keep the rest 2:end
grad = 1/m * (X' * (h_theta - y )) +  (lambda/m * theta)




% Option 2: use the existing costFunction() for non-regularized J and grad
%[J, grad] = costFunction(theta, X, y)

% new J = old cost J + regularization
% for matrix square  A^2 = A' * A in matlab notations
% we dont regularize the theta_0 i.e. intercept term
%theta(1) = 0
%J = J + ( lambda/(2*m) * theta' * theta )

% new grad = old grad + regularization for theta_1 onwards, make theta_0=0 and keep the rest 2:end
%grad = grad + (lambda/m * theta)




% =============================================================

end
