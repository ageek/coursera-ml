function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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

%size(Theta1)
%size(Theta2)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Theta1: 25 x 401
% Theta2: 10 x 26

K = num_labels;

%convert y to vectorized form for NN
% un-vectorized y: 5000 x 1
y_vec = zeros(m, num_labels);
for j=1:m;
	y_vec(j, y(j)) = 1;
end;

size(y_vec);

% Input/1st layer: 20x20 = 400 pixels + 1 bias unit = 401
% Hidden/2nd layer: 25 units
% Output/3rd layer: 10 units

% a1 = [1 X]   -- input with bias unit - shape: 401 x 25
a1 = [ones(m, 1) X];   % 5000 x 401
size(a1);

% a2 = sigmoid(a1*Theta1') : 5000 x 25
a2 = sigmoid(a1 * Theta1');  % 25x5000
size(a2);
a2 = [ones(size(a2,1), 1) a2];   % 5000 x 26


% h_theta = a3 = sigmoid(z3) : 5000 x 10
h_theta = sigmoid(a2 * Theta2');  % 5000 x 10
size(h_theta);

% we want the output to be 5000 x 10 [and not 5000x1, as y is a vector of 5000x10]
% for 5000 input examples
% and we don't want matrix multiplication of y_vec and log(h_theta)
% its just the element-wise multiplication, we want
J = 1/m * sum(sum(-y_vec .* log(h_theta) - (1 - y_vec) .* log(1-h_theta)));


% Calculate regularization 
Theta1_reg = Theta1;
Theta1_reg(:,1)=0;

Theta2_reg = Theta2;
Theta2_reg(:,1)=0;

regularization = (lambda/ (2*m)) * (sumsqr(Theta1_reg) + sumsqr(Theta2_reg));
J = J + regularization;

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
% forward pass
a1 = [ones(m, 1) X];   % 5000 x 401
z1 = a1 * Theta1';
a2 = sigmoid(z1);  % 25x5000

a2 = [ones(size(a2,1), 1) a2];   % 5000 x 26
z2 = a2 * Theta2';
a3 = sigmoid(z2);  % 5000 x 10
%h_theta=a3

% for accumulating deltas in each iteration
DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));

for i=1:m;
	% pick one example from X's 5000 entries
	% a1:  1x401
	a1 = X(i,:);
	a1 = [1, a1];
	%z2: 1x25
	z2 = a1 * Theta1';
	a2 = sigmoid(z2);
	%a2: 1x26
	a2 = [1, a2];
	%z3:  1x10
	z3 = a2 * Theta2';
	%a3: 1x10
	a3 = sigmoid(z3);
	
	% calculate delta : 1 x 10
	delta_3 = a3 - y_vec(i,:);
	
	%remove bias unit from delta_3
	delta3_x_Theta2 = delta_3 * Theta2;
	delta3_x_Theta2 = delta3_x_Theta2(:, 2:end);
	delta_2 = delta3_x_Theta2 .* sigmoidGradient(z2);

	
	% size of Theta2_grad: 10 x 26, so size of DELTA should be 10x26
	DELTA2 = DELTA2 + delta_3' * a2;
	
	% size of Theta1_grad: 25 x 401
	DELTA1 = DELTA1 + delta_2' * a1;
	
end;

% Actual Grad is DELTA * (1/m)
Theta1_grad = DELTA1 * ( 1/m);
Theta2_grad = DELTA2 * ( 1/m);

size(Theta1_grad);
size(Theta2_grad);	

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Theta1: 25 x 401
% Theta2: 10 x 26

Theta1_no_bias = Theta1;
Theta1_no_bias(:,1) = 0;
Theta1_grad_reg = (lambda/m) * Theta1_no_bias ;   % 25 x 401

Theta2_no_bias = Theta2;
Theta2_no_bias(:,1) = 0;
Theta2_grad_reg = (lambda/m) * Theta2_no_bias ;    % 10 x 26


Theta1_grad =  Theta1_grad + Theta1_grad_reg;
Theta2_grad =  Theta2_grad + Theta2_grad_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
