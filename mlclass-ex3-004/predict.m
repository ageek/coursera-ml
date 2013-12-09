function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% add column of 1's to X
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

	% size(X) : 5000 x 401
	% size(Theta1) : 25 x 401
	% size(Theta2) : 10 x 26
	
	% get the activations for 1st hidden layer
	% size(a) : 5000 x 25
	a1 = sigmoid( X * Theta1');
	
	% add a bias column to a
	a1_m = size(a1,1);
	a1 = [ones(a1_m, 1) a1];
	
	% get the activations for OUTPUT Layer
	% size(a1) : 5000 x 26
	% size(Theta2): 10 x 26
	% a2 is the actual probability for output layer
	% we need to pick the class with maximum value
	a2 = sigmoid(a1 * Theta2')
	
	% now we need to pick the max and its index from each row
	[C, I] = max(a2')
	p = I'
	
	








% =========================================================================


end
