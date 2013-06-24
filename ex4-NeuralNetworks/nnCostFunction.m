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
% Add ones to the X data matrix
X = [ones(m, 1) X];

% =============================================================

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

K = num_labels;

Y = eye(K)(y,:); % [5000, 10]

a1 = X; % results in [5000, 401]
z2 = a1*Theta1';
a2 = sigmoid(a1*Theta1'); % results in [5000,25]
a2 = [ones(size(a2, 1),1) a2]; % results in [5000,26]
a3 = sigmoid(a2*Theta2'); % results in [5000,10]
h = a3;

costPositive = -Y .* log(h);
costNegative =  (1 - Y) .* log(1 - h);
cost = costPositive - costNegative;

J = (1/m) * sum(cost(:));

Theta1Filtered = Theta1(:,2:end);
Theta2Filtered = Theta2(:,2:end);
reg = (lambda / (2*m)) * (sumsq(Theta1Filtered(:)) + sumsq(Theta2Filtered(:)));
J = J + reg;

Sigma3 = a3 - Y;
Sigma2 = (Sigma3*Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]))(:, 2:end);

Delta1 = Sigma2'*a1;
Delta2 = Sigma3'*a2;

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;


% -------------------------------------------------------------

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1Filtered);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2Filtered);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
