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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Input = [ones(m, 1), X];
A2 = sigmoid(Input * Theta1'); % 5000 * 25
ones_A2 = [ones(m, 1), A2];
A3 = sigmoid(ones_A2 * Theta2'); % 5000 * 10

new_Y = zeros(m, size(Theta2, 1));

for i = 1:size(y, 1)
    value = y(i);
    new_Y(i, value) = 1;
endfor



t1 = sum(sum(Theta1(:, 2:size(Theta1, 2)) .^ 2));
t2 = sum(sum(Theta2(:, 2:size(Theta2, 2)) .^ 2));

J = sum(sum(-1 / m * (new_Y .* log(A3) + (1 .- new_Y) .* log(1 .- A3)))) + lambda / (2 * m) * (t1 + t2);

% For loop implementation
%G1 = zeros(size(Theta1));
%G2 = zeros(size(Theta2));

%for i = 1:m
%    a1 = Input(i, :);
%    z2 = Theta1 * a1';
%    a2 = [1; sigmoid(z2)];
%    z3 = Theta2 * a2;
%    a3 = sigmoid(z3);
%
%    delta3 = a3 - new_Y(i, :)';
%    n_T2 = Theta2(:, 2:end);
%    delta2 = n_T2' * delta3 .* sigmoidGradient(z2);
%
%    G2 = G2 + delta3 * a2';
%    G1 = G1 + delta2 * a1;
%endfor

% Vectorized implementation
n_Theta1 = [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
n_Theta2 = [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];

delta3 = A3' - new_Y';
n_T2 = Theta2(:, 2:end);
delta2 = n_T2' * delta3 .* sigmoidGradient((Input * Theta1')');

G2 = delta3 * ones_A2;
G1 = delta2 * Input;
%
%
Theta1_grad = 1 / m .* G1 + lambda / m .* n_Theta1;
Theta2_grad = 1 / m .* G2 + lambda / m .* n_Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end