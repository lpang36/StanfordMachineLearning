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

H = sigmoid(X*theta);
Z = -1.*y.*log(H)-(-1.*y.+1).*log(-1.*H.+1);
J1 = 1/m*sum(Z);
J = J1+lambda/2/m*sum(theta(2:end,:).^2);
A = repmat(H-y,1,size(X,2)).*X;
grad = 1/m*sum(A,1)';
grad(2:end,:) = grad(2:end,:)+lambda/m*theta(2:end,:);




% =============================================================

end
