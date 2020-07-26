function [x, num_iter, m] = cubic_solver_gd(H,g,lipg,sigma,max_iters,tol)

% Gradient descend method for solving the cubic regularization
% subproblem:
%           minimize g'x + 1/2 x'Hx + sigma/3 x'Hx
% This is different implementation from the original algorithm. Here we
% first run all the lanczos iteration then solve the low-dimensional cubic
% problem with a tridiagonal matrix.
%
% input:
%       H,g,sigma           ---- cubic problem
%       lipg                ---- lipschitz constant of gradient
%       max_iters           ---- number of gradient descent iterations
%       tol                 ---- error tolarance
% output:
%       z                   ---- solution
%       m                   ---- objective value
%       num                 ---- number of Hessian-vector product
%
% written by Zhewei Yao, Peng Xu, Fred Roosta, 07/25/2020


size_grad = size(g);
step_size = 1. / (20. * lipg);

g_norm = norm(g);
num_iter = 0;

if g_norm >= lipg^2/sigma
    tmp = g'*H*g/(g_norm^2*sigma);
    RC = -tmp + sqrt(tmp^2+4*g_norm/sigma);
    x = -RC * g / g_norm/2;
    num_iter = num_iter + 1;
    m = x' * g + 1/2 * x'*H*x + sigma/3 * norm(x)^3;
else
    x = zeros(size_grad);
    step_size = 1. / (20. * lipg);
    g_tilde = g;
    
    for i=1:max_iters
        x = x - step_size*(g_tilde + H * x + sigma * norm(x) * x);
        num_iter = num_iter + 1;
    m = x' * g + 1/2 * x'*H*x + sigma/3 * norm(x)^3;
    num_iter = num_iter + 1;
    end
end
end
