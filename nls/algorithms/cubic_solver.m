function [z,m,num] = cubic_solver(H,g,sigma,max_iters,tol)
% Generialzied Lanczos method for solving the cubic regularization
% subproblem:
%           minimize g'x + 1/2 x'Hx + sigma/3 x'Hx
% This is different implementation from the original algorithm. Here we
% first run all the lanczos iteration then solve the low-dimensional cubic
% problem with a tridiagonal matrix.
%
% input:
%       H,g,sigma           ---- cubic problem
%       max_iters           ---- number of Lanczos iterations
%       tol                 ---- error tolarance
% output:
%       z                   ---- solution
%       m                   ---- objective value
%       num                 ---- number of Hessian-vector product
%
% written by Zhewei Yao, Peng Xu, Fred Roosta, 07/25/2020

d = size(g,1);
K = min(d, max_iters);
Q = zeros(d,K);

q = g + randn(d,1); q = q/norm(q);
T = zeros(K+1,K+1);
beta = 0;
q0 = 0;
tol = min(tol, tol*norm(g));
%
for i = 1:K
    Q(:,i) = q;
    v =H*q;% H(q);
    alpha = q' * v;
    T(i,i) = alpha;
    r = gsorth(v, Q(:,1:i));
    beta = norm(r);
    T(i,i+1) = beta;
    T(i+1,i) = beta;
    
    if beta < tol
        break;
    end
    q0 = q;
    q = r/beta;
end

T = T(1:i,1:i);
Q = Q(:,1:i);
num = i;

if norm(T) < tol && norm(g) < eps
    z = zeros(d,1);
    m = 0;
    return;
end

gt = Q'*g;

options = optimoptions(@fminunc,'SpecifyObjectiveGradient',true,...
    'Algorithm','quasi-newton','Display','off','OptimalityTolerance',tol);
z0 = zeros(i,1);
[z, m, flag, output] = fminunc(@(z)cubic_prob(@(x)T*x,gt,sigma, z), z0,options);
z = Q*z;

end

function [f,g] = cubic_prob(Hess,grad,sigma,z)
f = grad'*z + 1/2 * z'*Hess(z) + 1/3*sigma*norm(z)^3;
g = grad + Hess(z) + sigma * norm(z) *z;
end

function hv = HvFunc(v,x, Hess,sigma)
znorm = norm(x);
hv = Hess(v) + sigma*(x'*v/znorm*x + znorm*v);
end


function out = gsorth(v, M)
for i = 1 : size(M,2)
    v = v - v' * M(:, i) * M(:, i);
    out = v;
end

end
