function [sol, results] = subsampled_blc(X,Y,problem,params)
% subsample newton solver with full gradient for the following problem:
%       min_w sum_i l(w'*x_i,y_i) + lambda*norm(w)^2
% In this solver, we also add LBFGS, GD,AGD methods for comparison.
%
% input:
%       X,Y                     ---- data matrices
%       problem:
%           problem.loss        ---- loss, get_grad, get_hessian
%           problem.grad        ---- function to compute gradient
%           problem.hessian     ---- function to compute the diagonal D^2
%           problem.lambda      ---- ridge parameters
%           problem.w0          ---- initial start point (optional)
%           problem.w_opt       ---- optimal solution (optinal)
%           problem.l_opt       ---- minimum loss (optional)
%           problem.condition   ---- condition number (optional)
%       params:
%           params.method         ---- hessian sketch schemes
%           params.gradient_size  ---- sketching size for gradient  -zhewei:new
%           params.hessian_size   ---- sketching size for hessian
%           params.step_size      ---- step size
%           params.mh             ---- Hessian approximation frequency
%           params.constantlipg           ---- lipschitz constant of gradient
%           params.constantsigma          ---- lipschitz constant of hessian

%
% output:
%       sol ---- final solution
%       results:
%           results.t       ---- running time at every iteration
%           results.err     ---- solution error (if problem.w_opt given)
%           results.l       ---- objective value
%           results.sol     ---- solution at every iteration
%           results.oracle  ---- number of computation of gradient
%
%
%
% written by Zhewei Yao, Peng Xu, Fred Roosta, 07/25/2020



if nargin == 3
    params = 0;
end

loss = problem.loss;
get_grad = problem.grad;
get_hessian = problem.hessian;

[n,d] = size(X);

% default setting
lambda = 0;
method= 'Uniform';
niters = 1e4;            % total number of iterations
linesearch = false;
eta = 1;                % step size
w0 = zeros(d,1);        % initial point
delta = 1;
max_delta = Inf;
sigma = 1/delta;
min_sigma = 1e-6;
eta1 = 0.8;
eta2 = 1e-4;
gamma1 = 2;
gamma2 = 1.2;
solver = 'Steighaug';
max_props = 1e9;
max_iter = 500;
% check params
if isfield(problem, 'lambda')
    lambda = problem.lambda;
end

if isfield(problem, 'w0')
    w0 = problem.w0;
end

if isfield(params, 'method')
    method = params.method;
end

if isfield(params, 'gradient_size')   %zhewei:new
    gs = params.gradient_size;
end

if isfield(params, 'hessian_size')
    s = params.hessian_size;
end

if isfield(params, 'step_size')
    eta = params.step_size;
end

if isfield(params,'niters')
    niters = params.niters;
end

if isfield(params,'max_props')
    max_props = params.max_props;
end

if isfield(params, 'beta0')
    beta0 = params.beta0;
end

if isfield(params, 'delta')
    delta = params.delta;
end
if isfield(params, 'max_delta')
    max_delta = params.max_delta;
end

if isfield(params, 'sigma')
    sigma = params.sigma;
end
if isfield(params, 'min_sigma')
    min_sigma = params.min_sigma;
end
if isfield(params, 'constantlipg')
    constantlipg = params.constantlipg;
end
if isfield(params, 'constantsigma')
    constantsigma = params.constantsigma;
end


if isfield(params, 'eta1')
    eta1 = params.eta1;
    gamma1 = params.gamma1;
end

if isfield(params,'eta2')
    eta2 = params.eta2;
    gamma2 = params.gamma2;
end

if isfield(params,'solver')
    solver = params.solver;
end

if isfield(params,'linesearch')
    linesearch = params.linesearch;
    eta0 = eta;
end

w = w0;
t = zeros(niters,1);
sol = zeros(d, niters);
noProps = zeros(1,niters);
noOracles = zeros(1,niters);

noGs = zeros(1,niters);

nopProps_sofar = 1;
nopOracles_sofar =1; %zhewei

% algorithm start
fprintf('algorithm start ......\n');
tic;


rnorms = sum(X.^2,2);

%%%%%%%%% for adaptive gradient size
grad_norm_old = 0.;
delta_old = 0.;
%%%%%%%%%

for i = 1:niters
    % compute hessian
    switch method
        case {'Newton-TR', 'Newton-ARC'}
            D2 = get_hessian(X,Y,w);
            H = X'*bsxfun(@times, D2, X) + lambda*eye(d);H = (H + H')/2;
            hessianSamplesSize = size(X,1);
            
        case {'Uniform-TR', 'Uniform-ARC'} 
            XY = datasample([X,Y], s, 1, 'Replace', false);
            X_sub = XY(:,1:end-1);
            D2_sub = get_hessian(X_sub,XY(:,end),w);
            H = X_sub'*bsxfun(@times,D2_sub,X_sub)+lambda*eye(d);H = (H + H')/2;
            hessianSamplesSize = size(X_sub,1);
    end
    
    % compute gradient
    if isfield(params, 'gradient_size')   
        XY = datasample([X,Y], gs, 1, 'Replace', false);
        X_sub = XY(:, 1:end-1);
        Y_sub = XY(:, end);
        c = get_grad(X_sub, XY(:,end),w);
        grad = X_sub'*c + lambda*w; 
        %%%%%%%%% for adaptive gradient size
        grad_norm = norm(grad);
        if isfield(params, 'adaptive')
            if grad_norm_old == 0
                grad_norm_old = grad_norm;
            else
                if grad_norm > (1.2*grad_norm_old)
                    gs = ceil(max(gs / 1.2, 4999/5));
                    grad_norm_old = grad_norm;
                elseif grad_norm < (grad_norm_old/1.2);
                    gs = ceil(min(gs * 1.2, 4999*5));
                    gs = ceil(gs * 1.2);
                    grad_norm_old = grad_norm;
                else
                    gs = gs;
                end

            end
        end
        noGs(i) = gs;
        %%%%%%%%%
        nopProps_sofar = nopProps_sofar + 2*gs;
        nopOracles_sofar = nopOracles_sofar + gs;

    else
        c = get_grad(X,Y,w);
        grad = X'*c + lambda*w;
        nopProps_sofar = nopProps_sofar + 2*size(X,1);
        nopOracles_sofar = nopOracles_sofar + size(X,1);
    end

    switch method

        case {'Uniform-TR', 'Newton-TR'} 
            assert(eta == 1);
            fail_count = 0;
            tr_loss = loss(X,Y,w) + lambda * norm(w)^2;


            while true
               
                steihaugParams = [1e-15, max_iter, 0]; % parameters for Steighaug-CG
                if fail_count == 0
                    z0 = randn(d,1);
                    z0 = 0.99*delta*z0/norm(z0);
                else
                    z0 = [];
                end
                [z,m, num, iflag] = cg_steihaug (@(x)H*x, grad, delta, steihaugParams, z0 );
                nopProps_sofar = nopProps_sofar + num*2*hessianSamplesSize;
                nopOracles_sofar = nopOracles_sofar + num*hessianSamplesSize;
               
                if m>=0
                    z = 0;
                    break
                end

                newll = loss(X,Y, w + z);
                newll = newll + 0.5 * lambda * norm(w + z)^2;
                nopProps_sofar = nopProps_sofar + size(X,1);

                rho = (tr_loss - newll)/-m;
                if rho < eta2
                    fail_count = fail_count + 1;
                    delta = delta/gamma1;
                    z = 0;
                elseif rho < eta1
                    delta = min(max_delta, gamma2*delta);
                    break;
                else
                    delta = min(max_delta, gamma1*delta);
                    break;
                end

                if fail_count == 3
                    z=0;
%                     fprintf('Bad batch, choose another one\n')
                    break
                end
            end

        case {'Uniform-ARC', 'Newton-ARC'}
            assert(eta == 1);
            fail_count = 0;
            tr_loss = loss(X,Y,w) + lambda * norm(w)^2;
            
            switch solver
                case 'cubic_solver'
                    while true
                        [z,m, num] = cubic_solver(H,grad,sigma,max_iter,1e-12);

                        nopProps_sofar = nopProps_sofar + num*2*hessianSamplesSize;
                        nopOracles_sofar = nopOracles_sofar + num*hessianSamplesSize;
                        if m >= 0
                            z = 0;
                            break;
                        end
                        newll = loss(X,Y, w + z);
                        newll = newll + 0.5 * lambda * norm(w + z)^2;
                        nopProps_sofar = nopProps_sofar + size(X,1);
                        rho = (tr_loss - newll)/-m;

                        if rho ==0 && norm(grad) < 1e-6
                             sigma = max(min_sigma, sigma/gamma1);
                             break;
                        end

                        if rho < eta2
                            sigma = sigma*gamma1;
                            fail_count = fail_count + 1;
                            z = 0;
                        elseif rho < eta1
                            sigma = max(min_sigma, sigma/gamma2);
                            break;
                        else
                            sigma = max(min_sigma, sigma/gamma1);
                            break;
                        end

                        if fail_count == 3
                            z=0;
                            break
                        end

                    end
                case 'cubic_solver_gd'
                    tolarc = 1e-12;
                    [z, num, m] = cubic_solver_gd(H,grad,constantlipg,constantsigma,10,tolarc);

                    nopProps_sofar = nopProps_sofar + num*2*hessianSamplesSize;
                    nopOracles_sofar = nopOracles_sofar + num*hessianSamplesSize;
                    if m >= -0.01 * sqrt(tolarc^3/constantsigma)
                        z = cubic_solver_gdfinal(H,grad,constantlipg,constantsigma,250,tolarc);
                        nopProps_sofar = nopProps_sofar + num*2*hessianSamplesSize;
                        nopOracles_sofar = nopOracles_sofar + num*hessianSamplesSize;
                        break
                    end
                case 'cubic_solver_fixed'
                        [z,m, num] = cubic_solver(H,grad,constantsigma,max_iter,1e-12);
                        nopProps_sofar = nopProps_sofar + num*2*hessianSamplesSize;
                        nopOracles_sofar = nopOracles_sofar + num*hessianSamplesSize;
            end

    end

    alpha_prev = eta;
    w = w + eta*z;

    sol(:,i) = w;
    t(i) = toc;
    noProps(i) = nopProps_sofar;
    noOracles(i) = nopOracles_sofar;

    if nopProps_sofar >= max_props
        break;
    end
end
fprintf('main algorithm end\n');
iters = i;
% better improve this using vector operations
fprintf('Further postprocessing......\n')
t = [0;t(1:iters)];
sol = [w0,sol(:,1:iters)];
results.t = t;
results.sol = sol;
results.noProps = [1,noProps(1:iters)];
results.noOracles = [1, noOracles(1:iters)];
results.gs = noGs;
grads = zeros(iters+1,1);
l = zeros(iters+1,1);
for i = 1:iters+1
    w = sol(:,i);
    l(i) = (loss(X,Y,w) + lambda*(w'*w)/2)/n;
    c = get_grad(X,Y,w);
    grad = X'*c + lambda*w;
    grads(i) = norm(grad);
end
results.l = l;
results.grads = grads;
if isfield(problem, 'w_opt')
    w_opt = problem.w_opt;
    err = bsxfun(@minus, sol, w_opt);
    err = sqrt(sum(err.*err))';
    results.err = err;%/norm(w_opt,2);
end

if isfield(params,'name')
    results.name = params.name;
end
fprintf('DONE! :) \n');
end

