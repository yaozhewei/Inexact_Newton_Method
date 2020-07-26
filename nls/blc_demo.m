function blc_demo(filename, max_props)
%
% a demo of comparison among different methods for solving nonlinear least
% square problems
%
% filename   --- different dataset
% max_props: maximum number of propagations
%
% written by Zhewei Yao, Peng Xu, Fred Roosta, 07/25/2020


clc; close all; rehash;

addpath(genpath(pwd));
seed = 1234;
rng(seed);

if nargin < 2
    max_props = 1e8;
end

[type, X, Y, Xt, Yt, num_class] = readData(filename);

[n,d] = size(X);
Y(Y~=1) = 0;
Yt(Yt~=1) = 0;

%%
problem.loss = @compute_blc_loss;
problem.grad = @compute_blc_gradient;
problem.hessian = @compute_blc_hessian_diag;
problem.get_gn = @compute_blc_gn_diag;
lambda = 0;
problem.lambda = lambda;


w0 = randn(d,1) /sqrt(d);


problem.w0 = w0;
gradient_sample_factor = 10;
uniform_factor = 1;
nonuniform_factor = 1;
max_delta = Inf; 
delta = 10;
sigma = 10;
min_sigma = 1e-6; 

constantlipg = 1e-1; 
constantsigma = 1; 

lineWidth = 2;
legendLocation = 'bestoutside';

dir_name = ['./figs/blc_result/', filename, '/'];

if ~exist(dir_name, 'dir')
    status = mkdir(dir_name);
end

gradient_sample_size = ceil(gradient_sample_factor*n/100);
uniform_sample_size = ceil(uniform_factor*n/100);
rns_sample_size = ceil(nonuniform_factor*n/100); 

% TR method
methods{1} = struct('name',sprintf('TR Full Delta (%g)', delta),'method','Newton-TR', 'hessian_size',n, 'step_size',1,...
    'max_props', max_props,'delta', delta, 'max_delta', max_delta,'solver','Steihaug');
methods{2} = struct('name',sprintf('TR Uniform (%g%%) Delta (%g)', uniform_factor, delta), ...
    'method','Uniform-TR', 'hessian_size', uniform_sample_size, 'step_size',1,...
    'max_props', max_props,'delta', delta,'max_delta', max_delta,'solver','Steihaug');
methods{3} = struct('name',sprintf('TR Adap Uniform (%g%%) Uniform SUBH(%g%%)',gradient_sample_factor, uniform_factor), ...
    'method','Uniform-TR','gradient_size', ceil(gradient_sample_size/2), 'hessian_size', uniform_sample_size, 'step_size', 1, ...
    'max_props', max_props,'delta', delta,'max_delta', max_delta,'solver','Steihaug', 'adaptive', 1);
% ARC method
methods{4} = struct('name',sprintf('ARC FULL Sigma (%g)', sigma),'method','Newton-ARC', 'hessian_size',n, 'step_size',1,...
    'max_props', max_props,'sigma',sigma, 'min_sigma', min_sigma,'constantlipg', constantlipg, 'solver', 'cubic_solver');
methods{5} = struct('name',sprintf('ARC Uniform (%g%%) Sigma (%g)', uniform_factor, sigma),'constantlipg', constantlipg, ...
    'method','Uniform-ARC', 'hessian_size', uniform_sample_size, 'step_size',1,...
    'max_props', max_props,'sigma',sigma, 'min_sigma', min_sigma, 'solver', 'cubic_solver');
methods{6} = struct('name',sprintf('ARC Adap Uniform (%g%%) Uniform SUBH (%g%%) Sigma (%g)',gradient_sample_factor, uniform_factor, sigma), ...
    'method','Uniform-ARC', 'gradient_size', ceil(gradient_sample_size/2), ... 
    'hessian_size', uniform_sample_size, 'step_size',1,...
    'max_props', max_props,'sigma',sigma, 'min_sigma', min_sigma, 'solver', 'cubic_solver', 'adaptive', 1);


% GD ARC method
methods{7} = struct('name',sprintf('GDCR Uniform (%g%%) Uniform SUBH (%g%%) Step Size (%g), Lip Constant (%g)',gradient_sample_factor, uniform_factor,constantlipg, constantsigma), ...
    'method','Uniform-ARC', 'gradient_size', gradient_sample_size, ... 
    'hessian_size', uniform_sample_size, 'step_size',1,...
    'max_props', max_props,'sigma',sigma, 'min_sigma', min_sigma, 'constantlipg', constantlipg, ...
    'constantsigma', constantsigma, 'solver', 'cubic_solver_gd');

% Fixed Lanzos algorithm
methods{8} = struct('name',sprintf('CR Uniform (%g%%) Uniform SUBH (%g%%) Lip Constant (%g)',gradient_sample_factor, uniform_factor,constantsigma), ...
    'method','Uniform-ARC', 'gradient_size', gradient_sample_size, ... 
    'hessian_size', uniform_sample_size, 'step_size',1,...
    'max_props', max_props,'sigma',sigma, 'min_sigma', min_sigma, 'constantlipg', constantlipg, ...
    'constantsigma', constantsigma, 'solver', 'cubic_solver_fixed');

% for i = [1, 2, 3, 4, 5, 6, 7, 8]
for i = [4, 5, 6, 7, 8]
    rng(seed);
    disp([methods{i}.name, ':']);
    savename = sprintf([dir_name,'all_%s_%s.mat'],filename,methods{i}.name);
    if exist(savename,'file')
        load(savename,'result');    
    else
        [w, result] = subsampled_blc(X,Y,problem, methods{i});
        tr_acc = blc_eval(X,Y, result.sol);
        save(savename,'result');
    end
    figure(1)
    semilogx(result.noProps, n*result.l,'DisplayName', methods{i}.name, 'LineWidth', lineWidth ); hold on

end


%%
figure(1)
grid('on')
% legend('Full TR', 'SubH TR', 'Inexact TR', 'Full ARC', 'SubH ARC', 'Inexact ARC', 'SCR (Lanczos)', 'SCR (GD)', 'Location', 'SouthWest')
legend('Full ARC', 'SubH ARC', 'Inexact ARC', 'SCR (Lanczos)', 'SCR (GD)', 'Location', 'SouthWest')
set(gca, 'fontsize', 24)
hold off;
xlabel('# of Props')
ylabel('training loss')
title({sprintf('%s', filename)});

saveas(figure(1),sprintf([dir_name,'all_%s_noProps_tr_loss'],filename),'fig');
saveas(figure(1),sprintf([dir_name,'all_%s_noProps_tr_loss'],filename),'png');
saveas(figure(1),sprintf([dir_name,'all_%s_noProps_tr_loss'],filename),'pdf');




