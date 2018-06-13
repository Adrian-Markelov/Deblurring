clear all
close all

im0 = imread('peppers.png');
im0 = double(im0)/255;
siz = size(im0);

k0 = zeros(10, 10);
rp = randperm(100, 15);
k0(rp) =1/15;

fA = @(x) color_convolution(x, 1, k0, siz);
fAT = @(x) color_convolution(x, 0, k0, siz);


b0 = fA(im0);
b = b0+randn(size(b0))/100;
lambda = .001;
L = 100; %largest eigenvalue of A operator

% epsilon = norm(b)/100;
opt.max_iter = 1000;
opt.inner_iterations = 1;
opt.verbosity = 2;
x_k = tv_ineq_fista_color(b, fA, fAT, lambda, L, opt);
imshow([img x_k])

% 
% epsilon = norm(b)/100;
% opt.max_iter = 1000;
% opt.inner_iterations = 1;
% opt.verbosity = 2;
% 
% opt.beta1 = 1; opt.beta2 = 1; opt.eta = 1.5;
% opt.cgs_tol = 1e-6; opt.cgs_iterations = 100;
% opt.tv_opt.max_iter = 100; opt.tv_opt.verbosity = 0;
% x_k = tv_admm(fA, fAT, b, epsilon, opt);