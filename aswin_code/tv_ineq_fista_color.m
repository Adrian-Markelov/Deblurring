function x_k = tv_ineq_fista_color(b, funA, funAT, lambda, L, opt, x0)

siz = size(funAT(b));
t_k = 1;

if exist('x0', 'var')
    y_k = x0;
else
    y_k = zeros(siz);
end
x_k = zeros(siz); F_xk = norm(b)^2;

for iter = 1:opt.max_iter
    
    res = y_k - (2/L)*funAT(funA(y_k)-b);
    
    z_k = tv_shrinkage(res, 2*lambda/L, opt);
    
    t_k1 = (1 + sqrt(1 + 4*t_k^2))/2;
    
    F_zk = norm(b - funA(z_k))^2 + 2*lambda*tv_cost(z_k, opt);
    
    if (F_zk < F_xk)
        x_k1 = z_k;
        F_xk = F_zk;
    else
        x_k1 = x_k;
        F_xk = F_xk;
    end
    
    y_k = x_k1 + (t_k/t_k1)*(z_k - x_k1) + ((t_k-1)/t_k1)*(x_k1 - x_k);
    
    if opt.verbosity
        if mod(iter, opt.verbosity) == 1
            imshow(x_k, []) 
            title(sprintf('%d %1.3f', iter, 2*norm(x_k(:) - x_k1(:))/(norm(x_k(:))+norm(x_k1(:))) ));
            drawnow
        end
        if (mod(iter, 10) == 0)
            save junk.mat x_k
        end
    end
    
    x_k = x_k1;
    t_k = t_k1;
    

    
    
end

end

function z = tv_shrinkage(b, lamb, opt)

opt1.max_iter = opt.inner_iterations;
opt1.verbosity = 0;
z = fista_tv_proximity_color(b, lamb, opt1);

end

function tval = tv_cost(x, opt)
[gx, gy] = naive_forward_diff(x);
tval = sum(sum(sqrt(sum(gx.^2+gy.^2, 3))));
end


function [y_x, y_y] = naive_forward_diff(x)

y_x = x(:, 2:end, :)-x(:, 1:end-1, :); y_x(:,end+1, :) = 0;
y_y = x(2:end, :, :)-x(1:end-1, :, :); y_y(end+1, :, :) = 0;

end