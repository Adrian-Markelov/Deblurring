function x_prox = fista_tv_proximity_color(b, lambda, opt)

siz = size(b);
p_k = zeros(siz);
q_k = zeros(siz);
r_k = zeros(siz);
s_k = zeros(siz);
t_k = 1;

funDT = @(a, b) naive_reverse_diff(a, b);
funD = @(a) naive_forward_diff(a);

for kk=1:opt.max_iter
    
    res = b - lambda*funDT(r_k, s_k);
    %res = max(0, res);
    
    [tx, ty] = funD(res);
    
    [p_k1, q_k1] = shrinkage_isotropic(r_k + tx/(8*lambda), s_k + ty/(8*lambda));
    
    t_k1 = (1 + sqrt(1+4*t_k^2))/2;
    
    r_k = p_k1 + ((t_k-1)/t_k1)*( p_k1 - p_k);
    s_k = q_k1 + ((t_k - 1)/t_k1)*(q_k1 - q_k);
    
    if (opt.verbosity)
        if mod(kk, 5) == 1
            err = norm(vec(p_k - p_k1))+norm(vec(q_k - q_k1));
            
            fprintf('%d %2.3f \n', kk, err);
            imshow(res); drawnow
        end
    end
     
    
    t_k = t_k1;
    p_k = p_k1;
    q_k = q_k1;
    
    
    
end

x_prox = b - lambda*funDT(p_k, q_k);
%x_prox = max(0, x_prox); %%%%%%%

end

function [px, py] = shrinkage_isotropic(gx, gy)

g_norm = sqrt(sum(gx.^2+gy.^2, 3));

g_norm = repmat(g_norm,[1 1 size(gx, 3)]);

px = gx./max(1, g_norm);
py = gy./max(1, g_norm);


end
    

function [y_x, y_y] = naive_forward_diff(x)

y_x = x(:, 2:end, :)-x(:, 1:end-1, :); y_x(:,end+1, :) = 0;
y_y = x(2:end, :, :)-x(1:end-1, :, :); y_y(end+1, :, :) = 0;

end


function x = naive_reverse_diff(y_x, y_y)
y_x(:, end, :) = 0;
y_y(end, :, :) = 0;

x = circshift(y_x,[0 1])-y_x + circshift(y_y, [1 0])-y_y;

end


