function b = color_convolution(x, mode, k0, siz)
if length(siz) == 2
    siz(3) = 1;
end

if (mode == 1)
    x = reshape(x, siz);
    for ch=1:siz(3)
        b(:, :, ch) = conv2(x(:, :, ch), k0, 'valid');
    end
    b = b(:);
else
    x = reshape(x, siz-[size(k0)-1 0]);
    for ch=1:siz(3)
        b(:, :, ch) = conv2(x(:, :, ch), k0(end:-1:1, end:-1:1), 'full');
    end
end