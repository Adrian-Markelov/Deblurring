clear all
close all

img = imread('cameraman.tif');
img = double(img)/255;

x0 = img(128, :);
plot(x0); N = length(x0);


fil = randn(1, 15); fil = fil/norm(fil); M = length(fil);

y0 = conv(x0, fil, 'full');

len = (M+N-1)*2+1;
Fy0= fft(y0, len);
Ffil = fft(fil, len);
Fxhat = Fy0./Ffil;
xhat = real(ifft(Fxhat)); xhat = xhat(1:N);
hold on
plot(xhat+0.1,'r');
