% Project 2
% Andy Doran
%
% function wien(name,xdim,No)
%
% name = 'input image' (as in 'lenna.256')
% xdim = x dimension of input image (usually 256 or 512)
% No   = Variance of Noise
%
% This function takes an input image, runs it through a LSI filter h,
% and adds Gaussian noise to it.  The MSE between the degraded and
% original image is then calculated.  Weiner Filtering is then performed
% (using known Suu, Snn, and h) and the degraded image is restored using
% the filter.  The MSE between the restored and original image is then
% calculated and returned.
%

function wien(name,xdim,No,blur)


% Load and Plot image
pict = freadbin(name,xdim,xdim);

clf
subplot(221)
imagesc(pict);
colormap(gray);
txt = [num2str(name) ' before degradation'];
title(txt)
axis square
axis off

% Create LSI degradation model, need it to be phaseless
hi = 3.5^(-2);
h = zeros(256);
xl = 4;
xh = xdim - xl + 2;
h(1:xl,1:xl) = hi;
h(xh:xdim,1:xl) = hi;
h(1:xl,xh:xdim) = hi;
h(xh:xdim,xh:xdim) = hi;
% Create Gaussian noise, mean = 0, variance comes from input (No)
noise = sqrt(No)*randn(xdim,xdim);

% Run image through LSI Filter and then add noise
dpict = distimag(pict,h,noise);

% Plot degraded image
subplot(222)
imagesc(dpict);
colormap(gray);
txt = [num2str(name) ' with additive Gaussian Noise (mean=0, var=' , num2str(No), ')'];
title(txt)
axis square
axis off

% Calculate MSE of degraded image
error = dpict - pict;
sqerr = sum(sum(error.^2));
DMSE = sqerr/(xdim^2)

% Calculate power spectral density of input image
PICT = fft2(pict);
Suu = abs(PICT).^2;

% Calculate power spectral density of the AGN
NOISE = fft2(noise);
Snn = abs(NOISE).^2;

% Calculate Fourier Transform of LSI Filter
H = fft2(h,xdim,xdim);
H2 = abs(H).^2;

% Plot H to see type of filter we have
subplot(223)
imagesc(fftshift(abs(H)));
colormap(gray);
txt = ['Frequency Spectra of blurring filter'];
title(txt)
axis square
axis off

% Calculate thresholded 1/H
HINV = H.^(-1);
index = find(abs(H) < .2);
hzeros = length(index)       % Return number of elements below threshold
HINV(index) = 0;

% Calculate Wiener Filter
G = HINV.*(H2.*Suu)./((H2.*Suu) + Snn);

% Restore Image
DPICT = fft2(dpict);
RPICT = DPICT.*G;
rpict = ifft2(RPICT);

% Plot restored image
subplot(224)
imagesc(abs(rpict));
colormap(gray);
txt = [num2str(name) ' restored using Wiener Filter, known h, Snn, Suu'];
title(txt)
axis square
axis off

% Calculate MSE of restored image
error = abs(rpict) - pict;
sqerr = sum(sum(error.^2));
RMSE = sqerr/(xdim^2)

% Calculate MSE of restored image through Wiener filtering
block = [16 16];
wpict = wiener2(dpict,block);
error = abs(wpict) - pict;
sqerr = sum(sum(error.^2));
WMSE = sqerr/(xdim^2)

