% Project 2
% Andy Doran
%
% function jain(name,xdim,No,Snn,block)
%
% name  = 'input image' (as in 'lenna.256')
% xdim  = x dimension of input image (usually 256 or 512)
% No    = Variance of Noise
% Snn   = Power Spectral Density of Noise (Assumed White)
% block = Degraded image is broken up into block-by-block chunks
%         when computing Restoration Filter denominator
%
% This function takes an input image, runs it through a LSI filter h,
% and adds Gaussian noise to it.  The MSE between the degraded and
% original image is then calculated.  Weiner Filtering is then performed
% and the degraded image is restored using the filter.  The MSE between
% the restored and original image is then calculated and returned.
% Suu in the Wiener filter is assumed known (necessary for algorithm).
% Snn is set to a constant (assumes white noise).  h is estimated
% by breaking up the original image and the degraded image into blocks
% and using an approximation of the log of the FT of our blurring process.
% This method comes from Jain pp. 322-323 (eq 8.226).
% 
% The blurring filter is calculated using h = ones(4,4)/4^2;

function jain(name,xdim,No,Snn,block)

% Load image
pict = freadbin(name,xdim,xdim);

% Create LSI degradation model, need it to be phaseless
hi = 3.5^(-2);
h = zeros(256);
xl = 4;
xh = xdim - xl + 2;
h(1:xl,1:xl) = hi;
h(xh:xdim,1:xl) = hi;
h(1:xl,xh:xdim) = hi;
h(xh:xdim,xh:xdim) = hi;


% Plot abs(fft(h)) to see type of filter we have
H = fft2(h,xdim,xdim);
maximag = max(max(imag(H)))     % Make sure there are no complex parts

clf
subplot(222)
imagesc(fftshift(abs(H)));
txt = ['Magnitude of blurring filter'];
title(txt)
axis square
axis off

% Create Gaussian noise, mean = 0, variance comes from input (No)
noise = sqrt(No)*randn(xdim,xdim);

% Run image through LSI Filter and then add noise
dpict = distimag(pict,h,noise);

% Plot degraded image
subplot(221)
imagesc(dpict);
txt = ['Blurred 'num2str(name) ' with AGN (mean=0, var=' , num2str(No), ')'];
title(txt)
axis square
axis off

% Calculate MSE of degraded image
error = dpict - pict;
sqerr = sum(sum(error.^2));
DMSE = sqerr/(xdim^2)

% Calculate power spectral density of input image for Numerator
PICT = fft2(pict);
Suu = abs(PICT).^2;

% Estimate |H| for Restoration Filter
Hest = ones(xdim);    % Initialize denominator
iter = xdim/block;          % Get number of iterations for loops
M = iter^2;                % Calculate total number of blocks

for i = 0:(iter-1)
  for j = 0:(iter-1)
    x1 = (i*block) + 1;
    x2 = (i+1)*block;
    y1 = (j*block) + 1;
    y2 = (j+1)*block;
    V = fft2(dpict(x1:x2,y1:y2),xdim,xdim);   % Zero Pad out to xdim
    U = fft2(pict(x1:x2,y1:y2),xdim,xdim);   % Zero Pad out to xdim
    Hest = Hest.*((abs(V)./abs(U))).^(1/M);
  end
end


% Calculate thresholded 1/Hest
HINV = Hest.^(-1);
index = find(abs(Hest) < .1);
hzeros = length(index)       % Return number of elements below threshold
HINV(index) = 0;
H2 = Hest.^2;

% Calculate Wiener Filter
G = HINV.*(H2.*Suu)./((H2.*Suu) + Snn);

% Plot Estimated H Magnitude
subplot(224)
imagesc(fftshift(abs(Hest)));
txt = ['Estimated Degradation Filter Spectra'];
title(txt)
axis square
axis off

% Restore Image
DPICT = fft2(dpict);
RPICT = DPICT.*G;
rpict = ifft2(RPICT);

% Plot restored image
subplot(223)
imagesc(abs(rpict));
txt = [num2str(name) ' restored using Blind Deconv from Jain'];
title(txt)
axis square
axis off

% Calculate MSE of restored image
error = abs(rpict) - pict;
sqerr = sum(sum(error.^2));
RMSE = sqerr/(xdim^2)
