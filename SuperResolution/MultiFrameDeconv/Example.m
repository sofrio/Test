% This is a simulated example showing how to run the algorithm described
% in paper "Deconvolving PSFs for A Better Motion Deblurring using
% Multiple Images", ECCV 2012.
% Authors: Xiang Zhu, Filip Sroubek, Peyman Milanfar
%
% Written by Filip Sroubek and Xiang Zhu.
% All rights reserved.

% Load latent image and PSFs.
load kernel_data
im = imread('I21.bmp');
im = im(1:end,65:end-64,:);  % Crop the image.

% Add Gaussian noise.
sigma = 2;
randn('state',1);
im1 = imfilter(double(im),h{1},'same','conv')+sigma*randn(size(im));
im2 = imfilter(double(im),h{2},'same','conv')+sigma*randn(size(im));
im1 = im1(11:end-10,11:end-10,:);
im2 = im2(11:end-10,11:end-10,:);

% Estimate kernels from Y channel.
im1g = rgb2gray(im1/255)*255;
im2g = rgb2gray(im2/255)*255;
% Input image group, which may contain more than 2 images
ims_input = {im1g,im2g};
h_size = 27;

% Preliminary PSF estimation.
[im_output1, h_init] = MCRestoration(ims_input, h_size);

% PSF refinement.
[h_est, k_common] = findComKerfromH(h_init, 0.001, 1,[]);
h1 = h_est{1};
h1(h1<0)=0;
h2 = h_est{2};
h2(h2<0)=0;
hs = {h1./sum(h1(:)),h2./sum(h2(:))};

% Non-blind deconvolution.
parameters;
im_output = fftCGSRaL(ims_input, hs, PAR);

figure;
subplot(1, 3, 1); imshow(ims_input{1}./255); title('Input image 1');
subplot(1, 3, 2); imshow(ims_input{2}./255); title('Input image 2');
subplot(1, 3, 3); imshow(im_output./255); title('Deconvolved image');

figure;
subplot(1, 2, 1); imagesc(h{1}); colormap(gray); axis image; title('Latent PSF 1');
subplot(1, 2, 2); imagesc(h{2}); colormap(gray); axis image; title('Latent PSF 2');
figure
subplot(1, 2, 1); imagesc(h_init{1}); colormap(gray); axis image; title('Initially est. PSF 1');
subplot(1, 2, 2); imagesc(h_init{2}); colormap(gray); axis image; title('Initially est. PSF 2');
figure
subplot(1, 2, 1); imagesc(h_est{1}); colormap(gray); axis image; title('Refined PSF 1');
subplot(1, 2, 2); imagesc(h_est{2}); colormap(gray); axis image; title('Refined PSF 2');