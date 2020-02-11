%% Understading the parameters
% The algorithm runs in three steps: (1) registation, (2) PSFs (h) estimation and (3) final
% image (u) deconvolution
%
% Step (1) estimates parameters of an affine transform using the block-based phase
% correlation method.
%
% Step (2) is iterative (with maxiter iterations) and alternates between 
% min_u and min_h. It uses only a central section of the input images (maxROIsize)
% and can be multiscale (MSlevels)
%
% min_u and min_h are iterative with maximum of maxiter_u and maxiter_h
% iterations or stop earlier if the relative change is less than ccreltol
%
% The optimization tasks min_u and min_h are:
% min_u gamma/2 | Hu - g |^2 + alpha_u*PHI(vx,vy) + beta_u/2*| Dx u - vx -
% ax|^2 + beta_u/2*|Dy u - vy - ay |^2
%
% min_h gamma/2 | Uh - g |^2 + lambda/2*(h^T R h) + alpha_h*PSI(w) +
% beta_h/2*| h - w - b |^2
%
% Note that gamma depends on noise: 10dB -> 1e1, 20dB -> 1e2, etc.
% However in step (2) I always set gamma=1e1 (or 1e2) and in step (3) (see below) 
% I use the gamma corresponding to the noise level. 
%
% Step (3) uses estimated PSFs in step (2) to perform deconvolution on 
% the whole image. It performs min_u with parameters gamma_nonblind and
% beta_u_nonblind instead of gamma and beta_u.
%
% Note: You can try to replace in MCrestoration PSFestimaL with PSFestimaLnoRgrad 
% function. A relatively slow calculation of matrix R will not be done and
% min_h will be much faster. It works well, but you must use multiscale
% approach and I also multiply gamma by 1.5 after each iteration in step (2).
 




%% registation parameters
% perform registration flag {0,1}
doRegistration = 0;
% the number of non-overlapping blocks used in registration 
blocks = [6 6];
% gamma correction (if =1 no correction)
gamma_corr = 1; %2.2;

%% PSF estimation parameters
% perform PSF estimation flag {0,1}
doPSFEstimation = 1;
% size of the image central section, where PSF estimation will be calculated 
maxROIsize = [256 256]; %[ 256 256 ]; 
% multiscale levels {1,2,..} (if =1 do not use hierachical approach) 
MSlevels = 1;

%% parameters
PAR.verbose = 2; %{0 = no messages,1 = text messages,2 = text and graphs}

% common parameters to both PSF and image estimation
% data term weight gamma
PAR.gamma = 1e1;%1e1;
% which Lp norm to use
PAR.Lp = 1;
% Rel. tolerance uzed by CG minimizer
PAR.reltol = 1e-4;
% SR factor {1,2,3,...}
PAR.srf = 1;

% PSFs estimation
PAR.beta_h = 1e5*PAR.gamma;
PAR.alpha_h = 1;
% multichannel R weight lambda
PAR.lambda = 1e3*PAR.gamma; %%%%

% image estimation
PAR.beta_u = 1e-1*PAR.gamma;
PAR.alpha_u = 1;

% non-blind image estimation (final u-step)
% gamma_nonblind will be used at the end by the final u-step on 
% the full image (optional)
PAR.gamma_nonblind = 1e3*PAR.gamma;
PAR.beta_u_nonblind = 1e-1*PAR.gamma_nonblind;


% number of iterations
PAR.maxiter_u = 10;
PAR.maxiter_h = 10;
PAR.maxiter = 10;
PAR.ccreltol = 1e-3;