function U = fftCGSRaL(G,h,PAR)

% fast SR using augmented Lagrangian approach (split-Bregman iterations)
% CG implementation but in Fourier domain
% works also with color images (3D arrays)
%
% Problem formulation: g = D*H*u + n, 
% where g is a vector of LR images, D is decimation (with SR factor = srf),
% H is a set of blur matrices, u is the HR image, and n is noise.
%
% Solving:  min_u  { gamma/2*|| g - D*H*u ||^2 + alpha*||grad(u)||_p^p },
% where 0<=p<=1.
% Using additive half-quadratic linearization procedure:
% 1) min_u gamma/2*|| g_i - D*H_i*u ||^2 + beta/2+||grad(u) - v||^2
% 2) v = [ grad(u) - t ]*sign(grad(u)) if |grad(u)| >= u*
%    v = 0, if |grad(u)| < u*
%
% 
% G ... input LR images in a cell array
% h ... input blurs in a cell array, empty or PSF size 
% gamma ... weight of the fidelity term
%
% note: if srf = 1 (no SR) it switches to the faster method without CG


srf = PAR.srf;
if isfield(PAR,'spsf')
    spsf = PAR.spsf;
else
    if srf > 1
        [d is] = decmat(srf,[1 1],'o');
        spsf = full(unvec(d,is));
    else
        spsf = [1];
    end    
end
% number of iterations
maxiter = PAR.maxiter_u;

% alpha
alpha = PAR.alpha_u;

% which Lp norm to use (p in the above notation)
Lp = PAR.Lp;

% Rel. tolerance used by CG minimizer
reltol = PAR.reltol;
% rel. tolerace used as convergence criterion
ccreltol = PAR.ccreltol;

% use gamma if gamma_nonblind is not defined, otherwise use gamma_nonblind
if isfield(PAR,'gamma_nonblind')
    gamma = PAR.gamma_nonblind;
else
    gamma = PAR.gamma;
end

% use beta_u if beta_u_nonblind is not defined, otherwise use beta_u_nonblind
if isfield(PAR,'beta_u_nonblind')
    beta = PAR.beta_u_nonblind;
else
    beta = PAR.beta_u;
end

% number of input images
P = length(G);

% get the size (rectangular support) of blurs
if isempty(h)
    h = [3 3]; %default size of PSFs
end
if iscell(h)
  hsize = size(h{1});
else
  hsize = h;
end 



%%
% If no blurs are provided (h is empty or contains only the PSF size),  
% then estimate the blurs as shifted delta functions 
%
H = cell(1,P);
if ~iscell(h)
  disp(['Setting size of H to ',num2str(hsize(1)),'x',num2str(hsize(2))]);
  disp(['Initializing H to Dirac pulses at ...']);
  
  %% Determine shift between channels
  %% using optical flow with upwind scheme discretization
  for k = 1:P
     dcH(k,:) = srf*motionestuw(G{1},G{k});
  end
  if sum(ceil(max(dcH)-min(dcH)+1) > hsize)
    warning('BSR:warn','Positions out of bounds. Size of blurs is probably too small.');
    warning('BSR:warn',['Increasing the blur size to',num2str(ceil(max(dcH)-min(dcH)+1))]);
    hsize = ceil(max(dcH)-min(dcH)+1);
  end
  cc = (hsize+1)/2 - (max(dcH) + min(dcH))/2;
  dcH = (dcH+repmat(cc,P,1));
  disp(dcH);
  % dcH contains the translation vectors
  
  for k = 1:P
      % we should handle non-integer shifts correctly
      % initblur takes care of it 
      H{k} = initblur(hsize,dcH(k,:),[1 1]);
  end
  % if blurs are provided impose PSF constraints
else
  H = h;
end 

% Blur PSFs with sensor PSF and use them as new PSFs
for p=1:P
    H{p} = conv2(H{p},spsf,'full');
end

% size of image U
usize = size(G{1});
if ndims(G{1}) < 3
    usize(3) = 1;
end
usize(1:2) = usize(1:2)*srf;


% vrange ... range of intensity values in each color channel
vr = zeros(P,2);
vrange = zeros(usize(3),2);
for c=1:usize(3)
    for p=1:P
        vr(p,:) = [min(vec(G{p}(:,:,c))), max(vec(G{p}(:,:,c)))];
    end
    vrange(c,:) = [min(vr(:,1)), max(vr(:,2))];
end

% If we work with FFT, we have to move H center into the origin
hshift = zeros(size(H{1}));
hshift(floor(size(H{1},1)/2)+1, floor(size(H{1},2)/2)+1) = 1;
%hshift(1) = 1;

% FU ... FFT of u
FU = zeros(usize);
xD = zeros(usize);
yD = zeros(usize);

% FDx, FDx ... FFT of x and y derivative operators
FDx = repmat(fft2([1 -1],usize(1),usize(2)),[1 1 usize(3)]);
FDy = repmat(fft2([1; -1],usize(1),usize(2)),[1 1 usize(3)]);
% FH ... FFT of PSFs
% FHTH ... sum_i conj(FH_i)*FH_i
FHTH = zeros(usize(1),usize(2));
for p=1:P
    FTH = conj(fft2(hshift,usize(1),usize(2))).*fft2(H{p},usize(1),usize(2));
    FH{p} = repmat(FTH,[1 1 usize(3)]);
    FHTH = FHTH + conj(FTH).*FTH;
end

% FGs ... FFT of H^T*D^T*g
% Note that we use edgetaper to reduce border effect
FGs = zeros(usize);
FGu = zeros(usize);
for p=1:P
    eG = edgetaper(G{p},H{p});
    %eG = G{p};
    FGu = repmat(fft2(eG),[srf srf 1]);
    FGs = FGs + conj(FH{p}).*FGu;
end


DTD = conj(FDx).*FDx + conj(FDy).*FDy;


% the block size that repeats in FFT
blsize = usize(1:2)/srf;

if PAR.verbose > 1
    FIG_HANDLE = figure;
    axes('position',[0.25,0.94,0.5,0.01]);
    axis off; 
    title(['\gamma,\alpha,\beta = (',num2str([gamma,alpha,beta]),')']);
else
    FIG_HANDLE = [];
end

tic;
% extra variables for Bregman iterations
Bx = zeros(usize);
By = zeros(usize);
Vx = zeros(usize);
Vy = zeros(usize);
% main iteration loop, do everything in the FT domain
for i = 1:maxiter
    
    disp(['minU step ',num2str(i)]);
    
    FUp = FU;
    b = FGs + beta/gamma*(conj(FDx).*fft2(Vx+Bx) + conj(FDy).*fft2(Vy+By));     
    if srf > 1
        % CG solution
        [xmin,flag,relres,iter,resvec] = mycg(@gradcalcFU,vec(b),reltol,100,[],vec(FU));
        iterres(i,:) = {flag relres iter resvec};
        FU = unvec(xmin,usize);
        if PAR.verbose
            disp(['beta, flag, iter:',num2str([beta flag iter])]);
        end
    else
    % or if srf == 1, we can find the solution in one step
        FU = b./( repmat(FHTH,[1 1 usize(3)]) + beta/gamma*DTD);
        if PAR.verbose
            disp(['beta: ', num2str(beta)]);
        end
    end
    
    % Prepare my Lp prior
    Pr = asetupLnormPrior(Lp,alpha,beta);
    % get a new estimation of the auxiliary variable v
    % see eq. 2) in help above 
    xD = real(ifft2(FDx.*FU));
    yD = real(ifft2(FDy.*FU));
    xDm = xD - Bx;
    yDm = yD - By;
    nDm = repmat(sqrt(sum(xDm.^2,3) + sum(yDm.^2,3)),[1 1 usize(3)]);
    Vy = Pr.fh(yDm,nDm);
    Vx = Pr.fh(xDm,nDm);

    % update Bregman variables
    Bx = Bx + Vx - xD;
    By = By + Vy - yD;
   
    if PAR.verbose>1
        % we do not have to apply ifft after every iteration
        % this is only for convenience to display every new estimation
        U = real(ifft2(FU));
        %% impose constraints on U 
        U = uConstr(U,vrange);
        E = sqrt(Vy.^2+Vx.^2);
        updateFig(FIG_HANDLE,{[] By},{i, U, E},{[] Bx});
    end
    
    % we increase beta after every iteration
    % it should help converegence but probably not necessary
    %beta = 2*beta;
    % Calculate relative convergence criterion
    relcon = sqrt(sum(abs(FUp(:)-FU(:)).^2))/sqrt(sum(abs(FU(:)).^2));
    if PAR.verbose
        disp(['relcon:',num2str([relcon])]);
    end
    if relcon < ccreltol
        break;
    end
end
if PAR.verbose<2
    U = real(ifft2(FU));
    %% impose constraints on U 
    U = uConstr(U,vrange);
    %updateFig(FIG_HANDLE,[],{i, U, []});    
end
toc


% the part of gradient calculated in every CG iteration
    function g = gradcalcFU(x)
        X = unvec(x,usize);
        g = 0;
        for p=1:P
            T = FH{p}.*X;
            for j=1:usize(3)
                % implementation of D^T*D in FT
                T(:,:,j) = repmat(reshape(sum(im2col(T(:,:,j),blsize,'distinct'),2)/(srf^2),blsize),[srf,srf]);
            end
            g = g + conj(FH{p}).*T;
        end
        g = vec(beta/gamma*DTD.*X + g);
    end
end
