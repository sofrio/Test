function [H, U, Report] = findComKerfromH(G, regul, lp, Hstar)

% find Common Kernel from PSFs in G (Lp Norm Using Iterative Reweighted LS)
%
%
% note: trying to solve h-step in FT
% at this moment it works only for no SR srf=1
%
% Solving linear problem Ax=b that is (gamma*U'U  + delta*R)h = gamma*U'g
% using Gaussian elimination method
% or using fmincon to impose the contraint that the int. values 
% of PSFs must lie between 0 and 1. Fmincon is necessary if PSFs 
% are overestimated. 
%
% Using augmented Lagrangian approach
%
% note: If G is a cell array, output H will be a cell array as well

Report = [];

PAR.verbose = 0; %{0 = no messages,1 = text messages,2 = text and graphs}

% common parameters to both PSF and image estimation
% data term weight gamma
PAR.gamma = 1e1;
% which Lp norm to use
PAR.Lp = 1;

% PSFs estimation
PAR.beta_h = 1e1*PAR.gamma;
PAR.alpha_h = 1;

% image estimation
% 1e3*PAR.gamma
PAR.beta_u = 1e1*PAR.gamma;
PAR.alpha_u = 1;
% number of iterations
PAR.maxiter_u = 20;
PAR.maxiter_h = 20;
PAR.maxiter = 20;
PAR.ccreltol = 1e-4;

epsilon = regul;
theta = 0.001;
gamma = PAR.gamma;
Lp = 1;
ccreltol = PAR.ccreltol;

% conver cell input to 3D matrix
if iscell(G)
    inputIsCell = 1;
    G = reshape([G{:}],size(G{1},1),size(G{1},2),length(G));
else
    inputIsCell = 0;
end
% number of input images
P = size(G,3);
gsize = [size(G,1), size(G,2)];
usize = gsize;

hsize = gsize;
cen = (hsize+1)/2;
iH = zeros([hsize,P]);
for j=1:P
    iH(:,:,j) = initblur(hsize,cen,[1 1]);
end
% the block size that repeats in FFT
if PAR.verbose > 1
    FIG_HANDLE_H = figure;
    axes('position',[0.25,0.94,0.5,0.01]);
    axis off; 
    title(['\gamma, \alpha,\beta = (',num2str([gamma,PAR.alpha_h,PAR.beta_h]),')']);
    FIG_HANDLE_U = figure;
    axes('position',[0.25,0.94,0.5,0.01]);
    axis off; 
    title(['\gamma,\alpha,\beta = (',num2str([gamma,PAR.alpha_u,PAR.beta_u]),')']);
else
    FIG_HANDLE_H = [];
    FIG_HANDLE_U = [];
end

% if true PSF Hstar is provided -> calculate MSE
if exist('Hstar','var') && ~isempty(Hstar)
    doMSE = 0;
    Report.hstep.mse =  zeros(1,PAR.maxiter+1);
else
    doMSE = 0;
end

U = zeros(usize);
%U = iU;
U(1) = 1;

H = iH;

%% Initialization of variables for min_U step, which do not change
% If we work with FFT, we have to move H center into the origin
%hshift = zeros(floor(usize/2)+1); hshift(end) = 1;
hshift = 1;
%hshift(1) = 1;
% FU ... FFT of u
FU = fft2(U);
% FDx, FDx ... FFT of x and y derivative operators
FDx = fft2([1 -1],usize(1),usize(2));
FDy = fft2([1; -1],usize(1),usize(2));
DTD = conj(FDx).*FDx + conj(FDy).*FDy;

eG = zeros(size(G));

% auxiliary variables for image gradient and blurs
% initialize to zeros
Vx = zeros(usize);
Vy = zeros(usize);
Vu = zeros(usize);
Vh = zeros([usize P]);
% extra variables for Bregman iterations
Bx = zeros(usize);
By = zeros(usize);
Bu = zeros(usize);
Bh = zeros([usize P]);

if doMSE
        Report.hstep.mse(1) = calculateMSE(H,Hstar);
end
for p = 1:P
        eG(:,:,p) = G(:,:,p);
end
FeGu = fft2(eG);
for mI = 1:PAR.maxiter
    Hstep;
    Ustep;
    if doMSE
        Report.hstep.mse(mI+1) = calculateMSE(H,Hstar);
    end
end

if inputIsCell
    H = reshape(mat2cell(H,size(H,1),size(H,2),ones(1,size(H,3))),1,[]);
end
%% Initialization of variables for min_H step, which depend on U 
%%% ***************************************************************
%%% min_U step
%%% ***************************************************************
function Ustep
% FH ... FFT of  H
FH = repmat(conj(fft2(hshift,usize(1),usize(2))),[1 1 P])...
        .*fft2(H,usize(1),usize(2));

% FHTH ... sum_i conj(FH_i)*FH_i
FHTH = sum(conj(FH).*FH,3); 

% FGs ... FFT of H^T*D^T*g
% Note that we use edgetaper to reduce border effect
FGs = sum(conj(FH).*FeGu,3);

beta = PAR.beta_u;
alpha = PAR.alpha_u;

% main iteration loop, do everything in the FT domain
for i = 1:PAR.maxiter_u
    FUp = FU;
    b = FGs + beta/gamma*fft2(Vu+Bu);
    FU = b./( FHTH + beta/gamma);
    uD = real(ifft2(FU));
    uDm = uD - Bu;

    Vu = uDm;
    Vu(Vu<0) = 0;
    
    % update Bregman variables
    Bu = Bu + Vu - uD;

    %% impose constraints on U 
    E = abs(Vu);
    updateFig(FIG_HANDLE_U,{[] Bu},{i, [], E});
    % we increase beta after every iteration
    % it should help converegence but probably not necessary
    % beta = 2*beta;

    % Calculate relative convergence criterion
    relcon = sqrt(sum(abs(FUp(:)-FU(:)).^2))/sqrt(sum(abs(FU(:)).^2));
    if PAR.verbose
        disp(['relcon:',num2str([relcon])]);
    end
    if relcon < ccreltol
        break;
    end
end
disp(['min_U steps: ',num2str(i)]);
U = real(ifft2(FU));
%E = sqrt(Vy.^2+Vx.^2);
updateFig(FIG_HANDLE_U,[],{i, U, E});    


    
end
% end of Ustep

%%% ************************************************
%%% min_H step
%%% ************************************************
function Hstep


FUD = FeGu.*repmat(conj(FU),[1 1 P]);

FUTU = repmat(conj(FU).*FU,[1 1 P]);


iterres = cell(PAR.maxiter_h,2);
beta = PAR.beta_h;
alpha = PAR.alpha_h;

R_Delta = zeros(hsize(1)*hsize(2)*P);
hleng = hsize(1)*hsize(2);
Dx = eye(hsize(1)^2);
Dy = eye(hsize(1)^2);
for k = 1:P
    gg = H(:,:,k); gg=gg(:);
    A1x = (abs(Dx*gg).^(2-lp)+0.0001).^(-1);
    A1y = (abs(Dy*gg).^(2-lp)+0.0001).^(-1);
    R_Delta((k-1)*hleng+1:k*hleng,(k-1)*hleng+1:k*hleng)...
        = Dx'*diag(A1x)*Dx + Dy'*diag(A1y)*Dy;
end

Ap = fftconvcirc2matrix(U,hsize,1,1);
Ap = Ap + beta/gamma*eye(size(Ap));
Ap = kron(eye(P),Ap) + epsilon/gamma*R_Delta;

FH = fft2(H,usize(1),usize(2));

for i = 1:PAR.maxiter_h

FHp = FH; 
b = beta/gamma*fft2(Vh+Bh) + FUD;
B2D = real(ifft2(b));
B1D = B2D(:);
FH = fft2(unvec(Ap\B1D,size(FH)));

%flag
%%%%%
% Calculate relative convergence criterion
relcon = sqrt(sum(abs(FHp(:)-FH(:)).^2))/sqrt(sum(abs(FH(:)).^2));
    
hI = real(ifft2(FH));
hIm = hI - Bh;
nIm = abs(hIm);

Vh = hIm;
Vh(Vh<0) = 0; % Forcing positivity this way is a correct approach!!!

% update Bregman variables
Bh = Bh + Vh - hI;

H = hI(1:hsize(1),1:hsize(2),:);

E = abs(Vh);
updateFig(FIG_HANDLE_H, {[] reshape(Bh,size(Bh,1),size(Bh,2)*P)}, ...
    {i, reshape(H,hsize(1),hsize(2)*P), reshape(E,size(E,1),size(E,2)*P) },...
    []);

if PAR.verbose
    disp(['relcon:',num2str([relcon])]);
end
if relcon < ccreltol
    break;
end

end
disp(['min_H steps: ',num2str(i)]);

end
% end of Hstep

end

function r = calculateMSE(h,hs)
    hsize = size(hs);
    i = size(h)-hsize+1;
    R = zeros(prod(hsize(1:2)),prod(i(1:2)),hsize(3));
    h = h/sum(h(:))*sum(hs(:));
    for p = 1:hsize(3)
        R(:,:,p) = im2col(h(:,:,p),[size(hs,1) size(hs,2)],'sliding');
    end

    s = sqrt(sum(sum((R-repmat(reshape(hs,prod(hsize(1:2)),1,hsize(3)),1,prod(i(1:2)))).^2,3),1));
    r = s(ceil(prod(i(1:2))/2));
end

function [Dx Dy] = DDgenarate(ss)
    ddx = zeros(ss,ss);
    ddx(1,1) = 1;
    ddx(2,1) = -1;
    Dx = zeros(ss^2,ss^2);
    ddy = zeros(ss,ss);
    ddy(1,1) = 1;
    ddy(1,2) = -1;
    Dy = zeros(ss^2,ss^2);
    cnt = 1;
    for xx = 1:ss
        for yy = 1:ss
            tmp = circshift(ddx,[yy-1 xx-1]);
            Dx(cnt,:) = tmp(:);
            tmp = circshift(ddy,[yy-1 xx-1]);
            Dy(cnt,:) = tmp(:);
            cnt = cnt+1;
        end
    end
end
