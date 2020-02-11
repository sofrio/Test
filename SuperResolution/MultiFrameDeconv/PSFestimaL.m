function [H, U, Report] = PSFestimaL(G, iH, PAR, Hstar)

% PSFestim
%
% Estimating PSFs
%
% Solving linear problem Ax=b that is (gamma*U'U + lambda*L + delta*R)h = gamma*U'g
% using Gaussian elimination method
% or using fmincon to impose the contraint that the int. values 
% of PSFs must lie between 0 and 1. Fmincon is necessary if PSFs 
% are overestimated. 
%
% Using augmented Lagrangian approach
%
% gamma ... scalar; weight of the fidelity term
% mu ... scalar; weight of the blur consistency term
% lambda ... scalar; weight of the blur smoothing term (usually lambda=0)
% epsilon ... scalar; relaxation (only for TV) for numerical
% stability in the case |grad(U)| = 0

Report = [];

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
gamma = PAR.gamma;
lambda = PAR.lambda;
Lp = PAR.Lp;
reltol = PAR.reltol;
ccreltol = PAR.ccreltol;

% size of H
hsize = [size(iH,1) size(iH,2)];
% number of input images
P = size(G,3);
gsize = [size(G,1), size(G,2)];
usize = gsize*srf;
ssize = size(spsf);
hssize = hsize + ssize - 1;
% the block size that repeats in FFT
blsize = usize(1:2)/srf;
if PAR.verbose > 1
    FIG_HANDLE_H = figure;
    axes('position',[0.25,0.94,0.5,0.01]);
    axis off; 
    title(['\gamma,\lambda,\alpha,\beta = (',num2str([gamma,lambda,PAR.alpha_h,PAR.beta_h]),')']);
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
    doMSE = 1;
    Report.hstep.mse =  zeros(1,PAR.maxiter+1);
else
    doMSE = 0;
end

U = zeros(usize);
%U = iU;

H = iH;

%% fft of sensor PSF
Fspsf = fft2(spsf,usize(1),usize(2));

%%% Calculate R
disp('Calculating R...');
tic;
R = fftLapR2matrix(G,hsize,srf,spsf);
toc

%Z = reshape(mat2cell(G,gsize(1),gsize(2),ones(1,P)),1,P);
%R = mimoR({Z},hsize,1,srf);

disp('Done.');

%% Initialization of variables for min_U step, which do not change
% If we work with FFT, we have to move H center into the origin
%hshift = zeros(floor(hssize/2)+1); hshift(end) = 1;
hshift = 1;
%hshift(1) = 1;
% FU ... FFT of u
FU = fft2(U);
% FDx, FDx ... FFT of x and y derivative operators
FDx = fft2([1 -1],usize(1),usize(2));
FDy = fft2([1; -1],usize(1),usize(2));
DTD = conj(FDx).*FDx + conj(FDy).*FDy;

eG = zeros(size(G));
%eG = zeros([srf srf 1].*size(G));

% auxiliary variables for image gradient and blurs
% initialize to zeros
Vx = zeros(usize);
Vy = zeros(usize);
Vh = zeros(size(H));
% extra variables for Bregman iterations
Bx = zeros(usize);
By = zeros(usize);
Bh = zeros(size(H));

if doMSE
        Report.hstep.mse(1) = calculateMSE(H,Hstar);
end
for mI = 1:PAR.maxiter
    for p = 1:P
        %eG(:,:,p) = edgetaper(real(ifft2(repmat(fft2(G(:,:,p)),[srf srf]))),conv2(H(:,:,p),spsf,'full'));
        eG(:,:,p) = edgetaper(G(:,:,p),conv2(H(:,:,p),spsf,'full'));
        %eG(:,:,p) = G(:,:,p);
    end
    FeGu = repmat(fft2(eG),[srf srf 1]);
    %FeGu = fft2(eG);
    %tic;
    Ustep;
    %toc
    %tic;
    Hstep;
    %toc
    if doMSE
        Report.hstep.mse(mI+1) = calculateMSE(H,Hstar);
    end
end

%% Initialization of variables for min_H step, which depend on U 


%%% ***************************************************************
%%% min_U step
%%% ***************************************************************
function Ustep
% FH ... FFT of  conv(H,spsf)
FH = repmat(conj(fft2(hshift,usize(1),usize(2))).*Fspsf,[1 1 P])...
        .*fft2(H,usize(1),usize(2));

% FHTH ... sum_i conj(FH_i)*FH_i
FHTH = sum(conj(FH).*FH,3); 

% FGs ... FFT of H^T*D^T*g
% Note that we use edgetaper to reduce border effect
%FGs = zeros(usize);
%FGu = zeros(usize);
FGs = sum(conj(FH).*FeGu,3);

%for p=1:P
%    %eG = edgetaper(G(:,:,p),H(:,:,p));
%    eG = G(:,:,p);
%    FGu = repmat(fft2(eG),[srf srf 1]);
%    FGs = FGs + conj(FH(:,:,p)).*FGu;
%end
beta = PAR.beta_u;
alpha = PAR.alpha_u;

%tic;
% main iteration loop, do everything in the FT domain
for i = 1:PAR.maxiter_u
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
        FU = b./( FHTH + beta/gamma*DTD);
        %disp(['beta: ', num2str(beta)]);
    end
    
    % Prepare my Lp prior
    Pr = asetupLnormPrior(Lp,alpha,beta);
    % get a new estimation of the auxiliary variable v
    % see eq. 2) in help above 
    xD = real(ifft2(FDx.*FU));
    yD = real(ifft2(FDy.*FU));
    xDm = xD - Bx;
    yDm = yD - By;
    nDm = sqrt(xDm.^2 + yDm.^2);
    Vy = Pr.fh(yDm,nDm);
    Vx = Pr.fh(xDm,nDm);
    % update Bregman variables
    Bx = Bx + Vx - xD;
    By = By + Vy - yD;
    
    % we do not have to apply ifft after every iteration
    % this is only for convenience to display every new estimation
    %U = real((FU));
    %% impose constraints on U 
    %U = uConstr(U,vrange);
    E = sqrt(Vy.^2+Vx.^2);
    %updateFig(FIG_HANDLE,[],{i, [], []});
    updateFig(FIG_HANDLE_U,{[] By},{i, [], E},{[] Bx});
    % we increase beta after every iteration
    % it should help converegence but probably not necessary
    %beta = 2*beta;

    % Calculate relative convergence criterion
    relcon = sqrt(sum(abs(FUp(:)-FU(:)).^2))/sqrt(sum(abs(FU(:)).^2));
    
    if relcon < ccreltol
        break;
    end
end
if PAR.verbose
    disp(['min_U steps: ',num2str(i)]);
    disp(['relcon:',num2str([relcon])]);
end    

U = real(ifft2(FU));
%toc
%E = sqrt(Vy.^2+Vx.^2);
updateFig(FIG_HANDLE_U,[],{i, U, E});    

% the part of gradient calculated in every CG iteration
    function g = gradcalcFU(x)
        X = unvec(x,usize);
        g = 0;
        T = FH.*repmat(X,[1 1 P]);
        for p=1:P
            % implementation of D^T*D in FT
            T(:,:,p) = repmat(reshape(sum(im2col(T(:,:,p),blsize,'distinct'),2)/(srf^2),blsize),[srf,srf]);
        end
        g = sum(conj(FH).*T,3);
        g = vec(beta/gamma*DTD.*X + g);
    end
    
end
% end of Ustep

%%% ************************************************
%%% min_H step
%%% ************************************************
function Hstep

%UD = zeros([usize,P]);
%lmargin = floor((hssize-1)/2);
%rmargin = ceil((hssize-1)/2);
%UD(1:srf:end,1:srf:end,:) = G;
%UD = padarray(UD(1+lmargin(1):end-rmargin(1),1+lmargin(2):end-rmargin(2),:),hssize-1,'pre');
%FUD = repmat(conj(FU).*conj(Fspsf),[1 1 P]).*fft2(UD);
%iff=real(ifft2(FUD));
%bc = vec(real(iff(1:hsize(1),1:hsize(2),:)));

FUD = FeGu.*repmat(conj(FU).*conj(Fspsf),[1 1 P]);
iff = real(ifft2(FUD));
bc = vec(real(iff(1:hsize(1),1:hsize(2),:)));

%zTzc = sum(eG(:).^2);


%Ap = fftconv2matrix(U,hsize,srf,spsf);
Ap = fftconvcirc2matrix(U,hsize,srf,spsf);
Ap = kron(eye(P),Ap) + lambda/gamma*R;
% be sure it is symmetric
Ap = (Ap+Ap.')/2;

%lb = zeros(numel(H),1);
%ub = Inf*ones(numel(H),1);

iterres = cell(PAR.maxiter_h,2);
beta = PAR.beta_h;
alpha = PAR.alpha_h;

%%%%
for i = 1:PAR.maxiter_h

b = beta/gamma*vec(Vh+Bh) + bc(:);
%zTz = beta/gamma*sum((Vh(:)+Bh(:)).^2) + zTzc;
A = Ap + beta/gamma*eye(size(Ap)); %%% !!!!!we should add identity matrix and not scalar as before

% we can use the simplest solver, no need for fancy stuff!!!
xmin = A\b;
iterres(i,:) = {norm(b-A*xmin)/norm(b) 1 };
 
%[xmin,flag,relres,iter,resvec] = mycgcon(@Afun,b,reltol,1000,[],H(:));
%iterres(i,:) = {flag iter};

options = optimset('GradObj','on','Hessian','on','MaxIter',2000);
%[xmin,fval,flag,output] = fmincon(@minHcon,H(:),[],[],[],[],lb,ub,[],options);
%[xmin,fval,flag,output] = quadprog(A,-b,[],[],[],[],lb,ub,H(:),options);

%[xmin,fval,flag,output] = fminunc(@minHcon,H(:),options);

%options = optimset('LargeScale','off','TolPCG',reltol,'TolFun',reltol,'MaxIter',1000);
%options = optimset('LargeScale','off','MaxIter',1000);
%[xmin,fval,residual,flag,output] = lsqlin(A,b,[],[],[],[],lb,ub,H(:),options); 

%iterres(i,:) = { flag output.iterations};

%flag
%%%%%
% Calculate relative convergence criterion
relcon = sqrt(sum((H(:)-xmin).^2))/sqrt(sum(xmin.^2));

%toc;
H = unvec(xmin,size(H));
%H(H<0) = 0; % such projection does not work well!!!

Pr = asetupLnormPrior(Lp,alpha,beta);     
hI = H;
hIm = hI - Bh;
nIm = abs(hIm);
Vh = Pr.fh(hIm,nIm); % L1 norm 
%Vh = hIm; % no contraint
%Vh = beta*hIm/(alpha+beta); % L2 norm
Vh(Vh<0) = 0; % Forcing positivity this way is a correct approach!!!
% update Bregman variables
Bh = Bh + Vh - hI;


E = abs(Vh);

%updateFig(FIG_HANDLE, [], {[] [] reshape(E,hsize(1),hsize(2)*P)} , {i [reshape(convn(H,spsf,'full'),hssize(1),hssize(2)*P)]});
updateFig(FIG_HANDLE_H, {[] reshape(Bh,hssize(1),hssize(2)*P)}, ...
    {i, reshape(convn(H,spsf,'full'),hssize(1),hssize(2)*P), reshape(E,hssize(1),hssize(2)*P) },...
    []);

%beta = 2*beta;

if relcon < ccreltol
    break;
end

end
if PAR.verbose
    %disp(['beta, flag, iter, relerr:',num2str([beta iterres{i,1} iterres{i,2} relcon])]);
    disp(['min_H steps: ',num2str(i)]);
    disp(['relcon:',num2str([relcon])]);
end


    function [y,g,H] = minHcon(x)
        Ax = A*x; %Afun(x);
        %y = 0.5*(x'*Ax - 2*x'*b + zTz);
        y = 0.5*sum((Ax-b).^2);
        if nargout > 1
          g = Ax - b;
          if nargout > 2
              H = A;
          end
        end
    end

    function y = Afun(x)
        y = A*x;
    end
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
    %nhs = norm(hs(:));
    s = sqrt(sum(sum((R-repmat(reshape(hs,prod(hsize(1:2)),1,hsize(3)),1,prod(i(1:2)))).^2,3),1));
    %r = min(s);
    r = s(ceil(prod(i(1:2))/2));
end


  

