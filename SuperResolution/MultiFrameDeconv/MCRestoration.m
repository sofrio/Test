function [U H report] = MCRestoration(G,h,hstar)
% Multi-channel restoration.
% Input:
%  G: motion blurred image group;
%  h: blur kernel size;
%  hstar: latent PSFs;
%  U: deconvolved image;
%  H: deconvolved PSFs.

parameters;


srf = PAR.srf;

% number of input images
P = length(G);

% generate sensor PSF
if srf > 1
    [d is] = decmat(srf,[1 1],'o');
    spsf = full(unvec(d,is));
else
    spsf = [1];
end

% get the size (rectangular support) of blurs
if isempty(h)
    h = [3 3]; %default size of PSFs
end
if iscell(h)
  hsize = size(h{1});
else
  hsize = h;
end 

if ~exist('hstar','var')
    hstar  = [];
end
if iscell(hstar)
    hstar = reshape(cell2mat(hstar),size(hstar{1},1),size(hstar{1},2),length(hstar));
end

% Perform gamma correction
if gamma_corr ~= 1
    G = gammaCorr(G,gamma_corr);
end

%%% Register G 
%%% ***********************************************
% using phase correlation in small windows
if doRegistration
    disp('Registering...');
    [G] = registration(G,blocks);
    disp('Done.');
end

% normalize images
[G, norm_m, norm_v] = normimg(G);


H = cell(1,P);
%%% PSF estimation
%%% ***********************************************
if doPSFEstimation
disp('Estimating PSFs...');

L = MSlevels;
if (L<1)
    L = 1;
end
sr = [ maxROIsize(1)./(2.^(L-1:-1:0).'), maxROIsize(2)./(2.^(L-1:-1:0).')];

s = zeros(L,2);
s(1,:) = sr(1,:);
for i=2:L
    s(i,:) = 2*s(i-1,:) + 4;
end
% select ROI, on which PSF is calculated
ROI = cell(1,L);
hstarP = cell(1,L);

shift = [0 0];
ROI{L} = getROI(G,sr(L,:),shift);
if ~isempty(hstar)
    hstarP{L} = hstar;
end
for i = L-1:-1:1
    ROI{i} = imresize(ROI{i+1},0.5);
    if ~isempty(hstar)
        hstarP{i} = imresize(hstarP{i+1},0.5);
    end
end
% SR factor for downsampled images does not make sense, set it to one
% do SR if requested only on the finest scale
srf_list = ones(1,L); srf_list(L) = srf;
% initial PSF size and set them to delta functions
hsize = ceil(hsize/2^(L-1));
cen = (hsize+1)/2;
for i=1:P
    hi(:,:,i) = initblur(hsize,cen,[1 1]);
end
h = zeros([hsize P]); 
% main loop
report.ms = cell(1,L);
for i = 1:L
    disp(['hsize: ',num2str(size(hi))]);
    PAR.srf = srf_list(i);
    if PAR.srf == 1
        PAR.spsf = [1];
    else
        PAR.spsf = spsf;
    end
    hi = P*hi/sum(hi(:));
    
    [h u report.ms{i}] = PSFestimaL(ROI{i},hi,PAR,hstarP{i});
    hi = imresize(h,2,'lanczos3');
end

hsize = [size(h,1),size(h,2)];
h = reshape(mat2cell(h,size(h,1),size(h,2),ones(1,P)),1,P);

disp('Done.');
end
% End of PSF estimation code

%%
% If no blurs are provided (h is empty or contains only the PSF size),  
% then estimate the blurs as shifted delta functions 
%
if ~iscell(h)
  disp(['Setting size of H to ',num2str(hsize(1)),'x',num2str(hsize(2))]);
  disp(['Initializing H to Dirac pulses at ...']);

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
  % note: imposing constraint inside fftCGSRaL, not necessary here
else
  disp('Imposing constraints on H');
  H = hConstr(h);
  %H = h;
end


%% Run nonblind deconvolution
% augmented Lagrangian method
U = fftCGSRaL(G,H,PAR);

% denormalized th result
U = U*norm_v + norm_m;

% apply gamma 
if gamma_corr ~= 1
    % just a precaution; get rid of negative values
    U(U<0) = 0;
    U = U.^(1/gamma_corr);
end

report.par = PAR;

end

function R = gammaCorr(G,p)
R = cell(size(G));
for i = 1:length(G)
    R{i} =  exp(p*log(G{i}));
end
end

function R = getROI(G,win,shift)
isize = size(G{1});
gsize = isize(1:2);
P = length(G);
if size(G{1},3) > 1
    cind = 2; %green channel
else
    cind = 1;
end
% if the window size is larger then the image size, set win=isize.
if sum((gsize-win)<0)
    win = gsize;
    shift = 0;
end
R = zeros([win,P]);
margin = floor((gsize-win)/2) + shift;
for p=1:length(G)
    R(:,:,p) = G{p}(margin(1)+(1:win(1)),margin(2)+(1:win(2)),cind);
end    
end

function R = crop(G,win)
gsize = [size(G,1),size(G,2)];
margin = floor((gsize-win)/2);
R = G(margin(1)+(1:win(1)),margin(2)+(1:win(2)),:);
end