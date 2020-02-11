function R = fftLapR2matrix(G, hsize, scale, DM)
%
%
% R = fftLapR2matrix(g, hsize, scale,DM)
%
% Create R matrix for the partial-data case according to
% Harikumar-Bresler's article in IEEE Trans. Image Proces. 8 (1999)
% with FFT to speed up the calculation and 
% with upsampling extension if parameters DT, osize are specified.
%
% In addition Laplacian of G is used to construct R!!!.
%
% We use this matrix in the blur consistency regularization term.
%
% input:
% g ... cell array; blurred images (format: {image1, image2, ...} )
% hsize ... 1x2 vector; size of blurs [y,x]
% DT ... matrix; decimation matrix (optional)
% osize ... 1x2 vector; [y,x] output size of the decimation operation (optional); 
%                   y*x = size(DT,1)
%
% output:
% R ... matrix R
%
% note: Construction of this matrix is described in Sroubek, Flusser,
% IEEE Trans. Image Proces. 12 (2003). (R = Z' * Z)

% % written by Filip Sroubek (C) 2006

P = size(G,3);
% if scaling defined perform upsampling
if nargin > 2 && scale > 1
    %FG = conj(repmat(fft2(G),[scale scale 1]));
    %FG = repmat(fft2(DM,size(FG,1),size(FG,2)),[1 1 P]).*FG;
    %margin = size(DM)-1;
    
    FG = conj(fft2(iminterp2(G,scale)));
    margin = [0 0];
    
    %D = kron(eye(P),gendecmat(DM,scale,hsize));
    %hsize = ceil(hsize/scale);
    %FG = conj(fft2(G));
    %margin = [ 0 0 ];
else
    FG = conj(fft2(G));
    margin = [0 0];
end
FG = repmat(conj(fft2([0 0; 0 1],size(FG,1),size(FG,2))),[1 1 P]).*FG;
L = [0 -1 0; -1 4 -1; 0 -1 0]/4;
FL = repmat(fft2(L,size(FG,1),size(FG,2)),[1 1 P]);
FG = FG.*FL;
margin = margin+2;

G = real(ifft2(FG));
for i=1:P
    G(:,:,i) = fliplr(flipud(G(:,:,i)));
end
N = prod(hsize);

% construction of matrix R = Z'*Z 
R = zeros(N*P);
for i = 1:P
%  i
  r = localr2matrix(i,i,hsize,margin,G,FG);
  for k = [1:i-1, i+1:P]
    R(N*(k-1)+1:N*k,N*(k-1)+1:N*k) = R(N*(k-1)+1:N*k,N*(k-1)+1:N*k) + r;
  end
  for j = i+1:P
%    j
    r = -localr2matrix(i,j,hsize,margin,G,FG);
    R(N*(i-1)+1:N*i,N*(j-1)+1:N*j) = r; 
    R(N*(j-1)+1:N*j,N*(i-1)+1:N*i) = r.';
  end
end

if nargin > 2 && scale > 1
    %R = D.'*R*D;
end

end


function M = localr2matrix(p,q,s,mar,G,FG)

%
% M = localr2matrix(p,q,[m,n])
% returns matrix (M) so that 
%				unvec(M*x,m,n) = conv2(flipud(fliplr(q)),conv2(p,x,'valid'),'valid')
%

m = s(1);
n = s(2);
indi = kron(ones(1,n),[m:-1:1]);
indj = kron([n:-1:1], ones(1,m));
mn = m*n;
M = zeros(mn,mn);

for k=1:mn
  iff = ifft2(FG(:,:,p).*fft2(G(indi(k):end-m-mar(1)+indi(k),...
	indj(k):end-n-mar(2)+indj(k),q),size(FG,1),size(FG,2)));
  M(k,:) = vec(real(iff(end-m+1:end,end-n+1:end)))'; 
end
end

function D = gendecmat(h,s,o)
    rmargin = ceil((size(h)-1)/2);
    M = gconv2matrix(h,[1 1]+rmargin,o+rmargin,o);
    m = false(o); m(1:s:end,1:s:end) = 1;
    D = M(vec(m),:);
end
