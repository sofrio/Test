function M = fftconvcirc2matrix(p,hsize,scale,DM)
%
% M = fftconvcirc2matrix(p,[m,n],scale,dec)
% returns matrix (M) that performs convolution (circular) with p and p rotated
% by 180 degrees.
% size(p) must be divisible by scale

m = hsize(1);
n = hsize(2);
blsize = size(p)/scale;
indi = kron(ones(1,n),[0:1:m-1]);
indj = kron([0:1:n-1], ones(1,m));
mn = m*n;
M = zeros(mn,mn);
s2 = scale^2;
shift = reshape(eye(scale^2),[scale scale s2]);

%hshift = zeros(floor((hsize+size(DM)-1)/2)+1); hshift(end) = 1; 
FDM = fft2(shift,size(p,1),size(p,2)).*repmat(fft2(DM,size(p,1),size(p,2)),[1 1 s2]);
%or ... repmat(conj(fft2(hshift,size(p,1),size(p,2))).*fft2(DM,size(p,1),size(p,2)),[1 1 s2]);
Fp = repmat(fft2(p),[1 1 s2]).*FDM;
T = zeros([blsize,s2]);
for i = 1:s2
    T(:,:,i) = reshape(sum(im2col(Fp(:,:,i),blsize,'distinct'),2)/(scale^2),blsize);
end
T = repmat(T,[scale,scale]);
iff = real(ifft2(conj(Fp).*T));

% shift index
inds = repmat(reshape(1:s2,[scale scale]),ceil(hsize/scale));
inds = inds(1:m,1:n);

for i=1:mn
  C = circshift(iff(:,:,inds(i)),[indi(i),indj(i)]);
  M(i,:) = (vec(C(1:m,1:n)));
end
