function newU = uConstr(U,vrange)

%
% newU = uConstr(U,vrange)
%
% impose constraints on the image U
%
% The only constraint currently implemented is the intensity value range.
% We want to have our estimated HR image inside the itensity range of 
% input images.

newU = U;
for c = 1:size(U,3)
  m = logical(zeros(size(U)));
  m(:,:,c) = U(:,:,c)<vrange(c,1);
  newU(m) = vrange(c,1);
  m(:,:,c) = U(:,:,c)>vrange(c,2);
  newU(m) = vrange(c,2);
end


