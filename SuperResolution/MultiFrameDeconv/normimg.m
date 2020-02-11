function [I m v] = normimg(G)

% Normalize images in G
%
% I = normimg(G)
%
% G ... cell array; input images
%
% normalize the input images so that mean is zero and variance is 1
%

I = cell(size(G));
%m = mean(vec([G{:}]));
m = zeros(1,length(G));
%v = zeros(1,length(G));
for i=1:numel(G)
  m(i) = mean(G{i}(:));
  I{i} = double(G{i}) - m(i);
end
v = sqrt(var(vec([I{:}])));
for i=1:numel(G)
  %v(i) = sqrt(var(I{i}(:)));  
  I{i} = I{i}/v;
end
% get the index of the image with the smallest intensity
% and return its mean and variance, which can be used to 
% do denormalization; this guarantees that the demormalized image will be
% always positive if input G was positive
minv = zeros(1,length(I));
for i=1:numel(I)
    minv(i) = min(I{i}(:));
end
i = find(minv == min(minv),1);
m = m(i);
%v = v(i);



