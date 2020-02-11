function newH = hConstr(oldH)

%
% newH = hConstr(oldH)
%
% impose constraints on blur masks H
%
% e.g. energy preserving, centrosymmetric, positive
% 


if iscell(oldH)
    P = length(oldH);
    H = cell2mat(reshape(oldH,1,[]));
    H = reshape(H,size(oldH{1},1),size(oldH{1},2),P);
else
    P = size(oldH,3);
    H = oldH;
end

% positive condition
H(H<0) = 0;

% energy preserving condition
H = H/sum(H(:))*P;

if iscell(oldH)
    newH = reshape(mat2cell(H,size(H,1),size(H,2),ones(1,P)),1,P);
else
    newH = H;
end
