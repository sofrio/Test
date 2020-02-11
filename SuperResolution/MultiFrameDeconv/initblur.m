function [h] = initblur(s,d,hw)
%
% Calculate rectangular mask of size hw=[h w] shifted by vector d in 
% a window of size s   
%

if length(s) == 1
    s = [s s];
end
if length(hw) == 1
    hw = [hw hw];
end
sh = s;
center = [-0.5 -0.5]+d;
hr = diff(fpulse([0:sh(2)-1; 1:sh(2)],center(2),hw(2)));
if sum(hr) == 0
    if fpulse(1,center(2),hw(2)) > 0
        hr(1) = 1;
    else
        hr(end) = 1;
    end
end
hc = diff(fpulse([0:sh(1)-1; 1:sh(1)],center(1),hw(1))).';
if sum(hc) == 0
    if fpulse(1,center(1),hw(1)) > 0
        hc(1) = 1;
    else
        hc(end) = 1;
    end
end
h = hc*hr;
h = h/sum(h(:));  
end

% evaluate the integral of the pulse fce 
function y = fpulse(i,c,w)
y = zeros(size(i));
i = i-c;
mz = i >= 0;
mw = (i <= w/2) & (i > -w/2);
y(mw) = i(mw)/w;
y(mz & ~mw) = 0.5;
y(~mz & ~mw) = -0.5;
end

% evaluate the pulse function
function y = pulse(i,c,w)
y = zeros(size(i));
i = i-c;
mw = (i <= w/2) & (i > -w/2);
y = mw/w;
end
