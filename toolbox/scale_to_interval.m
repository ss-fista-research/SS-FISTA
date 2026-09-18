function y = scale_to_interval(x, a, b)
%SCALE_TO_INTERVAL Affinely scale all elements of an array into [a,b].

if nargin < 2
    a = 0;
end
if nargin < 3
    b = 1;
end

x_min = min(x(:));
x_max = max(x(:));

if x_max - x_min < eps
    y = x;
else
    y = (b - a) * (x - x_min) / (x_max - x_min) + a;
end
end
