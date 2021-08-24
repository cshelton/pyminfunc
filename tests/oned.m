function [f,g,H,T] = oned(x)

f = x.*x;

if nargout > 1
	g = 2*x;
end

if nargout > 2
	H = zeros(1,1);
end

if nargout > 3
	T = zeros(1,1,1);
end
