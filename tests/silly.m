function [f,g] = silly(x)
	sinx1 = sin(x(2));
	cosx1 = cos(x(2));
	f = x(1).^2 * sinx1.^2 + (x(2)-10).^2 + (x(1)-10).^2;
	g = [2*x(1) * sinx1.^2 + (x(1)-10), x(1).^2 * sinx1 * cosx1 + (x(2)-10)]';
