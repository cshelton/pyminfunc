function [f,g,H,T] = colville(x)

f = 100*(x(1).^2 - x(2)).^2 + (x(1)-1).^2 + (x(3)-1).^2 + 90*(x(3).^2 - x(4)).^2 + 10.1*((x(2)-1).^2 + (x(4)-1).^2) + 19.8*(x(2)-1)*(x(4)-1);

if nargout > 1
	g = [400*x(1).^3 - 400*x(1)*x(2) + 2*x(1) - 2;
                   -200*x(1).^2 + 220.2*x(2) + 19.8*x(4) - 40;
                   360*x(3).^3 - 360*x(3)*x(4) + 2*x(3) - 2;
                   19.8*x(2) - 180*x(3).^2 + 200.2*x(4) - 40];
end

if nargout > 2
	H = [1200*x(1).^2 - 400*x(2) + 2, -400*x(1), 0, 0;
		-400*x(1), 220.2, 0, 19.8;
		0,0,1080*x(3).^2 - 360*x(4) + 2, -360*x(3);
		0,19.8,-360*x(3),200.2];
end

if nargout > 3
	T = zeros(4,4,4);
	T(1,1,1) = 2400*x(1);
	T(1,1,2) = -400;
	T(1,2,1) = -400;
	T(2,1,1) = -400;
	T(3,3,3) = 2160*x(3);
	T(3,3,4) = -360;
	T(3,4,3) = -360;
	T(4,3,3) = -360;
end
