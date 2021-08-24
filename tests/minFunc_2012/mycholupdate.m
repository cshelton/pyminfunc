function R = mycholupdate(R,x,method)
	% below in attempt to get exactly same answer out as
	% python code for downdate case
	% patched to work when x is imaginary (but not complex)
	%Rold = R;
	%xold = x;
	if nargin < 3 || method=='+'
		%R = cholupdate(R,x);
		if any(imag(x))
			x = x*(-sqrt(-1));
		end
		p = size(R,1);
		for k = 1:p
			r = hypot(R(k,k),x(k));
			c = r/R(k,k);
			s = x(k) / R(k,k);
			R(k,k) = r;
			for i = k+1:p
				R(k,i) = (R(k,i) + s*x(i)) / c;
				x(i) = c*x(i) - s*R(k,i);
			end
		end
	else
		if any(imag(x))
			x = x*(-sqrt(-1));
		end
		p = size(R,1);
		for k = 1:p
			r = rypot(R(k,k),x(k));
			c = r/R(k,k);
			s = x(k) / R(k,k);
			R(k,k) = r;
			for i = k+1:p
				R(k,i) = (R(k,i) - s*x(i)) / c;
				x(i) = c*x(i) - s*R(k,i);
			end
		end

		%R = cholupdate(Rold,xold,'-');
	end
end


function ret = rypot(x,y)
	x = abs(x);
	y = abs(y);
	t = min(x,y);
	x = max(x,y);
	t = t/x;
	ret = x*sqrt(1-t*t);

end

function ret = hypot(x,y)
	x = abs(x);
	y = abs(y);
	t = min(x,y);
	x = max(x,y);
	t = t/x;
	ret = x*sqrt(1+t*t);
end

