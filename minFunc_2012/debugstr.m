function str = debugstr(x)

	[m,n] = size(x);
	if m==1 && n==1
		if imag(x)
			str = 'imaginary';
		else
			str = [sprintf('%.20g',x) ' (' num2hex(x) ')'];
		end
	elseif m==1
		str = '[';
		for i = 1:n
			if i>1
				str = [str ' '];
			end
			str = [str debugstr(x(1,i))];
		end
		str = [str ']'];
	elseif n==1
		str = '[';
		for i = 1:m
			if i>1
				str = [str ' '];
			end
			str = [str debugstr(x(i,1))];
		end
		str = [str ']'];
	else
		str = '[';
		inds = {1};
		sz = size(x);
		for d = 2:length(sz)
			inds = {inds{:}, 1:sz(d)};
		end
		for i = 1:m
			if i>1
				str = [str '\n'];
			end
			inds{1} = i;
			str = [str debugstr(shiftdim(x(inds{:}),1))];
		end
		str = [str ']'];
	end
