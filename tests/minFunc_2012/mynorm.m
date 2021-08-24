function n = mynorm(v)
	% Why do this, you might ask!
	% With the vector 
	% [-0.13113680909478608871 -0.053165565875915803129]
	% (hex repr as [bfc0c917492e9454 bfab38845d6bb87f])
	% this calculation differs very slightly from the official "norm"
	% in Matlab:
	% mynorm = 0.14150420521832102194 (3fc21ccf4ed45f66)
	%   norm = 0.1415042052183210497  (3fc21ccf4ed45f67)
	% (just one bit)
	% but this is enough to set the algorithm off in a different
	% direction than the python version (which is consistent with "mynorm")
	% So, this is the replacement for testing
	% Why not change the python?  Because I don't know how to duplicate
	% this behavior (and matlab's norm is built-in, so I cannot see the code)
	n = sqrt(sum(v.^2));
