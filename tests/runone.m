function runone(filename,andquit)

if nargin<2
	andquit=1
end

addpath('./minFunc_2012');
addpath('./minFunc_2012/autoDif');
addpath('./minFunc_2012/minFunc');

fid = fopen(filename);
line = fgetl(fid);
funname = textscan(line,'%s %s');
funname = funname{1}{1};
line = fgetl(fid);
x0 = str2num(line);
x0 = x0';
options = [];
args = {};
procargs = false;
line = fgetl(fid);
while ischar(line)
	if isempty(line)
		procargs = true;
	elseif procargs
		args{end+1} = line;
	else
		vals = textscan(line,'%s %s');
		fname = vals{1}{1};
		fvalstr = vals{2}{1};
		[fval,isnum] = str2num(fvalstr);
		if isnum
			options = setfield(options,fname,fval);
		else
			options = setfield(options,fname,fvalstr);
		end
	end
	line = fgetl(fid);
end


fprintf('===START HERE\n')
if length(procargs)==0
	evalstr = ['minFunc(@' funname ',x0,options);'];
else
	evalstr = ['minFunc(@' funname ',x0,options' strjoin(args,',') ');'];
end
if ~isfield(options,'noutputs') || options.noutputs == 3
	eval(['[x,f,exitflag] = ' evalstr]);
	nout = 3;
elseif options.noutputs == 1
	eval(['x = ' evalstr]);
	nout = 1;
elseif options.noutputs == 2
	eval(['[x,f] = ' evalstr]);
	nout = 2;
else
	eval(['[x,f,exitflag,output] = ' evalstr]);
	nout = 4;
end

fprintf('x [ ');
fprintf('%.10g ',x(1:end-1));
fprintf('%.10g ]\n',x(end));
if nout>1
	fprintf('f %.10g\n',f);
end
if nout>2
	fprintf('exitflag %d\n',exitflag);
end
if nout>3
	fprintf('output.iterations %d\n',output.iterations);
	fprintf('output.funcCount %d\n',output.funcCount);
	fprintf('output.algorithm %d\n',output.algorithm);
	fprintf('output.firstorderopt %.10g\n',output.firstorderopt);
	fprintf('output.message %s\n',output.message);
	fprintf('output.trace.fval ');
	fprintf('%.10g ',output.trace.fval);
	fprintf('\n');
	fprintf('output.trace.funcCount ');
	fprintf('%d ',output.trace.funcCount);
	fprintf('\n');
	fprintf('output.trace.optCond ');
	fprintf('%.10g ',output.trace.optCond);
	fprintf('\n');
end
fprintf('===END HERE\n');
if andquit==1
	exit(0);
end










