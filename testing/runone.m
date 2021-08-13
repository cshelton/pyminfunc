function runone(filename)

addpath('../minFunc_2012');
addpath('../minFunc_2012/autoDif');
addpath('../minFunc_2012/minFunc');

fid = fopen(filename);
line = fgetl(fid);
funname = line;
line = fgetl(fid);
x0 = str2num(line);
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
	line = fgetl(fid)
end

display(options)

fprintf('===START HERE\n')
if length(procargs)==0
	eval(['[x,f,exitflag,output] = minFunc(@' funname ',x0,options);'])
else
	eval(['[x,f,exitflag,output] = minFunc(@' funname ',x0,options' strjoin(args,',') ');'])
end

fprintf('x [')
fprintf('%f ',x(1:end-1))
fprintf('%f]\n',x(end))
fprintf('f %f\n',f)
fprintf('exitflag %d\n',exitflag)
fprintf('output.iterations %d\n',output.iterations)
fprintf('output.funcCount %d\n',output.funcCount)
fprintf('output.algorithm %d\n',output.algorithm)
fprintf('output.firstorderopt %f\n',output.firstorderopt)
fprintf('output.message %s\n',output.message)
fprintf('output.trace.fval ')
fprintf('%f ',output.trace.fval)
fprintf('\n')
fprintf('output.trace.funcCount ')
fprintf('%d ',output.trace.funcCount)
fprintf('\n')
fprintf('output.trace.optCond ')
fprintf('%f ',output.trace.optCond)
fprintf('\n')
fprintf('===END HERE\n')
exit(0)










