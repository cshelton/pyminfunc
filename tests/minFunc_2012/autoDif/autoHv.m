function [Hv] = autoHv(v,x,g,useComplex,funObj,varargin)
% [Hv] = autoHv(v,x,g,useComplex,funObj,varargin)
%
% Numerically compute Hessian-vector product H*v of funObj(x,varargin{:})
%  based on gradient values

if useComplex
    mu = 1e-150i;
else
    mu = 2*sqrt(1e-12)*(1+mynorm(x))/mynorm(v);
end
[f,finDif] = funObj(x + v*mu,varargin{:});
Hv = (finDif-g)/mu;
%format long
%format compact
%fprintf('-----\n')
%fprintf(debugstr(v))
%fprintf('\n')
%fprintf(debugstr(x))
%fprintf('\n')
%fprintf(debugstr(g)')
%fprintf('\n')
%fprintf(debugstr(mynorm(x)))
%fprintf('\n')
%fprintf(debugstr(mynorm(v)))
%fprintf('\n')
%fprintf(debugstr(mu))
%fprintf('\n')
%fprintf(debugstr(finDif))
%fprintf('\n')
%fprintf(debugstr(Hv))
%fprintf('\n')
