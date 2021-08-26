# minFunc API

This API is the same as the Matlab version, so that Matlab code using 
Mark Schmidt's minFunc (https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) can be directly translated to python.  The only difference is that the python version takes its arguments as a dictionary (mapping strings to parameter values), where as the Matlab version takes them as fields of a structure.

## minFunc specs

`minFunc(f,x0,options,*args)`:

- `f` is a function of the form `f(x,...)` where `...` are filled in with `*args`.  `x` will be of shape (N,).

It should return `(v,g)` where v is the value of the function at `x` and `g` is the gradient (with shape (N,)).  If a Hessian is required, it should return `(v,g,H)` where `H` is the Hessian at `x` (shape (N,N)).  If a tensor is required, it will should return `(v,g,H,T)` where `T` is the tensor of 3rd derviaties (shape (N,N,N)).

- `x0` starting point for optimization (shape (N,)).
- options: a dictionary mapping strings to values.  See below

- returns `(x,fval,exitflag,output)` (or subset of these -- see noutputs option): `x` is the "argmin of f" and fval is the corresponding "min of f."  `exitflag` is the exit condition:
	- -1: stopped by output function
	- 0: max number of iteration/evals reached
	- 1: optimality condition reached
	- 2: step size or value change too small
	output is a structure (of type `outputT`) that has fields
	- iterations: number of iterations
	- funcCount: number of function evaluations
	- algorithm: algorithm used (as a number)
	- firstorderopt: first-order optimality
	- message: exit message
	- trace: structure (of type `traceT`) that has the fields
		- fval: list of function values after each iteration
		- funcCount: list of function evalution counts after each iteration
		- optCond: list of optimality condition after each iteration

Below is a description of the possible parameters and defaults, as cribbed from the comments in Schmidt's code.  Capitalization does not matter.

### Method Options:

#### Method [default: lbfgs]
Specifies the method for determining the step direction

possible values (string):
- sd: Steepest Descent (no previous information used, not recommended)
- csd: Cyclic Steepest Descent (uses previous step length for a fixed length cycle)
- bb: Barzilai and Borwein Gradient (uses only previous step)
- cg: Non-Linear Conjugate Gradient (uses only previous step and a vector beta)
- scg: Scaled Non-Linear Conjugate Gradient (uses previous step and a vvector beta, and Hessian-vector products to intialize line search)
- pcg: Preconditioned Non-Linear Conjugate Gradient (uses only previous step and a vector beta, preconditioned version)
- lbfgs: Quasi-Newton with Limited-Memory BFGS Updating (uses a predetermined number of previous steps to form a low-rank Hessian approximation)
- newton0: Hessian-Free Newton (numerically computes Hessian-Vector products)
- pnewton0: Preconditioned Hessian-Free Newton (numerically computes Hessian-Vector products, preconditioned version)
- qnewton: Quasi-Newton approximation (uses dense Hessian approximation)
- mnewton: Newton's method with Hessian claculation after every user-specified number of iterations (needs user-supplied Hessian matrix)
- newton: Newton's method with Hessian calculation every iteration (needs user-supplied Hessian matrix)
- tensor: Tensor (needs user-supplied Hessian matrix and Tensor of 3rd partial derivates)

#### LS_type [default: 1, except for bb (default: 0)]
Specifies the method for line search for finding the step length

possible values (integer):
- 0: A backtracking line-search based on the Armijo condition
- 1: A bracketing line-search based on the strong Wolfe conditions

Note that in this python code, value "2" is not allowed (this corresponded to the Matlab Optimization Toolbox's `linesearch` method).

#### LS_interp [default: 2]
Specifies the interpolation method for the Armijo or Wolfe line-search

For Armijo, possible values (integer):
- 0: Step size halving
- 1: Polynomial interpolation using new fucntion values
- 2: Polynomial interpoluation using new fucntion and gradient values

For Wolfe, possible values (integer):
- 0: Step size doubling and bisection
- 1: Cubic interpolation/extrapolation using new funcation and gradient values
- 2: Mixed quadratic/cubic interpolation/extrapolation

[note: Comments in Matlab code state that 1 is the default for Wolfe.  However
this does not appear to be the case.]

#### LS_multi [default: 0]
Specifies the method of polynomial interpolation if using Armijo line-search
and LS_interp > 0

possible values (integer):
- 0: quadratic interpolation if LS_interp=1 or cubic interpolation if LS_interp=2
- 1: cubic interpolation if LS_interp=1 or quartic or qunitic interpolation is LS_interp=2

#### Fref [default: 1, except for bb (20) and csd (10)]
number of previous function values to store for Armijo

#### LS_init [default: 0, except pcg (2), csg (4), csd (2), sd (2)]
Specifies the strategy for choosign the initial step size for Wole

possible values (integer):
- 0: always try an initial step length of 1
- 1: use a step similar to the previous step: `t = t_old * min(2,g.T@d/g_gold*d_old)`
- 2: quadratic initialization using previous function value and new function value/gradient (use this if steps tend to be very long): `t = min(1,2*(f-f_old)/g)`
- 3: the minimum between 1 and twice and the previous step length: `t = min(1,2*t)`
- 4: the scaled conjudate gradient step length (may accelerate conjugate gradient methods, but requires a Hessian-vector product): `t = g.T@d/(d.T@H@d)`

#### Damped [default: 1]
whether to use a damped BFGS update [QNewton and LBFGS only]

#### Corr [default: 100]
number of corrections to store in memory [LBFGS only]

#### bbType [default: 0]
type of BB step [BB only]

possible values (integer):
- 0: min_alpha ||delta_x - alpha delta_g||_ 2
- 1: min_alpha ||alpha delta_x - delta_g||_ 2
- 2: Conic BB
- 3: Gradient method with retards

#### cycle [default: 3]
length of the cycle [CSD only]


### Method Options for CG/SCG/PCG only:

#### cgUpdate [default: 2, except for PCG (1)]
type of update

possible values (integer):
- 0: Fletcher Reeves
- 1: Polak-Ribiere
- 2: Hestenes-Stiefel (not supported for PCG)
- 3: Gilbert-Nocedal

#### HvFunc [default: None]
user-suplied function that returns Hessian-vector product [SCG only]

called as `HvFunc(v,x,*args)` where `v` is the vector and `x` is the
point (at which the Hessian should be evaluated).  `*args` are the supplied callback data passed into `minFunc`.

#### precFunc [default: None]
user-supplied preconditioner [for PCG only].  Default (of None) causes a L-BFGS preconditioner to be used.

called as `precFunc(v,x,*args)` where `v` is the vector to be multiplied and
`x` is the point at which the precondition should be applied.  `*args*` are the supplied callback data passed into `minFunc`.

### Method Options for Newton-like methods only:

#### cgSolve [default: 0]
Specifies whether (and how) the use a conjugate gradient solver instead of a direct solver. [all except QNewton]

possible values (integer or real-value):
- 0: Direct Solver
- 1: Conjugate gradient
- 2: Conjugate gradient with digaonl preconditioner
- 3: Conjugate gradient with LBFGS preconditioner
- real value x in [0,2]: Conjugate gradient with symmetric successive over relaxation preconditioner with parameter x
- real value x < 0: Conjugate gradient with incomplete Cholesky preconditioner with drop tolerance -x

[only values of 0 or 1 allowed for Newton0/PNewton0 method]

#### HessianModify [default: 0]
Specifies type of Hessian modification for direct solvers if Hessian is not positive definite. [Newton and MNewton only, and only if cgSolve=0]

possible values (integer):
- 0: Minimum Eucliden norm s.t. eigenvalues sufficiently large (requires eigenvalues on iterations where matrix is not pd)
- 1: Start with (1/2) * ||A||\_F and increment until Cholesky succeeds (an approximation to method 0, does not require eigenvalues)
- 2: Modified LDL factorization (only 1 generalized Cholesky factorization done and no eigenvalues required)
- 3: Modified Spectral Decomposition (requires eigenvalues)
- 4: Modified Symmetric Indefinite Factorization
- 5: Uses the eigenvector of the smallest eigenvalue as negative curvature direction

#### LS_saveHessiancomp [default: 1]
Whether to compute the Hessian at the first and last iteration of the line search only.

#### HessianIter [default: 1, except for MNewton (5)]
Number of iterations to use the same Hessian

#### initialHessType [default: 1]
Whether to scale the initial Hessian approximation [QNewton only]

#### qnUpdate [default: 3]
type of quasi-Newton update [QNewton only]

possible values (integer):
- 0: BFGS
- 1: SR1 (when it is positive-definite, otherwise BFGS)
- 2: Hoshino
- 3: Self-scaling BFGS
- 4: Oren's Self-scaling variable metric method
- 5: McCormick-Huang asymmetric update

#### useNegCurv [default: 1]
whether to use a negative curvature direction as the descent direction if one is encourntered during the CG iterations [Newton0/PNewton0 only]


### Convergence Options:

#### MaxFunEvals [default: 1000]
maximum number of funtion evaluations allowed

#### MaxIter [default: 500]
maximum number of iterations allowed

#### optTol [default: 1e-5]
termination tolerance on the first-order optimality

#### progTol [default: 1e-9]
termination tolerance on process in terms of function/parameter changes

#### c1 [default: 1e-4]
sufficent decrease for Armijo condition

#### c2 [default: .9, except for pcg, scg, cg, csd (0.2)]
curvature decrease for Wolfe conditions

### Deriviative Options:

#### numDiff [default: 0]
how to compute derviatives.  If a Newton method, whether to compute Hessian numerically.  Otherwise, as below.

possible values (integer):
- 0: user-supplied
- 1: numerically using forward-differencing
- 2: numberically using central-differencing

#### useComplex [default: 0]
whether to use complex differentials if computing numerical derivates

#### DerivativeCheck [default: 'off']
whether to compute derivatives numerically at initial point and compare to user-supplied derivates.  Either 1 or 'on' for "true" and 0 or 'off' for "false."

### Other Options

#### noutputs [default: 3]
How many of the outputs to return from the normal tuple `(x,f,exitflag,output)`.
[Original Matlab code could detect number of used outputs and only calculate those needed.  x, f, and exitflag don't take any extra time/space to return.  output takes a bit of extra space to store, so by default it is not returned.]

#### outputFcn [default: None]
function to be called after every iteration; called as
`outputFcn(x,loc,i,funEvals,f,t,gtd,g,d,optCond,*args)`; should return `True` if the optimization should be stopped (and `False`) otherwise.  Args:
- `x`: current point
- `loc`: 'init' if initial call (i=0), 'iter' if middle of iterations, 'done' if last call
- `i`: iteration number
- `funEvals`: number of function evals so far
- `f`: value of function at `x`
- `t`: step size (None for loc='init')
- `gtd`: inner product of gradient and step direction (None for loc='init')
- `g`: gradient
- `d`: step direction (None for loc='init')
- `optCond`: optimality condition
- `*args`: args as per call to `minFunc`

#### useMex [default: 1]
not used in this code.  In original Matlab code, indicated whether to use mex code (ie compiled C code) instead of Matlab code for some routines.








