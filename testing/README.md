# Testing code

### Reproducibility

The goal of testing is to assure that the algorithms implemented in python are
the same as those implemented in Matlab (including all options).  The exact answers
will necessarily vary (see below) and there isn't a "correct answer," given how 
floating point computations in a computer work.  However, getting bit-level "sameness"
is reassuring that the same calculations were being performed.

A few notes on that attempt:

- some operations produce different results in Matlab and numpy.  For example, for the
  matrix and vector

	A = [[3.5621300779392863767 (400c7f3e0de6c6ff) -9.1781279034222578161 (c0225b33949f6844)]   
          [0 (0000000000000000) 9.8451410606317875107 (4023b0b6543fd72f)]]   
	x = [0.13113680895928436776 (3fc0c91748e41628) 0.053165565863502003974 (3fab38845d506c26)]
	
  where the hex values inside the parentheses are the raw bits of the underlying double-precision
  floating point number,


  Matlab gives A\*x as
  
     [ -0.020833992034196655019 (bf95558189ba5dfc) 0.52342249549448727208 (3fe0bfe088847708) ]

  whereas Python gives A@x as

	[-0.020833992034196668897 (bf95558189ba5e00) 0.52342249549448727208 (3fe0bfe088847708)]

  (note the difference in the first value).

  This first value is (in Matlab notation) A(1,1)\*x(1) + A(1,2)\*x(2).

  Calculating this explicitly gives the same value as the Python code.  The difference *seems* to be that the Matlab code is using extended prevcision (say of the Intel internal registers) during the computation (probably extending out to 80 bits).  I cannot find a way to supress this (just for testing), and replacing matrix multiplication with an explicit version would be difficult and costly, even for testing.

  Matlab's norm versus Pythons numpy.norm have the same issue, sometimes differing in the least significant bits.

  The same is true of inner products.  For instance the following two vectors 
  have different inner products (by one bit) in Matlab and numpy:

      [ -199.29870031430442623 (c068e98ef3f627c2) 138.32340922732635136 (40614a595e4ed2ec) -45.549624050743560133 (c046c65a14b584f2) 19.213496343336956329 (403336a7b24472a4) ]

	 [ -0.21694196198103754547 (bfcbc4c113eb5848) 0.10955569080042396046 (3fbc0bd77d14192a) -0.037693665894205273525 (bfa34c958c905d5b) 0.078641461218126110233 (3fb421d8c80aa001) ]

- The problem is that while these are very small changes, they can magnify over the course of an optimization run, resulting in the algorithm taking a different path.  This makes it hard to test if the algorithms are really "the same" if we cannot rely on bit-wise comparisons *or even "approximately equal"*.

- Never-the-less, I did replace some of the more complex operations in Matlab with algorithms that (more closely) mimic the way they are done in Matlab (as the reverse is not possible): mynorm and mycholupdate.  
  

Again, this is not to say that floating point arthmetic is not exact (I think most know that)!  This is to say that it is not reproducible across systems, because of the looseness of the IEEE standard, which only guarentees very simple operations be bit-level identical.  This is known among many, but not all.

### Tests that fail (as of Aug 23, 2021)

Failures that are "okay" (seem to be due to slight variations in numeric processing, not because algorithm is incorrect or more unstable than Matlab version):
- Tests A03c and B03c fail because it depends on random choices and I cannot get the randomness to be the same between python and matlab easily
- Test A11f fails because no matter which sparse Cholesky decomp I pick in Python, the sparsity (for a given level) is chosen differently than in Matlab.  
- Test B10e fails, seemingly because of small (one-bit) differences in inner products early that propagate.
- Test B11f also fails (with a more dramatic deviation) due to small bit changes early.  In this case, Hessian and grad slight differences cause algorithm to choose different path.
- Test B12 similarly fails
- Test B14b fails because the "Adjusting Hessian" portion is unstable (even in the original Matlab).  Adding approximate 1e-12 to the diagonal isn't enough to really stablize the inversion.  Trying different methods in python, none produce the same answer as the Matlab code.

