# Testing code

## Notes:

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

- The problem is that while these are very small changes, they can magnify over the course of an optimization run, resulting in the algorithm taking a different path.  This makes it hard to test if the algorithms are really "the same" if we cannot rely on bit-wise comparisons *or even "approximately equal"*.

- Never-the-less, I did replace some of the more complex operations in Matlab with algorithms that (more closely) mimic the way they are done in Matlab (as the reverse is not possible): mynorm and mycholupdate.  
  

Again, this is not to say that floating point arthmetic is not exact (I think most know that)!  This is to say that it is not reproducible across systems, because of the looseness of the IEEE standard, which only guarentees very simple operations be bit-level identical.  This is known among many, but not all.
