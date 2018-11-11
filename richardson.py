#!/usr/bin/env python

"""

	Richardson Extrapolation


	Approximation of second derivative...

	f( x + h )    =    f(x)  +  (h) * f'(x)  +  (h^2/2) * f''(x)    +    (h^3/3!) * f'''(x)  +  ...        eq (1)
	f( x - h )    =    f(x)  -  (h) * f'(x)  +  (h^2/2) * f''(x)    -    (h^3/3!) * f'''(x)  +  ...        eq (2)


	Add eq's (1) and (2)...

	f( x + h ) + f( x - h )   =   2 * f(x)  +  (h^2) * f''(x)    +    2 * [ (h^4/4!) * f^(4)(x)  +  (h^6/6!) * f^(6)(x)  +  ... ]


	Rearrange:


	f''(x)   =  (1/h^2) * [ f( x + h )  -  2 * f(x)  +  f( x - h ) ]    -2 * [ (h^4/4!) * f^(4)(x)  +  (h^6/6!) * f^(6)(x)  +  ... ]


	Let,
		 	r4( h )   =   (1/h^2) * [ f( x + h )  -  2 * f(x)  +  f( x - h ) ]


	Then,

	f''(x)   =   r4( h )  +  O(h^4)


	Continuing...


	r6( h )   =   -1/(2^4 - 1) * r4( h )   +   (2^4)/(2^4 - 1) * r4( h/2 )

	r8( h )   =   -1/(2^6 - 1) * r6( h )   +   (2^6)/(2^6 - 1) * r6( h/2 )

	r10( h )  =   -1/(2^8 - 1) * r6( h )   +   (2^8)/(2^8 - 1) * r6( h/2 )


	...


	rn( h )  =   -1/(2^(n-1) - 1) * rn-1( h )   +   (2^(n-1))/(2^(n-1) - 1) * rn-1( h/2 )


	error in rn( h ) apprixmation of f''(x) is  O(h^n)

"""

__author__  = "LJ Brown"
__file__ = "richardson.py"

import math
import numpy as np

def richardson(f, x, h, order=10):
	""" second derivative approximation using richardson extrapolation """

	# O(h^4)
	init_order = 4

	ddf_approx = lambda h0: ( 1.0 / h0**2 ) * ( f( x + h0 )  -  2 * f(x)  +  f( x - h0 ) )
	get_ddf_approx = lambda prev_ddf_approx, o: \
		lambda hi: -1.0/(2.0**(o-2) - 1.0) * prev_ddf_approx(hi) + (2.0**(o-2))/(2.0**(o-2) - 1.0) * prev_ddf_approx(hi/2) 

	assert order >= init_order

	if order == init_order:
		return ddf_approx(h)

	ddf_approx = ddf_approx
	for i in range(init_order+2, order+2, 2):
		ddf_approx = get_ddf_approx(ddf_approx, i)


	return ddf_approx(h)


def magnitude(x):
	"""order of magnitude of x"""
	return int(math.log10(x))

def display_results(aproximations, hs, order=None, ddf_true=None):

	h = hs[-1]
	ddf_approx = aproximations[-1]
	k = len(aproximations) - 1

	# realvitve solution error
	get_relative_solution_error = lambda k: abs(aproximations[k] - aproximations[k-1])/abs(aproximations[k])

	# convergence rate approximation
	get_convergence_rate_approx =  lambda k: (math.log(get_relative_solution_error(k)) - math.log(get_relative_solution_error(k-1))) / (math.log(hs[k]) - math.log(hs[k-1]))


	if ddf_true is not None:

		print("\n\nh = %s" % h)
		print("\nddf true = ", ddf_true)
		print("ddf approximation = ", ddf_approx)

		true_error = abs(ddf_true-ddf_approx)
		print("true error = ", true_error)

	else:

		print("\n\nh: %s" % h)
		print("ddf approximation = ", ddf_approx)


	if k > 0:

		# relative solution errors
		relative_solution_error = get_relative_solution_error(k)
		print("relative solution error: |ek - ek-1|/|ek| = ", relative_solution_error)


	if k > 1:

		# convergence rate approximation
		convergence_rate_approx = get_convergence_rate_approx(k)
		print("rate of convergence estimate: (log(ek) - log(ek-1))/(log(hk) - log(hk-1)) = ", convergence_rate_approx)

	if order is not None:

		# error order of magnitude vs. theoretical error order of magnitude
		expected_error_order = magnitude(h**order)
		error_order = magnitude(true_error)
		print("Expected error order of magnitude >= returned error order of magnitude: ", expected_error_order >= error_order)


if __name__ == "__main__":

	#  Test function
	def f(x):
		return np.exp(np.sin(x)) 

	ddf_true = 0.46956439926573407 #grad(grad(f))(x)

	# test specs
	x = 0.5

	# order of precision O(h^order)
	order = 10

	ns = [-3, -2, -1, 0, 1, 2, 3]
	prev_ddf_approx = None

	hs = []
	ddf_approximations = []

	for n in ns:

		h = 0.8**n
		hs += [h]

		ddf_approx = richardson(f, x, h)
		ddf_approximations += [ddf_approx]

		# output
		display_results(ddf_approximations, hs)




