import numpy as np
import matplotlib.pyplot as plt

def pixel_mapping(x, x1, x2, R, e, s1, s2):
    # DESCRIPTION: The function constructs a piecewise continuous and differentiable curve provided the points of slope
    # change, the overall range of the function, and slope definitions of the piecewise components. 
    # INPUT arguments to the function:
    # x     : Input vector
    # x1    : first point of slope change
    # x2    : second point of slope change
    # R     : x Range of function
    # e     : delta(y) in the range [x1, x2)
    # s1, s2: exponential multiplier - controls slope of sigmoids (>0)

    # Perform sanity check on inputs
    if x1 >= R or x2 >= R:
        print('x1 and x2 cannot be greater than the total range')
        exit()

    if e <= 0:
        print('slope in range (x1, x2] must be > 0')
        exit()

    if s1 <= 0 or s2 <= 0:
        print('s1 and s2 must be > 0')
        exit()

    # Function computation
    m  = np.arctan(e/(x2-x1))
    c1 = (((1+np.exp(s1*x1))**2)/(2*s1*np.exp(s1*x1)))*m
    f1 = lambda X: c1*(1 - np.exp(-s1*X))/(1 + np.exp(-s1*X))

    f2 = lambda X: m*(X-x1) + f1(x1)

    c2 = m*((1+np.exp(-((s2*x2)-R)))**2)/(s2*np.exp(-((s2*x2)-R)))
    c3 = f2(x2) - c2/(1+np.exp(-(s2*x2-R)))

    f3 = lambda X: (c2/(1+np.exp(-(s2*X-R)))) + c3

    # Define Masks
    X1_Mask = (x < x1)
    X2_Mask = ((x1 <= x) & (x < x2))
    X3_Mask = ((x2 <= x) & (x <= R))

    # Define ranges for piecewise function
    X1 = X1_Mask*x
    X2 = X2_Mask*x
    X3 = X3_Mask*x

    # Evaluate piecewise functions
    Y1 = X1_Mask*f1(X1)
    Y2 = X2_Mask*f2(X2)
    Y3 = X3_Mask*f3(X3)

    # Combine results
    Y  = Y1 + Y2 + Y3

    # Optional plot command
    # plt.plot(x, Y)
    # plt.show()
    return Y

if __name__ == '__main__':
    R  = 5
    x  = np.arange(0, 5, 0.01)
    x1 = 2
    x2 = 4
    e  = 0.001
    s1 = 1
    s2 = 1
    pixel_mapping(x, x1, x2, R, e, s1, s2)
