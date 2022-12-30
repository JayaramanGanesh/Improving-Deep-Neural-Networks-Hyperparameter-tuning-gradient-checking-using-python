import numpy as np
from testCases import *
from gc_utils import sigmoid
from gc_utils import relu
from gc_utils import dictionary_to_vector
from gc_utils import vector_to_dictionary
from gc_utils import gradients_to_vector


# GRADED FUNCTION: forward_propagation
def forward_propagation(x, theta):
    J = np.dot(theta, x)
    return J
x, theta = 2, 4
J = forward_propagation(x, theta)
print ("J = " + str(J))




# GRADED FUNCTION: backward_propagation
def backward_propagation(x, theta):
    dtheta = x
    return dtheta
x, theta = 2, 4
dtheta = backward_propagation(x, theta)
print ("dtheta = " + str(dtheta))


# GRADED FUNCTION: gradient_check
def gradient_check(x, theta, epsilon=1e-7):
    # Compute gradapprox using left side of formula (1). epsilon is small enough, you don't need to worry about the limit.
    thetaplus = theta + epsilon                               
    thetaminus = theta - epsilon                              
    J_plus = forward_propagation(x, thetaplus)               
    J_minus = forward_propagation(x, thetaminus)            
    gradapprox = (J_plus - J_minus) / (2 * epsilon)          
    

    # Check if gradapprox is close enough to the output of backward_propagation()
    grad = backward_propagation(x, theta)
    numerator = np.linalg.norm(grad - gradapprox)                     
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)    
    difference = numerator / denominator                             
    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")
    
    return difference
x, theta = 2, 4
difference = gradient_check(x, theta)







