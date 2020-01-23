# EOSC213 Assignment 1 Exercise 1
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def misfit(a,b):
    """Computes the distance between a and b. We will use this to compare how close
    your gradients are to the true gradients.
    
    Arguments:
        a {torch.Tensor}
        b {torch.Tensor}
    
    Returns:
        [torch.Tensor] -- Mean squared error between a and b
    """
    n = a.numel()
    return torch.norm(a-b)/n       


def getGradients(T,x,y):

    dummy = torch.zeros(3,4)
    Tx = dummy
    Ty = dummy
    # your code here to compute Tx, Ty
    
    return Tx, Ty

def getAbsGrad(Tx,Ty):
    
    dummy = torch.zeros(3,4)
    absGradT = dummy
    # your code here to compute absGradT = sqrt(Tx^2 + Ty^2)
    
    return absGradT


if __name__ == "__main__":
     
    # load topo file
    x = torch.load('x.pt')
    # load y and topo here
    #
    #
    
    #plt.imshow(topo, extent=[x.min(), x.max(), y.min(), y.max()])
    #plt.show()
    
    # Compute the gradient
    Tx, Ty =  getGradients(topo,x,y)
    
    
    # Compute |\grad T|
    absGradT = getAbsGrad(Tx,Ty)


    # load correct answer
    #absGradTtrue = 
    
    # compare
    epsilon = 0.001
    pss = 0
    if misfit(absGradT,absGradTtrue)<epsilon:
        pss = 1
        
    if pss == 1:
        print('Test Pass')
    else:
        print('Test Fail')
 

