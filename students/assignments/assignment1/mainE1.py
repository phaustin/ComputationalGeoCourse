# EOSC213 Assignment 1.1
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
    """ Compute dT/dx and dT/dy using finite difference methods.
    
    Arguments:
        T {torch.Tensor} -- A 2-dimensional tensor of data measured on a grid.
        x {torch.Tensor} -- 1-dimensional tensor of x grid point locations.
        y {torch.Tensor} -- 1-dimensional tensor of y grid point locations.
    
    Returns:
        [torch.Tensor] -- dT/dx
        [torch.Tensor] -- dT/dy
    """

    # These are place holders, you will overwrite them in your code.
    dummy = torch.zeros(3,4)
    Tx = dummy
    Ty = dummy

    # TODO: your code here to compute Tx, Ty
    
    return Tx, Ty


def getAbsGrad(Tx,Ty):
    """ Compute the partials Tx and Ty, compute the absolute value of the gradients.
    
    Arguments:
        Tx {torch.Tensor} -- dT/dx
        Ty {torch.Tensor} -- dT/dy
    
    Returns:
        [torch.Tensor] -- |\nabla T|
    """
    
    # These are place holders, you will overwrite them in your code.
    dummy = torch.zeros(3,4)
    absGradT = dummy

    # TODO: your code here to compute absGradT = sqrt(Tx^2 + Ty^2)
    
    return absGradT


if __name__ == "__main__":

    # In this exercise you are going to use finite difference methods to compute
    # the absolute values of the gradient of a topographic map. In the assignment 
    # directory there are file 'x.pt', 'y.pt', and 'topo.pt' that contain torch.Tensors.
    # topo.pt contains data that was measured on a grid, as described by the grid points 
    # in 'x.pt' and 'y.pt'
     
    # TODO: Finish loading y and topo here
    # An example of how to load a torch tensor
    x = torch.load('x.pt')
    # y = 
    # topo = 
    
    # Uncomment these lines to plot the data, you should do this at least once!
    #plt.imshow(topo, extent=[x.min(), x.max(), y.min(), y.max()])
    #plt.show()
    
    # Compute the gradient
    Tx, Ty =  getGradients(topo,x,y)
    
    # Compute |\grad T|
    absGradT = getAbsGrad(Tx,Ty)

    # TODO: load correct answer
    #absGradTtrue = 
    
    # Here we are going to measure the distance between your AbsGrad and the true answer.
    # Don't change epsilon here.
    epsilon = 0.001
    pss = 0
    if misfit(absGradT,absGradTtrue)<epsilon:
        pss = 1
        
    if pss == 1:
        print('Test Pass')
    else:
        print('Test Fail')
 

