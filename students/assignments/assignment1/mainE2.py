import torch
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# In this assignnmet you will need to compute the second derivative of f
# We do it by the following steps
# 1. f(x+h) = f(x) + ...
# 2. f(x-h) = f(x) - ...
# 3. Add the lines
# 4. Isolate d^2f/dx^2
# 5. Note the error

def d2dx2(f,h):
    # compute the second derivative of f
    # Your code here
    #d2f =  
    
    return d2f       



if __name__ == "__main__":
    
    pi = 3.1415926535
    d  = torch.zeros(8) 
    for i in np.arange(2,10):
        n = 2**i
        x = torch.tensor(np.arange(n+2)/(n+1))
        h = x[2]-x[1]
    
        f      = torch.sin(2*pi*x)
    
        d2fTrue = -(4*pi**2)*torch.sin(2*pi*x)
        d2fComp = d2dx2(f,h)
    
        # dont use boundaries
        d2fTrue = d2fTrue[1:-1]
    
        res = torch.abs(d2fTrue - d2fComp)
        d[i-2] = torch.max(res)
        print(h.item(),  '      ',d[i-2].item()   )
      
    pss = d[-2]/d[-1]   
    
    if pss > 3.5:
        print('Test Pass')
    else:
        print('Test Fail')
 
