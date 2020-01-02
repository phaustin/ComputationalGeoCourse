# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import torch
import numpy as np
import timeit
import matplotlib.pyplot as plt


# %% [markdown]
# # What will we do here
#
# 1. Review derivatives
# 2. Learn how to numerically compute them
# 3. Code a little using numpy

# %% [markdown]
# # Review and a little bit more
#
# This section surge as a short review for some of the background material that we need.
# It also introduced basic programming in Julia.
#
#

# %% [markdown]
# ## Derivatives and their approximation
#
# One of the fundamental tools we will use in this course is derivatives.
# We recall that the definition of the derivative is simply
# \begin{eqnarray}
# \label{der}
# {\frac {df}{dx}} |_{x_{0}} =\lim_{h\rightarrow 0} {\frac { f(x_{0} +h) - f(x_{0})}{h}}
# \end{eqnarray}
# Similarly, for multivariable functions we have that
# \begin{eqnarray}
# \label{derxy}
# {\frac {\partial f(x,y)}{ \partial x}} |_{x_{0},y_{0}} =\lim_{h\rightarrow 0} {\frac { f(x_{0} +h,y_{0}) - f(x_{0},y_{0})}{h}}
# \end{eqnarray}
#
# We can also define a directional derivative in direction $\vec n = [n_{1},n_{2}]$ with
# $\|\vec n\|^{2} = n_{1}^{2}+n_{2}^{2} = 1$ as
#
# $$
# {\frac {\partial f(x,y)}{ \partial \vec n}} |_{x_{0},y_{0}} =\lim_{h\rightarrow 0} {\frac { f(x_{0} +h n_{1},y_{0} +h n_{2}) - f(x_{0},y_{0})}{h}} = \nabla f(x_{0},y_{0}) \cdot \vec n
# $$
#
# Here we define the gradient as
# $$ \nabla f(x_{0},y_{0}) = \begin{pmatrix} {\frac {\partial f(x,y)}{ \partial x}} |_{x_{0},y_{0}} \\
# {\frac {\partial f(x,y)}{ \partial y}} |_{x_{0},y_{0}} \end{pmatrix}.$$
# and we use the dot as the dot product between 2 vectors.
#
#

# %% [markdown]
# If the function $f$ is given explicitly, then we can compute the derivative using the tools that
# are taught in calculus. The problem arises when the function is not given analytically. For example,
# consider the case that $f(t)$ is the price of a stock or the temperature in a particular place.
# This is where we need to use {\em numerical differentiation}. The idea is to {\em approximate} the quantity
# we are after in a controlled way.
#
# To start, we use a small $h$ and approximate the derivative as
# \begin{eqnarray}
# \label{ader}
# {\frac {df}{dx}} |_{x_{0}} \approx {\frac { f(x_{0} +h) - f(x_{0})}{h}}
# \end{eqnarray}
# The question is, what is ``small'' means? and how would we control the error?
#
# To answer this question we use the Taylor expansion
# $$ f(x+h) = f(x) + h {\frac {df}{dx}} + \frac 12 h^{2} {\frac {d^{2}f}{dx^{2}}} + {\frac 16} h^{3} {\frac {d^{3}f}{dx^{3}}} + ...
# $$
# which implies that
# $$ {\frac { f(x +h) - f(x)}{h}} = {\frac {df}{dx}} + \frac 12 h {\frac {d^{2}f}{dx^{2}}} + {\frac 16} h^{2} {\frac {d^{3}f}{dx^{3}}} + ... $$
#
#
# If $h$ is small then we have that the terms after $h$ are much smaller than leading term and therefore
# we can say that the leading error behaves like $h$ or
#  $$ {\frac { f(x +h) - f(x)}{h}} = {\frac {df}{dx}} + {\cal O}(h).$$
# The symbol ${\cal O}(h)$ implies that the error is of order $h$.
#
# We can obtain a better expression (in terms of accuracy) by combining the following
# \begin{eqnarray}
# \nonumber
# && {\frac { f(x +h) - f(x)}{h}} = {\frac {df}{dx}} + \frac 12 h {\frac {d^{2}f}{dx^{2}}} + {\frac 16} h^{2} {\frac {d^{3}f}{dx^{3}}} + ... \\
# \nonumber
# && {\frac { f(x) - f(x-h)}{h}} = {\frac {df}{dx}} - \frac 12 h {\frac {d^{2}f}{dx^{2}}} + {\frac 16} h^{2} {\frac {d^{3}f}{dx^{3}}} + ...
# \end{eqnarray}
# and adding the expressions to have
#  $$ {\frac { f(x +h) - f(x-h)}{2h}} = {\frac {df}{dx}} + {\cal O}(h^{2}).$$
#
#  Using the point $x+h$ and $x$ to approximate the derivative is often refers as the forward difference
#  while using the point $x-h$ is referred as the backward difference. Using the points $x+h$ and $x-h$
#  is referred to as the central or long difference.

# %%

# %% [markdown]
# ## Computing derivatives
#
# Consider now the computation of the derivative for a function $f$. First we need to sample
# $f$ at some points. To this end we define the {\bf grid function} ${\bf f} = [f(x_{1}),\ldots,f(x_{n})]^{\top}$.
# The grid function is the function $f$ discretized on the points $x_{1},\ldots,x_{n}$.
# For simplicity, assume that the interval $x_{j+1} - x_{j} = h$ is constant. Using the formulas above we
# obtain that the upwind approximation is
# $$ {\frac {\partial f}{\partial x}}|_{x_{i}} \approx {\frac 1h} ({\bf f}_{i+1} - {\bf f}_{i}) + {\cal O}(h), $$
# the downwind approximation is
# $$ {\frac {\partial f}{\partial x}}|_{x_{i}} \approx {\frac 1h} ({\bf f}_{i} - {\bf f}_{i-1}) + {\cal O}(h), $$
# and the central approximation is
# $$ {\frac {\partial f}{\partial x}}|_{x_{i}} \approx {\frac 1{2h}} ({\bf f}_{i+1} - {\bf f}_{i-1}) + {\cal O}(h^{2}). $$
#
# There is one important thing to note and this is the treatment of the derivative on the boundary.
# The upwind approximation cannot be used for the end of the grid while the downwind cannot be used
# for the first point on the grid. Finally, the central difference can be used only inside the grid.
# If we wish to use the central difference also on the boundary then we need boundary conditions.
#
# At this point it is useful to add the concept of a staggered grid. The idea is to use second order
# accurate derivative using only two neighbors. To this end we introduce another grid at points
# $[x_{3/2},x_{5/3},\ldots,x_{n-\frac 12}]$ and note that
# $$ f(x_{i+\frac 12}) = {\frac 1h} ({\bf f}_{i+1} - {\bf f}_{i}) + {\cal O}(h^2). $$
#
#
# Coding the derivative in Python is straight forward. Here we code the long difference.

# %%
def computeDerivative(f,h):
    
    df = (f[2:] - f[0:-2])/(2*h)
    return df


# %% [markdown]
# # Code testing
#
# It is important to be able to test the code and see that  it works as suggested by the theory.
# When reviewing code it is important to be skeptic and not believe that the code is working
# until proven otherwise.
#
# To this end we conduct a simple experiment to show that our code works.
# We pick a function $f$ and computes its derivatives for different $h$'s. Our goal
# is to see that the error of the second order behaves as $h^{2}$.
#  

# %%
pi = 3.1415926535
for i in np.arange(2,10):
    n = 2**i
    x = np.arange(n+2)/(n+1)
    h = 1/(n+1)
    
    f      = np.sin(2*pi*x)
    
    dfTrue = 2*pi*np.cos(2*pi*x)
    dfComp = computeDerivative(f,h)
    
    # dont use boundaries
    dfTrue = dfTrue[1:-1]
    
    res = np.abs(dfTrue - dfComp)
    print(h,  '      ',   np.max(res))


# %%

# %% [markdown]
# # More accurate and one sided derivatives
#
# We have computed deribvatives in rhe interior of our domain using a central difference giving us accuracy of ${\cal O}(h^2). However, this cannot be used for the first or the last point in our array. We therefore want to derive a second order formula that uses points only in one side of the interval.
#
# Consider the last point in our domain $x_n$ with the function value ${\bf f}_n$. We use the same trick as before
# and apply Taylor's theorem to obtain
# \begin{eqnarray}
# {\bf f}_{n-1} &=& {\bf f}_n - h {\bf f}_x + {\frac {h^2}2} {\bf f}_{xx} + {\cal O}(h^3) \\
# {\bf f}_{n-2} &=& {\bf f}_n - 2h {\bf f}_x + 2h^2 {\bf f}_{xx} + {\cal O}(h^3) 
# \end{eqnarray}
#
# Multiplying the first equation by 4 we obtain that
# \begin{eqnarray}
# 4{\bf f}_{n-1} &=& 4{\bf f}_n - 4h {\bf f}_x + 2h^2 {\bf f}_{xx} + {\cal O}(h^3) \\
# {\bf f}_{n-2} &=& {\bf f}_n - 2h {\bf f}_x + 2h^2 {\bf f}_{xx} + {\cal O}(h^3) 
# \end{eqnarray}
#
# We then subtract the equations to obtain
# \begin{eqnarray}
# 4{\bf f}_{n-1} - {\bf f}_{n-2} = 3{\bf f}_n - 2h {\bf f}_x  + {\cal O}(h^3) \\
# \end{eqnarray}
#
# We can isolate ${\bf f}_x$ and obtain
# \begin{eqnarray}
#  {\bf f}_x = {\frac {3{\bf f}_n - 4{\bf f}_{n-1} + {\bf f}_{n-2}}{ 2h}}   + {\cal O}(h^2) 
# \end{eqnarray}
#
# $$. $$
#
# Similarly you can find that it is possible to use points to the right of ${\bf f}_n$ to compute the derivatives 
# \begin{eqnarray}
#  {\bf f}_x = {\frac {-3{\bf f}_n + 4{\bf f}_{n+1} - {\bf f}_{n+2}}{ 2h}}   + {\cal O}(h^2) 
# \end{eqnarray}
#
#
#
#
#

# %% [markdown]
# # Class assignmets 
#
# Derive the above formula

# %% [markdown]
# # Class assignmets
#
# Modify the following code to handle boundary points 

# %%
def computeDerivativeBC(f,h):
    
    n  = f.shape
    df = np.zeros(n)
    df[1:-1] = (f[2:] - f[0:-2])/(2*h)
    
    # Your code here
    #df[0] = 
    #df[-1] = 
    
    return df

# %% [markdown]
# # Class assignmets
# Design a test for the code similar to our test above abd verify its working as planned

# %%

# %%
