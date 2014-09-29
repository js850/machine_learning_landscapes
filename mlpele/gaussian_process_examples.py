"""This is simply for testing the gaussian process.  It plots curves
generated from a gausian process for a variety of kernals.  The idea
for this was coppied from 

http://youtu.be/e7as_wY0hUI
"""
import numpy as np
from matplotlib import pyplot as plt

def brownian_kernal(x1, x2):
    return min(x1, x2)

def squared_exponential(x1, x2, L=.1):
    return np.exp(- np.dot(x1-x2, x1-x2) / L**2)

def linear_kernal(x1, x2):
    return np.dot(x1, x2)

def ornstein_uhlenbeck(x1, x2, theta=.1):
    return np.exp(- theta * np.sqrt(np.dot(x1-x2, x1-x2)))

def periodic_kernal(x1, x2, l=1.3, p=.13):
    return np.exp(- 2./l**2 * np.sin(np.pi * (x1-x2) / (2.*p))**2)

def power_decay(x1, x2, L=.2):
    return L**2 / ((x1-x2)**2 + L**2)

def run(kernal):
    
    x = np.arange(0,1,.005)
    
    N = x.size
    
    cov = np.zeros([N,N])
    for i in xrange(N):
        for j in xrange(N):
            cov[i,j] = kernal(x[i], x[j])
    print cov
    
    A, Svec, B = np.linalg.svd(cov)
    sqrtS = np.zeros([N,N])
    np.fill_diagonal(sqrtS, np.sqrt(Svec))
#     print S
#     sqrtS = sqrtm(np.asmatrix(S))
    
    u = np.random.normal(0,1,N)
    z = A.dot(sqrtS).dot(u)
    
    plt.plot(x,z, '.-')
    plt.show()
    

if __name__ == "__main__":
#     run(brownian_kernal)    
    run(squared_exponential)    
#     run(linear_kernal)
#     run(ornstein_uhlenbeck)    
#     run(periodic_kernal)
#     run(power_decay)
    