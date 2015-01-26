import os
import sys
import time
import gzip
import cPickle

import numpy
import numpy as np

import theano
import theano.tensor as T

from pele.potentials import BasePotential
from pele.systems import BaseSystem

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer, MLP


def get_data():
    print "hi"
    dataset = "mnist.pkl.gz"
    with gzip.open(dataset) as f:
        train_set, valid_set, test_set = cPickle.load(f)
    
    print train_set
    print train_set[0].shape
    print 28*28
    print train_set[1].shape
    
    # training data
    x = train_set[0]
    ndata, n_features = x.shape
    
    # labels
    t = train_set[1]
    assert ndata == t.size
    
    return train_set


class NNPotential(BasePotential):
    def __init__(self, ndata=1000, n_hidden=10, L1_reg=0.00, L2_reg=0.0001):
        
        train_x, train_t = get_data()
        train_x = train_x[:ndata,:]
        train_t = train_t[:ndata]
        train_t = np.asarray(train_t, dtype="int32")
    
        # allocate symbolic variables for the data.  
        # Make it shared so it cab be passed only once 
        x = theano.shared(value=train_x, name='x')  # the data is presented as rasterized images
        t = theano.shared(value=train_t, name='t')  # the labels are presented as 1D vector of
                            # [int] labels
    
        
        rng = numpy.random.RandomState(1234)
        
        # construct the MLP class
        classifier = MLP(
            rng=rng,
            input=x,
            n_in=28 * 28,
            n_hidden=n_hidden,
            n_out=10
        )
        self.classifier = classifier
    
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = (
            classifier.negative_log_likelihood(t)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
        )
    
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in classifier.params]
    
        outputs = [cost] + gparams
        self.theano_cost_gradient = theano.function(
               inputs=(),
               outputs=outputs
               )
        
    #    res = get_gradient(train_x, train_t)
    #    print "result"
    #    print res
    #    print ""
    
        self.nparams = sum([p.get_value().size for p in classifier.params])
        self.param_sizes = [p.get_value().size for p in classifier.params]
        self.param_shapes = [p.get_value().shape for p in classifier.params]
    
                  
    
    def _cost_gradient(self):
        ret = self.theano_cost_gradient()
        cost = float(ret[0])
        gradients = ret[1:]
                
        grad = np.zeros(self.nparams)
        
        i = 0
        for g in gradients:
            npar = g.size
            grad[i:i+npar] = g.ravel()
            i += npar
        return cost, grad
        
    def get_params(self):
        params = np.zeros(self.nparams)
        i = 0
        for p in self.classifier.params:
            p = p.get_value()
            npar = p.size
            params[i:i+npar] = p.ravel()
            i += npar
        return params
    
    def set_params(self, params_vec):
        assert params_vec.size == self.nparams
        i = 0
        for count, p in enumerate(self.classifier.params):
            npar = self.param_sizes[count]
            p.set_value(params_vec[i:i+npar].reshape(self.param_shapes[count]))
            i += npar
    
    def getEnergyGradient(self, params):
        # the params are stored as shared variables so we have to update
        # them in memory before computing the cost.
        self.set_params(params)
        return self._cost_gradient()

class NNSystem(BaseSystem):
    def __init__(self, *args, **kwargs):
        self.potential = NNPotential(*args, **kwargs)
    
    def get_potential(self):
        return self.potential

def test():
    system = NNSystem()
    t = system.get_potential()
            
    print "get_energy_gradient"
    newparams = np.random.uniform(-.05, .05, t.nparams)
    e, g = t.getEnergyGradient(newparams)
    print "cost", e
    
    print "\n\nagain\nget_energy_gradient"
    newparams = np.random.uniform(-.05, .05, t.nparams)
    e, g = t.getEnergyGradient(newparams)
    print "cost", e
    
    params = t.get_params()
    print params
    dx = np.max(np.abs(params - newparams))
    print dx
    assert dx < 1e-8
    
    # do minimization
    from pele.optimize import lbfgs_py
    res = lbfgs_py(newparams, t, iprint=10, tol=1e-4)
    print res

            
        
     
    

if __name__ == "__main__":
    test()