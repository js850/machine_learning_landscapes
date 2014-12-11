import numpy as np
from pele.potentials import BasePotential

class NNCostFunction(BasePotential):
    def __init__(self, net, inputs, targets, reg=False):
        self.inputs = inputs
        self.targets = targets
        self.net = net
        """ regularization"""
        self.reg = reg
        
    def calc_reg(self):
        
        return 0.5 * self.reg * np.dot(self.net.weights, self.net.weights)
    
    def calc_reg_der(self):
        
        return self.reg * self.net.weights
    
    def getEnergy(self, coords, inputs=None, targets=None):
        
        if inputs == None: inputs = self.inputs
        if targets== None: targets= self.targets
        
        self.net.weights = coords
        
        if self.reg==False:
            return self.net.sqerror(inputs, targets)
        return self.net.sqerror(inputs, targets) + self.calc_reg()

    def getValidationEnergy(self, coords, inputs, targets):
        
        return self.getEnergy(coords, inputs, targets)
    
    def getEnergyGradient(self, coords):

        self.net.weights = coords
        
        grad = self.net.sqgrad(self.inputs, self.targets)
        if self.reg==True:
            grad += self.calc_reg_der()
            
        return self.net.sqerror(self.inputs, self.targets), grad