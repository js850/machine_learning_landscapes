import numpy as np
from pele.potentials import BasePotential

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));

def sigmoid_grad(z):
    s = sigmoid(z)
    return s*(1.-s)


def cost(X, y, Theta1, Theta2, lambda_):
    # forward propagation of the neural net
    
    # input layer
    a1 = np.vstack([np.ones(m), X.transpose()])
    
    # hidden layer
    z2 = np.dot(Theta1,a1)
    a2 = sigmoid(z2)
    m2 = a2.shape[1]
    a2 = np.vstack([np.ones(m2), a2])
    
    # output layer
    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3)
    h = a3
    
    # calculate the cost matrix
    J=0.
    for c in xrange(10):
        yc = (y == c+1)*1
        J -= (np.dot(np.log(h[c,:]),yc)   + np.dot(np.log(1. - h[c,:]),1. - yc))
        
    # regularization    
    J += 0.5*lambda_*(np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
    return J/m

def cost_grad(X, y, Theta1, Theta2, lambda_):
    # forward propagation of the neural net
    
    # input layer
    a1 = np.vstack([np.ones(m), X.transpose()])
    
    # hidden layer
    z2 = np.dot(Theta1,a1)
    a2 = sigmoid(z2)
    m2 = a2.shape[1]
    a2 = np.vstack([np.ones(m2), a2])
    
    # output layer
    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3)
    h = a3
    
    # calculate the cost matrix
    J=0.
    for c in xrange(10):
        yc = (y == c+1)*1
        J -= (np.dot(np.log(h[c,:]),yc)   + np.dot(np.log(1. - h[c,:]),1. - yc))
        
    # regularization    
    J += 0.5*lambda_*(np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
    
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)
    
    # calculate gradient via backward propagation
    for c in xrange(10):
        yc = (y == c+1)*1; 
        # this needs to be written as a matrix stuff
        for i in xrange(y.size):        
            delta3 = a3[c,i] - yc[i]
            Theta2_grad[c,:] += delta3*a2[:,i].transpose()
        
            tmp = np.concatenate([[1.], z2[:,i]])
            delta2 = np.dot(Theta2[c,:].transpose(), delta3) * sigmoid_grad(tmp)
            Theta1_grad +=  np.outer(delta2[1:], a1[:,i])
    
    return J/m, Theta1_grad/m, Theta2_grad/m


class NNCostFunction(BasePotential):
    def __init__(self, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels=10):
        self.X = X
        self.y = y
        self.lambda_ = lambda_
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.num_labels=num_labels       
        
    def getEnergy(self, coords):
        input_layer_size = self.input_layer_size
        hidden_layer_size = self.hidden_layer_size
        num_labels = 10
        
        Theta1 = coords[0:(input_layer_size+1)*hidden_layer_size]\
                    .reshape(hidden_layer_size,input_layer_size+1)
        Theta2 = coords[Theta1.size:].reshape(num_labels,hidden_layer_size+1)
        
        return cost(self.X, self.y, Theta1, Theta2, self.lambda_)

    def getEnergyGradient(self, coords):
        input_layer_size = self.input_layer_size
        hidden_layer_size = self.hidden_layer_size
        num_labels = 10
        
        Theta1 = coords[0:(input_layer_size+1)*hidden_layer_size]\
                    .reshape(hidden_layer_size,input_layer_size+1)
        Theta2 = coords[Theta1.size:].reshape(num_labels,hidden_layer_size+1)
        
        J,g1,g2 =  cost_grad(self.X, self.y, Theta1, Theta2, self.lambda_)
        
        return J, np.concatenate([g1.flatten(), g2.flatten()])

X = np.loadtxt("X.dat")
y = np.loadtxt("y.dat")

Theta1 = np.loadtxt("Theta1.dat")
Theta2 = np.loadtxt("Theta2.dat")

lambda_ = 1.0
m = X.shape[0]

pot = NNCostFunction(X, y, 0.0, 400, 25)
coords = np.concatenate([Theta1.flatten(), Theta2.flatten()])

E, grad = pot.getEnergyGradient(coords)
eps=1e-4
print len(coords)
for i in xrange(0,2):
    coords[i]+=eps
    E1 = pot.getEnergy(coords)
    coords[i]-=2*eps
    E2 = pot.getEnergy(coords)
    coords[i]+=eps
    gn = (E1-E2)/2./eps
    print gn, grad[i]
    
