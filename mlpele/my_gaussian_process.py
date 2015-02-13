import numpy as np
from pele.systems import BaseSystem
from pele.potentials import BasePotential

class GPPotential(BasePotential):
    def __init__(self, gaussian_process):
        self.gp = gaussian_process
        
    def getEnergy(self, coords):
        """ 
        returns: negative Log Likelihood of GP (Bishop Eq. 6.69)
        """
        
        """ Sets parameters and updates covariance matrix"""
        self.gp.set_params(coords)
        
        """ determinant of covariance matrix"""
#         detcov = np.linalg.det(self.gp.cov)
        detcov = np.linalg.det(self.gp.Lambda)

        """ inverse of covariance matrix"""
        covinv = self.gp.cov_inv
        
        t = self.gp.t
        N = self.gp.N
        
        return 0.5 * np.log(detcov) + 0.5 * t.T.dot(covinv).dot(t) + 0.5*N*np.log(2.*np.pi)
        
    def getEnergyGradient(self, coords):
        """ 
        returns: 1) negative Log Likelihood of GP (Bishop Eq. 6.69), 
                 2) and its gradient w.r.t. parameters (Bishop Eq. 6.70)
        """
#         self.gp.set_params(coords)
        energy = self.getEnergy(coords)
        
        t = self.gp.t
        covinv = self.gp.cov_inv
        dthetaC = self.gp.dthetaC
        
        grad = np.zeros(self.gp.kernel.Nparams)
        for fi,f in enumerate(range(self.gp.kernel.Nparams)):
            grad[fi] = 0.5 * np.trace(covinv.dot(dthetaC[:,:,fi])) - 0.5 * t.T.dot(covinv).dot(dthetaC[:,:,fi]).dot(covinv).dot(t)
        
        return energy, grad
        
class MyGaussianProcess():
    def __init__(self, kernel, X, t, noise=1.0):
        
        self.kernel = kernel
        self.X = X
        self.t = t
        self.N = X.shape[0]
        self.noise = noise
        
        """ 
        covariance matrix 
        """
        self.cov = None
        """ 
        inverse of covariance matrix
        """        
        self.cov_inv = None
        """ 
        diagonal matrix of eigenvalues of covariance matrix
        """
        self.Lambda = None
        """ 
        matrix of eigenvectors for covariance matrix
        """
        self.U = None
        """ 
        derivative of covariance matrix with respect to theta vector
        """
        self.dthetaC = None
        
        self.update_covariance_matrix(self.X)
        
        
    def get_params(self):
        return self.kernel._theta
    
    def set_params(self, coords):
        """
        sets parameters of kernel model 
             and updates covariance matrix
        """
        self.kernel._theta = coords    
        self.update_covariance_matrix(self.X)
        
    def update_covariance_matrix(self, X):
        
        N = self.N
        
        cov = np.zeros([N,N])
        dthetaC = np.zeros([N,N,len(self.kernel._theta)])
        
        for i in xrange(N):
            for j in xrange(N):
                cov[i,j] = self.kernel.evaluate(X[i], X[j])
                dthetaC[i,j,:] = self.kernel.der(X[i], X[j])
        
        cov = cov + self.noise * np.identity(N)
        UT, Lambdavec, self.U = np.linalg.svd(cov)
        self.Lambda = np.zeros([N,N])
        
        np.fill_diagonal(self.Lambda, Lambdavec)

        self.cov = cov
        self.dthetaC = dthetaC

        linv = np.linalg.inv(self.Lambda)
        self.cov_inv = self.U.T.dot(linv).dot(self.U)
    
    def predict(self, Xnew):
        
        kvec = np.zeros(self.N)
        for i in xrange(self.N):
            kvec[i] = self.kernel.evaluate(self.X[i],Xnew)

        cov_inv = self.cov_inv
        
        return kvec.T.dot(cov_inv).dot(self.t) 
    
    
class Kernel():
    def __init__(self, nfeatures=1):
        
        self._theta = np.random.random(nfeatures+1)
        self.Nparams = len(self._theta)
        
    def evaluate(self, X1, X2):
        """ return kernel evaluation """
        return self._theta[0] * np.exp(-0.5 * np.dot(self._theta[1:], (X1-X2)**2))
     
    def der(self, X1, X2):
        
        k = self.evaluate(X1, X2)
        kder = np.array([1./self._theta[0]])

        """ Here I'm really taking dK/d(ln theta) to ensure theta > 0"""
        kder = np.append(kder, -0.5 * self._theta[1:] * (X1-X2)**2)
#         kder = np.append(kder, -0.5 * (X1-X2)**2)
        
        return k * kder
        
def minimize(pot, coords0):

    from pele.optimize import lbfgs_cpp as quench
    
    return quench(coords0, pot, iprint=1, tol=1.0e-1)    

def plot_predictions(gp, ttest, Xtest):
    
    import matplotlib.pyplot as plt
    
    order = np.argsort(ttest)
    predictions = [gp.predict(x) for x in Xtest]
    print ttest.sort()
    plt.plot(np.sort(ttest))
    plt.plot([predictions[i] for i in order])
    plt.show()

def gradient_check(pot, coords):
            
    # gradient testing
    e,g = pot.getEnergyGradient(coords)
    dcoords = np.array([0.0, 0.01])
    Em = pot.getEnergy(coords - dcoords)
    Ep = pot.getEnergy(coords + dcoords)
    print "numerical der: ", (Ep-Em)*coords[1]/(2*dcoords[1])
#         print "numerical der: ", (Ep-Em)*1./(2*dcoords[1])
    e,g = pot.getEnergyGradient(coords)
    print " analytic: ", g

def test():

    noise = 0.1
    X = np.random.random(100) - 0.5
    t = np.exp(-X**2/0.2) + noise * np.random.normal(size=len(X))    

    kernel = Kernel()
    gp = MyGaussianProcess(kernel, X, t, noise = noise)
    pot = GPPotential(gp)
        
    import matplotlib.pyplot as plt
  
    plt.plot(X,t,'x')

    Xt = np.arange(-0.5,0.5,0.05)
    legend = ["bla"]
    for i in xrange(3):
        coords = 2.5 * np.random.random(size=kernel.Nparams)
        print "initial: ", i, pot.getEnergy(coords), gp.get_params()

        gradient_check(pot, coords)
        
        ret = minimize(pot, coords)
        print "quenched coords, energy: ", ret.coords, ret.energy        
        
        legend.append(pot.getEnergy(coords))
        plt.plot(Xt, [gp.predict(xi) for xi in Xt],'-')
    
    plt.legend(legend)
    plt.show()
    
    
if __name__=="__main__": 
    test()
#     main()   