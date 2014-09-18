"""
=============================================
Density Estimation for a mixture of Gaussians
=============================================

Plot the density estimation of a mixture of two Gaussians. Data is
generated from two Gaussians with different centers and covariance
matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import matplotlib as mpl

from pele.potentials import BasePotential
from pele.optimize import LBFGS

n_samples = 300

def make_ellipses(gmm, ax):
    nmixtures = gmm.means_.shape[0]
    if nmixtures == 2:
        colors = 'rg'
    else:
        colors = 'rgb'
    for n, color in enumerate(colors):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        print v
        print w
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def generate_data_from_gaussians():
    # generate random sample, two components
    np.random.seed(0)
    
    # generate spherical data centered on (20, 20)
    shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
    
    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
    
    # generate zero centered stretched Gaussian data
    C = np.array([[0., 0.7], [-3.5, .7]])
    com = np.array([15,7])
    stretched_gaussian2 = np.dot(np.random.randn(n_samples, 2), C) * 5 + com[np.newaxis,:]
    stretched_gaussian2 = []

    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian, stretched_gaussian])#, stretched_gaussian2])

    print X_train.shape
    return X_train

def random_data(n=200):
    data = np.random.uniform(-20,20, n)
    return data.reshape([-1,2])

class CoordsConverter(object):
    def __init__(self, ndim, nmix, params=None):
        self.ndim = ndim
        self.nmix = nmix
        self.params = params
        nparams = nmix + ndim*nmix + self.n_covar_dof()
        if self.params is None:
            self.params = np.zeros(nparams)
        assert self.params.size == nparams

    def single_covar_dof(self):
        return self.ndim * (self.ndim + 1) / 2

    def n_covar_dof(self):
        return self.nmix * self.single_covar_dof()
    
    def get_weights(self):
        return self.params[:self.nmix]
    
    def get_means(self):
        istart = self.nmix
        return self.params[istart : istart + self.nmix*self.ndim]

    def unflatten_covars(self, M):
        n = self.ndim
        M = M.reshape([self.nmix, self.single_covar_dof()])
        A = np.zeros([self.nmix, n, n])
        for i in xrange(self.nmix):
            jstart = 0
            for j in xrange(n):
                dj = n-j
                A[i,j,j:] = M[i, jstart : jstart + dj]
                jstart += dj
            for k in xrange(n):
                for j in xrange(k):
                    A[i,k,j] = A[i,j,k]
        return A

    def flatten_covars(self, M):
        M = M.reshape([self.nmix, self.ndim, self.ndim])
        A = np.zeros(self.n_covar_dof()).reshape([self.nmix, -1])
        for i, mat in enumerate(M):
            a = np.concatenate([mat[j:-1] for j in xrange(self.ndim-1)])
            a = np.append(a, mat[-1,-1])
            A[i,:] = a.reshape(-1)
        return A.reshape(-1)

    def get_covars_flat(self):
        istart = self.nmix + self.ndim*self.nmix
        iend = istart + self.n_covar_dof()
        return self.params[istart:iend]

    def get_covars_matrix(self):
        return self.unflatten_covars(self.get_covars_flat())
    
#    def set_covars(self):
        

class GMMPotential(BasePotential):
    def __init__(self, gmm, data):
        self.gmm = gmm
        self.data = data
        self.ndim = data.shape[1]
    
    def get_coords_adaptor(self, params=None):
        return CoordsConverter(self.ndim, self.gmm.n_components, params=params)
    
    def get_random_coords(self):
        ca = self.get_coords_adaptor()
        weights = ca.get_weights()
        weights[:] = np.random.rand(weights.size)
        weights /= weights.sum()
        print weights.shape
        
        means = ca.get_means()
        means[:] = np.random.uniform(-1,1,means.size)
        print means.shape
        
        cov = ca.get_covars_flat()
        print cov.shape
        print ca.params.shape, ca.n_covar_dof()#
        c = np.array([np.eye(self.ndim) for i in xrange(ca.nmix)])
        print "c orig"
        print c
        cflat = ca.flatten_covars(c)
        print "cflat"
        print cflat
        
        cov[:] = cflat
        
        return ca.params
        
    
    def get_parameter_array(self):
        ca = self.get_coords_adaptor()
        ca.get_means()[:] = self.gmm.means_
        raise NotImplementedError
    
    def set_parameters(self, params):
        ca = self.get_coords_adaptor(params)
#        print ca.get_weights()
#        print ca.get_means()
#        print ca.get_covars_flat()
#        print ca.get_covars_matrix()
        self.gmm.weights_ = ca.get_weights().copy() / ca.get_weights().sum()
        self.gmm.means_ = ca.get_means().copy().reshape([-1,self.ndim])
        self.gmm.covars_ = ca.get_covars_matrix()
    
    def getEnergy(self, params):
#        print params
        self.set_parameters(params)
        
        
        forbidden_energy = 100000
        try:
            logprob, responsibilities = self.gmm.score_samples(self.data)
        except np.linalg.LinAlgError:
            print "hit linalg error"
            return forbidden_energy
        return -np.mean(logprob)

def print_event(coords=None, **kwargs):
    print coords

def run(X_train):
    # fit a Gaussian Mixture Model with two components
    clf = mixture.GMM(n_components=2, covariance_type='full')
    
    pot = GMMPotential(clf, X_train)
    params = pot.get_random_coords()
    print params
    e, g = pot.getEnergyGradient(params)
    print "energy", e
    print "grad", g
    opt = LBFGS(params, pot, tol=1e-5, maxstep=1., iprint=1)#, events=[print_event])
    res = opt.run()
    
    print "finished"
    e, g = pot.getEnergyGradient(res.coords)
    print "energy", e
    print "grad"
    print "grad", g

    
#    raise Exception("exiting early")
    
#    clf.fit(X_train)
    
    print "weights"
    print clf.covars_
    
    print "\nmeans"
    print clf.means_
    
    print "\ncovariances"
    print clf.covars_
    
    # display predicted scores by the model as a contour plot
    x = np.linspace(-20.0, 30.0)
    y = np.linspace(-20.0, 40.0)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)[0]
    Z = Z.reshape(X.shape)
    
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)
    
    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    make_ellipses(clf, plt.gca())
    plt.show()

if __name__ == "__main__":
    data = generate_data_from_gaussians()
#    data = random_data()
    run(data)


