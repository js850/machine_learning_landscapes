from pele.mindist._minpermdist_policies import TransformPolicy,MeasurePolicy
import numpy as np
class TransformRegression(TransformPolicy):
    ''' transformation rules for atomic clusters '''
    
    def __init__(self, can_invert=True):
        self._can_invert = can_invert
    
    @staticmethod
    def translate(X, d):
        X+=d
    
    @staticmethod
    def rotate(X, mx,):
        Xtmp = np.dot(mx, X.transpose()).transpose()
        X[:] = Xtmp.reshape(X.shape)
    
    @staticmethod        
    def permute(X, perm):
        assert len(perm) == 2
        Xtmp = np.copy(X)
        Xtmp[perm[0]] = X[perm[1]]
        Xtmp[perm[1]] = X[perm[0]]
        X = Xtmp
        return X
    
    def can_invert(self):
        return self._can_invert
    
    @staticmethod
    def invert(X,inversion_indices):
        #print inversion_indices
        X[inversion_indices] = -X
        return X
        
class MeasureRegression(MeasurePolicy):
    ''' measure rules for atomic clusters '''
    
    def __init__(self, permlist=None):
        self.permlist = permlist
    
    def get_com(self, X):
        raise NotImplementedError

    def get_dist(self, X1, X2):
        return np.linalg.norm(X1.flatten()-X2.flatten())
    
    def find_permutation(self, X1, X2):
        raise NotImplementedError
        #return find_best_permutation(X1, X2, self.permlist)
    
    def find_rotation(self, X1, X2):
        raise NotImplementedError
        #dist, mx = findrotation(X1, X2)
        #return dist, mx