from pele.mindist._minpermdist_policies import TransformPolicy,MeasurePolicy
import numpy as np




class TransformRegression(TransformPolicy):
    ''' transformation rules for regression system '''
    
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
    ''' measure rules for regression system '''
    
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
        
class SymmetryChecker():
    def __init__(self,db,transform=TransformRegression(),tol=0.001):
        self.db = db
        self.transform = transform
        self.tol = tol
        
        self.symmetries = [self.check_inversions,self.check_singlesin]
        #self.symmetries = [self.check_inversions,self.check_permutations,self.check_translations,
        #                   self.check_singlesin]        
    
    def check_singlesin(self,coordsA,coordsB):
        
        sign = np.sign([coordsB[1],coordsB[3]]) == np.sign([coordsA[1],coordsA[3]])
        inversion_indices = [2*i + np.arange(1,3) for i,s in enumerate(sign) if s==True]

        for i,s in enumerate(sign):
            if s: coordsB = self.transform.invert(coordsB,[]) 
        
    def check_inversions(self,coordsA,coordsB):
        
        coordsB = self.transform.invert(coordsB,range(1,5))
        
        if np.linalg.norm(coordsB-coordsA) < self.tol: 
            return True
        
        return False
    
    def find_degeneracies(self):
        
        minima = self.db.minima()
        # make sure ordered by energy
        
        
        bottomset = []
        bottomset.append([minima[0]])
        bottomenergy = [minima[0].energy]

        
        for i,m in enumerate(minima[1:]):
            if np.abs(m.energy-bottomenergy[-1])<self.tol:
                bottomset[-1].append(m)
            else: 
                bottomset.append([m])
                bottomenergy.append(m.energy)
            #bottomset.append(mset)
                
        print bottomset
        print bottomenergy
        return bottomset
    
    def check(self):
        
        # make list of sets of degenerate minima
        duplicate_sets = self.find_degeneracies()
                
        # check permutational alignment
        for d in duplicate_sets:
            self.check_symmetries(d)
            
    def check_symmetries(self,degenerate_set):
        
        for i,di in enumerate(degenerate_set):
            for j,dj in enumerate(degenerate_set[i+1:]):
        
                for s in self.symmetries:
                    issymm = s(di.coords,dj.coords) 
                    print "symmetric: ",issymm
                    if issymm: 
                        self.redundancies.append([di,dj])
                        break
        
        #for r in self.redundancies:
        #    self.database.mergeMinima(r[0],r[1])

            