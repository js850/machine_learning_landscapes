import numpy as np
from pele.takestep import RandomDisplacement, RandomCluster
from pele.systems import BaseSystem
from NNpotential import NNCostFunction

class NNSystem(BaseSystem):
    def __init__(self, net, inputs, targets, reg=False, *args, **kwargs):
        super(NNSystem, self).__init__()
        self.params.database.accuracy =0.01
#         self.params.double_ended_connect.local_connect_params.tsSearchParams.hessian_diagonalization = True
        self.params.double_ended_connect.local_connect_params.verbosity = 10
        
        self.pot = NNCostFunction(net, inputs, targets, reg=reg);

    def get_potential(self):
        return self.pot
    
    def get_minimizer(self, **kwargs):
        from pele.optimize import lbfgs_cpp as quench
        return lambda coords: quench(coords, self.get_potential(),tol=0.00001,**kwargs)
        #return lambda coords: myquench(coords, self.get_potential(),**kwargs)

    def get_mindist(self):
        # no permutations of parameters
        
        return lambda x1,x2: np.linalg.norm(x1-x2),x1,x2
        

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
    
    def get_random_configuration(self):
        
        return np.random.random(len(self.pot.net.weights))

#     def get_basinhopping(self,**kwargs):
#         step = self.get_takestep(verbose=True, interval=10)
#         return BaseSystem.get_basinhopping(takestep=step, **kwargs)