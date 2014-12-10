import numpy as np

from pele.potentials import BasePotential
from pele.takestep import RandomDisplacement, RandomCluster
from pele.systems import BaseSystem

from ffnet import mlgraph, ffnet
import networkx as NX
import matplotlib.pyplot as plt
from pele.gui import run_gui

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
#         self.net.weights = coords
        
        if self.reg==False:
            return self.getEnergy(coords, inputs, targets)

        return self.getEnergy(coords, inputs, targets) + self.calc_reg()
#         return self.net.sqerror(inputs,targets)
    
    def getEnergyGradient(self, coords):

        self.net.weights = coords
        
        if self.reg==False:
            return self.net.sqerror(self.inputs, self.targets), self.net.sqgrad(self.inputs, self.targets)
        
        return (self.net.sqerror(self.inputs, self.targets) + self.calc_reg(), 
                self.net.sqgrad(self.inputs, self.targets) + self.calc_reg_der()
                )
#     def getEnergyGradientHessian(self, coords):
#         """The hessian is calculated using the approximate Levenberg-Marquardt algorithm"""
#         self.net.weights = coords
#         error = self.net.sqerror(self.inputs,self.targets)
#         grad = self.net.sqgrad(self.inputs,self.targets)
#         hess = np.outer(grad,grad)
#         return error,grad,hess
    
def mydist(x1,x2):
    return np.linalg.norm(x1-x2),x1,x2

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
        
        #return mindist_with_discrete_phase_symmetry
        #permlist = []
        return lambda x1,x2: mydist(x1,x2)
        return lambda x1,x2: np.linalg.norm(x1,x2)

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
    
    def get_random_configuration(self):
        
        return np.random.random(len(self.pot.net.weights))
#     def get_basinhopping(self,**kwargs):
#         step = self.get_takestep(verbose=True, interval=10)
#         return BaseSystem.get_basinhopping(takestep=step, **kwargs)


def connect(system, database):
    from pele.landscape import ConnectManager

    manager = ConnectManager(database, strategy="gmin")
    for i in xrange(3):
        try:
            m1, m2 = manager.get_connect_job()
        except manager.NoMoreConnectionsError:
            break
        connect = system.get_double_ended_connect(m1, m2, database)
        connect.connect()
        database.session.commit()
    
    
    for m in database.minima():
        print m.energy,m.coords         
        
    database.session.commit()
    

def run_basinhopping(system,database,coords0,nsteps):

    #x0 = np.random.uniform(-1,1,[porder])
#     step = RandomDisplacement(step=1.0)
#     step = RandomCluster(volume=5.0)
    step = system.get_takestep(verbose=True, interval=10)
    bh = system.get_basinhopping(database=database, takestep=step,
                                 coords=coords0,temperature = 10.0)
    #bh.stepsize = 20.
    bh.run(nsteps)
    print "found", len(database.minima()), "minima"
    min0 = database.minima()[0]
    print "lowest minimum found has energy", min0.energy
    m0 = database.minima()[0]
    mindist = system.get_mindist()
    for m in database.minima():
        #dist = mindist(m0.coords, m.coords.copy())
        #print "   ", m.energy, dist, m.coords
        print "   ", m.energy#, m.coords
    return system, database

def main():
    """ load training data"""
    inputs  = np.loadtxt("../handwriting/X2_100samples.dat")
    targets = np.loadtxt("../handwriting/y2_100samples.dat")
    
    """ define network topology """
    conec = mlgraph((inputs.shape[1],10,1))

    net = ffnet(conec)
    system = NNSystem(net, inputs, targets, reg=1.0)
    
    database = system.create_database(
                    db="/home/ab2111/machine_learning_landscapes/neural_net/db_ffnet_100samples_reg.sqlite"
                )
    run_gui(system, database)
    
#     run_basinhopping(system,database,system.pot.net.weights,1000)
#     connect(system,database)
    
if __name__=="__main__":
    main()  
#     pass 
