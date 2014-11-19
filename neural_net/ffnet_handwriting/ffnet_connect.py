import numpy as np

from pele.potentials import BasePotential
from pele.takestep import RandomDisplacement, RandomCluster
from pele.systems import BaseSystem

from ffnet import mlgraph, ffnet
import networkx as NX
import matplotlib.pyplot as plt

class NNCostFunction(BasePotential):
    def __init__(self, net, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.net = net
    
    def getEnergy(self, coords):
        
        self.net.weights = coords
        return self.net.sqerror(self.inputs,self.targets)

    def getEnergyGradient(self, coords):

        self.net.weights = coords
        return self.net.sqerror(self.inputs,self.targets),self.net.sqgrad(self.inputs,self.targets)

def mydist(x1,x2):
    return np.linalg.norm(x1-x2),x1,x2

class NNSystem(BaseSystem):
    def __init__(self,net,inputs,targets,*args,**kwargs):
        super(NNSystem, self).__init__()
        self.params.database.accuracy =0.01
        self.params.double_ended_connect.local_connect_params.tsSearchParams.hessian_diagonalization = True
        
        self.pot = NNCostFunction(net,inputs,targets);

    def get_potential(self):
        return self.pot
    
    def get_minimizer(self, **kwargs):
        from pele.optimize import lbfgs_cpp as quench
        return lambda coords: quench(coords, self.get_potential(),tol=0.00001,iprint=20,**kwargs)
        #return lambda coords: myquench(coords, self.get_potential(),**kwargs)

    def get_mindist(self):
        # no permutations of parameters
        
        #return mindist_with_discrete_phase_symmetry
        #permlist = []
        return lambda x1,x2: mydist(x1,x2)
        return lambda x1,x2: np.linalg.norm(x1,x2)

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
    
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
    
    
    for m in database.minima():
        print m.energy,m.coords         
        
    database.session.commit()
    

def run_basinhopping(system,database,coords0,nsteps):

    #x0 = np.random.uniform(-1,1,[porder])
    
    step = RandomCluster(volume=1.0)
    bh = system.get_basinhopping(database=database, takestep=step,coords=coords0,temperature = 10.0)
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
    inputs  = np.loadtxt("../handwriting/X2_10samples.dat")
    targets = np.loadtxt("../handwriting/y2_10samples.dat")
    
    """ define network topology """
    conec = mlgraph((inputs.shape[1],20,1))

    net = ffnet(conec)
    system = NNSystem(net,inputs,targets)
    
    database = system.create_database(db="/home/ab2111/machine_learning_landscapes/neural_net/db_ffnet.sqlite")
    
    run_basinhopping(system,database,system.pot.net.weights,10)
    connect(system,database)
    
if __name__=="__main__":
    main()    
