import numpy as np

from ffnet_connect import NNSystem
# from pele.potentials import BasePotential
# from pele.takestep import RandomDisplacement, RandomCluster
# from pele.systems import BaseSystem
# 
from ffnet import mlgraph, ffnet
# import networkx as NX
import matplotlib.pyplot as plt
# from pele.gui import run_gui

def get_validation_data():

    """ load validation set"""
    xdat  = np.loadtxt("../handwriting/X.dat")
    ydat  = np.loadtxt("../handwriting/y.dat")
    inputs = xdat[600:1000]
    targets = ydat[600:1000]
    inputs = np.append(inputs, xdat[1000:1400],axis=0)
    targets = np.append(targets,ydat[1000:1400],axis=0)
    print inputs.shape, targets.shape
    return inputs, targets
# testme = np.loadtxt("../handwriting/X2_10samples.dat")[-10:]

def check_its_a_minimum(system, database):
    
    quench = system.get_minimizer()
    dist = system.get_mindist()
    for m in database.minima():
        ret = quench(m.coords)
        print dist(ret.coords, m.coords)[0]
        
def main():
    """ load training data"""
    inputs  = np.loadtxt("../handwriting/X2_100samples.dat")
    targets = np.loadtxt("../handwriting/y2_100samples.dat")    
    
    ValInputs, ValTargets = get_validation_data()

    """ define network topology """
    conec = mlgraph((inputs.shape[1],10,1))
    
    net = ffnet(conec)
    system = NNSystem(net, inputs, targets)
    
    pot = system.get_potential()
            
    database = system.create_database(db="/home/ab2111/machine_learning_landscapes/neural_net/db_ffnet_100samples.sqlite")
    # database = system.create_database(db="/home/ab2111/machine_learning_landscapes/neural_net/db_ffnet_me3.sqlite")
    # run_gui(system, database)
    
#     check_its_a_minimum(system, database)

    energies = np.array([])
    for m in database.minima():
        coords = m.coords
        testenergy = pot.getValidationEnergy(coords,ValInputs,ValTargets)/len(ValTargets)
        energies = np.append(energies,testenergy)
#         plt.plot(m.coords,'o')
#         np.max(m.coords)
         

#     plt.plot([m._id for m in database.minima()], np.array([m.energy for m in database.minima()])/100., 'o')
    plt.plot(np.array([m.energy for m in database.minima()])/100)
    plt.plot(energies)
    plt.plot(np.array([np.max(m.coords) for m in database.minima()])/1000, 'x')
    
    plt.legend(["Etrain","Evalidation","max(params)"])
    plt.show()
    
if __name__=="__main__":
#     pass
    main()