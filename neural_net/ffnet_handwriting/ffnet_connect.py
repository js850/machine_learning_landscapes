import numpy as np

from ffnet import mlgraph, ffnet
import networkx as NX
import matplotlib.pyplot as plt
from pele.gui import run_gui

from NNSystem import NNSystem

def connect(system, database):
    from pele.landscape.connect_manager import ConnectManagerGMin as ConnectManager

    mymin = database.minima()[-2]
    print mymin.energy
#     manager = ConnectManager(database, strategy="gmin", alternate_min=mymin)
    manager = ConnectManager(database, alternate_min=mymin)

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

#     reg = 0.1
    reg=False
    net = ffnet(conec)
    system = NNSystem(net, inputs, targets, reg=reg)
    
    database = system.create_database(
#                     db="/home/ab2111/machine_learning_landscapes/neural_net/db_ffnet_100samples_reg"+str(reg) +".sqlite"
                    db="/home/ab2111/machine_learning_landscapes/neural_net/db_ffnet_100samples.sqlite"
                )
    run_gui(system, database)
    
#     run_basinhopping(system,database,system.pot.net.weights,1000)
#     connect(system,database)
    
if __name__=="__main__":
    main()  
#     pass 
