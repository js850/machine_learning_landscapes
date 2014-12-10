import numpy as np
from ffnet import mlgraph, ffnet
import matplotlib.pyplot as plt
from ffnet_validation import get_validation_data

def make_disconnectivity_graph(system, database):
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    import matplotlib.pyplot as plt
 
    inputs, targets = get_validation_data()
   
    """ make validation_energy a minimum object"""    
    pot = system.get_potential()
    validation_energy = lambda m: pot.getValidationEnergy(m.coords, inputs, targets)
#     for m in database.minima():
# #         m.validation_energy = validation_energy(m)
# #         m.energy = validation_energy(m)
#         m.energy = pot.getEnergy(m.coords)
# #         setattr(m, "validation_energy", validation_energy(m))
# #         m.validation_energy = lambda: None
#     for t in database.transition_states():
# #         t.energy = validation_energy(t)
#         t.energy = pot.getEnergy(t.coords)
        
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, nlevels=5, center_gmin=True, energy_attribute="validation_energy")
    dg.calculate()
    
#     minimum_to_validation_energy = lambda m: pot.getValidationEnergy(m.coords, inputs, targets)
    
#     dg.color_by_value(validation_energy)
    
    dg.plot()
    plt.show()
#     dg.savefig("/home/ab2111/machine_learning_landscapes/neural_net/dg.png")

from ffnet_connect import NNSystem

""" load training data"""
inputs  = np.loadtxt("../handwriting/X2_100samples.dat")
targets = np.loadtxt("../handwriting/y2_100samples.dat")
# from ffnet_validation import get_validation_data
# inputs, targets = get_validation_data()
    
""" define network topology """
conec = mlgraph((inputs.shape[1],10,1))
print inputs.shape
# exit()
net = ffnet(conec)
system = NNSystem(net, inputs, targets)
        
database = system.create_database(db="/home/ab2111/machine_learning_landscapes/neural_net/db_ffnet_100samples.sqlite")

for m in database.minima():
    m.validation_energy=11.0
#     setattr(m, "validation_energy", 11.0)

for m in database.minima():
    print m.validation_energy

#     plt.plot(ts.coords,'x')
#     plt.plot(ts.eigenvec,'o')
# plt.show()
# make_disconnectivity_graph(system, database)
