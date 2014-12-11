import numpy as np
from ffnet import mlgraph, ffnet
import matplotlib.pyplot as plt
from ffnet_validation import get_validation_data

def make_disconnectivity_graph(system, database, vinputs, vtargets):
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    import matplotlib.pyplot as plt
    
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, nlevels=5, center_gmin=True, Emax=50.0, subgraph_size=3)
    dg.calculate()

    pot = system.get_potential()
    validation_energy = lambda m: pot.getValidationEnergy(m.coords, vinputs, vtargets)
    vmin = min(graph.nodes(), key=lambda m: validation_energy(m))
    labels = {vmin : "vmin"}    
    print vmin.energy, min(database.minima(), key = lambda m : validation_energy(m)).energy
#     for m in graph.nodes():
#         print m.energy
    for t in database.transition_states():
        print t.minimum1.energy, t.minimum2.energy, t.energy
        if abs(t.minimum1.energy-12.1884413947)<0.1 or abs(t.minimum2.energy-12.1884413947)<0.1:
            print t.minimum1.energy, t.minimum2.energy, t.energy
#     for u,v,data in graph.edges(data=True):
#         ts = data["ts"]
#         if abs(u.energy-12.1884413947)<0.1 or abs(v.energy-12.1884413947)<0.1:
#             print u.energy, v.energy, ts.energy  
    dg.plot()
    dg.label_minima(labels)

    plt.show()   
    
def make_validation_disconnectivity_graph(system, database):
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    import matplotlib.pyplot as plt
 
    inputs, targets = get_validation_data()
   
    """ make validation_energy a minimum object"""    
    pot = system.get_potential()
    validation_energy = lambda m: pot.getValidationEnergy(m.coords, inputs, targets)
    graph = database2graph(database)
    for m in graph.nodes():
        m.validation_energy = validation_energy(m)
    for u,v,data in graph.edges(data=True):
        ts = data["ts"]
        ve = max([validation_energy(ts), u.validation_energy, v.validation_energy])
        ts.validation_energy = ve

    gmin = min(graph.nodes(), key=lambda m:m.energy)
#     smin = graph.nodes().sort(key=lambda m:m.energy)
    smin = sorted(graph.nodes(), key=lambda m:m.energy)
#     gmax = max(graph.nodes(), key=lambda m:m.energy)
    
    labels = dict()
    for i,s in enumerate(smin):
        if i % 10 == 0: labels[s] = str(i)
#     labels = {gmin : "gmin"}
    dg = DisconnectivityGraph(graph, nlevels=10, center_gmin=True, energy_attribute="validation_energy", subgraph_size=3)
    dg.calculate()
    
#     minimum_to_validation_energy = lambda m: pot.getValidationEnergy(m.coords, inputs, targets)
    
#     dg.color_by_value(validation_energy)
    
    dg.plot()
    dg.label_minima(labels)
    print labels
    plt.show()
#     dg.savefig("/home/ab2111/machine_learning_landscapes/neural_net/dg.png")

from NNSystem import NNSystem

""" load training data"""
inputs  = np.loadtxt("../handwriting/X2_100samples.dat")
targets = np.loadtxt("../handwriting/y2_100samples.dat")
from ffnet_validation import get_validation_data
vinputs, vtargets = get_validation_data()
    
""" define network topology """
conec = mlgraph((inputs.shape[1],10,1))
print inputs.shape
# exit()
net = ffnet(conec)
system = NNSystem(net, inputs, targets)
        
database = system.create_database(db="/home/ab2111/machine_learning_landscapes/neural_net/db_ffnet_100samples.sqlite")

# make_disconnectivity_graph(system, database, vinputs, vtargets)

#     plt.plot(ts.coords,'x')
#     plt.plot(ts.eigenvec,'o')
# plt.show()
make_validation_disconnectivity_graph(system, database)
