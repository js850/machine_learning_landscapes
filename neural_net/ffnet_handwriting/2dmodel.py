import numpy as np

from ffnet_connect import NNSystem, run_gui
# from pele.potentials import BasePotential
# from pele.takestep import RandomDisplacement, RandomCluster
# from pele.systems import BaseSystem
# 
from ffnet import mlgraph, ffnet
# import networkx as NX
import matplotlib.pyplot as plt
    
def test_optimized_data(Niterations,net):
    points = []
    for i in range(Niterations):
        point = 3.0*(np.random.random(2)-0.5)
        out = net(point)
        print point, out
        points.append(np.concatenate((point,out)))
        
    return np.array(points)

""" load training data"""
dir="/scratch/ab2111/dellcp10/projects/BDynam2d/LONGER/TPanalysis/"
tp  = np.loadtxt(dir+"tpout",usecols=(1,2))
ntp = np.loadtxt(dir+"nottpout",usecols=(1,2))
print tp.shape
inputs = np.concatenate((tp,ntp))
targets = np.concatenate(([1.0 for a in range(len(tp))],[0.0 for a in range(len(ntp))]))
    
""" define network topology """
conec = mlgraph((inputs.shape[1],5,1))

net = ffnet(conec)
system = NNSystem(net,inputs,targets)
    
database = system.create_database(db="/home/ab2111/machine_learning_landscapes/neural_net/db.2dmodel.sqlite")
# run_gui(system, database)
m = database.minima()[0]
net.weights = m.coords
predicts = test_optimized_data(100,net)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Axes3D.plot_surface(predicts[:,0],predicts[:,1],predicts[:,2])
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(predicts[:,0],predicts[:,1],predicts[:,2])
# plt.plot_surface(predicts[:,0],predicts[:,1],predicts[:,2])
plt.show()
#     run_basinhopping(system,database,system.pot.net.weights,1000)
#     connect(system,database)