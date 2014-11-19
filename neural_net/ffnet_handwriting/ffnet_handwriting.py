import numpy as np
from ffnet import mlgraph, ffnet
import networkx as NX
import matplotlib.pyplot as plt

# X = np.loadtxt("handwriting/X.dat")[:1000]
# y = np.loadtxt("handwriting/y.dat")[:1000]
X = np.loadtxt("../handwriting/X2_100samples.dat")
y = np.loadtxt("../handwriting/y2_100samples.dat")

conec = mlgraph((X.shape[1],20,1))

Errors=[]

for i in xrange(3):
    net = ffnet(conec)
    # 
    # NX.draw_graphviz(net.graph, prog='dot')
    # plt.show()
    
    net.train_tnc(X,y,messages=10)
    
    output,regress = net.test(X,y)
    Errors.append(net.sqerror(X,y))
    print Errors[-1]
print Errors
