import numpy as np
import time
from gaussian_process_curve_fit import TrainingSet,GaussianProcess
from gaussian_process_examples import periodic_kernal,squared_exponential

from pele.systems import LJCluster
from pele.takestep import RandomDisplacement

natoms=13
x0 = np.random.random(natoms*3)#.reshape(natoms,3)


def initialize_system():
    
    system = LJCluster(natoms)    
    x0 = np.random.random(natoms*3)#.reshape(natoms,3)
    db = system.create_database()
    step = RandomDisplacement(stepsize=1.0)
    bh = system.get_basinhopping(database=db, 
                                 takestep=step,
                                 coords=x0,temperature = 10.0)
    
    return system, db, bh

def run_basinhopping(ts,system,db,x):
    
    if x == None: x = 0.1*np.random.random()+0.003
    
    E0 = -44.5
    step = RandomDisplacement(stepsize=x)
    bh = system.get_basinhopping(database=db, 
                                 takestep=step,
                                 coords=x0,temperature = 1.0)    
    t = time.time()
    while True:
        bh.run(1)
        eps = np.abs(bh.trial_energy/E0 - 1.)
        print "\n\n",bh.trial_energy, bh.takeStep.stepsize,"\n\n"
        if eps < 0.02:
            print "Finished\n\n",bh.trial_energy, bh.takeStep.stepsize
            print bh.trial_energy
            break
    
    dt = time.time()-t
    #ts.points = (ssize,dt)    
    ts.points = np.concatenate((ts.points,[[x,dt]]))
        
def main():
    
    points = np.zeros((1,2))
    
    system,db,bh = initialize_system()
    #get_point(bh)
    
    ts = TrainingSet(points=points)
    ts.gen_new_point = lambda ts,system,db,stepsize : run_basinhopping(ts,system,db,stepsize)
    
    l=0.4
    p=1.1
    
    kernel = lambda x1,x2 : squared_exponential(x1,x2,L=l)
    gp = GaussianProcess(ts,kernel=kernel,system=system,beta=1.)
    #gp.get_new_point = lambda ts,bh,stepsize : run_basinhopping(ts,bh,x=stepsize)
    #gp.run_single_iteration = my_run_single_iteration
    for i in xrange(200):
        gp.run(1)

    x = np.arange(0,10.0,0.1)
    gp.generate_curve(x)

if __name__=="__main__":
    main()  
