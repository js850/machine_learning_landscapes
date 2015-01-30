import numpy as np
from pele_mpl import NNSystem

def run_basinhopping(system):
    
#     quench = system.get_minimizer()
    db = system.create_database("/home/ab2111/machine_learning_landscapes/neural_net/theano_db/"+"reg"+str(system.potential.L2_reg)+".sqlite")
#     db = system.create_database()
    
    pot = system.get_potential()
    coords0 = np.random.random(pot.nparams)
    pot.set_params(coords0)
    
    stepsize=5.0
#     stepsize=2.0
    step = system.get_takestep(verbose=True, interval=10, stepsize=stepsize)
    temperature=10.0
    bh = system.get_basinhopping(database=db, takestep=step,
                                 temperature = temperature,
                                 coords = coords0
                                 )
    bh.run(500)
    
#     from pele.thermodynamics._normalmodes import normalmodes
#     import matplotlib.pyplot as plt
    
    import matplotlib.pyplot as plt
    for m in db.minima():
        print system.potential.L2_reg, m.energy, pot.getValidationError(m.coords)
        plt.plot(m.coords)
        plt.show()
#         e,g,h = pot.getEnergyGradientHessian(m.coords)
#         evals, evecs = normalmodes(h)
#         print evals
#         plt.plot(evals)
#     plt.show()
#     exit()

def get_minima_stats(system):
    database = system.create_database("/home/ab2111/machine_learning_landscapes/neural_net/"+"reg0.001sqlite")
    
    for m in database.minima():
        print m.energy, system.potential.L2_reg * np.dot(m.coords, m.coords) * 1./len(m.coords)
    
def main():
    
#     for p in range(10):
    p = 3
    system = NNSystem(ndata=100,L2_reg=0.0)#np.power(1.0e1, -p))
#     get_minima_stats(system)
    run_basinhopping(system)     
    
if __name__=="__main__":
    main()
