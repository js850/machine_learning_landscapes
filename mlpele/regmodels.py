"""
an example of how to create a new potential.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from pele.potentials import BasePotential
from pele.takestep import RandomDisplacement, RandomCluster
from pele.landscape import ConnectManager

def generate_points_on_circle(self,R=1.0):
    
    theta = 2*np.pi*np.random.random(self.npoints)    
    r = np.random.random(10*self.npoints)
    rnew = []
    for ri in r:
        if np.random.random()<(ri/R):   
            rnew.append(ri)
        if len(rnew)==self.npoints:
            break
    r = rnew

    x = r * np.cos(theta)
    y = r * np.sin(theta)       
    
    #plt.plot(x,y,'x')
    #plt.show() 
    #exit()
    return zip(x,y)

class BaseModel(object):
    
    def __init__(self,params0=None,points=None,x=None,npoints=100,sigma=0.3):        

        self.params0 = params0
        self.sigma = sigma
        self.points = points
        self.npoints = npoints     
        
        """ optional x-values to generate training data """
        self.xvals = x
        
        if self.points == None:
            assert self.params0 != None
            self.nparams = len(self.params0)
            self.points = self.generate_points(x=self.xvals)
        else:
            self.npoints = len(self.points)
            
    def generate_points(self,x=None):
        
        if x==None:
            x = np.random.random(self.npoints)

        y = self.model(x,self.params0) + np.random.normal(scale=self.sigma,size=self.npoints)
        
        return zip(x,y)
                
class LinearModel(BaseModel):
    
    def __init__(self,*args,**kwargs):
        super(LinearModel,self).__init__(*args,**kwargs)
    
    def model(self,x,params):
        return params[0]*x  
    

class WaveModel(BaseModel):

    def __init__(self,*args,**kwargs):

        super(WaveModel,self).__init__(*args,**kwargs)
        self.xvals = 3*np.pi*np.random.random(self.npoints)
        self.points = self.generate_points(x=self.xvals)
        
    def model(self,x,params):
        
        return np.exp(-params[0]*x) * np.sin(params[1]*x+params[2]) * np.cos(params[3]*x+params[4])
    
    def model_batch(self, x, params):
        """evaluate multiple data points at once
        
        this can be much faster than doing them individually
        """
        return np.exp(-params[0]*x) * np.sin(params[1]*x + params[2]) * np.cos(params[3]*x + params[4])

    def model_gradient_batch(self, x, params):
        """return a matrix of gradients at each point
        
        Returns
        -------
        grad : array
            grad[i,j] is the gradient w.r.t. param[i] at point[j]
        """
        t1 = np.exp(-params[0]*x)
        t2 = np.sin(params[1]*x+params[2])
        t2der = np.cos(params[1]*x+params[2])
        t3 = np.cos(params[3]*x+params[4])
        t3der = -np.sin(params[3]*x+params[4])
        
        grad = np.zeros([params.size, x.size])
        grad[0,:] = -x * t1 * t2 * t3
        
        grad[1,:] = x * t1 * t2der * t3
        grad[2,:] = t1 * t2der * t3
        grad[3,:] = x * t1 * t2 * t3der
        grad[4,:] = t1 * t2 * t3der
        
        return grad
        
               
class PolynomialModel(BaseModel):
    
    def __init__(self,Porder,*args,**kwargs):

        super(PolynomialModel,self).__init__(*args,**kwargs)
        
        self.nparams = Porder
        self.Porder = Porder
        self.xvals = 3*np.pi*(np.random.random(self.npoints)-0.5)
        self.points = self.generate_points(x=self.xvals)
       
    def model(self,x,params):
        
        y = np.zeros(shape=np.shape(x))
        
        #for i,p in enumerate(params):
        for i,p in enumerate(params[:self.nparams/2]):
            #y += p*np.power(x,int(params[self.nparams/2+i]))
            y += p*np.power(x,params[self.nparams/2+i])
    
        return y
    
class MyModel(BaseModel):
    
    def __init__(self,Nfeatures=2,*args,**kwargs):
        super(MyModel,self).__init__(*args,**kwargs)

        self.Nfeatures = Nfeatures
        self.xvals = np.random.random((self.Nfeatures,self.npoints))
        self.points = self.generate_points(x=self.xvals)
    
    def model(self,x,params):
        
        #return params[0] + np.sqrt((params[1] - x)**2 + (params[2] - z)**2)         
        #return np.sqrt((params[0] - x)**2 + (params[1] - z)**2)         
        r = np.linalg.norm(x)
        #return np.tanh(params[0]+params[1]*r)
        #return params[0]*r/(params[1]+r)
        return r*np.abs(params[0])*1./(np.dot(params,params)+r)

class ErrorFunction(BasePotential):
    """a quadratic error function for fitting
    
    V(xi,yi|alpha) = 0.5 * (yi-(f(xi|alpha))**2
    where 
    """
    def __init__(self, model):
        """ instance of model to fit"""
        self.model = model
        self.points = np.array(self.model.points).reshape([-1,2])

    def getEnergyBatch(self, params):
        x = self.points[:,0]
        y = self.points[:,1]
        model_x = self.model.model_batch(x, params)
        
        E = 0.5 * np.sum((y - model_x)**2)
        return E
                
    def getEnergy(self, params):
        """return the derivative of the error function with respect to the params"""
        if hasattr(self.model, "model_batch"):
            return self.getEnergyBatch(params)
        E = 0.
        for xi,yi in self.points:
            E += 0.5 * (yi - self.model.model(xi,params))**2
        
        # regularization
        #E = E + 0.1*np.dot(params,params)
        return E

    def getEnergyGradient(self, params):
        if not hasattr(self.model, "model_gradient_batch"):
            return self.getEnergy(params), self.NumericalDerivative(params)
        
        x = self.points[:,0]
        y = self.points[:,1]
        model_y = self.model.model_batch(x, params)
        energy = 0.5 * np.sum((y - model_y)**2)

        model_grad = self.model.model_gradient_batch(x, params)
        
        grad = -model_grad.dot(y - model_y)
        assert grad.size == params.size
        
        return energy, grad.ravel()
        

def do_nothing_mindist(x1, x2):
    # align the center of mases
    #dr = np.mean(x1) - np.mean(x2)
    #x2 += dr
    dist = np.linalg.norm(x2 - x1)
    return dist, x1, x2

def my_orthog_opt(vec,coords):
    
    return vec

from regression_utils import TransformRegression,MeasureRegression

class MinPermDistWaveModel(object):
    
    def __init__(self, niter=10, verbose=False, tol=0.01, accuracy=0.01,
                 phase_translations=[2,4],point_inversions=[1,2,3,4],perm_groups=[[1,2],[3,4]],
                 measure=MeasureRegression(),transform=TransformRegression()):
        self.niter = niter
        
        self.verbose = verbose
        self.measure=measure
        self.transform=transform
        self.accuracy = accuracy
        self.tol = tol

        self.phase_translations = np.array(phase_translations)
        self.point_inversions = np.array(point_inversions)
        self.perm_groups = np.array(perm_groups)
        
    def check_discrete_translational_symmetry(self,x1, x2):
        """ paramlist:
            list of parameter indices with symmetry about translation by  2*pi*n
        """
        for p in self.phase_translations:
            eps = np.abs(x2[p]-x1[p])/(2*np.pi)
            x2[p] -= np.floor(eps)*2*np.pi
        dist = np.linalg.norm(x2 - x1)
        return dist, x2
    
    def __call__(self,coords1,coords2):
        
        # we don't want to change the given coordinates
        check_inversion = False
        coords1 = coords1.copy()
        coords2 = coords2.copy()
        
        x1 = np.copy(coords1)
        x2 = np.copy(coords2)
    
        self.distbest = self.measure.get_dist(x1, x2)
        self.x2best = x2.copy()
        
        if self.distbest < self.tol:
            return self.distbest, coords1, x2
        
        """translational symmetry of phase factors"""
        dist,x2 = self.check_discrete_translational_symmetry(x1,x2)
        if dist < self.distbest: 
            self.distbest = dist
            self.x2best = x2
                
        """permutational symmetry"""
        self.transform.permute(x2,self.perm_groups)
        dist = self.measure.get_dist(x1, x2)
        if dist < self.distbest: 
            self.distbest = dist
            self.x2best = x2
        if self.distbest < self.tol:
            return self.distbest, coords1, x2        
        
        """point inversion symmetry"""
        self.transform.invert(x2,self.point_inversions)
        dist = self.measure.get_dist(x1, x2)
        if dist < self.distbest: 
            self.distbest = dist
            self.x2best = x2
        if self.distbest < self.tol:
            return self.distbest, coords1, x2        
        
        
        return self.distbest, coords1, self.x2best
                    
from pele.systems import BaseSystem
from pele.mindist import MinPermDistAtomicCluster, ExactMatchAtomicCluster
from pele.transition_states import orthogopt, orthogopt_translation_only


class RegressionSystem(BaseSystem):
    def __init__(self, model):
        super(RegressionSystem, self).__init__()
        self.model = model
        self.params.database.accuracy =0.01
        self.params.double_ended_connect.local_connect_params.tsSearchParams.hessian_diagonalization = True

    def get_potential(self):
        return ErrorFunction(self.model)
    
    def get_mindist(self):
        # no permutations of parameters
        
        #return mindist_with_discrete_phase_symmetry
        #permlist = []
        return MinPermDistWaveModel( niter=10)

    def get_orthogonalize_to_zero_eigenvectors(self):
        return None
        return my_orthog_opt
        #return orthogopt
        #return orthogopt_translation_only
    
    def get_minimizer(self, **kwargs):
        return lambda coords: myMinimizer(coords, self.get_potential(),**kwargs)
    
    def get_compare_exact(self, **kwargs):
        # no permutations of parameters
        mindist = self.get_mindist()
        return lambda x1, x2: mindist(x1, x2)[0] < 1e-3

def myMinimizer(coords,pot,**kwargs):
    from pele.optimize import lbfgs_cpp as quench
    #print coords
    return quench(coords,pot,**kwargs)


def run_basinhopping(model, nsteps):
    

    system = RegressionSystem(model)
    #print system.get_potential().getEnergy(alpha)

    database = system.create_database()
    #x0 = np.random.uniform(-1,1,[porder])
    x0 = np.random.uniform(0.,3,[model.nparams])
    
    step = RandomCluster(volume=1.0)
    bh = system.get_basinhopping(database=database, takestep=step,coords=x0,temperature = 10.0)
    #bh.stepsize = 20.
    bh.run(nsteps)
    print "found", len(database.minima()), "minima"
    min0 = database.minima()[0]
    print "lowest minimum found has energy", min0.energy
    m0 = database.minima()[0]
    mindist = system.get_mindist()
    for m in database.minima():
        dist = mindist(m0.coords, m.coords.copy())[0]
        print "   ", m.energy, dist, m.coords
    return system, database

def run_double_ended_connect(system, database):
    # connect the all minima to the lowest minimum
    from pele.landscape import ConnectManager
    manager = ConnectManager(database, strategy="gmin")
    for i in xrange(database.number_of_minima()-1):
        min1, min2 = manager.get_connect_job()
        connect = system.get_double_ended_connect(min1, min2, database)
        connect.connect()

def make_disconnectivity_graph(database):
    from pele.utils.disconnectivity_graph import DisconnectivityGraph, database2graph
    import matplotlib.pyplot as plt
    
    graph = database2graph(database)
    dg = DisconnectivityGraph(graph, nlevels=3, center_gmin=True)
    dg.calculate()
    dg.plot()
    plt.show()

def main():    
    params=[0.1,1.0,0.0,0.0,0.0]
    model = WaveModel(params0=params,sigma=0.1)
    mysys, database = run_basinhopping(model,10)

    #multiple = False
    #while multiple==False:
    #    mysys, database = run_basinhopping(model,20)
    #    multiple=True
    #    if len(database.minima()) > 1: multiple=True
    
    for m in database.minima():
        x = np.array(mysys.model.points)[:,0]
        plt.plot(x,[mysys.model.model(xi,m.coords) for xi in x],'o')

    plt.plot(x,np.array(mysys.model.points)[:,1],'x')#

    plt.plot(x,mysys.model.model(x,mysys.model.params0),'x')
    plt.show()
    #print mysys.model.points
    #np.savetxt("points",mysys.model.points)
    
    #R = np.sqrt(mysys.model.points[:,0]**2+mysys.model.points[:,1]**2)
    #plt.plot(R,mysys.model.points[:,2],'x')
    #plt.show()
    #exit()
    #exit()
    #run_double_ended_connect(mysys, database)
    #make_disconnectivity_graph(database)
    
    for m in database.minima():
        print m.coords
    exit()

    #alpha = [0.0]
    #nparams = len(alpha)
    #natoms = nparams    
    #model = MyModel(alpha)
    #model = LinearModel(npoints=200)
    #model = mysys.model
    #system = RegressionSystem(natoms,model)

    Es = []
#a = np.arange(-1.0,1.0,0.1)
    #A.append(a)
    
    #Theta=np.arange(-0.5*np.pi,0.5*np.pi,0.01)
    #for t in Theta:
    #    Es.append(system.get_potential().getEnergy([t]))    #print mysys.get_potential().getEnergy((1.0,1.05,1.05))    
    lowest = database.minima()[0].coords
    highest = database.minima()[-1].coords
    #lowest = [1.,1.]
    #highest = [1.,1.]
    #B=np.arange(lowest[0]-1,highest[0]+1,(highest[0]-lowest[0]+2)/100.)
    #M=np.arange(lowest[1]-1,highest[1]+1.,(highest[1]-lowest[1]+2)/100.)
    B=np.arange(0.,1.,.01)
    M=np.arange(0.,1.,.01)
    
    Es = np.zeros((len(B),len(M)))
    for i,b in enumerate(B):
        for j,m in enumerate(M):
            Es[i,j] = mysys.get_potential().getEnergy([b,m])    #print mysys.get_potential().getEnergy((1.0,1.05,1.05))    
    
    B,M = np.meshgrid(B,M)
    fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')
    ax = Axes3D(fig)
    #ax.contour(B,M,Es, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=False)
    #ax.plot_surface(B,M,Es, rstride=4, cstride=4, linewidth=0)
    #plt.show()
    #ax.zaxis.set_scale('log')
    ax.set_zlim([0.,10.])
    ax.plot_surface(B,M,Es, rstride=4, cstride=4, linewidth=0)
    plt.show()
    
    exit()
    #plt.plot(np.tan(Theta),Es,'x')
    #plt.plot(Theta,Es,'x-')
    #plt.xlim([-10,10])
    #plt.show()
    #exit()
    #print mysys.get_potential().getEnergy((1.0,-1.05,1.04))    
    #print mysys.get_potential().getEnergy((m.coords[0],m.coords[1]))    
    #print mysys.get_potential().getEnergy((m.coords[0],-m.coords[1]))    
    points = np.array(mysys.model.points)
    #plt.plot(points[:,0],np.sqrt(np.power(points[:,1],2)+np.power(points[:,2],2)),'x')
    plt.plot(points[:,0],points[:,1],'x')
    fit = [points[:,0],np.array([m.coords[0]*p for p in points[:,0]])]
    plt.plot(fit[0],fit[1],'-')
    

    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.contour(points[:,0],points[:,1],np.diag(points[:,2]), rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=False)
    #plt.show()
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    #p = ax.plot_surface(points, rstride=4, cstride=4, linewidth=0)
    
    #plt.plot(points[:,2],m.coords[0]+np.sqrt(np.power(points[:,0]-m.coords[1],2)+np.power(points[:,1]-m.coords[2],2)),'x')
    #fit = [points[:,0],np.array([m.coords[0]+m.coords[1]*p for p in points[:,0]])]

    #plt.plot(fit[0],fit[1])
    #plt.show()

def main2():
    """a copy of main to clean it up a bit"""
    np.random.seed(0)
    params=[0.1,1.0,0.0,0.0,0.0]
    model = WaveModel(params0=params,sigma=0.1)
    system = RegressionSystem(model)
    database = system.create_database()
    pot = system.get_potential()

    # do basinhopping
    x0 = np.random.uniform(0.,3,[model.nparams])
    print pot.getEnergy(x0)
    pot.test_potential(x0)
    step = RandomCluster(volume=1.0)
    bh = system.get_basinhopping(database=database, takestep=step,coords=x0,temperature = 10.0)
    bh.run(20)
    
    
    # connect the minima
    manager = ConnectManager(database, strategy="gmin")
    for i in xrange(2):
        try:
            m1, m2 = manager.get_connect_job()
        except manager.NoMoreConnectionsError:
            break
        connect = system.get_double_ended_connect(m1, m2, database)
        connect.connect()
        
        
    make_disconnectivity_graph(database)
    
    for m in database.minima():
        print m.energy,m.coords/(2*np.pi)
        
    m1 = database.minima()[0]
    m2 = database.minima()[1]
    print mindist_with_discrete_phase_symmetry(m1.coords,m2.coords)[0]


if __name__ == "__main__":
    main2()
    