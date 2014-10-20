import numpy as np
import matplotlib.pyplot as plt

from gaussian_process_examples import periodic_kernal,squared_exponential

class TrainingSet():
    def __init__(self,points=[],genpoints=None,noise=0.1):
        self.points = np.array(points)
        self.noise = noise
        if genpoints != None: 
            self.initialize_array()
            self.get_points(n=genpoints)
    
    def model(self,x):
        return np.sin(2.*np.pi*x)
    
    def initialize_array(self):
        x=np.random.random()        
        self.points = np.array([[x,self.model(x)+self.noise*np.random.normal()]])
    
    def get_points(self,n=1):
        for i in xrange(n): self.gen_new_point()
    
    def gen_new_point(self,x=None):
        if x==None:
            #x = gen_biased_x()
            x=np.random.random()
            if 0.7 <= x <= 0.8 and np.random.random()<0.99:
                x = x - 0.2
        self.points = np.concatenate((self.points,[[x,self.model(x)+self.noise*np.random.normal()]]))
    
class GaussianProcess():
    def __init__(self,ts,kernel=None,beta=1.0):
        self.ts = ts
        self.beta = beta
        self.kernel = kernel

        self.C = self.initialize_covariance_matrix()
        
    def compute_kernel(self,xlist,x):
        
        if type(xlist) is np.float64:
            return self.kernel(xlist,x)
        
        K=np.array([])
        for i,xi in enumerate(xlist):
            K = np.append(K,self.kernel(xi,x))
       
        return K

    
    def run(self,niterations):
        for i in xrange(niterations):
            self.run_single_iteration()

    def run_single_iteration(self):
        self.get_new_point()
        self.update_covariance_matrix()
    
    def get_new_point(self,x=None):
        return self.ts.gen_new_point()
    
    def visualize(self):
        plt.plot(self.ts.points[:,0],self.ts.points[:,1],'x') 
        plt.show()
           
    def initialize_covariance_matrix(self):
        
        x = self.ts.points[:,0]
        C = np.zeros(shape=(len(x),len(x)))
        
        for i,xi in enumerate(x):
            for j,xj in enumerate(x):
                C[i,j] = self.compute_kernel(xi,xj)
                if i==j: C[i,j] += 1./self.beta
        
        return C
        
    def update_covariance_matrix(self):
      
        #xnarray = np.array([self.ts.points[-1] for i in range(len(self.ts.points[:,0]))])
        xlast = self.ts.points[-1,0]
        K = self.compute_kernel(self.ts.points[:,0],xlast)

        #c = self.compute_kernel(xlast,xlast) + 1./self.beta
        #print self.C
        #print K
        self.C = np.append(self.C,[K[:-1]],axis=0)
        self.C = np.append(self.C,np.array([K]).T,axis=1)
        self.C[-1,-1] += 1./self.beta
        
        print "det: ", np.linalg.det(self.C),np.shape(self.C)
        
        #exit()
        
    def query_point(self,x):
        
        Cinv = np.linalg.inv(self.C)
        t = self.ts.points[:,1]
        k_me = self.compute_kernel(self.ts.points[:,0],x)
        c = self.compute_kernel(x,x)+1./self.beta
        
        mu = np.dot(k_me.T,Cinv.dot(t))
        var = c - np.dot(k_me.T,Cinv.dot(k_me))

        return mu, var
    
    def generate_curve(self,xvals):
        
        vals = np.zeros(shape=(len(xvals),3))
        
        for i,x in enumerate(xvals):
            mu,var = self.query_point(x)
            vals[i,:] = x,mu,var
            #vals = np.append(vals,[x,mu,val])
     
        plt.plot(vals[:,0],vals[:,1])
        plt.plot(vals[:,0],vals[:,1]+np.sqrt(vals[:,2]))
        plt.plot(vals[:,0],vals[:,1]-np.sqrt(vals[:,2]))
        plt.plot(self.ts.points[:,0],self.ts.points[:,1],'x')
        #plt.ylim([-1.5,1.5])
        #plt.show()
        #exit()
        
def main():
    ts = TrainingSet(genpoints=100,noise=0.1)
    
    l=0.4
    p=1.1
    
    #kernel = lambda x1,x2 : squared_exponential(x1,x2,L=l)
    kernel = lambda x1,x2 : periodic_kernal(x1,x2,l=l,p=p)
    
    x = np.arange(0,1.0,0.01)

    gp = GaussianProcess(ts,kernel=kernel,beta=1./ts.noise)
    
    #plt.plot(x,gp.ts.model(x),'--')
    gp.generate_curve(x)
    plt.show()
    exit()
    
    for i in range(6,8,2):
        gp.run(2)
    
        x = np.arange(0,1.0,0.01)

        gp.generate_curve(x)
        plt.plot(x,gp.ts.model(x),'--')

        #evals = np.linalg.eigvals(gp.C)
        #plt.plot(evals)
    
    #evals,evecs = np.linalg.eig(gp.C)
    
    #print evals[-1]
    #print evecs[:,-1]
    #exit()
    #plt.yscale('log')
    plt.ylim([-1.5,1.5])
    plt.show()


if __name__=="__main__":
    main()  
