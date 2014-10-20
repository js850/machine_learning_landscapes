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
            x=np.random.random()
        self.points = np.concatenate((self.points,[[x,self.model(x)+self.noise*np.random.normal()]]))
            
class GaussianProcess():
    def __init__(self,ts,beta=1.0,p=1.0,l=1.0):
        self.ts = ts
        self.beta = beta
        self.p_kernel = p
        self.l_kernel = l
        #self.kernel = kernel
        self.C = self.initialize_covariance_matrix()
        
    def compute_kernel(self,xlist,x):
        
        if type(xlist) is np.float64:
            return self.kernel(xlist,x,L=self.l_kernel)
            #return self.kernel(xlist,x,l=self.l_kernel,p=self.p_kernel)
        
        K=np.array([])
        for i,xi in enumerate(xlist):
            K = np.append(K,self.kernel(xi,x,L=self.l_kernel))
            #K = np.append(K,self.kernel(xi,x,l=self.l_kernel,p=self.p_kernel))
            #K[i] = squared_exponential(xi,x,L=self.l_kernel)
        
        return K
        #return periodic_kernal(*args,l=self.l_kernel,p=self.p_kernel)
        #return squared_exponential(*args,L=self.l_kernel)
    
    def kernel(self,*args,**kwargs):
        #return periodic_kernal(*args,**kwargs)
        return squared_exponential(*args,**kwargs)
    
    def run(self,niterations):
        for i in xrange(niterations):
            self.run_single_iteration()
            #if i % 10==0: self.visualize()
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
        
        return C
        
    def update_covariance_matrix(self):
      
        #xnarray = np.array([self.ts.points[-1] for i in range(len(self.ts.points[:,0]))])
        xlast = self.ts.points[-1,0]
        K = self.compute_kernel(self.ts.points[:,0],xlast)

        c = self.compute_kernel(xlast,xlast) + self.beta
        
        self.C = np.append(self.C,[K[:-1]],axis=0)
        self.C = np.append(self.C,np.array([K]).T,axis=1)
        
        print "det: ", np.linalg.det(self.C),np.shape(self.C)
        
        
    def query_point(self,x):
        

        Cinv = np.linalg.inv(self.C)

        #print Cinv
        #print self.C

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
     
        plt.plot(self.ts.points[:,0],self.ts.points[:,1],'x')
        plt.ylim([-1.5,1.5])
        #plt.show()
        #exit()
        
def main():
    ts = TrainingSet(genpoints=1)

    gp = GaussianProcess(ts,l=1.0,p=1.0,beta=ts.noise)
    #gp.run(100)

    
    for i in range(6,20,2):
        gp.run(2)
    
        x = np.arange(0,1.0,0.01)

        gp.generate_curve(x)
        plt.plot(x,gp.ts.model(x),'--')

        #evals = np.linalg.eigvals(gp.C)
        #plt.plot(evals)
    
    evals,evecs = np.linalg.eig(gp.C)
    
    print evals[-1]
    print evecs[:,-1]
    exit()
    #plt.yscale('log')
    plt.ylim([-1.5,1.5])
    plt.show()


if __name__=="__main__":
    main()  
