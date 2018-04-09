from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from scipy.optimize import fmin
from scipy.integrate import quad
import math
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os


def get_plot_palette(colors,n,cmap=None):
    if cmap==None:
        cmap = plt.get_cmap('cubehelix_r')
    cmap.set_bad('white',1.)
    cols=[cmap(i) for i in np.linspace(0.1,1,n)]
    if colors!=None:
        if all([isinstance(a,float) for a in colors]):
            cols=[cmap(i) for i in colors]
        elif all([isinstance(a,str) for a in colors]):
            cols=colors
        else:
            print("Warning, color palette not recognized")
    return cols

class smodel(object):
    def __init__(self,a0,q,p,d):
                self.a0=a0
                self.q=q
                self.p=p
                self.xl=0.
                self.xu=1.
                self.d=d
    def f(self,x):
        fitness=(self.a0*min(1,x+self.d)**self.q+(1-self.a0)*(min(1,(1-x)+self.d))**self.q)*self.p
        return ((fitness/self.a0)**(1/self.q))
    def fg(self,x):
        fitness=(self.a0*min(1,x+self.d)**self.q+(1-self.a0)*(min(1,(1-x)+self.d))**self.q)*self.p
        return (fitness)
    def fs(self,x):
        s0=1
        fitness=self.a0*s0**self.q
        return fitness

class smodel_adaptation(object):
    """
    Considers the fitness of agents after they have reached specialization.
    Now delta can be seen as the ratio between the length of the season and a constant speed of learning:
    If delta is low, the season is short and so learning can adapt only up to a certain point, if delta is high the season is long and so the agent can specialize and profit from specialization for the remaining time.
    """
    def __init__(self,a0,q,p,d):
                self.a0=a0
                self.q=q
                self.p=p
                self.xl=0.
                self.xu=1.
                self.d=d
    def __skill2fitness(self,x):
        """
        Assume the relation between skill and fitness is a polinomial of power self.q
        """
        adaptation=[x,min(1,x+self.d)] # extremes where it is still adapting
        adapted=[1,max(1,x+self.d)]    # for how long it is adapted
        skill_adaptation=quad(lambda x: x**self.q,adaptation[0],adaptation[1])[0] # the area below the curve
        skill_adapted=adapted[1]-adapted[0]                               # the area of the rectangle with height 1 and base 'adapted'
        return skill_adaptation+skill_adapted
    def f(self,x):
        # compute the fitness of the adaptation as the area of a triangle of base min(1,x+delta)-x
        fitness=(self.a0*self.__skill2fitness(x)+(1-self.a0)*self.__skill2fitness(1-x))*self.p
        return ((fitness/self.a0)**(1/self.q))
    def fg(self,x):
        fitness=(self.a0*self.__skill2fitness(x)+(1-self.a0)*self.__skill2fitness(1-x))*self.p
        return (fitness)
    def fs(self,x):
        s0=1
        fitness=self.a0*s0**self.q
        return fitness

#########################################################
# Code to plot fitness comparisons with varying deltas  #
#########################################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator

@np.vectorize
def f(x,a0,q,d):
    return a0*min(1,x+d)**q+(1-a0)*min(1,1-x+d)**q

wd="./"
a0=0.5
x=np.linspace(0,1,1000)
qs=[1,2,3,10]
for q in qs:
    ds=[0.5,0.52,0.55,0.58,0.6,0.7]
    fig=plt.figure()
    for d in ds:
        m=smodel_adaptation(a0,q,1,d)
        y=[m.f(i) for i in x]
        plt.plot(x,y,label="Delta: "+str(d))
        idxs=[i for i,j in enumerate(y) if round(j,4)==round(max(y),4)]        # find maxes
        idxs=list(np.unique([idxs[0],idxs[-1]])) # find first and last occurrence, take only one value if the extremes are the same
        for i in idxs:
            c=plt.Circle((x[i],y[i]),0.01,color="red",fill=False)
            plt.gca().add_artist(c)
    plt.title("Q: "+str(q))
    plt.gca().axhline(a0,label="Specialist",linestyle="dotted",color="grey")
    plt.gca().legend()
    fig.savefig(os.path.join(wd,"comparison_"+str(q)+".pdf"),format='pdf')
