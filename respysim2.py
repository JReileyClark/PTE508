import numpy as np


# class Reservoir(object):
#     def __init__(self, N=None,rw=None,re=None, h=None, pi=None, pa=None,pwf=None,k=None,phi=None,muo=None,muw=None,c_phi=None,c_o=None,c_w=None, Boi=None, Bwi=None, Swi=None, Sawi=None, BC=None):
#         self.N = N
#         self.rw = rw
#         self.re = re
#         self.h = h
#         self.pi = pi
#         self.pa = pa
#         self.pwf = pwf
#         self.k = k
#         self.phi = phi
#         self.muo = muo
#         self.muw = muw
#         self.c_phi = c_phi
#         self.c_o = c_o
#         self.c_w = c_w
#         self.Boi = Boi
#         self.Bwi = Bwi
#         self.Swi = Swi
#         self.Sawi = Sawi
#         self.Soi = 1-Swi
#         self.dchi = (np.log(self.re) - np.log(self.rw))/(N-1)
#         self.ra = self.re*np.exp(self.dchi)
#         self.X = self.chinodes()
#         self.R = self.rnodes()
#         self.BC = BC
#     def chinodes(self):
#         n = np.zeros(self.N+2)
#         n[1:-1] = np.arange(0.5,0.5+self.N,1)
#         n[-1] = self.N
#         n *= self.dchi
#         n += np.log(self.rw)
#         return n
#     def rnodes(self):
#         return np.exp(self.X)
#     def bo(self,p):
#         return self.Boi*np.exp(-self.c_0*(p-self.pi))
#     def bw(self,p):
#         return self.Bwi*np.exp(-self.c_w*(p-self.pi))
#     def phi(self,p):
#         return phi*np.exp(self.c_phi*(p-self.pi))
#     def kro(self,Sw):
#         return 1.77*(1-Sw)**2 if 0.25 < Sw <= 1 else 1
#     def krw(self,Sw):
#         return 2.37*(Sw - 0.25)**3 if 0.25 < Sw <= 1 else 0
#     def mobility(self,Sw,p):
#         return [self.k*kro(Sw)/(muo*bo(p)), self.k*krw(Sw)/(muw*bw(p))]


#Define Reservoir  Properties
res_props = dict(rw=0.25, #wellbore radius
                 re=1500, #reservoir radius
                 h=50, #reservoir thickness
                 pi=3000, #initial pressure
                 pa=3000, #pressure at the aquifer
                 pwf=1000,#pressure at the wellbore
                 k=100,#permeability (isotropic)
                 phii=0.2,#porosity
                 muo=2, #oil viscosity, cp
                 muw=1, #water viscosity, cp
                 c_phi=1.0e-5, #formation compressibility
                 c_o=1.5e-5, #oil compressibility
                 c_w=0.3e-5, #water compressibility
                 Boi=1.25, #initial oil formation volume factor
                 Bwi=1.05, #initial water formation volume factor
                 Swi=0.25, #initial water saturation
                 Sawi=1, #initial water saturation in aquifer
                 BC=[1000,3000]) #pressure boundary conditions, [pwf, pe]

sim_props = dict(T=30, #total time into the future to simulation
                 tstep=1, # time step size
                 N=11) # number of nodes (1d)


#constructing functions for mobility calculations

def kbar(kdown,kup): #kdown is the downstream-pressure perm, kup is the upstream-pressure perm
    return 2*kdown*kup/(kup +kdown)

def bo(boi,coi,pi):
    def func(p):
        return  boi* np.exp(-coi * (p - pi))
    return func

def bw(bwi, cwi, pi):
    def func(p):
        return bwi* np.exp(-cwi * (p - pi))
    return func

def kro(Sw):
    return 1.77 * (1 - Sw) ** 2 if 0.25 < Sw <= 1 else 1


def krw(Sw):
    return 2.37 * (Sw - 0.25) ** 3 if 0.25 < Sw <= 1 else 0

def mobility(Swup,pdown,pup,kdown,kup,b,mu,fluid): #b is a function here
    p=(pdown+pup)/2
    if fluid == 'w':
        return kbar(kdown,kup)*krw(Swup)/(mu*b(p))
    elif fluid == 'o':
        return kbar(kdown,kup)*kro(Swup)/(mu*b(p))

def compressibility(cf,co,cw): #total compressibility. This returns a function
     def func(so,sw):
        return cf + co*so + cw*sw
     return func

def ra(re,dchi):
    return re*np.exp(dchi)

def dchi(N,rw,re):
    return np.log(re/rw)/(N-1)

def chinodes(N,dchi,rw):
    n = np.zeros(N+2)
    n[1:-1] = np.arange(0.5,0.5 + N,1)
    n[-1] = N
    n *= dchi
    n += np.log(rw)
    return n

def phi(phii,c_phi,pi):
    def func(p):
        return phii*np.exp(c_phi*(p-pi))
    return func

#initialize the closure, compressibility. ct is a function that takes so and sw as parameters
ct = compressibility(res_props['c_phi'],res_props['c_o'],res_props['c_w'])
delchi = dchi(sim_props['N'],res_props['rw'],res_props['re'])
chi = chinodes(sim_props['N'],delchi,res_props['rw'])

#constructing the matrix of pressures our simulation results will go into
Pmat = np.zeros((sim_props['T']//sim_props['tstep']+1,sim_props['N'],10))

#setting the outer columns (boundaries) to their respective conditions
#Pmat[:,0] = res_props['BC'][0]
#Pmat[:,-1] = res_props['BC'][1]

#intialize closures
bo_p = bo(res_props['Boi'],res_props['c_o'],res_props['pi'])
bw_p = bw(res_props['Bwi'],res_props['c_w'], res_props['pi'])

#setting the initial condition (t=0)
Pmat[0 , : , 0] = res_props['pi']
Pmat[0,:-1,1] = res_props['Swi']
Pmat[0,-1,1] = res_props['Sawi']
Pmat[0,:,2] = [kro(x) for x in Pmat[0,:,1]]
Pmat[0,:,3] = [krw(x) for x in Pmat[0,:,1]]

#Initialize Left Node Lambda o West
Pmat[0,0,4] = res_props['k']*Pmat[0,0,2]/(res_props['muo']*bo_p(Pmat[0,0,0]))

#Initialize Remaining Lambda o Wests
Pmat[0,1:,4] = res_props['k']*Pmat[0,1:,2]/(res_props['muo']*bo_p(Pmat[0,1:,0]))

#Initialize Right Node Lambda o East
Pmat[0,-1,6] = res_props['k']*Pmat[0,0,3]/(res_props['muw']*bw_p(Pmat[0,0,0]))

#Initialize Remaining Lambda o Easts
Pmat[0,:-1,6] = res_props['k']*Pmat[0,1:,3]/(res_props['muw']*bw_p(Pmat[0,1:,0]))

#Intialize Right  Lambda w East
Pmat[0,-1,]
# Sw_mat = np.zeros((sim_props['T']//sim_props['tstep']+1,sim_props['N']))
# Sw_mat[0,:-1] = res_props['Swi']
# Sw_mat[0,-1] = res_props['Sawi']

#print(Sw_mat)
print("Initial Sw: \n",Pmat[0,:,1])
print("Nodes in Chi space: ", chi,"Nodes in r space: ", np.exp(chi), sep='\n')
print("Initial Pressures: \n",Pmat[0,:,0])
print("All Initial Conditions: ",Pmat[0,:,:], sep=' \n')









# if __name__ == '__main__':
#     res_props = dict(N=10,rw=0.25,re=1500,h=50,pi=3000,pa=3000,pwf=1000,k=10,phi=0.2,muo=2,muw=1,c_phi=1.0e-5,c_o=1.5e-5,c_w=0.3e-5,Boi=1.25,Bwi=1.05,Swi=0.25,Sawi=1,BC=[1000,3000])
#     print(res_props)
#     res = Reservoir(**res_props)
#     print(res.X,np.exp(res.X),sep=' ')
#     print(res.dchi)
#     print(res.rnodes())
#     print(res.kro(0.33),res.krw(0.25))