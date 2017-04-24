import numpy as np
from tdma import tdma
from scipy import sparse as sp
from matplotlib import pyplot as pl
import operator
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
res_props = dict(rw=100, #wellbore radius
                 re=1500, #reservoir radius
                 h=50, #reservoir thickness
                 pi=3000, #initial pressure
                 pa=3000, #pressure at the aquifer
                 pwf=1000,#pressure at the wellbore
                 k=10,#permeability (isotropic)
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

def compressibility(c_phi,co,cw): #total compressibility. This returns a function
     def func(sw,so):
        return c_phi + co*so + cw*sw
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

#Initialize nodes
delchi = dchi(sim_props['N'],res_props['rw'],res_props['re'])
chi = chinodes(sim_props['N'],delchi,res_props['rw'])

print(delchi)
print(chi,chi.shape)
print(np.exp(chi))
print(ra(1500,delchi))



#intialize closures
bo_p = bo(res_props['Boi'],res_props['c_o'],res_props['pi'])
bw_p = bw(res_props['Bwi'],res_props['c_w'], res_props['pi'])
phi_p = phi(res_props['phii'],res_props['c_phi'],res_props['pi'])
ct = compressibility(res_props['c_phi'],res_props['c_o'],res_props['c_w'])

#constructing the matrix of pressures our simulation results will go into
Pmat = np.zeros((sim_props['T']//sim_props['tstep']+1,sim_props['N'],12))
print(Pmat.shape)
print(Pmat[0,1:-1,0].shape)
#0 Pressure
#1 Sw
#2 Kro
#3 Krw
#4 Bo
#5 Bw
#6 Lambda_oW
#7 Lambda_oE
#8 Lambda_wW
#9 Lambda_wE
#10 Phi
#11 Ct

#setting the initial condition (t=0)
Pmat[0 , : , 0] = res_props['pi']
Pmat[0,:-1,1] = res_props['Swi']
Pmat[0,-1,1] = res_props['Sawi']
Pmat[0,:-1,2] = [kro(x) for x in Pmat[0,1:,1]]
Pmat[0,:-1,3] = [krw(x) for x in Pmat[0,1:,1]]
Pmat[0,-1,3] = 1
Pmat[0,-1,2] = 0
#Pmat[0,-1,2] = kro(Pmat[0,-1,1])
Pmat[0,-1,3] = 1 #krw(Pmat[0,-1,1])
print("KRO: \n",Pmat[0,:,2])
print("KRW: \n", Pmat[0,:,3])
#Initialize Bo
Pmat[0,:,4] = bo_p(Pmat[0,:,0])
#Initialize Bw
Pmat[0,:,5] = bw_p(Pmat[0,:,0])

#Initialize Left Node Lambda o West
Pmat[0,0,6] = res_props['k']*Pmat[0,0,2]/(res_props['muo']*bo_p(Pmat[0,0,0]))
#Initialize Remaining Lambda o Wests
Pmat[0,1:,6] = res_props['k']*Pmat[0,1:,2]/(res_props['muo']*bo_p((Pmat[0,1:,0] + Pmat[0,:-1,0])/2))
print("Pmat[i,:,6], Lambda_o West: \n",Pmat[0,:,6])
#Initialize Right Node Lambda o East
Pmat[0,-1,7] = res_props['k']*Pmat[0,-1,2]/(res_props['muo']*bo_p(Pmat[0,-1,0]))

#Initialize Remaining Lambda o Easts
Pmat[0,:-1,7] = res_props['k']*Pmat[0,:-1,2]/(res_props['muo']*bo_p((Pmat[0,1:,0] + Pmat[0,:-1,0])/2))
print("Pmat[i,:,7], Lambda_o East: \n",Pmat[0,:,7])

#Initialize Left Node Lambda w West
Pmat[0,0,8] = res_props['k']*Pmat[0,0,3]/(res_props['muw']*bw_p(Pmat[0,0,0]))

#Initialize Remaining Lambda w Wests
Pmat[0,1:,8] = res_props['k']*Pmat[0,1:,3]/(res_props['muw']*bw_p((Pmat[0,1:,0] + Pmat[0,:-1,0])/2))

#Initialize Right Node Lambda w East
Pmat[0,-1,9] = res_props['k']*Pmat[0,-1,3]/(res_props['muw']*bw_p(Pmat[0,-1,0]))

#Initialize Remaining Lambda w Easts
Pmat[0,:-1,9] = res_props['k']*Pmat[0,:-1,3]/(res_props['muw']*bw_p((Pmat[0,1:,0] + Pmat[0,:-1,0])/2))

#Initialize phi
Pmat[0,:,10] = phi_p(Pmat[0,:,0])

#Initialize ct
Pmat[0,:,11] = ct(Pmat[0,:,1],1 - Pmat[0,:,1])

#Solving for pressure
# for i in range(sim_props['T']/sim_props['tstep'])

for i in range(30):
    ################ PRESSURE CALCULATION STEP ################################
    alpha = (158*np.exp(2*chi[1:-1])*Pmat[i,:,10]*Pmat[i,:,11]*(delchi*delchi))
    alpha[0] *= 3
    alpha[-1] *= 3
    #print(chi[1:-1])
    print("ALPHA: \n", alpha)
    print("SW: \n", Pmat[i,:,1])
    print("CT:  \n", Pmat[i,:,11])
    #print("ALPHA[0]: \n", alpha[0])
    #print("ALPHA[-1]: \n", alpha[-1])
    A = Pmat[i,1:,6]*Pmat[i,1:,4] + Pmat[i,1:,8]*Pmat[i,1:,5]
    A[-1] *=4
    B = -(Pmat[i,:,4]*(Pmat[i,:,6] + Pmat[i,:,7]) + Pmat[i,:,5]*(Pmat[i,:,8] + Pmat[i,:,9]))
    B[0] =  -(Pmat[i,0,4]*(8*Pmat[i,0,6] + 4*Pmat[i,0,7]) + Pmat[i,0,5]*(8*Pmat[i,0,8] + 4*Pmat[i,0,9]))
    B[-1] = -(Pmat[i,-1,4]*(4*Pmat[i,-1,6] + 8*Pmat[i,-1,7]) + Pmat[i,-1,5]*(4*Pmat[i,-1,8] + 8*Pmat[i,-1,9]))
    B -= alpha
    #print(B)
    C =  Pmat[i,:-1,7]*Pmat[i,:-1,4] + Pmat[i,:-1,9]*Pmat[i,:-1,5]
    C[0] *= 4
    D = -alpha*Pmat[i,:,0]
    #print(D)
    D[0] -= 8*(((Pmat[i,0,6]*Pmat[i,0,4]) + (Pmat[i,0,8]*Pmat[i,0,5]))*res_props['pwf'])
    #print(D)
    D[-1] -= 8*(((Pmat[i,-1,7]*Pmat[i,-1,4]) + (Pmat[i,-1,9]*Pmat[i,-1,5]))*res_props['pi'])
    #print(D)
    E = tdma(A,B,C,D)
    #print("E: \n",E)
    #F = np.eye(sim_props['N'],k=-1)*A + np.eye(sim_props['N'],k=0)*B + np.eye(sim_props['N'],k=1)*C
    #print("NP LINALG SOLVE SOLUTION: \n",np.linalg.solve(F,D))
    Pmat[i+1,:,0] = E
    ###########VALUE UPDATE STEPS########################
    print((Pmat[i, 1:, 0] + Pmat[i, :-1, 0])/2)
    # Update phi
    Pmat[i+1, :, 10] = phi_p(Pmat[i+1, :, 0])
    # Update Bo
    Pmat[i+1, :, 4] = bo_p(Pmat[i+1, :, 0])
    # Update Bw
    Pmat[i+1, :, 5] = bw_p(Pmat[i+1, :, 0])
    #Update Sw
    Pmat[i+1,0,1] = (Pmat[i+1,0,5]/Pmat[i+1,0,10])*((Pmat[i,0,10]*Pmat[i,0,1]/Pmat[i,0,5]) +
                    (8*Pmat[i,0,8]*res_props['pwf'] - (8*Pmat[i,0,8] + 4*Pmat[i,0,9])*Pmat[i+1,0,0] +
                    4*Pmat[i,0,9]*Pmat[i+1,1,0])/(158*np.exp(2*chi[1])*3*(delchi**2)))

    Pmat[i+1,1:-1,1] = (Pmat[i+1, 1:-1, 5] / Pmat[i+1, 1:-1, 10]) * ((Pmat[i, 1:-1, 10] * Pmat[i, 1:-1, 1] / Pmat[i, 1:-1, 5]) +
                         (Pmat[i, 1:-1, 8] * Pmat[i+1,:-2,0] - ( Pmat[i, 1:-1, 8] + Pmat[i, 1:-1, 9]) * Pmat[i+1, 1:-1, 0] + Pmat[i, 1:-1, 9]
                         *Pmat[i+1, 2:, 0]) / (158 * np.exp(2 * chi[2:-2]) * (delchi ** 2)))
    Pmat[i+1,-1,1] = 1

    Pmat[i+1, :-1, 2] = [kro(x) for x in Pmat[i+1, 1:, 1]]
    Pmat[i+1, :-1, 3] = [krw(x) for x in Pmat[i+1, 1:, 1]]
    Pmat[i+1,-1,2] = 0
    Pmat[i+1,-1,3] = 1
    #print("KRO: \n", Pmat[0, :, 2])
    #print("KRW: \n", Pmat[0, :, 3])



    # Initialize Left Node Lambda o West
    Pmat[i+1, 0, 6] = res_props['k'] * Pmat[i+1, 0, 2] / (res_props['muo'] * bo_p(Pmat[i+1, 0, 0]))
    # Initialize Remaining Lambda o Wests
    Pmat[i+1, 1:, 6] = res_props['k'] * Pmat[i+1, 1:, 2] / (res_props['muo'] * bo_p((Pmat[i+1, 1:, 0] + Pmat[i+1, :-1, 0]) / 2))
    #print("Pmat[i,:,6], Lambda_o West: \n", Pmat[i+1, :, 6])
    # Initialize Right Node Lambda o East
    Pmat[i+1, -1, 7] = res_props['k'] * Pmat[i+1, -1, 2] / (res_props['muo'] * bo_p(Pmat[0, -1, 0]))

    # Initialize Remaining Lambda o Easts
    Pmat[i+1, :-1, 7] = res_props['k'] * Pmat[i+1, 1:, 2] / (res_props['muo'] * bo_p((Pmat[i+1, 1:, 0] + Pmat[i+1, :-1, 0]) / 2))
    #print("Pmat[i,:,7], Lambda_o East: \n", Pmat[i+1, :, 7])

    # Update Left Node Lambda w West
    Pmat[i+1, 0, 8] = res_props['k'] * Pmat[i+1, 0, 3] / (res_props['muw'] * bw_p(Pmat[i+1, 0, 0]))

    # Update Remaining Lambda w Wests
    Pmat[i+1, 1:, 8] = res_props['k'] * Pmat[i+1, 1:, 3] / (res_props['muw'] * bw_p((Pmat[i+1, 1:, 0] + Pmat[0, :-1, 0]) / 2))

    # Update Right Node Lambda w East
    Pmat[i+1, -1, 9] = res_props['k'] * Pmat[i+1, -1, 3] / (res_props['muw'] * bw_p(3000))

    # Update Remaining Lambda w Easts
    Pmat[i+1, :-1, 9] = res_props['k'] * Pmat[i+1, 1:, 3] / (res_props['muw'] * bw_p((Pmat[i+1, 1:, 0] + Pmat[0, :-1, 0]) / 2))



    # Update ct
    Pmat[i+1, :, 11] = ct(Pmat[i+1, :, 1], 1 - Pmat[i+1, :, 1])
    ############################################################################################
#pl.plot(chi[1:-1],E)
#pl.show()
#for j in range(12):
#    print(j,Pmat[0,:,j])
#Test = np.linalg.solve(E,D)
#print(Test)
#Pmat[i+1,:,0]= tdma(A,B,C,D)
#print(Pmat[i+1,:,0])
print("A \n",A)
print("B \n",B)
print("C \n",C)
print("D \n",D)
print("Pmat[i,:,6], Bo: \n",Pmat[i,:,6])
# Sw_mat = np.zeros((sim_props['T']//sim_props['tstep']+1,sim_props['N']))
# Sw_mat[0,:-1] = res_props['Swi']
# Sw_mat[0,-1] = res_props['Sawi']

#print(Sw_mat)
print("Initial Sw: \n",Pmat[0,:,1])
print("Nodes in Chi space: ", chi,"Nodes in r space: ", np.exp(chi), sep='\n')
print("Initial Pressures: \n",Pmat[0,:,0])

print(Pmat[:,:,0])
for row in Pmat[:,:,0]:
    pl.plot(chi,[1000,*row,3000])
pl.show()

for row in Pmat[:,:,0]:
    pl.plot(np.exp(chi),[1000,*row,3000])
pl.show()




# if __name__ == '__main__':
#     res_props = dict(N=10,rw=0.25,re=1500,h=50,pi=3000,pa=3000,pwf=1000,k=10,phi=0.2,muo=2,muw=1,c_phi=1.0e-5,c_o=1.5e-5,c_w=0.3e-5,Boi=1.25,Bwi=1.05,Swi=0.25,Sawi=1,BC=[1000,3000])
#     print(res_props)
#     res = Reservoir(**res_props)
#     print(res.X,np.exp(res.X),sep=' ')
#     print(res.dchi)
#     print(res.rnodes())
#     print(res.kro(0.33),res.krw(0.25))