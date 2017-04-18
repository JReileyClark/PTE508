import numpy as np
from  matplotlib import pylab as pl
from scipy import special as sp
pl.style.use('fivethirtyeight')
# define a function to calculate our alpha values

def alpha(phi,mu,ct,k,chi,dx,dt):
    """alpha(phi,mu,ct,k,chi,dx,dt)
    Calculates the chi coefficient for the Project 1 discretization. i.e It has Chi = log(r) as an input"""
    return ((158*phi*mu*ct)/k)*np.exp(2*chi)*(dx**2)/dt

def ss_press(pwf,pe,rw,re,r):
    """ss_press(pwf, pw, rw, re, r) 
    Calculates the steady state pressure at a given radius r"""
    return pwf + ((pe - pwf)/(np.log(re/rw)))*np.log(r/rw)

def ss_rate(pwf,pe,rw,re,k,h,mu,B):
    """ss_rate(pwf, pe, rw, re, r, B) 
    Calculates the steady state flow rate"""
    return  (7.08*10**(-3))*(k*h/(mu*B))*(pe-pwf)/(np.log(re/rw))

def trans_rate(k,h,phi,mu,ct,B,rw,pwf,pi,t):
    """trans_rate(k, h, phi, mu, ct, B, rw, pwf, pi, t) 
    Calculates the transient flow rate"""
    td = (2.637*10**(-4))*k*(t*24)/(phi*mu*ct*(rw**2))
    return (7.08*10**(-3))*(k*h/(mu*B))*((pi - pwf)/(0.5*sp.exp1(1/(4*td))))


#print(alpha(.2,2,3*10**(-5),10,np.linspace(np.log(0.25),np.log(1500),100),(np.log(1500) - np.log(0.25))/100,1))

#calculate our delta chi for #nodes = 50

dchi = (np.log(1500) - np.log(0.25))/50
# create a vector of locations in chi space
chi = np.linspace(0,51,52)
#print(chi)
#print(chi[1:52])

#modify chi values according to place vector at appropriate node locations in chi space
#these calculations are based off of the recurrence relation for X as a function of dchi, node index, and initial position ln(rw)
chi[1:] = (chi[1:]-1)*dchi + dchi/2 + np.log(0.25)
chi[0] = np.log(0.25)
chi[51] = chi[51] - dchi/2

#print(dchi)
#print(chi)
#print(np.log(1500))



#print(alpha(.2,2,3*10**(-5),10,chi[1:51],dchi,1))

#use our alpha function to construct the central vector
# of our tridiagonal matrix and output vector. Both involved alpha, while the off diagonal elements do not.
b  = alpha(.2,2,3*10**(-5),10,chi[1:51],dchi,1)
d = alpha(.2,2,3*10**(-5),10,chi[1:51],dchi,1)
#print(b[[0,-1]])
# modifying our central diagonal vector to account for our boundary and non boundary behaviors
# they are not the same due to our half step dchi at the nodes just before the boundary
b[[0,-1]] = -(12 +3*b[[0,-1]])
b[1:49] = -(2 + b[1:49])
#print(b, len(b[1:49]))

# here we make a similar adjustment for the output vector
d[[0,-1]] = -3*d[[0,-1]]
d[1:49] = -d[1:49]

#now we define our off diagonal coefficients. We start with a vector of length n-1 of ones, and modify to meet our needs
a = np.ones(49)
c = np.ones(49)
a[-1] = 4
c[0] = 4

#print(b,d,c,a, len(a))

from  tdma import tdma
lbc = 1000
rbc = 3000
ic = 3000

results = np.zeros((366,52))
#print(results.shape)
results[0,:] = ic
results[1:,0] = lbc
results[1:,-1] = rbc
#print(results[:5,:])
#print("LOOP BEGINS HERE")
for i in range(1,366):
    e = d*results[i-1,1:51]
    e[0] = e[0] - 8*lbc
    e[-1] = e[-1] - 8*rbc
    #print(e)
    results[i,1:51] = tdma(a,b,c,e)
    #print(results[i])
#print(results[1:12])
i=0
pl.figure(1)
pl.subplot(211)
for row in results[0:366:10]:
    pl.plot(np.exp(chi),row,label="t = {}".format(i))
    i=i+10

pl.plot(np.exp(chi),ss_press(1000,3000,0.25,1500,np.exp(chi)),'rs',label="Steady State")
pl.legend(loc='best',ncol=3,prop={'size': 6})
pl.ylabel('Pressure')
pl.xlabel('Radius')
pl.title('365 Days - plotting every 10th day')

pl.subplot(212)
pl.plot(np.linspace(1,365,365),ss_rate(results[1:,1],results[1:,2],np.exp(chi[1]),np.exp(chi[2]),10,50,2,1.25),label="Numerical Rate")
pl.plot(np.linspace(1,365,365),trans_rate(10,50,0.2,2,3*10**(-5),1.25,0.25,1000,3000,np.linspace(1,365,365)),label="Transient Rate")
pl.plot(np.linspace(1,365,365),np.ones(365)*ss_rate(1000,3000,0.25,1500,10,50,2,1.25),label="SteadyState")
pl.ylabel('Q - Rate')
pl.xlabel('Days')
pl.ylim([300,500])
pl.legend()
pl.title('Flow Rate for 365 Days of Production')
pl.subplots_adjust(hspace=0.30)
pl.show()

