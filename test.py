__author__ = 'Bleep'

import numpy as np
import matplotlib.pyplot as plt



# WELL TEST SIMULATOR #
class welltest(object):

    def __init__( self,p, delta_t, pi, tp, q, Bo, mu,kt, phi, ct, rw, lamda, omega, h,  testtype = "BUILDUP", model = "homogeneous", boundarydist = -1, faulttype = "noflow" ):
        self.p = p
        self.delta_t = delta_t
        self.pi = pi
        self.Bo = Bo
        self.q = q
        self.mu = mu
        self.kt = kt
        self.phi = phi
        self.ct = ct
        self.testtype = testtype
        self.model = model
        self.boundarydist = boundarydist
        self.faulttype = faulttype
        self.rw = rw
        self.lamda = lamda
        self.omega = omega

    def press_NFR(self, delta_t,lamda, omega):
        pi=4500
        q=800
        ct=0.000005465
        mu=0.5653
        k2=900
        h=200
        #lamda=0.000007976
        #omega=(0.029)
        rw=0.5
        B=1.347
        tp=24

        dt = delta_tao(delta_t,ct,mu,rw,k2)

        taop = delta_tao(tp,ct,mu,rw,k2)

        m = (162.6*q*mu*B/(k2*h))

        expi1 = np.special.expi(-lamda*dt/(omega*(1-omega)))
        expi2 = np.special.expi(-lamda*dt/(1-omega))
        expi3 = np.special.expi(-lamda*(taop+dt)/(1-omega))
        expi4 = np.special.expi(-lamda*(taop + dt)/(omega*(1-omega)))
        pws = pi - m*(np.math.log10((taop + dt)/dt) - 0.435*(expi1 - expi2 + expi3 - expi4))

        return pws


params = {'pi': 4500,
          'p': [4450 + i for i in range(51)],
          'delta_t': [],
          'q': 800,
          'ct': 0.000005465,
          'mu': 0.5653,
          'kt': 900,
          'h': 200,
          'lamda': 0.000007976,
          'omega': .029,
          'rw': 0.5,
          'Bo': 1.347,
          'tp': 24,
          'phi': .23}
print(type(params))
wt1 = welltest(**params, testtype='BUILDUP',model='homogeneous', boundarydist=-1, faulttype = 'noflow')

print(wt1.testtype)
print(wt1.Bo)
print(wt1.p)