from scipy import special
import matplotlib.pyplot as plt
import csv
import math
from scipy.optimize import curve_fit
import numpy as np

with open('/volumes/sandisk/export2.csv',mode='r') as mycsv:
    datatable = list(csv.reader(mycsv))

time = [float(row[1]) for row in datatable[1:]]
horner_time = [(24+moment)/moment for moment in time]
print(time)


def delta_tao(delta_t, ct=0.00001465, mu=0.5658, rw=0.5,k2=900):

    dt = 0.0002637*k2 * delta_t/(ct*mu*(rw**2))

    return dt

def press_NFR( delta_t,lamda, omega):
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

    expi1 = special.expi(-lamda*dt/(omega*(1-omega)))
    expi2 = special.expi(-lamda*dt/(1-omega))
    expi3 = special.expi(-lamda*(taop+dt)/(1-omega))
    expi4 = special.expi(-lamda*(taop + dt)/(omega*(1-omega)))
    pws = pi - m*(math.log10((taop + dt)/dt) - 0.435*(expi1 - expi2 + expi3 - expi4))

    return pws


print(list(moment for moment in time))

delta_taovec = [delta_tao(x, ct=0.00001465, mu=0.5658, rw=0.5, k2=900) for x in time]
press = []
for i,j in enumerate(range(-10,11)):
    press.append([press_NFR(x,0.000003976 + 0.000001*i, 0.029 + 0.00001*j ) for x in time])
    plt.plot(horner_time,press[i])



 popt, pcov = curve_fit(press_NFR, xdata=[([time],0.00003657),([time],0.002)], ydata=[row[4] for row in datatable[1:]])


print(delta_taovec)
print(press)
print(list(float(row[4]) for row in datatable[1:]))
ActualData = plt.plot(horner_time, [float(row[4]) for row in datatable[1:]])
plt.legend(['Mine'])
plt.xscale('log')
plt.gca().invert_xaxis()
plt.show()