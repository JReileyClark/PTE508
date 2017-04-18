import math
import matplotlib.pyplot as plt



#GAS Correlations
rho_w = 62.6 # density of freshwater, 62.4 lb/ft^3

def density(Pressure, Temperature, Gravity, zfactor):
    rho = 2.7 * Gravity * Pressure / (zfactor * Temperature)
    return rho

def Ppc_Ahmed(GasGravity, Composition={'N2': 0,'CO2': 0,'H2S': 0}):
    '''Calculate Critical Pressure from Gas Specific Gravity and a dictionary of gases with composition %
    :param GasGravity:
    :param Composition:
    :return:
    '''
    Ppc = 678 - 50*(GasGravity - 0.5) - 206.7*Composition['N2'] + 440*Composition['CO2'] + 606.7*Composition['H2S']
    return Ppc

def Tpc_Ahmed(GasGravity, Composition={'N2': 0,'CO2': 0,'H2S': 0}):
    '''Calculate Critical Temperature from Gas Specific Gravity and a dictionary of gases with composition %'''
    Tpc = 326 + 315.7*(GasGravity - 0.5) - 240*Composition['N2'] - 83.3*Composition['CO2'] + 133.3*Composition['H2S']
    return Tpc

def Brill_Beggs(Pressure, Temperature, Gravity, Composition={'N2': 0.02,'CO2': 0.02, 'H2S': 0.02}):
    '''Pressure in Psia, Temperature in Farenheit, Specific Gas Gravity at standard conditions'''
    Tpr = (Temperature + 459.67)/Tpc_Ahmed(Gravity,Composition)
    Ppr = Pressure/Ppc_Ahmed(Gravity,Composition)
    A = 1.39*((Tpr - 0.92)**0.5) - 0.36*Tpr - 0.101
    E = 9*(Tpr -1)
    B = (0.62 - 0.23*Tpr)*Ppr + (0.066/(Tpr - 0.86) - 0.037)*Ppr**2 + 0.32 * (Ppr**6)/(10**E)
    C = 0.132 - 0.32*math.log10(Tpr)
    F = 0.3106 - 0.49*Tpr + 0.1824*Tpr**2
    D = 10**F
    z = A + (1-A)/(math.exp(B)) + C*(Ppr**D)
    rho = density(Pressure,Temperature,Gravity,z)
    return z, rho

def grav_to_api(Gravity):
    '''Convert Specific Gravity to API'''
    API = 141.5/Gravity - 131.5
    return API

def api_to_grav(API):
    '''Convert API to Specific Gravity'''
    Gravity = 141.5/(API + 131.5)
    return Gravity

def Hall_Yarborough(Pressure, Temperature, Gravity, Composition={'N2': 0.02,'CO2': 0.02, 'H2S': 0.02}):
    Tpr = (Temperature + 459.67)/Tpc_Ahmed(Gravity,Composition)
    Ppr = Pressure/Ppc_Ahmed(Gravity,Composition)
    tr = 1/Tpr
    A = 0.06125*tr*math.exp(-1.2*(1-tr)**2)
    B = tr*(14.76 - 9.76*tr +4.58*tr**2)
    C = tr*(90.7 - 242.2*tr + 42.4*tr**2)
    D = 2.18 + 2.82*tr
    Y = 0
    fy = 1
    while abs(fy) > 0.0000001:
        fy = (Y + Y**2 + Y**3 - Y**4)/(1-Y)**3 - A*Ppr - B*Y**2 + C*Y**D
        dfdy = (1 + 4*Y + 4*Y**2 - 4*Y**3 + Y**4)/(1 - Y)**4 - 2*B*Y + C*D*(Y**(D-1))
        Y = Y - fy/dfdy
    z = A*Ppr/Y
    rho = density(Pressure,Temperature,Gravity,z)
    return z, rho



def solution_GOR(Pressure, Temperature, gas_specgrav, oil_apigrav):
    Rs = gas_specgrav*((Pressure/18)*(10**(0.0125*oil_apigrav))/10**(0.00091*Temperature))
    return Rs

def oil_density_Ahmed(oil_specgrav, gas_specgrav, Temperature,  GOR,):
    rho_o = 62.4*oil_specgrav + 0.0136*GOR*gas_specgrav/(0.972 + 0.000147*(GOR*((gas_specgrav/oil_specgrav)**0.5) + 1.25*Temperature)**1.175)
    return rho_o

def FVF(oil_specgrav, gas_specgrav, Temperature, GOR):
    Bo = 0.9759 + 0.00012*(GOR*((gas_specgrav/oil_specgrav)**0.5)*1.25*Temperature)**1.2
    return Bo

# Oil Viscosity Correlations by Standing 1981
def viscosity_deadoil_Standing(Temperature,API):
    A = 10**(0.43 + (8.33/API))
    mu_o = (0.32 + (1.8*(10**7)/(API**4.53)))*(360/(Temperature + 200))**A
    return mu_o

def viscosity_satoil_Standing(Temperature, API, GOR):
    a = GOR*(2.2*GOR*(10**(-7)) - 7.4*(10**(-4)))
    c = GOR*8.62*10**(-5)
    d = GOR*1.1*10**(-3)
    e = GOR*3.74*10**(-3)
    b = 0.68/(10**c) + 0.25/(10**d) + 0.062/(10**e)
    mu_o = (10**a)*(viscosity_deadoil_Standing(Temperature, API)**b)
    return mu_o

def viscosity_unsatoil_Standing(Temperature, Pressure, Bubblepoint_Press, API, GOR):
    mu_sat = viscosity_satoil_Standing(Temperature, API, GOR)
    mu_o = mu_sat + 0.001*(Pressure - Bubblepoint_Press)*(0.024*(mu_sat**1.6) + 0.38*(mu_sat**0.560))
    return mu_o






if __name__ == '__main__':
    print(viscosity_unsatoil_Standing(140, 4475, 2745, 35, 600))
    zoverP = zip([Brill_Beggs(x, 180.0, 0.65, {'N2': 0.05, 'CO2': 0.08, 'H2S': 0.02}) for x in range(50, 5000, 10)],
                 range(50, 5000, 10))
    y, x = zip(*zoverP)
    y, rho = zip(*y)
    HYzoverP = zip(
        [Hall_Yarborough(x, 180.0, 0.65, {'N2': 0.05, 'CO2': 0.05, 'H2S': 0.02}) for x in range(50, 5000, 10)],
        range(50, 5000, 10))
    hy, hx = zip(*HYzoverP)
    hy, hrho = zip(*hy)
    plt.plot(x,y,color='g', label = 'Brill_Beggs')
    plt.plot(hx,hy, color='r', label = 'Hall_Yarborough')
    plt.xlabel('Presure')
    plt.ylabel('z-Factor')
    plt.legend()
    plt.title('Brill_Begs and Hall Yarborough \n T={}, grav={}, composition = {}'.format(180,0.65,{'N2':0.05,'CO2':0.05,'H2S':0.02}))
    plt.grid()
    plt.show()

    plt.plot(x,rho, color = 'g', label = 'Brill_Beggs')
    plt.plot(hx,hrho, color = 'r', label = 'Hall_Yarborough')
    plt.xlabel('Pressure')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Brill_Begs and Hall Yarborough \n T={}, grav={}, composition = {}'.format(180,0.65,{'N2':0.05,'CO2':0.05,'H2S':0.02}))
    plt.grid()
    plt.show()
