import numpy as np
import scipy.constants

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0
Z_0 = np.sqrt(MU_0/EPS_0)

class Antenna():
    '''
    Defines base class for antenna construction.

    args:
        element: Element object
        f: operating frequency
        Lz: length of antenna
        delta_z: antenna element spacing
    '''

    def __init__(self, element, f, Lz, delta_z):
        
        self.params['f'] = f
        self.params['Lz'] = Lz
        self.params['delta_z'] = delta_z
        self.calculate_parameters()

        self.element = element

    def make_element_ABCD(self):
        A = ( (1 + self.element.S11)*(1 - self.element.S22) + self.element.S12*self.element.S21 ) / (2*self.element.S21) 
        B = self.params['Z'] * ( (1 + self.element.S11)*(1 + self.element.S22) - self.element.S12*self.element.S21 ) / (2*self.element.S21)
        C = 1 / self.params['Z'] * ( (1 - self.element.S11)*(1 - self.element.S22) - self.element.S12*self.element.S21 ) / (2*self.element.S21)
        D = ( (1 - self.element.S11)*(1 + self.element.S22) + self.element.S12*self.element.S21 ) / (2*self.element.S21)
        self.element.A = np.transpose(np.array([[A, B], [C, D]]), (2, 0, 1))

    def make_ABCD(self, S):
        A = ( (1 + S[:,0,0])*(1 - S[:,1,1]) + S[:,0,1]*S[:,1,0] ) / (2*S[:,1,0]) 
        B = self.params['Z'] * ( (1 + S[:,0,0])*(1 + S[:,1,1]) - S[:,0,1]*S[:,1,0] ) / (2*S[:,1,0])
        C = 1 / self.params['Z'] * ( (1 - S[:,0,0])*(1 - S[:,1,1]) - S[:,0,1]*S[:,1,0] ) / (2*S[:,1,0])
        D = ( (1 - S[:,0,0])*(1 + S[:,1,1]) + S[:,0,1]*S[:,1,0] ) / (2*S[:,1,0])
        return np.transpose(np.array([[A, B], [C, D]]), (2, 0, 1))
    
    def make_S(self, A):
        denominator = A[:,0,0] + A[:,0,1]/self.params['Z'] + A[:,1,0]*self.params['Z'] + A[:,1,1]
        S11 = (A[:,0,0] + A[:,0,1]/self.params['Z'] - A[:,1,0]*self.params['Z'] - A[:,1,1]) / denominator
        S12 = 2 * (A[:,0,0]*A[:,1,1] - A[:,0,1]*A[:,1,0]) / denominator
        S21 = 2 / denominator
        S22 = (-A[:,0,0] + A[:,0,1]/self.params['Z'] - A[:,1,0]*self.params['Z'] + A[:,1,1]) / denominator
        return np.transpose(np.array([[S11, S12], [S21, S22]]), (2, 0, 1))

    def compute_fields(self, tuning_state):
        return
    
    def calculate_parameters(self):
        return
    
    def __str__(self):
        param_table = '{:<25} {:<25}\n'.format('PARAMETER', 'VALUE')     # print column names
    
        # print each parameter
        for key, value in self.params.items():
            param_table += '{:<25} {:<25}\n'.format(key, value)
        return param_table
    
class RectangularWaveguide(Antenna):
    '''
    Defines a (dielectric-filled) rectangular waveguide antenna.

    args:
        element: Element object
        f: operating frequency
        Lz: length of antenna
        delta_z: antenna element spacing
        a: waveguide width
        b: waveguide height
    kwargs:
        m: waveguide mode number (default: 1)
        n: waveguide mode number (default: 0)
        epsilon_r: relative permittivity of waveguide material (default: 1)
        tan_delta: loss tangent of waveguide material (default: 0)
        sigma_wall: conductivity of waveguide walls (default: 1E7)
    '''
    
    def __init__(self, element, f, Lz, delta_z, a, b, **kwargs):
        
        self.params = locals()
        self.params.pop('self')
        self.params.pop('element')
        self.calculate_parameters()

        self.element = element
        if self.element is not None:
            if self.element.alpha_m is None:
                alpha_mx = (1j*self.params['a']*self.params['b']*self.params['beta_g']) / (2*self.params['k']) * (self.element.S21 + self.element.S11 - 1)
                alpha_ey = (1j*self.params['a']*self.params['b']) / (2*self.params['beta_g']) * (self.element.S21 + self.element.S11 - 1)
                self.element.alpha_m = np.stack((alpha_mx, np.zeros_like(alpha_mx), np.zeros_like(alpha_mx)), axis=1)
                self.element.alpha_e = np.stack((np.zeros_like(alpha_ey), alpha_ey, np.zeros_like(alpha_ey)), axis=1)
            elif self.element.S21 is None:
                self.element.S11 = ( - 1j*(self.params['k']**2*self.element.alpha_e[:,1])/(self.params['a']*self.params['b']*self.params['beta_g'])
                                     + 1j*(self.params['beta_g']*self.element.alpha_m[:,0])/(self.params['a']*self.params['b']) )
                self.element.S21 = ( 1 - 1j*(self.params['k']**2*self.element.alpha_e[:,1])/(self.params['a']*self.params['b']*self.params['beta_g'])
                                       - 1j*(self.params['beta_g']*self.element.alpha_m[:,0])/(self.params['a']*self.params['b']) )
                self.element.S12 = self.element.S21
                self.element.S22 = self.element.S11
                self.element.S = np.stack((self.element.S11, self.element.S12, self.element.S21, self.element.S22), axis=1)

        self.element.A = self.make_ABCD(self.element.S)
        self.A0 = np.array([[np.cosh(self.params['gamma']*self.params['delta_z']), self.params['Z']*np.sinh(self.params['gamma']*self.params['delta_z'])],
                            [1/self.params['Z']*np.sinh(self.params['gamma']*self.params['delta_z']), np.cosh(self.params['gamma']*self.params['delta_z'])]])
                
    def calculate_parameters(self):

        kwargs = self.params.pop('kwargs')
        self.params['m'] = kwargs.get('m', 1)
        self.params['n'] = kwargs.get('n', 0)
        self.params['epsilon_r'] = kwargs.get('epsilon_r', 1)
        self.params['tan_delta'] = kwargs.get('tan_delta', 0)
        self.params['sigma_wall'] = kwargs.get('sigma_wall', 1E7)

        self.params['omega'] = 2*np.pi * self.params['f']
        self.params['k'] = self.params['omega'] * np.sqrt(MU_0*EPS_0*self.params['epsilon_r'])
        self.params['lambda'] = 2*np.pi / self.params['k']
        self.params['beta_g'] = np.sqrt(self.params['k']**2 - (np.pi*self.params['m']/self.params['a'])**2 - (np.pi*self.params['n']/self.params['b'])**2)
        self.params['lambda_g'] = 2*np.pi / self.params['beta_g']
        self.params['f_c'] = 1/(2 * np.sqrt(MU_0*EPS_0*self.params['epsilon_r'])) * np.sqrt((self.params['m']/self.params['a'])**2 + (self.params['n']/self.params['b'])**2)
        self.params['R_s'] = np.sqrt(self.params['omega']*MU_0 / (2*self.params['sigma_wall']))
        self.params['alpha_d'] = self.params['k']**2 * self.params['tan_delta'] / (2 * self.params['beta_g'])
        self.params['alpha_c'] = ( (self.params['R_s'] / (self.params['a']**3 * self.params['b'] * self.params['beta_g'] * self.params['k'] * Z_0))
                                    * (2 * self.params['b'] * np.pi**2 + self.params['a']**3 * self.params['k']**2) )
        self.params['alpha'] = self.params['alpha_d'] + self.params['alpha_c']
        self.params['gamma'] = self.params['alpha'] + 1j * self.params['beta_g']
        self.params['Z'] = self.params['k'] * Z_0 / self.params['beta_g']

class SIW(RectangularWaveguide):
    '''
    Defines a (dielectric-filled) substrate-integrated waveguide antenna. 
    Inherits from RectangularWaveguide, but modifies waveguide width according to via pitch and diameter.

    args:
        element: Element object
        f: operating frequency
        Lz: length of antenna
        delta_z: antenna element spacing
        a: waveguide width
        b: waveguide height
        via_pitch: pitch of vias
        via_diameter: diameter of vias
    kwargs:
        m: waveguide mode number (default: 1)
        n: waveguide mode number (default: 0)
        epsilon_r: relative permittivity of waveguide material (default: 1)
        tan_delta: loss tangent of waveguide material (default: 0)
        sigma_wall: conductivity of waveguide walls (default: 1E7)
    '''

    def __init__(self, element, f, Lz, delta_z, a, b, via_pitch, via_diameter, **kwargs):

        a = a - 1.08 * (via_diameter**2)/via_pitch + 0.1 * (via_diameter**2)/a
        super().__init__(element, f, Lz, delta_z, a, b, **kwargs)
        self.params['via_pitch'] = via_pitch
        self.params['via_diameter'] = via_diameter