import numpy as np
import scipy.constants

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0
Z_0 = np.sqrt(MU_0/EPS_0)

class Antenna():

    def __init__(self, f, element, Lx, delta_x):
        
        self.params = {}
        self.element = element
        
        self.params['f'] = f
        self.params['Lx'] = Lx
        self.params['delta_x'] = delta_x

    def compute_fields(self, tuning_state):
        print('TBD')
        return
    
    def __str__(self):
        param_table = '{:<25} {:<25}\n'.format('PARAMETER', 'VALUE')     # print column names
    
        # print each parameter
        for key, value in self.params.items():
            param_table += '{:<25} {:<25}\n'.format(key, value)
        return param_table
    
class RectangularWaveguide(Antenna):
    
    def __init__(self, f, element, Lx, delta_x, a, b, **kwargs):
        
        super().__init__(f, element, Lx, delta_x)
        
        self.params['a'] = a
        self.params['b'] = b
        self.params['m'] = kwargs.get('m', 1)
        self.params['n'] = kwargs.get('n', 0)
        self.params['epsilon_r'] = kwargs.get('epsilon_r', 1)
        self.params['tan_delta'] = kwargs.get('tan_delta', 0)
        self.params['sigma_wall'] = kwargs.get('sigma_wall', 0)
        self.params['omega'] = 2*np.pi * self.params['f']
        self.params['k'] = self.params['omega'] * np.sqrt(MU_0*EPS_0*self.params['epsilon_r'])
        self.params['lambda'] = 2*np.pi / self.params['k']
        self.params['beta_g'] = self.beta_g()
        self.params['lambda_g'] = 2*np.pi / self.params['beta_g']
        self.params['f_c'] = self.cutoff_frequency()
        self.params['R_s'] = self.surface_resistance()
        self.params['alpha_d'] = self.alpha_d()
        self.params['alpha_c'] = self.alpha_c()
        self.params['alpha'] = self.params['alpha_d'] + self.params['alpha_c']

    def beta_g(self):
        '''
        Calculate the propagation constant for TE_mn mode.
        '''
        return np.sqrt(self.params['k']**2 - (np.pi*self.params['m']/self.params['a'])**2 - (np.pi*self.params['n']/self.params['b'])**2)
    
    def cutoff_frequency(self):
        '''
        Calculate the cutoff frequency for TE_mn mode.
        '''
        return 1/(2 * np.sqrt(MU_0*EPS_0*self.params['epsilon_r'])) * np.sqrt((self.params['m']/self.params['a'])**2 + (self.params['n']/self.params['b'])**2)
    
    def alpha_d(self):
        '''
        Calculate the dielectric attenuation constant.
        '''
        return self.params['k']**2 * self.params['tan_delta'] / (2 * self.params['beta_g'])
    
    def surface_resistance(self):
        '''
        Calculate the surface resistance.
        '''
        return np.sqrt(self.params['omega']*MU_0 / (2*self.params['sigma_wall']))
    
    def alpha_c(self):
        '''
        Calculate the conductor attenuation constant.
        '''
        return ((self.params['R_s'] / (self.params['a']**3 * self.params['b'] * self.params['beta_g'] * self.params['k'] * Z_0))
                * (2 * self.params['b'] * np.pi**2 + self.params['a']**3 * self.params['k']**2))

class SIW(RectangularWaveguide):

    def __init__(self, f, element, Lx, delta_x, a, b, via_pitch, via_diameter, **kwargs):

        super().__init__(f, element, Lx, delta_x, a, b, **kwargs)