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

    def __init__(self, element, f, Lz, delta_z=None, N=None):
        
        self.params['f'] = f
        self.params['Lz'] = Lz
        self.params['delta_z'] = delta_z
        self.params['N'] = N
        if self.params['delta_z'] is None:
            self.params['delta_z'] = Lz / N
        elif self.params['N'] is None:
            self.params['N'] = int(Lz / delta_z)
        self.z = np.linspace(0, Lz, self.params['N'])
        self.calculate_parameters()

        self.element = element

    def make_ABCD(self, S):
        '''Converts scattering matrix to ABCD matrix.'''
        A = ( (1 + S[:,0,0])*(1 - S[:,1,1]) + S[:,0,1]*S[:,1,0] ) / (2*S[:,1,0]) 
        B = self.params['Z'] * ( (1 + S[:,0,0])*(1 + S[:,1,1]) - S[:,0,1]*S[:,1,0] ) / (2*S[:,1,0])
        C = 1 / self.params['Z'] * ( (1 - S[:,0,0])*(1 - S[:,1,1]) - S[:,0,1]*S[:,1,0] ) / (2*S[:,1,0])
        D = ( (1 - S[:,0,0])*(1 + S[:,1,1]) + S[:,0,1]*S[:,1,0] ) / (2*S[:,1,0])
        return np.transpose(np.array([[A, B], [C, D]]), (2, 0, 1))
    
    def make_A0(self, delta_z0):
        '''Defines ABCD matrix for unperturbed transmission line propagation.'''
        A0 = np.array([[np.cosh(self.params['gamma']*delta_z0), self.params['Z']*np.sinh(self.params['gamma']*delta_z0)],
                       [1/self.params['Z']*np.sinh(self.params['gamma']*delta_z0), np.cosh(self.params['gamma']*delta_z0)]])
        return A0
    
    def make_S(self, A):
        '''Converts ABCD matrix to scattering matrix.'''
        denominator = A[:,0,0] + A[:,0,1]/self.params['Z'] + A[:,1,0]*self.params['Z'] + A[:,1,1]
        S11 = (A[:,0,0] + A[:,0,1]/self.params['Z'] - A[:,1,0]*self.params['Z'] - A[:,1,1]) / denominator
        S12 = 2 * (A[:,0,0]*A[:,1,1] - A[:,0,1]*A[:,1,0]) / denominator
        S21 = 2 / denominator
        S22 = (-A[:,0,0] + A[:,0,1]/self.params['Z'] - A[:,1,0]*self.params['Z'] + A[:,1,1]) / denominator
        return np.transpose(np.array([[S11, S12], [S21, S22]]), (2, 0, 1))
    
    def cascade(self, indices, tuning_state):
        Ai = np.empty((indices.size, 2, 2), dtype=complex)
        Sn_plus = np.empty((indices.size, 2, 2), dtype=complex)
        Sn_minus = np.empty((indices.size, 2, 2), dtype=complex)
        for i in range(indices.size):
            Ai[i,:,:] = self.A0 @ self.element.A[tuning_state[indices[i]],:,:]   # make Ai_tilde for each element depending on tuning state

        for n in range(indices.size):
            if n == 0:
                An = np.eye(2)
            else:
                An = np.eye(2)
                for i in range(n-1):
                    An = An @ Ai[i,:,:]
                An = An @ self.A0
            Sn_plus[n,:,:] = self.make_S(An[None,:,:])[0,:,:]

            An = np.eye(2)
            for i in range(n, indices.size):
                An = An @ Ai[i,:,:]
            Sn_minus[n,:,:] = self.make_S(An[None,:,:])[0,:,:]

        V_plus = Sn_plus[:,1,0]
        V_minus = V_plus * Sn_minus[:,0,0]
        C_10 = V_plus + V_minus
        Ey = -1j*self.params['omega'] * MU_0 * np.pi / (self.params['k_c']**2 * self.params['a']) * np.sin(np.pi*self.element.x[indices]/ self.params['a']) * C_10
        Hx = 1j*self.params['beta_g'] * np.pi / (self.params['k_c']**2 * self.params['a']) * np.sin(np.pi*self.element.x[indices]/ self.params['a']) * C_10
        Hz = np.cos(np.pi*self.element.x[indices]/ self.params['a']) * C_10
        E = np.stack((np.zeros_like(Ey), Ey, np.zeros_like(Ey)), axis=1)
        H = np.stack((Hx, np.zeros_like(Hx), Hz), axis=1)
        return E, H
    
    def compute_antenna_S(self, tuning_state):
        '''Calculates scattering matrix for antenna array.'''
        #### NEED TO FIX THIS
        indices = np.arange(self.params['N'])
        Ai = np.empty((indices.size, 2, 2), dtype=complex)
        for i in range(indices.size):
            Ai[i,:,:] = self.A0 @ self.element.A[tuning_state[indices[i]],:,:]   # make Ai_tilde for each element depending on tuning state
        
        for n in range(indices.size):
            if n == 0:
                # A = self.A0
                A = np.eye(2)
            else:
                A = A @ Ai[i,:,:]
        return self.make_S(A[None,:,:])[0,:,:]

    def compute_fields(self, tuning_state, feed_position='left'):
        '''
        Calculates electric and magnetic fields for each element in the antenna array using a cascaded ABCD matrix approach.

        args:
            tuning_state: array of tuning states for each element. Defined as length N vector of integers indexing S and alpha matrices.
            feed_position: position of feed ('left', 'right', 'both', or 'center'. default: 'left')
        '''
        if feed_position == 'left':
            indices = np.arange(self.params['N'])
            E, H = self.cascade(indices, tuning_state)
            self.E = E
            self.H = H
        elif feed_position == 'right':
            indices = np.flip(np.arange(self.params['N']))
            E, H = self.cascade(indices, tuning_state)
            self.E = np.flip(E, axis=0)
            self.H = np.flip(H, axis=0)
        elif feed_position == 'both':
            indices = np.arange(self.params['N'])
            E_left, H_left = self.cascade(indices, tuning_state)
            indices = np.flip(np.arange(self.params['N']))
            E_right, H_right = self.cascade(indices, tuning_state)
            self.E = E_left + np.flip(E_right, axis=0)
            self.H = H_left + np.flip(H_right, axis=0)
        elif feed_position == 'center':
            indices = np.arange(self.params['N']//2, self.params['N'])
            E_right, H_right = self.cascade(indices, tuning_state)
            indices = np.flip(np.arange(self.params['N']//2))
            E_left, H_left = self.cascade(indices, tuning_state)
            self.E = np.concatenate((np.flip(E_left, axis=0), E_right), axis=0)
            self.H = np.concatenate((np.flip(H_left, axis=0), H_right), axis=0)

    def compute_dipoles(self, tuning_state):
        self.m_e = self.element.alpha_e[tuning_state,:,:] @ self.E[:,:,None]
        self.m_m = self.element.alpha_m[tuning_state,:,:] @ self.H[:,:,None]
        self.m_e = self.m_e[:,:,0]
        self.m_m = self.m_m[:,:,0]

    def propagate(self, r_target):
        '''
        Numerically propagates computed dipoles to target points in the antenna near field.
        '''
        R_vec = r_target[None,:,:] - np.stack((np.zeros_like(self.z), np.zeros_like(self.z), self.z), axis=1)[:,None,:]
        R_norm = np.linalg.norm(R_vec, axis=2, keepdims=True)
        R_hat = R_vec / R_norm
        k = 2*np.pi*self.params['f'] / C
        
        G1 = -(1 + 1j*k*R_norm - k**2 * R_norm**2)/(R_norm**3)
        G2 = (3 + 3*1j*k*R_norm - k**2 * R_norm**2)/(R_norm**5)
        
        if bool(np.any(self.m_m)):
            J_m = (1j*2*np.pi*self.params['f'] * MU_0 * self.m_m)[:,None,:]
            E_F_integrand = ( (-1/(4*np.pi)) 
                            * np.cross(J_m, R_hat, axisa=2, axisb=2, axisc=2)
                            * (1 + 1j*k*R_norm)/R_norm**2
                            * np.exp(-1j*k*R_norm) )
            E_F = np.trapz(E_F_integrand, self.z, axis=0)
        
        if bool(np.any(self.m_e)):
            J_e = (1j*2*np.pi*self.params['f'] * self.m_e)[:,None,:]
            E_A_integrand = ( (-1j*Z_0/(4*np.pi*k))
                                * (G1 * J_e + 
                                   G2 * R_vec * np.sum(R_vec * J_e, axis=2, keepdims=True))
                                * np.exp(-1j*k*R_norm) )
            E_A = np.trapz(E_A_integrand, self.z, axis=0)
            
        return E_A + E_F
    
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
    
    def __init__(self, element, f, Lz, a, b, delta_z=None, N=None, **kwargs):
        
        self.params = locals()
        self.params.pop('self')
        self.params.pop('element')
        self.calculate_parameters()

        self.element = element
        if self.element is not None:
            if self.element.alpha_m is None:
                alpha_mx = (1j*self.params['a']*self.params['b']) / (2*self.params['k']**2) * (self.element.S21 - self.element.S11 - 1)
                alpha_ey = (1j*self.params['a']*self.params['b']*self.params['beta_g']) / (2*self.params['k']**2) * (self.element.S21 + self.element.S11 - 1)
                self.element.alpha_m = np.array([[alpha_mx, np.zeros_like(alpha_mx), np.zeros_like(alpha_mx)],
                                                 [np.zeros_like(alpha_mx), np.zeros_like(alpha_mx), np.zeros_like(alpha_mx)],
                                                 [np.zeros_like(alpha_mx), np.zeros_like(alpha_mx), np.zeros_like(alpha_mx)]])
                self.element.alpha_m = np.transpose(self.element.alpha_m, (2, 0, 1))
                self.element.alpha_e = np.stack([[np.zeros_like(alpha_ey), np.zeros_like(alpha_ey), np.zeros_like(alpha_ey)],
                                                 [np.zeros_like(alpha_ey), alpha_ey, np.zeros_like(alpha_ey)],
                                                 [np.zeros_like(alpha_ey), np.zeros_like(alpha_ey), np.zeros_like(alpha_ey)]])
                self.element.alpha_e = np.transpose(self.element.alpha_e, (2, 0, 1))

            elif self.element.S21 is None:
                if self.element.type == 'analytic':
                    self.element.alpha_m = self.element.alpha_m / (1 + 1j*self.element.alpha_m * ( self.params['beta_g']/(self.params['a']*self.params['b']) + (self.params['k']**3)/(3*np.pi)))
                self.element.S11 = ( - 1j*(self.params['k']**2*self.element.alpha_e[:,1,1])/(self.params['a']*self.params['b']*self.params['beta_g'])
                                     + 1j*(self.params['beta_g']*self.element.alpha_m[:,0,0])/(self.params['a']*self.params['b']) )
                self.element.S21 = ( 1 - 1j*(self.params['k']**2*self.element.alpha_e[:,1,1])/(self.params['a']*self.params['b']*self.params['beta_g'])
                                       - 1j*(self.params['beta_g']*self.element.alpha_m[:,0,0])/(self.params['a']*self.params['b']) )
                self.element.S12 = self.element.S21
                self.element.S22 = self.element.S11
                self.element.S = np.transpose(np.array([[self.element.S11, self.element.S12], [self.element.S21, self.element.S22]]), (2, 0, 1))

            self.element.rotate()
        
            self.element.x_offset = kwargs.get('x_offset', 0)
            if np.array(self.element.x_offset).ndim == 0:
                self.element.x_offset = self.element.x_offset * np.ones(self.params['N'])
            self.element.x = self.params['a']/2 + self.element.x_offset

            self.element.A = self.make_ABCD(self.element.S)
            delta_z0 = self.params['delta_z'] - self.element.delta_z_element        # defining distance between elements for general nonzero element size
        else:
            delta_z0 = self.params['delta_z']

        self.A0 = self.make_A0(delta_z0)
                
    def calculate_parameters(self):

        kwargs = self.params.pop('kwargs')
        self.params['m'] = kwargs.get('m', 1)
        self.params['n'] = kwargs.get('n', 0)
        self.params['epsilon_r'] = kwargs.get('epsilon_r', 1)
        self.params['tan_delta'] = kwargs.get('tan_delta', 0)
        self.params['sigma_wall'] = kwargs.get('sigma_wall', 1E7)

        if self.params['delta_z'] is None:
            self.params['delta_z'] = self.params['Lz'] / self.params['N']
        elif self.params['N'] is None:
            self.params['N'] = int(self.params['Lz'] / self.params['delta_z'])
        self.z = np.linspace(0, self.params['Lz'], self.params['N'])
        self.params['omega'] = 2*np.pi * self.params['f']
        self.params['k'] = self.params['omega'] * np.sqrt(MU_0*EPS_0*self.params['epsilon_r'])
        self.params['lambda'] = 2*np.pi / self.params['k']
        self.params['beta_g'] = np.sqrt(self.params['k']**2 - (np.pi*self.params['m']/self.params['a'])**2 - (np.pi*self.params['n']/self.params['b'])**2)
        self.params['lambda_g'] = 2*np.pi / self.params['beta_g']
        self.params['k_c'] = np.sqrt(self.params['k']**2 - self.params['beta_g']**2)
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

    def __init__(self, element, f, Lz, a, b, via_pitch, via_diameter, delta_z=None, N=None, **kwargs):

        a = a - 1.08 * (via_diameter**2)/via_pitch + 0.1 * (via_diameter**2)/a
        super().__init__(element, f, Lz, a, b, delta_z, N, **kwargs)
        self.params['via_pitch'] = via_pitch
        self.params['via_diameter'] = via_diameter