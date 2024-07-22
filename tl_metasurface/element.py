import numpy as np
from matplotlib import pyplot as plt
import skrf as rf
import scipy.constants
import scipy.io
from tl_metasurface import toolbox as tb
tb.set_font()

C = scipy.constants.c
EPS_0 = scipy.constants.epsilon_0
MU_0 = scipy.constants.mu_0

class Element:
    '''
    Defines element for antenna construction.

    args:
        filepath: path to Touchstone file
        alpha: polarizability vector vs. frequency. By default, assigns
        f: operating frequency
        f0: resonance frequency vector
    kwargs:
        rotation: rotation of element polarization (default: 0)
    '''

    def __init__(self, filepath_list=None, f=None, alpha_m=None, alpha_e=None, f0=None, **kwargs):
        self.quiet = kwargs.get('quiet', False)

        self.rotation = kwargs.get('rotation', 0)

        self.f = f
        if self.f is None:
            if not self.quiet:
                print('No frequency vector provided, defaulting to 10 GHz')
            self.f = 10E9
        
        if filepath_list is not None:
            self.S21 = []
            self.S11 = []
            for filepath in filepath_list:
                filetype = filepath.split('.')[-1]
                if filetype == 's2p' or filetype == 's3p':
                    touchstone = rf.Network(filepath)
                    f_index = np.argmin(np.abs(touchstone.f - self.f))
                    self.S21.append(touchstone.s21.s[f_index,0,0])
                    self.S11.append(touchstone.s11.s[f_index,0,0])
                else:
                    raise ValueError('Filetype not supported. Must supply Touchstone file.')
            self.S21 = np.array(self.S21)
            self.S11 = np.array(self.S11)
        
        else:
            self.f0 = f0
            if alpha_m is None:
                if self.f0 is None:
                    self.f0 = np.linspace(self.f - 0.1*self.f, 
                                        self.f + 0.1*self.f, 
                                        101)
                self.F = kwargs.get('F', 1)
                self.Q = kwargs.get('Q', 20)
                omega = 2*np.pi * self.f
                omega_0 = 2*np.pi * self.f0
                Gamma = omega / self.Q
                alpha_m = (self.F * omega**2) / (omega_0**2 - omega**2 + 1j*Gamma*omega)
            if alpha_e is None:
                alpha_e = np.zeros_like(alpha_m)
            
            if alpha_m.ndim == 1:
                self.alpha_m = np.stack((alpha_m, np.zeros_like(alpha_m), np.zeros_like(alpha_m)), axis=1)
            if alpha_e.ndim == 1:
                self.alpha_e = np.stack((np.zeros_like(alpha_e), alpha_e, np.zeros_like(alpha_e)), axis=1)

    def plot_alpha(self, plot_dict=None, dipole_type='magnetic', component='x'):
        if plot_dict is None:
            plot_dict = ['magnitude', 'phase', 'complex']

        if dipole_type == 'magnetic':
            alpha = self.alpha_m
        elif dipole_type == 'electric':
            alpha = self.alpha_e

        component_dict = {'x': 0, 'y': 1, 'z': 2}
        alpha = alpha[:, component_dict[component]]

        _, axes = plt.subplots(1, len(plot_dict), figsize=(len(plot_dict)*5, 5))
        if len(plot_dict) == 1:
            axes = [axes]
        for i, p in enumerate(plot_dict):
            if p == 'magnitude':
                axes[i].plot(np.abs(alpha))
                axes[i].set_title('Magnitude')
            elif p == 'phase':
                axes[i].plot(np.angle(alpha))
                axes[i].set_title('Phase')
            elif p == 'complex':
                axes[i].scatter(np.real(alpha), np.imag(alpha))
                axes[i].set_title('Complex')
            elif p == 'real':
                axes[i].plot(np.real(alpha))
                axes[i].set_title('Real')
            elif p == 'imag':
                axes[i].plot(np.imag(alpha))
                axes[i].set_title('Imaginary')
            plt.tight_layout()

    def plot_S(self, plot_dict=None, parameter='S21'):
        if plot_dict is None:
            plot_dict = ['magnitude', 'phase', 'complex']

        S_dict = {'S21': self.S21, 'S11': self.S11}
        S = S_dict[parameter]

        _, axes = plt.subplots(1, len(plot_dict), figsize=(len(plot_dict)*5, 5))
        if len(plot_dict) == 1:
            axes = [axes]
        for i, p in enumerate(plot_dict):
            if p == 'magnitude':
                axes[i].plot(np.abs(S))
                axes[i].set_title('Magnitude')
            elif p == 'phase':
                axes[i].plot(np.angle(S))
                axes[i].set_title('Phase')
            elif p == 'complex':
                axes[i].scatter(np.real(S), np.imag(S))
                axes[i].set_title('Complex')
            elif p == 'real':
                axes[i].plot(np.real(S))
                axes[i].set_title('Real')
            elif p == 'imag':
                axes[i].plot(np.imag(S))
                axes[i].set_title('Imaginary')
            plt.tight_layout()