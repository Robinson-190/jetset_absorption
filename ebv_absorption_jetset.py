# Author: Joshua robinson, directly adapted from work by Andrea Tramacere

from scipy import interpolate

import numpy as np

from astropy import units as u
from astropy.units import spectral
from astropy.table import Table
from astropy.io import fits as pf

import os

from jetset.plot_sedfit import PlotSED,PlotSpectralMultipl

from jetset.model_parameters import ModelParameter
from jetset.base_model import  Model, MultiplicativeModel

from jetset.template_2Dmodel import TemplateTable2D


__all__=['GalacticReddeningTemplate']


class GalacticReddeningTemplate(TemplateTable2D,MultiplicativeModel):
    """

    """
    def __init__(self,
                 reddening_array,
                 energy_array,
                 tau_array,
                 applied_model=None,
                 ebv=1.0,
                 template_name='tau_tbabs_template',
                 nu_size=100):
        
        """

        Parameters
        ----------
        hydrogen_column_density_array
        energy_array
        tau_values
        applied_model
        E(B-V)
        template_name
        nu_size
        """
        super().__init__(
            x_values=reddening_array,
            y_values=energy_array,
            z_values=tau_array,
            log_input_grid=False,
            log_log_interp=False,
            nu_size=nu_size,
            name=template_name
        )

        self.parameters.add_par(ModelParameter(name='scale_factor',
                                                par_type='scale_factor',
                                                val=1.0,
                                                val_min=0,
                                                val_max=None,
                                                units='',
                                                frozen=True,
                                                log=False))

        if ebv is None and applied_model is not None:
            self.parameters.add_par(applied_model.get_par_by_type('reddening'))
        elif applied_model is None and ebv is not None:
            self.parameters.add_par(ModelParameter(name='E(B-V)',
                                                   par_type='reddening',
                                                   val=ebv,
                                                   val_min=0,
                                                   val_max=None,
                                                   units='',
                                                   frozen=True,
                                                   log=False))
        elif  ebv is not None and applied_model is not None:
            raise RuntimeError('either you provide a reddening (E(B-V)) or model with reddening parameter')

    @property
    def ebv(self):
        if hasattr(self,'_ebv'):
            if isinstance(self._ebv,ModelParameter):
                return self._ebv.val
            else:
                return self._ebv
        else:
            return  None

    @ebv.setter
    def ebv(self,ebv):
        self._ebv = ebv

    def apply_to(self,model):
        p_m = model.parameters.get_par_by_type('reddening')
        _p= self.get_par_by_name('E(B-V)')
        _p.val=p_m.val

    def _check(self):
        pass

    @property
    def reddening_array(self):
        return self.x_values
    
    @property
    def energy_array(self):
        return self.y_values
    
    @property
    def tau_array(self):
        return self.z_values
    
    @property
    def reddening_array(self):
        return self.x_values

    def eval(self, fill_SED=True, nu=None, get_model=False, loglog=False):
        """

        Parameters
        ----------
        fill_SED
        nu
        get_model
        loglog
        E(B-V)

        Returns
        -------

        """

        out_model=None

        if nu is None:
            log_nu = np.copy(self.y_values)

        else:
            if loglog is True:
                log_nu = nu
            else:
                log_nu = np.log10(nu)


        ebv=self.parameters.get_par_by_name('E(B-V)').val
        s=self.parameters.scale_factor.val

        model = np.exp(-s*self._func(ebv,log_nu).T).flatten()
        model[np.isnan(model)]=self._zero
        if get_model == True:
            if loglog == False:
                out_model=model
            else:
                out_model=  np.log10(model)

        if fill_SED is True:
            if loglog is False:
                _nu=np.power(10.,log_nu)
            else:
                _nu=np.copy(log_nu)

            self.nu=_nu
            self.tau=model

        return out_model

    def plot_model(self, plot_obj=None,  label=None, line_style='-',color=None,frame='obs'):
        if plot_obj is None:
            plot_obj = PlotSpectralMultipl()

        if label is None:
            label = '%s'%self.name
            label += ' ,E(B-V)=%5.5f'%self.parameters.get_par_by_name('E(B-V)').val

        plot_obj.plot(nu=self.nu,y=self.tau, y_label=r'$ log10(\exp^{- \tau}) $', line_style=line_style, label=label, color=color)


        return plot_obj
