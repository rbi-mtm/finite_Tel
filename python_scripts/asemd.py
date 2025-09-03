from re import I
from ase.units import kB
from matplotlib.pyplot import get
import ase.io as ase_io

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from copy import deepcopy
import json

from ..utils import BaseIO, load_obj, to_dict, from_dict

from ase.calculators.calculator import Calculator, all_changes
from copy import deepcopy
from ..neighbourlist.structure_manager import AtomsList, unpack_ase
from rascal.models.genericmd import GenericMDCalculator

from scipy.integrate import quad


class ASEMLCalculator(Calculator, BaseIO):
    """Wrapper class to use a rascal model as an interatomic potential in ASE

    Parameters
    ----------
    model : class
        a trained model of the rascal library that can predict the energy and
        derivaties of the energy w.r.t. atomic positions
    representation : class
        a representation calculator of rascal compatible with the trained model
    """

    implemented_properties = ["energy", "forces", "stress"]
    "Properties calculator can handle (energy, forces, ...)"

    default_parameters = {}
    "Default parameters"

    nolabel = True

    def __init__(self, model, representation, **kwargs):
        super(ASEMLCalculator, self).__init__(**kwargs)
        self.model = model
        self.representation = representation
        self.kwargs = kwargs
        self.manager = None

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "stress"],
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.manager is None:
            #  happens at the begining of the MD run
            at = self.atoms.copy()
            at.wrap(eps=1e-11)
            self.manager = [at]
        elif isinstance(self.manager, AtomsList):
            structure = unpack_ase(self.atoms, wrap_pos=True)
            structure.pop("center_atoms_mask")
            self.manager[0].update(**structure)

        self.manager = self.representation.transform(self.manager)

        energy = self.model.predict(self.manager)
        self.results["energy"] = float(energy[0])
        self.results["free_energy"] = float(energy[0])
        if "forces" in properties:
            self.results["forces"] = self.model.predict_forces(self.manager)
        if "stress" in properties:
            self.results["stress"] = extract_stress_matrix(self.model.predict_stress(self.manager).flatten())

    def _get_init_params(self):
        init_params = dict(model=self.model, representation=self.representation)
        init_params.update(**self.kwargs)
        return init_params

    def _set_data(self, data):
        self.manager = None
        super()._set_data(data)

    def _get_data(self):
        return super()._get_data()

class ASEFiniteTCalculator(GenericMDCalculator, Calculator):
    
    implemented_properties = ["energy", "forces", "stress"]
    "Properties calculator can handle (energy, forces, ...)"

    default_parameters = {}
    "Default parameters"
    
    def __init__(self, model_json, is_periodic, xdos, temperature, ref_temperature, structure_template, choice, nelectrons=None, ref_nelectrons=None, atomic_numbers=None, contribution="all"):
        super().__init__(model_json, is_periodic, structure_template=structure_template, atomic_numbers=atomic_numbers)    

        try:
            self.xdos = np.load(xdos)
        except:
            print("cannot load the energy axis, please make sure it is *.npy")
        self.temperature = float(temperature)       # target temperature
        self.beta = 1. / (self.temperature * kB)
        self.temp_0 = ref_temperature               # reference temperature
        self.beta_0 = 1. / (self.temp_0 * kB) 
        self.natoms = len(self.atoms)
        self.choice = choice
        self.positions = ase_io.read(structure_template).get_positions()
        self.cell_matrix = ase_io.read(structure_template).get_cell()
        if nelectrons == None:
            raise ValueError(
                "please enter the number of valence electrons per atom"
            )
        self.nelectrons = float(nelectrons)         # target number of electrons
        self.ref_nelectrons = float(ref_nelectrons) # reference number of electrons
        self.xdos = np.load(xdos)
        if contribution not in ["all", "band_T", "band_0", "entr_T", "entr_0"]:
            raise ValueError(
                "please provide the correct contribution, choose between: all, band_T, band_0, entr_T and entr_0"
            )
        self.contribution = contribution

        # Duplicate the weights and the self_contributions of model so it can be restored at the end of the force calculation
        self.model.unmodified_weights = deepcopy(self.model.weights)
        self.model.unmodified_self_contributions = deepcopy(self.model.self_contributions)

    def calculate(
            self,
            atoms=None,
            properties=["energy", "forces", "stress"],
            system_changes=all_changes,
        ):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.manager is None:
            #  happens at the begining of the MD run
            at = self.atoms.copy()
            at.wrap(eps=1e-11)
            self.manager = [at]
        elif isinstance(self.manager, AtomsList):
            structure = unpack_ase(self.atoms, wrap_pos=True)
            structure.pop("center_atoms_mask")
            self.manager[0].update(**structure)

        self.manager = self.representation.transform(self.manager)
       
        # Quick consistency checks
        if self.positions.shape != (len(self.atoms), 3):
            raise ValueError(
                "Improper shape of positions (is the number of atoms consistent?)"
            )
        if self.cell_matrix.shape != (3, 3):
            raise ValueError("Improper shape of cell info (expected 3x3 matrix)")
            
        self.dos_pred = self.model.predict(self.manager)[0]
        
        if self.contribution == "band_0":
            energy, force, stress = get_band_contribution(self.model, self.manager, self.dos_pred, self.beta_0, self.nelectrons, self.xdos)
            reset_model(self.model)
            extras = json_string(self.contribution, energy, force, stress, self.dos_pred)
            self.results["energy"] = float(energy[0])
            self.results["free_energy"] = float(energy[0])
            if "forces" in properties:
                self.results["forces"] = force
            if "stress" in properties:
                self.results["stress"] = stress
        
        elif self.contribution == "band_T":
            energy, force, stress = get_band_contribution(self.model, self.manager, self.dos_pred, self.beta, self.nelectrons, self.xdos)
            reset_model(self.model) 
            extras = json_string(self.contribution, energy, force, stress, self.dos_pred)
            self.results["energy"] = float(energy[0])
            self.results["free_energy"] = float(energy[0])
            if "forces" in properties:
                self.results["forces"] = force
            if "stress" in properties:
                self.results["stress"] = stress
        
        elif self.contribution == "entr_0":
            energy, force, stress = get_entropy_contribution(self.model, self.manager, self.dos_pred, self.beta_0, self.temp_0, self.nelectrons, self.xdos)
            reset_model(self.model) 
            extras = json_string(self.contribution, energy, force, stress, self.dos_pred)
            self.results["energy"] = float(energy[0])
            self.results["free_energy"] = float(energy[0])
            if "forces" in properties:
                self.results["forces"] = force
            if "stress" in properties:
                self.results["stress"] = stress
        
        elif self.contribution == "entr_T":
            energy, force, stress = get_entropy_contribution(self.model, self.manager, self.dos_pred, self.beta, self.temperature, self.nelectrons, self.xdos)
            reset_model(self.model) 
            extras = json_string(self.contribution, energy, force, stress, self.dos_pred)
            self.results["energy"] = float(energy[0])
            self.results["free_energy"] = float(energy[0])
            if "forces" in properties:
                self.results["forces"] = force
            if "stress" in properties:
                self.results["stress"] = stress
        
        else:
            energy_band_0, force_band_0, stress_band_0 = get_band_contribution(self.model, self.manager, self.dos_pred, self.beta_0, self.ref_nelectrons, self.xdos)
            reset_model(self.model)
            
            energy_band_T, force_band_T, stress_band_T = get_band_contribution(self.model, self.manager, self.dos_pred, self.beta, self.nelectrons, self.xdos)
            reset_model(self.model) 
            
            energy_entr_0, force_entr_0, stress_entr_0 = get_entropy_contribution(self.model, self.manager, self.dos_pred, self.beta_0, self.temp_0, self.ref_nelectrons, self.xdos)
            reset_model(self.model) 
            
            energy_entr_T, force_entr_T, stress_entr_T = get_entropy_contribution(self.model, self.manager, self.dos_pred, self.beta, self.temperature, self.nelectrons, self.xdos)
            reset_model(self.model)
            
            if self.choice == "energy_entr_0":
                energy = energy_band_T - energy_band_0 + energy_entr_T - energy_entr_0
                force = force_band_T - force_band_0 + force_entr_T - force_entr_0
                stress = stress_band_T - stress_band_0 + stress_entr_T - stress_entr_0

                extras = json_string(self.contribution, energy, force, stress, self.dos_pred)
                
            elif self.choice == "no_energy_entr_0":
                energy = energy_band_T - energy_band_0 + energy_entr_T
                force = force_band_T - force_band_0 + force_entr_T
                stress = stress_band_T - stress_band_0 + stress_entr_T

                extras = json_string(self.contribution, energy, force, stress, self.dos_pred)

            elif self.choice == "only_0":
                energy = - energy_band_0 - energy_entr_0
                force = - force_band_0 - force_entr_0
                stress = - stress_band_0 - stress_entr_0

                extras = json_string(self.contribution, energy, force, stress, self.dos_pred)

            elif self.choice == "only_T":
                energy = energy_band_T + energy_entr_T
                force = force_band_T + force_entr_T
                stress = stress_band_T + stress_entr_T

                extras = json_string(self.contribution, energy, force, stress, self.dos_pred)

            self.results["energy"] = float(energy[0])
            self.results["free_energy"] = float(energy[0])
            if "forces" in properties:
                self.results["forces"] = force
            if "stress" in properties:
                self.results["stress"] = stress

# Some helper functions for the finite temperature calculator
def fd_distribution(x, mu, beta):
    """Fermi-Dirac distribution"""
    y = (x-mu)*beta
    ey = np.exp(-np.abs(y))
    if hasattr(x,"__iter__"):
        negs = (y<0)
        pos = (y>=0)
        try:
            y[negs] = 1 / (1+ey[negs])        
            y[pos] = ey[pos] / (1+ey[pos])
        except:
            print (x, negs, pos)
            raise
        return y
    else:
        if y<0: return 1/(1+ey)
        else: return ey/(1+ey)

def derivative_fd_fermi(x, mu, beta):
    """the derivative of the Fermi-Dirac distribution wrt
    the Fermi energy (or chemical potential)
    For now, only cases of T>10K are handled by using np.float128"""
    y = (x-mu)*beta
    y = y.astype(np.float128)
    ey = np.exp(y)
    return beta * ey * fd_distribution(x, mu, beta)**2

def nelec(dos, mu, beta, xdos):
    """ computes the number of electrons covered in the DOS """
    return trapezoid(dos * fd_distribution(xdos, mu, beta), xdos)

def getmu(dos, beta, xdos, n=2.):
    """ computes the Fermi energy of structures based on the DOS """
    return brentq(lambda x: nelec(dos, x ,beta, xdos)-n, xdos.min(), xdos.max())

def get_shift(dos, xdos, deriv_fd):
    return trapezoid(xdos * dos * deriv_fd, xdos) / trapezoid(dos * deriv_fd, xdos)

def get_entropy(f):
    """ computes the f*log(f) term in the expession of the entropy and return the integrand 
    and a mask to determine the valid energy interval"""
    entr = f * np.log(f) + (1. - f) * np.log(1. - f)
    valid = np.logical_not(np.isnan(entr))

    return entr[valid], valid

def extract_stress_matrix(stress_voigt):
    """logic extracted from parent class"""

    matrix_indices_in_voigt_notation = (
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    )
    stress_matrix = np.zeros((3, 3))
    stress_matrix[tuple(zip(*matrix_indices_in_voigt_notation))] = stress_voigt
    # Symmetrize the stress matrix (replicate upper-diagonal entries)
    stress_matrix += np.triu(stress_matrix).T
    stress_matrix[np.diag_indices_from(stress_matrix)] *= 0.5
    return stress_matrix

def get_band_contribution(model, manager, dos, beta, nelectrons, xdos):
    mu = getmu(dos, beta, xdos, n=nelectrons)
    
    deriv_fd = derivative_fd_fermi(xdos, mu, beta)
    fd = fd_distribution(xdos, mu, beta)

    shift = get_shift(dos, xdos, deriv_fd)

    weights = trapezoid(xdos * fd * model.unmodified_weights,
                                     xdos, axis=1)
    f_weights = trapezoid((xdos - shift) * fd * model.unmodified_weights,
                                     xdos, axis=1)
    
    model.is_scalar = True

    for key in model.self_contributions.keys():
            model.self_contributions[key] = trapezoid(xdos * fd * model.unmodified_self_contributions[key],
                    xdos)
    
    model.weights = weights
    energy = model.predict(manager)

    model.weights = f_weights
    force = model.predict_forces(manager)

    stress = model.predict_stress(manager)
    stress = extract_stress_matrix(stress)

    return energy, force, stress

def get_band_contribution_0(model, manager, dos, beta, nelectrons, xdos):
    mu = getmu(dos, beta, xdos, n=nelectrons)
    
    deriv_fd = derivative_fd_fermi(xdos, mu, beta)
    fd = fd_distribution(xdos, mu, beta)

    shift = get_shift(dos, xdos, deriv_fd)

    weights = trapezoid(xdos * np.heaviside(mu-xdos, 1) * model.unmodified_weights,
                                     xdos, axis=1)
    f_weights = trapezoid((xdos - shift) * np.heaviside(mu-xdos, 1) * model.unmodified_weights,
                                     xdos, axis=1)
    
    model.is_scalar = True

    for key in model.self_contributions.keys():
            model.self_contributions[key] = trapezoid(xdos * np.heaviside(mu-xdos, 1) * model.unmodified_self_contributions[key],
                    xdos)
    
    model.weights = weights
    energy = model.predict(manager)

    model.weights = f_weights
    force = model.predict_forces(manager)

    stress = model.predict_stress(manager)
    stress = extract_stress_matrix(stress)

    return energy, force, stress

def get_entropy_contribution(model, manager, dos, beta, temperature, nelectrons, xdos):
    """ computes the contribution from (-TS)"""

    mu = getmu(dos, beta, xdos, n=nelectrons)
    
    deriv_fd = derivative_fd_fermi(xdos, mu, beta)
    fd = fd_distribution(xdos, mu, beta)

    shift = get_shift(dos, xdos, deriv_fd)

    s, x_mask = get_entropy(fd)

    model.is_scalar = True

    for key in model.self_contributions.keys():
            model.self_contributions[key] = trapezoid(model.unmodified_self_contributions[key][x_mask] * s,
                                                           xdos[x_mask])
            model.self_contributions[key] *= (-kB)
            model.self_contributions[key] *= (-temperature)
    
    weights = trapezoid(s * model.unmodified_weights[:, x_mask], xdos[x_mask], axis=1)
    weights *= (-kB)
    model.weights = -temperature * weights
    energy = model.predict(manager)

    f_weights = weights
    f_weights += (kB * beta * (mu - shift) * trapezoid(fd * model.unmodified_weights, xdos, axis=1))
    model.weights = -temperature * f_weights
    force = model.predict_forces(manager)

    stress = model.predict_stress(manager)
    stress = extract_stress_matrix(stress)

    return energy, force, stress

def json_string(label, energy, force, stress, dos):
    return json.dumps({label+"energy": "{:.8f}".format(energy[0]),
                       label+"DOS": dos.tolist(),
                       label+"force": force.flatten().tolist(),
                       label+"stress": stress.flatten().tolist()})

def reset_model(model):
        model.weights = deepcopy(model.unmodified_weights)
        model.self_contributions = deepcopy(model.unmodified_self_contributions)
        model.is_scalar = False
