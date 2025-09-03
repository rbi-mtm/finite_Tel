import numpy as np
import scipy
from scipy.integrate import trapezoid
from scipy.optimize import brentq, minimize
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
import ase
from ase.units import kB as kb

def fd_distribution(x, mu, beta):
    """Fermi-Dirac distribution
        INPUTS:
        =======
        x: array energy axis (eV)
        mu: Fermi energy (eV)
        beta: inverse temperature (eV)
        """
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
    """ computes the number of electrons from the DOS 
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        beta: inverse temperature
        xdos: array energy axis
        """
    return trapezoid(dos * fd_distribution(xdos, mu, beta), xdos)

def getmu(dos, beta, xdos, n=2.):
    """ computes the Fermi energy of structures based on the DOS 
        INPUTS:
        =======
        dos: array of the DOS
        beta: inverse temperature
        xdos: array energy axis
        n: number of electrons
        """
    return brentq(lambda x: nelec(dos, x ,beta, xdos)-n, xdos.min(), xdos.max())

def get_dos_fermi(dos, mu, xdos):
    """retrun the DOS value at the Fermi energy for one structure
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        xdos: array energy axis
        """
    idos = interp1d(xdos, dos)
    dos_fermi = idos(mu)
    return dos_fermi

def get_band_energy(dos, mu, xdos, beta):
    """compute the band energy of one structure
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        xdos: array energy axis
        beta: inverse temperature
        """
    return trapezoid(dos * xdos * fd_distribution(xdos, mu, beta), xdos)

def get_entropy(dos, mu, xdos, beta):
    """compute the electronic entropy of one structure
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        xdos: array energy axis
        beta: inverse temperature
        """
    f = fd_distribution(xdos, mu, beta)
    minus_f = 1.- f
    func = dos*(f*np.log(f) + (minus_f)*np.log(minus_f))
    valid = np.logical_not(np.isnan(func))
    if len(dos.shape) == 1:
        return -kb * trapezoid(func[valid], xdos[valid])
    elif len(dos.shape) == 2:
        func = np.ma.masked_array(func, mask=~valid)
        xd = np.ma.masked_array(xdos, mask=~valid[0])
        return -kb * trapezoid(func, xd, axis=1)

def get_aofd(ldos, mu, xdos, beta):
    """compute the exciataion spectrum of one structure"""
    dx = xdos[1] - xdos[0]
    xxc = np.asarray(range(len(xdos)), float)*dx
    lxc = np.zeros(len(xxc))
    for i in range(len(xdos)):
        lxc[i] = np.sum(ldos[:len(xdos)-i] * fd_distribution(xdos[:len(xdos)-i], mu, beta) *
                              ldos[i:] * (1 - fd_distribution(xdos[i:], mu, beta)))
    lxc *= dx
    return xxc, lxc

def get_charge(local_dos, mu, xdos, beta, nel):
    """compute the local charges of one srtucture
        INPUTS:
        =======
        local_dos: array of the LDOS
        mu: Fermi energy
        xdos: array energy axis
        beta: inverse temperature
        nel: number of valence electrons
        """
    return nel - trapezoid(local_dos * fd_distribution(xdos, mu, beta), xdos, axis=1)


def gauss(x):
    return np.exp(-0.5*x**2)

def build_dos(sigma, eeigv, dx, emin, emax, natoms=None, weights=None):
    """build the DOS (per state) knowing the energy resolution required in eV
        works with FHI-aims, needs to be modified for QuantumEspresso
        INPUTS:
        =======
        sigma: Gaussian broadening
        eeigv: list of eigenergies of all the structures
        dx: energy grid spacing
        emin: minimum energy value on the grid
        emax: maximum energy value on the grid
        natoms: array of the number of atoms per structure
        weights: if you are using FHI-aims, keep value equal to None. If you are using QuantumEspresso, provide the the k-point weights. 
        
        OUTPUTS:
        xdos: energy grid
        ldos: array containing the DOS"""
    
    if natoms is None:
        raise Exception("please provide 'natoms' array containing the number of atoms per structure")
        
    beta = 1. / sigma

    ndos = int((emax-emin+3) / dx)
    xdos = np.linspace(emin-1.5, emax+1.5, ndos) # extend the energy grid by 3eV 
    ldos = np.zeros((len(eeigv), ndos))
    
    if weights == None:
        for i in range(len(eeigv)):    
            for ei in eeigv[i].flatten():
                iei = int((ei-(emin-1.5))*2/sigma)
                ldos[i] += np.exp(-0.5*((xdos[:]-ei)/sigma)**2)
            ldos[i] *= 1/np.sqrt(2*np.pi*sigma**2)/natoms[i]/len(eeigv[i])
            
    else:
        for i in range(len(eeigv)):
            for j in range(len(eeigv[i])):
                for ei in eeigv[i][j].flatten():
                    ldos[i,: ] += weights[i][j]*gauss((xdos[:]-ei)/sigma)
            ldos[i] *= 1/np.sqrt(2*np.pi*sigma**2)
    return xdos, 2*ldos #xdos, ldos

def getnrgs(dir_scf_out):
    """ Will get the list of energy levels per k point plus its weight 
    for one structure from a Quantum ESPRESSO output file """
    nrgs = []
    w = []
    kks = []
    ## Remember that we have twice as many kpoints sets, we have to take the first half
    with open(dir_scf_out,'r') as scfout:
        scf = scfout.readlines()
        for i,line in enumerate(scf):
            if line.startswith( '        k(' ):
                w.append(np.float64(line.split()[-1]))
                
            if line.startswith( '          k =' ):
                cc = []
                j = 2
                while ('ocuupation' not in scf[i+j] and scf[i+j] != '\n'):
                    cc += scf[i+j].split()
                    j += 1
                #nrgs.append([np.float(n) for n in scf[i+2].split()+scf[i+3].split() if n != '\n'])
                nrgs.append([np.float64(n) for n in cc if n != '\n'])
    w = w[:len(w)//2]

    return np.array(w),np.array(nrgs)

def get_rmse(a, b, xdos=None, perc=False):
    """ computes  Root Mean Squared Error (RMSE) of array properties (DOS/aofd).
         a=pred, b=target, xdos, perc: if False return RMSE else return %RMSE"""
    
    if xdos is not None:
        rmse = np.sqrt(trapezoid((a - b)**2, xdos, axis=1).mean())
        if not perc:
            return rmse
        else:
            mean = b.mean(axis=0)
            std = np.sqrt(trapezoid((b - mean)**2, xdos, axis=1).mean())
            return 100 * rmse / std
    else:
        rmse = np.sqrt(((a - b)**2).mean())
        if not perc:
            return rmse
        else:
            return 100 * rmse / b.std(ddof=1)

def build_truncated_dos(basis, coeffs, mean, n_pc=10):
    """ builds an approximate DOS providing the basis elements and coeffs""" 
    return coeffs @ basis[:, :n_pc].T + mean

def build_pc(dos, dosmean, n_pc=10):
    """
    n_pc: the number of prinicpal components to keep
    """
   
    #dosmean = dos.mean(axis=0)
    cdos = dos - dosmean
    doscov = (cdos.T @ cdos) / len(dos)
    doseva, doseve = np.linalg.eigh(doscov)
    doseva = np.flip(doseva, axis = 0)
    doseve = np.flip(doseve, axis = 1)     
    print('Variance covered with {} PCs is = {}'.format(n_pc, doseva[:n_pc].sum()/doseva.sum()))
    return doseva, doseve[:, :n_pc]
        
def build_coeffs(dos, doseve):
    """ finds basis elements and projection coefs of the DOS 
        INPUTS:
        =======
        dos: DOS of the strcutures, should be centered wrt to training set
        doseve: the principal components
        OUPUTS:
        dosproj: projection coefficients on the retained """
    
    dosproj = dos @ doseve 
    return dosproj

def get_regression_weights(train_target, 
                           regularization1=1e-3, 
                           regularization2=None, 
                           kMM=None,
                           transfMat=None,
                           kNM=None, 
                           gradients=False, 
                           nn=None):
    
    """using the RKHS-QR solver"""
    nrkhs = transfMat.shape[1]
    if not gradients:
        KNM = kNM.copy()
        tr = train_target.copy()
        #tr = tr[:len(train)]
        
        delta = np.var(tr) / kMM.trace() / len(kMM)
        KNM /= regularization1 / delta
        tr /= regularization1 / delta
        A = np.vstack((KNM @ transfMat, np.eye(nrkhs)))
        Q, R = np.linalg.qr(A)
        if len(train_target.shape) == 2:
            b = np.vstack([tr, np.zeros((nrkhs, train_target.shape[1]))])
        else:
            tr = tr[:, np.newaxis]
            b = np.vstack([tr, np.zeros((nrkhs, 1))])
        w = transfMat @ scipy.linalg.solve_triangular(R, Q.T @ b)
        
        if len(train_target.shape) == 2:
            return w
        else:
            return w[:, 0]
    else:
        KNM = kNM.copy()
        tr = train_target.copy()

        delta = np.var(tr[:nn]) / kMM.trace() / len(kMM)
        KNM[:nn] /= regularization1 / delta
        tr[:nn] /= regularization1 / delta
        
        KNM[nn:] /= regularization2 / delta
        tr[nn:] /= regularization2 / delta
            
        A = np.vstack((KNM @ transfMat, np.eye(nrkhs)))
        Q, R = np.linalg.qr(A)
        if len(train_target.shape) == 2:
            b = np.vstack([tr, np.zeros((nrkhs, train_target.shape[1]))])
        else:
            tr = tr[:, np.newaxis]
            b = np.vstack([tr, np.zeros((nrkhs, 1))])
        w = transfMat @ scipy.linalg.solve_triangular(R, Q.T @ b)
        
        if len(train_target.shape) == 2:
            return w
        else:
            return w[:, 0]
    
def get_finiteT_force(DOS, DOS_grad, xdos, ref_temperature, temperature, nele=1.):
    
    beta_0 = 1. / (kb * ref_temperature)
    
    beta = 1. / (kb * temperature)

    mu_0 = getmu(DOS, beta_0, xdos, n=nele)

    ## compute the chemical potential
    mu_T = getmu(DOS, beta, xdos, n=nele)

    ## store the drivative of FD statistics wrt the chemical potenital
    deriv_fd = derivative_fd_fermi(xdos, mu_T, beta)
    deriv_fd_0 = derivative_fd_fermi(xdos, mu_0, beta_0)


    ## compute the shift in the integral of the band energy
    shift_T = trapezoid(xdos*DOS*deriv_fd, xdos) / trapezoid(DOS*deriv_fd, xdos)
    #self.shift_0 = self.mu_0
    shift_0 = trapezoid(xdos*DOS*deriv_fd_0, xdos) / trapezoid(DOS*deriv_fd_0, xdos)

    ## compute the "useful" band energy at finite T
    band_energy = get_band_energy(DOS, mu_T, xdos, beta) - get_band_energy(DOS, mu_0, xdos, beta_0)
    bt = get_band_energy(DOS, mu_T, xdos, beta)

    ## compute the band energy contribution to the total force
    grad_be_0 = trapezoid((xdos-shift_0) * DOS_grad * fd_distribution(xdos, mu_0, beta_0) , xdos)
    grad_be_T = trapezoid((xdos-shift_T) * DOS_grad * fd_distribution(xdos, mu_T, beta) , xdos)

    ## compute the entropy
    entropy_T = get_entropy(DOS, mu_T, xdos, beta)
    entropy_0 = get_entropy(DOS, mu_0, xdos, beta_0)

    ## compute the gradient of the entropy
    grad_entr_T = get_entropy(DOS_grad, mu_T, xdos, beta)
    grad_entr_T += (beta * kb * (mu_T - shift_T) * trapezoid(DOS_grad * fd_distribution(xdos, mu_T, beta), xdos))

    grad_entr_0 = get_entropy(DOS_grad, mu_0, xdos, beta_0)
    grad_entr_0 += (beta_0 * kb * (mu_0 - shift_0) * trapezoid(DOS_grad * fd_distribution(xdos, mu_0, beta_0), xdos))


    ## compute the free energy
    free_energy = band_energy - temperature * entropy_T + ref_temperature * entropy_0

    ## compute the total force
    force = grad_be_T - grad_be_0 - temperature * grad_entr_T + ref_temperature * grad_entr_0 #grad_entr_0 / (kb * beta_0)
    #force = force.reshape((self.natoms, 3))
    force = -force

    return force

def get_free_energy(DOS, xdos, ref_temperature, temperature, nele=1.):
    
    beta_0 = 1. / (kb * ref_temperature)
    
    beta = 1. / (kb * temperature)

    ## compute the chemical potential
    mu_0 = getmu(DOS, beta_0, xdos, n=nele)
    mu_T = getmu(DOS, beta, xdos, n=nele)

    ## compute the "useful" band energy at finite T
    band_energy = get_band_energy(DOS, mu_T, xdos, beta) - get_band_energy(DOS, mu_0, xdos, beta_0)

    ## compute the entropy
    entropy_T = get_entropy(DOS, mu_T, xdos, beta)
    entropy_0 = get_entropy(DOS, mu_0, xdos, beta_0)

    ## compute the free energy
    free_energy = band_energy - temperature * entropy_T + ref_temperature * entropy_0

    return free_energy

def free_energy(DOS, xdos, ref_temperature, temperature, nele=1.):
    
    beta_0 = 1. / (kb * ref_temperature)
    
    beta = 1. / (kb * temperature)

    ## compute the chemical potential
    mu_0 = getmu(DOS, beta_0, xdos, n=nele)
    mu_T = getmu(DOS, beta, xdos, n=nele)

    ## compute the "useful" band energy at finite T
    band_energy = get_band_energy(DOS, mu_T, xdos, beta) - get_band_energy(DOS, mu_0, xdos, beta_0)

    ## compute the entropy
    entropy = get_entropy(DOS, mu_T, xdos, beta)

    ## compute the free energy
    free_energy = band_energy - temperature * entropy

    return free_energy
