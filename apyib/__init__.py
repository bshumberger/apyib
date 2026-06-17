"""Python package for VCD spectra via RHF, MP2, CID, and CISD methods."""

from apyib import config, utils, hamiltonian, integrals
from apyib import hf_wfn, mp2_wfn, ci_wfn
from apyib import energy, fin_diff, aats, parallel, analytic_aats, analytic_apts, analytic_hessian, ps_analytic_hessian, freq, vcd


from ._version import __version__
