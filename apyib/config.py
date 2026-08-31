"""Parameter validation and default-filling for apyib calculations."""

_REQUIRED_KEYS = (
    'geom', 'basis', 'method', 'e_convergence', 'd_convergence',
    'DIIS', 'freeze_core', 'F_el', 'F_mag', 'max_iterations',
)

_VALID_METHODS = (
    'RHF', 'MP2', 'CID', 'CISD', 'MP2_SO', 'CID_SO', 'CISD_SO',
)

# Optional keys and their defaults; all default to None so existing .get() calls are unaffected.
_OPTIONAL_DEFAULTS = {
    'isotopes': None,
    'P_nuc': None,
    'hamiltonian': None,
    'ps_formulation': None,
    'ps_ao_locality_factor(w)': None,
    'ps_bf_partitioning_prefactor': None,
    'ps_bf_locality_factor(beta)': None,
    'ps_bf_partitioning_factor(sigma)': None,
    'magnetic_gauge_origin': None,
    'gauge_origin': None,
    'F_mom': [0.0, 0.0, 0.0],
}


def validate_parameters(parameters):
    """Validate a parameters dictionary and fill optional keys with defaults.

    Parameters
    ----------
    parameters : dict
        Calculation parameters dictionary (modified in-place).

    Returns
    -------
    dict
        The same dict with optional keys defaulted to ``None`` where absent.

    Raises
    ------
    KeyError
        If one or more required keys are missing.
    ValueError
        If ``parameters['method']`` is not a recognised method string.
    """
    missing = [k for k in _REQUIRED_KEYS if k not in parameters]
    if missing:
        raise KeyError(
            f"Missing required parameter key(s): {missing}. "
            f"Required keys: {list(_REQUIRED_KEYS)}"
        )

    if parameters['method'] not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{parameters['method']}'. "
            f"Valid methods: {_VALID_METHODS}"
        )

    for key, default in _OPTIONAL_DEFAULTS.items():
        parameters.setdefault(key, default)

    return parameters
