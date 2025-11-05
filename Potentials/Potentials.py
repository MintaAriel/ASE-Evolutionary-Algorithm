import matplotlib.pyplot as plt
import numpy as np

'''
-------------------------------------------------------------------------------
Atom  Types   Potential         A         B         C         D     Cutoffs(Ang)
  1     2                                                            Min    Max 
--------------------------------------------------------------------------------
O    c Mg   c Lennard 12  6  1.50      0.00      0.00     0.00      0.000  6.000
O    c Al   c Lennard 12  6  1.50      0.00      0.00     0.00      0.000  6.000
O    c O    c Lennard 12  6  1.50      0.00      0.00     0.00      0.000  6.000
Mg   c Mg   c Lennard 12  6  1.50      0.00      0.00     0.00      0.000  6.000
Mg   c Al   c Lennard 12  6  1.50      0.00      0.00     0.00      0.000  6.000
Al   c Al   c Lennard 12  6  1.50      0.00      0.00     0.00      0.000  6.000
O    c Mg   c Buckingham    0.143E+04 0.295      0.00     0.00      0.000 10.000
O    c Al   c Buckingham    0.111E+04 0.312      0.00     0.00      0.000 10.000
O    c O    c Buckingham    0.202E+04 0.267      0.00     0.00      0.000 10.000
'''

def V_LJ_12_6(r: np.ndarray, A: float, C: float) -> np.ndarray:
    """
    Lennard-Jones-like 12-6 in GULP A/r^12 - C/r^6 form (rho ignored when given as 0).
    """
    r6 = np.power(r, 6)
    r12 = np.power(r, 12)
    return A / r12 - C / r6

def V_Buck(r: np.ndarray, A: float, rho: float, C: float) -> np.ndarray:
    """
    Buckingham: V(r) = A * exp(-r / rho) - C / r^6
    """
    r6 = np.power(r, 6)
    rep = A * np.exp(-r / rho) if rho != 0 else 0.0
    att = (C / r6) if (C != 0) else 0.0
    return rep - att
