from __future__ import division

import numpy as np
import scipy.special

def normalize_histogram(values, errors = None):
  integral = np.sum(values)
  values   = np.array(values, dtype='float') / integral
  if not errors == None:
    errors   = errors / integral
  return values, errors

def poissonian_cls(y):
  yerr_lo = scipy.special.gammaincinv(y, 0.5 * 0.317)
  yerr_hi = scipy.special.gammaincinv(y + 1, 1 - 0.5 * 0.317)
  yerr_lo = np.nan_to_num(yerr_lo)
  yerr_lo = yerr_lo - y
  yerr_hi = yerr_hi - y
  yerrors = abs(np.vstack([yerr_lo, yerr_hi]))
  return yerrors
