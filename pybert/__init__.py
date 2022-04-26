# -*- coding: utf-8 -*-
"""
These are the python bindings for libbert

We recommend:

import pybert as pb

"""
from __future__ import print_function

import os
import sys

if sys.platform == 'win32':
    os.environ['PATH'] = __path__[0] + ';' + os.environ['PATH']

import locale
# print(locale.localeconv()['decimal_point'])

if locale.localeconv()['decimal_point'] == ',':
    print("Found locale decimal_point ',', change it to: decimal point '.':",
          end=' ')
    locale.localeconv()['decimal_point']
    locale.setlocale(locale.LC_NUMERIC, 'C')

try:
    import pygimli as pg
except ImportError as e:
    print(e)
    import traceback
    traceback.print_exc(file=sys.stdout)
    sys.stderr.write("ERROR: cannot import pygimli'.\n")

from pybert.data import showData
from pybert.data import showData as show
#from pybert.data import plotERTData
#from pybert.data import createData
from pybert.importer import exportData, importData
from pybert.importer import importData as load
from pybert.fdip import FDIP, FDIPdata
from pybert.tdip import TDIP, TDIPdata

# Versioning using versioneer
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


def version():
    """Shortcut to show and return current version."""
    pg.info('pybert: ' + __version__ + " pygimli:" + pg.__version__)
    return __version__


# inject print function for DataContainerERT
def Data_str(self):
    return "Data: Electrodes: " + str(self.sensorCount()) + " data: " + str(
        self.size())


pg.DataContainerERT.__str__ = Data_str


def readCFG(fileName):
    """Read BERT CFG file and return a dictionary."""
    opts = {}
    with open(fileName) as fid:
        lines = fid.readlines()
        for l in lines:
            l = l.replace('\n', '')
            l = l.split('#')[0]
            l = l.strip()
            kv = l.split('=')
            if len(kv) == 2:
                try:
                    if '.' in kv[1]:
                        opts[kv[0]] = float(kv[1])
                    else:
                        opts[kv[0]] = int(kv[1])
                except:
                    opts[kv[0]] = kv[1]
    return opts


def writeCFG(cfg, fileName):
    """Write dictionary as BERT CFG file."""
    with open(fileName, 'w') as fid:
        for k, v in cfg.items():
            val = v
            fid.write('{0}={1}\n'.format(k, val))


#########################
# here could additional functions go keep compatibility

try:
    DCSRMultiElectrodeModelling = pg.core.DCSRMultiElectrodeModelling
    DCMultiElectrodeModelling = pg.core.DCMultiElectrodeModelling
    DataMap = pg.core.DataMap
    DCParaDepth = pg.core.DCParaDepth
    geometricFactors = pg.core.geometricFactors
    coverageDCtrans = pg.core.coverageDCtrans
except:
    DCSRMultiElectrodeModelling = pg.DCSRMultiElectrodeModelling
    DCMultiElectrodeModelling = pg.DCMultiElectrodeModelling
    DataMap = pg.DataMap
    DCParaDepth = pg.DCParaDepth
    geometricFactor = pg.geometricFactor  # DEPRECATED
    geometricFactors = pg.geometricFactors
    coverageDCtrans = pg.coverageDCtrans

DataContainerERT = pg.DataContainerERT
