#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral induced polarization (SIP) data handling and inversion."""

# general system imports
import sys
import os.path
from math import pi, sqrt

# numpy and plotting imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LogNorm, Normalize

# pygimli library, utility functions and standard (lab) stuff
import pygimli as pg
from pygimli.physics import ert  # module
from pygimli.physics.SIP import SIPSpectrum
from pygimli.physics.SIP.models import modelColeColeRho
# try:  # pg > 1.1
from pygimli.physics.SIP.importData import readSIP256file, fstring
# except ImportError:  # pre 1.1
# from pygimli.physics.SIP.importexport import readSIP256file, fstring

from pygimli.viewer.mpl import setCbarLevels, drawSensors

# pybert stuff for ERT-specific stuff
import pybert as pb
# from pybert.data import plotERTData
from .sipmodelling import ERTTLmod, ERTMultiPhimod, DCIPMModelling


class FDIP(object):
    """Class for managing spectral induced polarization (SIP) field data."""

    def __init__(self, fileName=None, **kwargs):
        """Initialize class with optional data to be loaded.

        Parameters
        ----------
        fileName: str [None]
            Single fileName

        Other Parameters
        ----------------
        **kwargs:
            * paraGeometry: PLC
                plc for the 2d inversion domain
            * verbose: bool
                Be verbose.
            * takeall: bool
                Don't delete any data while reading res files.
        """
        self.verbose = kwargs.get('verbose', False)

        # CR: The name of the members should be a little bit more literate.
        # TG: I agree, do we have to stick to the pylint/pyflakes conventions?
        # CR: would by nice .. btw. we defined own
        self.basename = 'base'  # for saving results and images
        self.figs = {}  # figure container
        self.freq = kwargs.pop('f', None)  # frequency vector
        self.RHOA = kwargs.pop('RHOA', None)  # app. resistivity matrix [Ohm m]
        self.PHIA = kwargs.pop('PHIA', None)  # app. phases matrix [Grad, deg]
        self.RHOA_E = None  # relative rhoa in %/100
        self.PHIA_E = None  # absolute phia error in rad
        self.data = kwargs.pop('data', None)  # data container
        self.ERT = None  # Resistivity manager class instance
        self.sINV = None  # single inversion instance
        self.RES = None  # matrix of (inverted) resistivities (redundant?)
        self.PHI = None  # matrix of (inverted) phases
        self.pd = None  # paraDomain
        self.res = None  # (single-frequency) resistivity
        self.phi = None  # (single-frequency) phase
        self.coverage = None  # coverage vector (from single inversion)
        # Cole-Cole model
        self.m = None  # chargeability (from model spectrum)
        self.tau = None  # time constant (from model spectrum)
        self.c = None  # Cole-Cole exponent (from model spectrum)
        self.fitChi2 = None  # vector of fitting chi^2 for each model cell
        self.header = {}  # some useful header information (any instrument)

        # TODO: SIP256C/D internals (to be removed!)
        self.DATA = None  # dto.
        self.AB = None  # dto.
        self.RU = None  # dto. (will all be thrown away)
        self.nc = 0  # number of current injections (redundant)
        self.nRU = 0  # number of remove units RU (voltage) units (SIP256 only)

        self.customParaGeometry = kwargs.pop('paraGeometry', None)
        self.customParaMesh = kwargs.pop('paraMesh', None)

        if fileName is not None:
            self.load(fileName, **kwargs)
            self.data['k'] = pg.core.geometricFactors(self.data, dim=2)

    def __repr__(self):  # for print function
        """Return string representation of the class."""
        out = ['SIP data: nf=' + str(len(self.freq)) + ' nc=' +
               str(self.nc) + ' nv=' + str(self.nRU) + " " +
               self.data.__str__()]
        if hasattr(self, 'header'):
            for key in self.header:
                val = self.header[key]
                if isinstance(val, int) or isinstance(val, float):
                    out.append(key + ' = ' + str(val))
                else:
                    out.append(key + ' = array(' + str(val.shape) + ')')
        return "\n".join(out)

    def load(self, fileName, verbose=False, f=None, instr='SIP256',
             electrodes=None, takeall=False, **kwargs):
        """Load SIP data from file.

        Load SIP data from file. (either Radic RES, MPT or single files)

        Parameters
        ----------
        fileName: str, [str, ]
            single fileName, basename or fileName list for shm/rhoa/phia
        f: array
            frequency vector (not in all instrument data files)
        instr: str
            instrument name (as alternative to the frequency vector)
        electrodes: [[x,y],]
            Overrides sensor positions
        verbose: bool
            Be verbose.
        takeall: bool
            Don't delete any data while reading res files.
        """
        if isinstance(fileName, list):  # data, RHOA and PHIA files
            self.data = pg.DataContainerERT(fileName[0])
            self.RHOA = np.loadtxt(fileName[1])
            self.PHIA = np.loadtxt(fileName[2])

        elif isinstance(fileName, str):
            if fileName.lower().rfind('.res') >= 0:  # SIP 256 or Fuchs file
                self.header, self.DATA, self.AB, self.RU = \
                    readSIP256file(fileName, verbose)

                self.basename = fileName.replace('.res', '').replace('.RES',
                                                                     '')

                self.nc = self.header['Number_of_Readings']
                self.nRU = self.header['Number_of_Remote_Units']
                self.organiseSIP256data(electrodes, takeall=takeall, **kwargs)

            elif (fileName.lower().endswith('.mpt') or
                  fileName.endswith('.Data')):  # MPT file
                self.header = {}
                self.loadMPTData(fileName)
                self.sortFrequencies()

            elif os.path.isfile(fileName):  # full file name
                self.data = pg.DataContainerERT(fileName)
                self.basename = fileName[:-4]
            else:
                self.basename = fileName
                if os.path.isfile(fileName + '.shm'):
                    self.data = pg.DataContainerERT(fileName + '.shm')

                if os.path.isfile(fileName + '.rhoa'):
                    self.RHOA = np.loadtxt(fileName + '.rhoa', skiprows=1)
                if os.path.isfile(fileName + '.phia'):
                    A = np.loadtxt(fileName + '.phia')
                    self.PHIA = A[1:, :]
                    self.freq = A[0, :]
            self.sortFrequencies()
        if f is not None:
            self.freq = f
        if not hasattr(self, 'freq'):
            if instr == 'Fuchs':
                stf = 12000./2**np.arange(25)
            else:
                stf = [1000, 500, 266, 125, 80, 40, 20, 10, 5, 2.5, 1.25,
                       0.625, 0.3125, 0.156, 0.078, 0.039, 0.02, 0.01, 5e-3,
                       2.5e-3, 1.25e-3]
            self.freq = stf[self.RHOA.shape[1]-1::-1]
        if not hasattr(self, 'nc'):
            ab = self.data('a')*1000 + self.data('b')
            self.nc = len(np.unique(ab))
        if not hasattr(self, 'nv'):
            mn = self.data('m')*1000 + self.data('n')
            self.nRU = len(np.unique(mn))

    def loadMPTData(self, fileName):
        """Read Multi-phase technology (MPT) phase SIP field data files."""
        with open(fileName) as fid:
            dataact = False
            elecact = False
            ELEC, DATA = [], []
            a, b, m, n = [], [], [], []
            elmap = np.arange(256)
            elnum = 0
            for line in fid:
                sp = line.split()
                if line.find("#elec_start") >= 0:
                    elecact = True
                if line.find("#elec_end") >= 0:
                    elecact = False
                if elecact and line.lower().find("elec") < 0:
                    ELEC.append([float(sp[i]) for i in range(1, 5)])
                    elmap[int(sp[0].split(',')[1])] = elnum
                    elnum += 1
                if line.find("#data_start") >= 0:
                    dataact = True
                if line.find("#data_end") >= 0:
                    dataact = False
                if dataact and line.find("Frequency =") >= 0:
                    self.freq = np.array(sp[4::5], dtype=np.float)
                if (dataact and line.find("!") < 0 and line.find("#") < 0 and
                        line.find("**") < 0):
                    a.append(elmap[int(sp[1].split(",")[1])])
                    b.append(elmap[int(sp[2].split(",")[1])])
                    m.append(elmap[int(sp[3].split(",")[1])])
                    n.append(elmap[int(sp[4].split(",")[1])])
                    DATA.append(np.array(sp[5:-7], dtype=np.float))

            DATA = np.array(DATA)
            self.data = pg.DataContainerERT()
            for elec in ELEC:
                self.data.createSensor(pg.Pos(elec[:3]))
            self.data.resize(len(a))
            self.data.set("a", pg.Vector(np.asarray(a)))
            self.data.set("b", pg.Vector(np.asarray(b)))
            self.data.set("m", pg.Vector(np.asarray(m)))
            self.data.set("n", pg.Vector(np.asarray(n)))
            self.data.set("valid", pg.Vector(self.data.size(), 1))
            self.data.set("k", pg.core.geometricFactors(self.data, dim=2))

            self.basename = fileName.replace('.mpt', '').replace('.MPT', '')
            # self.data.save(self.basename + ".shm", "a b m n k")
            nf = DATA.shape[1] // 5
            kk = np.reshape(self.data('k'), (-1, 1))
            self.RHOA = kk * DATA[:, 0:nf*5:5]
            self.PHIA = -DATA[:, 2:nf*5+2:5] * 1e-3

    def organiseSIP256data(self, electrodes=None, eScale=1.0, takeall=None,
                           extraCurrentRow=False):
        """Build up empty data container with the quadrupoles.

        Parameters
        ----------
        electrodes : list [None]
            Overwrite the electrodes positions given in the SIP265.res file.

        takeall : bool [False]
            Don't delete any data while reading res files.

        extraPowerRow : bool [False]
            SIP256 can be operated with separated current electrodes. If set
            electrode positions need to be specified twice (voltage, current).
        """
        self.freq = []
        for line in self.header['FrequencyParameter']:
            if (len(line) < 7) or (line[6] == 1):
                self.freq.append(line[0].round(3))
        self.freq = np.array(self.freq)
        # assemble measurement logics
        aa, bb, mm, nn, ii, iu = [], [], [], [], [], []

        if takeall is None:  # not specified
            takeall = len(self.DATA) > 1
        for ir in range(len(self.DATA)):
            readings = self.header['Readings'][ir]
            leftout = readings[3:]
            iA, iB = self.AB[ir]
            if ir < len(self.RU):
                ru = self.RU[ir]
                for iru, _ in enumerate(ru):
                    iM = ru[iru]
                    iN = iM + 1
                    while iN in leftout:
                        iN += 1
                    if (iM > iB and iN-iM == iB-iA) or takeall:
                        aa.append(iA)
                        bb.append(iB)
                        mm.append(iM)
                        nn.append(iN)
                        ii.append(ir)
                        iu.append(iru)

        # create data container
        self.data = pg.DataContainerERT()

        for line in self.header['Layout']:
            self.data.createSensor(pg.Pos(line[2], 0., 0.))

        if electrodes is not None:
            if len(electrodes) >= self.data.sensorCount():
                self.data.setSensorPositions(electrodes)
            else:
                print(self.data.sensorCount(), len(electrodes))
                raise Exception("Electrode count mismatch. "
                                "Cannot not overwrite Electrodes.")
        else:
            for line in self.header['Layout']:
                self.data.createSensor([line[2]*eScale, line[3]*eScale, 0.])

        self.data.resize(len(aa))
        if self.data.size() == 0:
            pg.critical("No data found.")

        self.data.set('a', pg.Vector(aa) - 1)  # np.array(aa)-1)
        self.data.set('b', pg.Vector(bb) - 1)
        self.data.set('m', pg.Vector(mm) - 1)
        self.data.set('n', pg.Vector(nn) - 1)
        self.data.markValid(self.data('a') > -1)

        # assemble data matrices
        self.RHOA = np.ones((self.data.size(), len(self.freq))) * np.nan
        self.PHIA = np.ones((self.data.size(), len(self.freq))) * np.nan
        self.RHOA_E = np.ones((self.data.size(), len(self.freq))) * np.nan
        self.PHIA_E = np.ones((self.data.size(), len(self.freq))) * np.nan
        self.K = np.ones((self.data.size(), len(self.freq))) * np.nan
        self.I = np.ones((self.data.size(), len(self.freq))) * np.nan
        self.T = np.ones((self.data.size(), len(self.freq))) * 0.0

        for i, _ in enumerate(ii):
            if ii[i] < len(self.DATA) and iu[i] < len(self.DATA[ii[i]]):
                A = self.DATA[ii[i]][iu[i]]
                for ifr in range(len(self.freq)):
                    line = A[A[:, 0].round(3) == self.freq[ifr]]
                    # print(line)
                    # print(self.tMeas[ifr])
                    # sys.exit()
                    if len(line):
                        self.RHOA[i, ifr] = np.abs(line[0][1])
                        self.RHOA_E[i, ifr] = np.abs(line[0][3]) / 100.  # in %
                        # grad->neg.rad
                        self.PHIA[i, ifr] = -line[0][2] * pi / 180.

                        # if i == 2:
                        #     print(-line[0][2])
                        #     print(-line[0][2] * pi / 180.)
                        self.PHIA_E[i, ifr] = np.abs(line[0][4]) * pi / 180.
                        # line[0][5] is calibration annot.
                        if len(line[0]) > 6:
                            self.I[i, ifr] = line[0][6] * 1e-3
                        if len(line[0]) > 7:
                            self.K[i, ifr] = line[0][7]
                        if len(line[0]) > 8:
                            self.T[i, ifr] = line[0][8]

            else:
                print("RU " + str(iu[i]) + " not present, RI=" + str(ii[i]))

        self.sortFrequencies()
        if electrodes is not None:
            self.RHOA = self.RHOA / self.K

            if extraCurrentRow:
                self.data.set('a', self.data('a') + self.nRU)
                self.data.set('b', self.data('b') + self.nRU)

            for i in range(len(self.K[0])):
                self.K[:, i] = pg.core.geometricFactors(self.data, dim=2)
            self.RHOA = abs(self.RHOA * self.K)

        self.RHOA = np.ma.masked_invalid(self.RHOA)
        self.PHIA = np.ma.masked_invalid(self.PHIA)

    def addData(self, name):
        """Add data from another file or sip class.

        Second data can contain additional frequencies (horizontal stacking) or
        additional quadrupoles (vertical stacking).
        """
        if isinstance(name, str):
            sip2 = FDIP(name)
        else:
            sip2 = name

        if self.RHOA.shape[1] == sip2.RHOA.shape[1]:  # same frequencies
            self.data.add(sip2.data)
            for field in ['RHOA', 'PHIA', 'RHOA_E', 'PHIA_E', 'K', 'I', 'T']:
                if hasattr(self, field) and hasattr(sip2, field):
                    F1 = getattr(self, field)
                    F2 = getattr(sip2, field)
                    if (F1 is not None and F2 is not None):
                        setattr(self, field, np.vstack((F1, F2)))
                    else:
                        setattr(self, field, None)
                        print('Ignoring partial values for '+field)
        elif self.RHOA.shape[0] == sip2.RHOA.shape[0]:  # same data
            self.freq = np.hstack((self.freq, sip2.freq))
            for field in ['RHOA', 'PHIA', 'RHOA_E', 'PHIA_E', 'K', 'I', 'T']:
                if hasattr(self, field) and hasattr(sip2, field):
                    F1 = getattr(self, field)
                    F2 = getattr(sip2, field)
                    if (F1 is not None and F2 is not None):
                        setattr(self, field, np.hstack((F1, F2)))
                    else:
                        setattr(self, field, None)
                        print('Ignoring partial values for '+field)
        else:
            pg.error("Neither number of data nor frequencies is equal. " +
                     "Don't know how to combine data.")

    def sortFrequencies(self):
        """Sort frequencies (and data) in increasing order."""
        ind = np.argsort(self.freq)
        self.freq.sort()
        self.RHOA = self.RHOA[:, ind]
        self.PHIA = self.PHIA[:, ind]
        if self.RHOA_E is not None:
            self.RHOA_E = self.RHOA_E[:, ind]
        if self.PHIA_E is not None:
            self.PHIA_E = self.PHIA_E[:, ind]

        if hasattr(self, 'K'):
            self.K = self.K[:, ind]
        if hasattr(self, 'I'):
            self.I = self.I[:, ind]
        if hasattr(self, 'T'):
            self.T = self.T[:, ind]

    def filter(self, nr=[], fmin=0, fmax=1e9, kmax=1e6, electrode=None,
               a=None, b=None, m=None, n=None, ab=None, mn=None, corrSID=1,
               forward=False):
        """Filter data with respect to frequencies and geometric factor.

        Parameters
        ----------
        fmin : double
            minimum frequency
        fmax : double
            maximum frequency
        kmax : double
            maximum (absolute) geometric factor
        electrode : int
            electrode to be removed completely
        a/b/m/n : int
            delete data with specific current or potential dipole lengths
        ab/mn : int
            delete data with specific current or potential dipole lengths
        corrSID: int [1]
            correct sensor index (like in data files)
        """
        print("filtering: nd={:d}, nf={:d}".format(*self.RHOA.shape))

        ind = (self.freq >= fmin) & (self.freq <= fmax)
        self.RHOA = self.RHOA[:, ind]
        self.PHIA = self.PHIA[:, ind]
        if self.RHOA_E is not None:
            self.RHOA_E = self.RHOA_E[:, ind]
        if self.PHIA_E is not None:
            self.PHIA_E = self.PHIA_E[:, ind]

        if self.RES is not None:
            self.RES = self.RES[:, ind]

        if self.PHI is not None:
            self.PHI = self.PHI[:, ind]

        if hasattr(self, 'K'):
            self.K = self.K[:, ind]
        if hasattr(self, 'I'):
            self.I = self.I[:, ind]
        if hasattr(self, 'T'):
            self.T = self.T[:, ind]

        self.freq = self.freq[ind]
        ind = (np.abs(self.data('k')) <= kmax)  # maximum geometric factor
        ind[nr] = False  # individual numbers
        am = self.data("m") - self.data("a")

        if ab is not None:
            ind[np.isclose(np.abs(self.data("b")-self.data("a")), ab)] = False

        if mn is not None:
            ind[np.isclose(np.abs(self.data("n")-self.data("m")), mn)] = False

        print("Sum(id):", sum(ind))
        if forward:
            ind[am < 0] = False  # reverse measurements
            print("Sum(id):", sum(ind))

        for name in ['a', 'b', 'm', 'n']:
            u = list(np.atleast_1d(eval(name)))
            if electrode is not None:
                u.extend(list(np.atleast_1d(electrode)))
            for uu in u:
                ind = ind & np.not_equal(self.data(name) + corrSID, uu)

        self.RHOA = self.RHOA[ind, :]
        self.PHIA = self.PHIA[ind, :]
        if self.RHOA_E is not None:
            self.RHOA_E = self.RHOA_E[ind, :]
        if self.PHIA_E is not None:
            self.PHIA_E = self.PHIA_E[ind, :]

        if hasattr(self, 'K'):
            self.K = self.K[ind, :]
        if hasattr(self, 'I'):
            self.I = self.I[ind, :]
        if hasattr(self, 'T'):
            self.T = self.T[ind, :]

        self.data.set('valid', pg.Vector(self.data.size()))
        self.data.markValid(pg.find(ind))
        self.data.removeInvalid()

        if electrode is not None:
            self.data.removeUnusedSensors()

        print("filtered: nd={:d}, nf={:d}".format(*self.RHOA.shape))

    def mask(self, rhomin=0, rhomax=9e99, phimin=-9e99, phimax=9e99):
        """Mask (mark invalid but not delete) single data of RHOA/PHIA cubes.

        Parameters
        ----------
        rhomin : float
            minimum apparent resistivity
        rhomax : float
            maximum apparent resistivity
        phimin : float
            minimum apparent phase
        phimax : float
            maximum apparent phase
        """
        self.RHOA = np.ma.masked_outside(self.RHOA, rhomin, rhomax)
        self.PHIA = np.ma.masked_outside(self.PHIA, phimin/1000, phimax/1000)

    def sortFreq(self):
        """Old version of sortFrequency (for backward compatibility)."""
        raise BaseException("sortFreq() deprecated, "
                            "use sortFrequencies() instead")

    def simulate(self, mesh, rhovec, mvec, tauvec, cvec, **kwargs):
        """Synthetic simulation based on Cole-Cole model."""
        if "scheme" in kwargs:
            self.data = kwargs["scheme"]
        if "fr" in kwargs:
            self.freq = kwargs["fr"]

        noiseLevel = kwargs.pop('noiseLevel', 0)  # Ca: 0.01
        noiseAbs = kwargs.pop('noiseAbs', 1e-5)  # Ca: 1e-5
        verbose = kwargs.pop('verbose', False)
        # ert = pb.Resistivity()
        ert = pg.physics.ERTManager()
        self.RHOA = np.zeros((self.data.size(), len(self.freq)))
        self.PHIA = np.zeros((self.data.size(), len(self.freq)))

        for i, fr in enumerate(self.freq):
            res = modelColeColeRho(fr, rhovec, mvec, tauvec, cvec)
            if verbose:
                pg.info(i, fr, res)

            rhoai, phiai = ert.simulate(mesh,
                                        res=res[mesh.cellMarkers()],
                                        scheme=self.data,
                                        noiseLevel=noiseLevel,
                                        noiseAbs=noiseAbs,
                                        returnArray=True,
                                        verbose=True,
                                        sr=kwargs.get('sr', True)
                                        )
            phiai[phiai > pi/2] = pi - phiai[phiai > pi/2]
            # phiai.setVal(pi - phiai[phiai > pi/2], pg.find(phiai > pi/2))
            if verbose:
                phi = -np.angle(res)
                print('{:d}\t{:5e}\t{:.2f}\t{:.2f}'.format(
                    i, fr, max(phi)*1000, max(phiai)*1000))
            self.RHOA[:, i] = rhoai
            self.PHIA[:, i] = -phiai  # convention

        return self.RHOA, self.PHIA

    def saveData(self, basename=None, withTimes=False):
        """Save data shm and .rhoa/.phia matrices."""
        if basename is not None:
            self.basename = basename
        self.data.save(basename + '.shm', 'a b m n k')
        self.writeDataMat(withTimes=withTimes)

    def writeDataMat(self, fmt='%10.6f', withTimes=False, basename=None):
        """Output the data as matrices called basename + ending rhoa/phia."""
        if basename is not None:
            self.basename = basename

        np.savetxt(basename + '.rhoa',
                   np.vstack((self.freq, self.RHOA)), fmt=fmt)
        np.savetxt(basename + '.phia',
                   np.vstack((self.freq, self.PHIA)), fmt=fmt)
        if self.RHOA_E is not None:
            np.savetxt(basename + '.rhoaE',
                       np.vstack((self.freq, self.RHOA_E)), fmt=fmt)
        if self.PHIA_E is not None:
            np.savetxt(basename + '.phiaE',
                       np.vstack((self.freq, self.PHIA_E)), fmt=fmt)
        if withTimes is True:
            np.savetxt(basename + '.times',
                       np.vstack((self.freq, self.T)), fmt='%i')

    def exportTX3(self, fileName=None, **kwargs):
        """Export data for AarhusInv spectral inversion (tx3) format.

        Parameters
        ----------
        filename : str [None]
            filename to save file, if None then basename is extended by .tx3
        amplitudeError : float [0.02]
            amplitude error (in 1)
        phaseError : float [3]
            phase error in mrad
        """
        if fileName is None:
            fileName = self.basename+'.tx3'

        nf = len(self.freq)
        xE = pg.x(self.data)
        xABMN = np.column_stack((xE[self.data('a')], xE[self.data('b')],
                                 xE[self.data('m')], xE[self.data('n')]))
        dA = np.zeros((self.data.size(), 4))
        one = np.ones((self.data.size(), 1))
        left = np.hstack((xABMN, dA, xABMN, xABMN*0, dA, one, one*nf))

        sABMN = ['A', 'B', 'M', 'N']
        fields = ['x'+s for s in sABMN]
        fields.extend(['d'+s for s in sABMN])
        fields.extend(['UTMx'+s for s in sABMN])
        fields.extend(['UTMy'+s for s in sABMN])
        fields.extend(['s'+s for s in sABMN])
        fields.append('FID')
        fields.append('Nfreq')
        for ss in ['Freq', 'STDA', 'STDP', 'Amp', 'Phase', 'FlagA', 'FlagP']:
            fields.extend([ss+str(i) for i in range(nf)])

        one = np.ones(self.data.size())
        FF = np.array([one*ff for ff in self.freq]).T
        aerr = kwargs.pop('amplitudeError', 0.02)  # percent
        perr = kwargs.pop('phaseError', 3)  # mrad
        R = self.RHOA / np.reshape(self.data('k'), (-1, 1))
        ER = np.ones_like(R) * aerr  # in 1.0
        EP = np.ones_like(self.PHIA) * perr / (self.PHIA*1000)  # in 1.0
        if isinstance(self.RHOA, np.ma.masked_array):
            FA = self.RHOA.mask*1
        else:
            FA = np.zeros_like(self.RHOA)  # take mask if np.masked_array

        if isinstance(self.PHIA, np.ma.masked_array):
            FP = self.PHIA.mask*1
        else:
            FP = np.zeros_like(self.PHIA)  # take mask if np.masked_array

        ALL = np.hstack((left, FF, ER, EP, R, self.PHIA*1000, FA, FP))
        np.savetxt(fileName, ALL, fmt='%g', delimiter='\t',
                   header='\t'.join(fields))

    def singleFrequencyData(self, ifr=0, kmax=None):
        """Return filled ERT data container for one frequency.

        Note that the data token 'ip' contains negative phase angles (mrad).

        Parameters
        ----------
        ifr : int | float
            Frequency index (type int), or nearest frequency (type float)

        Returns
        -------
        dat : DataContainerERT

        """
        if isinstance(ifr, float):  # choose closest frequency
            ifr = np.argmin(np.abs(self.freq - ifr))
            pg.info('use frequency index: {0} for {1} Hz'.format(
                ifr, self.freq[ifr]))

        data1 = pg.DataContainerERT(self.data)
        # data1.set('rhoa', self.RHOA[:, ifr].filled())
        # data1.set('ip', self.PHIA[:, ifr].filled() * 1000)
        data1.set('rhoa', np.array(self.RHOA[:, ifr]))
        data1.set('ip', np.array(self.PHIA[:, ifr] * 1000))

        if self.RHOA_E is not None:
            data1.set('err', np.array(self.RHOA_E[:, ifr]))

        if self.PHIA_E is not None:
            data1.set('iperr', np.array(self.PHIA_E[:, ifr] * 1000))

        if hasattr(self, 'K'):
            data1.set('k', np.array(self.K[:, ifr]))
            data1.set('r',
                      np.array(self.RHOA[:, ifr]) / np.array(self.K[:, ifr]))

        if hasattr(self, 'I'):
            data1.set('i', np.array(self.I[:, ifr]))
            data1.set('u', data1('r')*data1('i'))

        return data1

    def writeSingleFrequencyData(self, kmax=None):
        """Write single frequency data in unified data format."""
        for ifr, fri in enumerate(self.freq):
            data1 = self.singleFrequencyData(ifr, kmax=kmax)
            data1.checkDataValidity()
            if fri > 1.:
                fname = '{:02d}-{:d}Hz.ohm'.format(ifr, int(np.round(fri)))
            else:
                fname = '{:02d}-{:d}mHz.ohm'.format(ifr,
                                                    int(np.round(fri*1e3)))
#            data1.save(self.basename+'_'+fname, 'a b m n rhoa k u i ip')
            if self.RHOA_E is not None:
                data1.save(self.basename + '_' + fname,
                           'a b m n rhoa err ip iperr')
            else:
                data1.save(self.basename + '_' + fname, 'a b m n rhoa ip')

    def showDataSpectra(self, nr=[], ax=None, ab=None, mn=None, verbose=True,
                        **kwargs):
        """Show decay curves."""
        data = self.data
        bs = kwargs.pop('basename', 'abmn')
        labelgiven = 'label' in kwargs
        if ab is not None:
            a = np.minimum(data('a'), data('b'))
            b = np.maximum(data('a'), data('b'))
            # nr.extend(pg.find((a == min(ab)-1) & (b == max(ab)-1)))
            nr = np.nonzero(np.isclose(a, min(ab)-1) &
                            np.isclose(b, max(ab)-1))[0]

        if mn is not None:
            m = np.minimum(data('m'), data('n'))
            n = np.maximum(data('m'), data('n'))
            # fi = pg.find((m == min(mn)-1) & (n == max(mn)-1))
            fi = np.nonzero(np.isclose(m, min(mn)-1) &
                            np.isclose(n, max(mn)-1))[0]
            if ab is not None:  # already chose AB dipole => select
                nr = np.intersect1d(nr, fi)
            else:
                nr.extend(fi)

        if verbose:
            print("nr=", nr)

        kwargs.setdefault('marker', 'x')
        if len(nr) > 0:
            if ax is None:
                fig, ax = plt.subplots()
            if isinstance(nr, int):
                nr = [nr]
            for nn in nr:
                abmn = [int(self.data(t)[nn]+1) for t in ['a', 'b', 'm', 'n']]
                if not labelgiven:
                    kwargs['label'] = (bs+': '+'{:d} '*4).format(*abmn)

                ax.semilogx(self.freq, self.PHIA[nn]*1000, **kwargs)

            ax.grid(True)
            ax.legend()
            ax.set_xlabel(kwargs.pop('xlabel', 'f (Hz)'))
            ax.set_ylabel(kwargs.pop('ylabel', r'-$\phi$ (mrad)'))
            return ax

    def generateSpectraPDF(self, useall=False, maxphi=100., rlim=None,
                           maxdist=999, figsize=(8.5, 11), **kwargs):
        """Make pdf file containing all spectra.

        Parameters
        ----------
        useall : bool [False]
            use all data, also skewed dipole-dipole data with MN!=AB

        maxdist : float [999]
            maximum distance between current and voltage dipoles

        maxphi : float [100]
            maximum phase in mrad

        rlim : [float, float]
            limit for resistivity axis

        figsize : (float, float)
            figure size in inches
        """
        pdf = PdfPages(self.basename + '-spectra.pdf')
        fig, ax = plt.subplots(figsize=figsize, nrows=2, sharex=True)
        colors = 'bgrcmyk'
        markers = ('x', 'o', 'v', '^', 's', 'p', '>', '<', '+', 'd')
        minphi = kwargs.pop('minphi', 0)
        phiScale = kwargs.pop('phiScale', 'linear')
        cind = np.asarray((self.data('a')+1) * 100 + self.data('b')+1)
        for ci in np.unique(cind):
            ind = np.nonzero(cind == ci)[0]
            for ii in ind:
                rhoa, phia = self.RHOA[ii, :], self.PHIA[ii, :]
                j = int(self.data('m')[ii])
                co = colors[j % 7]
                marker = markers[j // 7]
                lab = 'MN={:d}-{:d}'.format(j+1, int(self.data('n')[ii])+1)
                ax[0].semilogx(self.freq, np.abs(rhoa), label=lab,
                               color=co, marker=marker, **kwargs)
                ax[1].semilogx(self.freq, phia*1000, color=co,
                               marker=marker, label=lab)
            ax[0].set_yscale('log')
            ax[1].set_yscale(phiScale)
            ax[0].set_xlim(min(self.freq), max(self.freq))
            # ax[0].set_xlabel('f in Hz')  # shared x
            ax[1].set_xlabel('f in Hz')
            ax[0].set_ylabel(r'$\rho_a$ in $\Omega$m')
            ax[1].set_ylabel(r'-$\phi_a$ in mrad')
            ax[1].set_ylim(minphi, maxphi)
            if rlim is not None:
                ax[0].set_ylim(rlim)
            ax[0].grid(True)
            ax[1].grid(True)
            ax[0].legend(numpoints=1, ncol=2)
            ax[0].set_title('AB={:d}-{:d}'.format(int(ci)//100,
                                                  int(ci) % 100))
            fig.savefig(pdf, format='pdf')
            ax[0].cla()
            ax[1].cla()

        fig.savefig(pdf, format='pdf')
        pdf.close()

    def generateDataPDF(self, kmax=None, ipmin=0, ipmax=None, rmin=None,
                        rmax=None, figsize=(8, 10), **kwargs):
        """Generate multipage pdf document for all data as pseudosections.

        Each page contains app. res. and phase pseudosections for single phase.

        Parameters
        ----------
        Colorscales:

        rmin : float [minvalues]
            minimum apparent resistivity in mrad

        rmax : float [maxvalues]
            minimum apparent resistivity in mrad

        ipmin : float [0]
            minimum apparent phase in mrad

        ipmax : float [maxvalues]
            minimum apparent phase in mrad

        figsize : tuple(width, height)
            figure size in inches

        **kwargs
            options to be passed to pb.show()
        """
        if self.header is not None:
            if 'Layout' in self.header:
                xl = self.header['Layout'][[0, -1], 1]
        else:
            xp = pg.x(self.data.sensorPositions())
            xl = [min(xp), max(xp)]
        if ipmax is None:
            ipmax = np.max(self.PHIA)*0.8*1000

        if rmin is None:
            rmin = np.min(self.RHOA)
        if rmax is None:
            rmax = np.max(self.RHOA)

        pdf = PdfPages(self.basename + '-data.pdf')
        fig, ax = plt.subplots(nrows=2, figsize=figsize, sharex=True)

        for i, fri in enumerate(self.freq):
            if self.RHOA is not None and self.PHIA is not None:
                data = self.data
                rhoa = self.RHOA[:, i]
                phia = self.PHIA[:, i] * 1000.
            else:
                data = self.singleFrequencyData(fri, kmax=kmax)
                rhoa = data('rhoa')
                phia = data('ip')

            for axi in ax:
                axi.clear()

            ert.showERTData(data, ax=ax[0], vals=rhoa, logScale=True,
                            colorBar=True, cMap='Spectral_r',
                            label=r'apparent resistivity in $\Omega$m',
                            cMin=rmin, cMax=rmax, **kwargs)

            ert.showERTData(data, ax=ax[1], vals=phia, logScale=False,
                            colorBar=True, cMap='viridis',
                            label='-apparent phase in mrad',
                            cMin=ipmin, cMax=ipmax, **kwargs)

            ax[0].set_title(fstring(fri))
            if 0:
                ax[0].set_xlim(xl)
            plt.pause(0.01)
            fig.savefig(pdf, format='pdf')

        pdf.close()

    def removeEpsilon(self, mode=2, verbose=False):
        """Remove high-frequency parts by fitting static epsilon."""
        we0 = self.freq * 2 * np.pi * 8.854e-12  # Omega epsilon_0
        for i in range(self.RHOA.shape[0]):
            # imaginary conductivity
            sigmai = 1/self.RHOA[i, :] * np.sin(self.PHIA[i, :])
            epsr = sigmai / we0  # relative permittivity
            if mode == 0:
                er = 2*epsr[-1] - epsr[-2]  # extrapolation
            else:
                er = np.mean(epsr[-mode:])  # mean of last ones

            print(er)
            sigmai -= max([er, 0]) * we0  # correct for static epsilon term
            if verbose:
                print(i, er)
            self.PHIA[i, :] = np.arcsin(sigmai*self.RHOA[i, :])

        if hasattr(self, 'DATA'):
            delattr(self, 'DATA')  # make sure corrected spectra are plotted

    def showSingleFrequencyData(self, fr=0, ax=None, what=None, **kwargs):
        """Show pseudosections of a single frequency."""
        if ax is None:
            if what is None:  # plot both
                fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
            else:
                fig, ax = plt.subplots()
        data = self.singleFrequencyData(fr)
        if hasattr(ax, '__iter__'):  # iterable, i.e. 2 axes
            pb.show(data, vals=data('rhoa'), ax=ax[0], **kwargs)
            pb.show(data, vals=data('ip'), ax=ax[1], **kwargs)
        else:
            if what is None:
                what = 'ip'
            pb.show(data, vals=data(what), ax=ax, **kwargs)

        return ax

    def showAllFrequencyData(self, **kwargs):
        """Show pseudesections for all data in one plot with subplots."""
        fig, ax = plt.subplots(ncols=2, nrows=len(self.freq), figsize=(10, 15),
                               sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        for i, f in enumerate(self.freq):
            self.showSingleFrequencyData(f, ax=ax[i, :], **kwargs)

    def createERTManager(self, **kwargs):
        """Create an ERT manager to do the ERT inversion with."""
        self.ERT = pg.physics.ERTManager()
        self.ERT.data = self.data

        if self.customParaMesh is not None:
            self.ERT.setMesh(self.customParaMesh, refine=True)
#        elif self.customParaGeometry is not None:
        else:
            kwargs.setdefault('quality', 34.5)
            self.ERT.createMesh(plc=self.customParaGeometry, **kwargs)

        self.ERT.fop.setVerbose(False)
        self.pd = pg.Mesh(self.ERT.fop.regionManager().paraDomain())
        self.pd.setCellMarkers(pg.Vector(self.pd.cellCount(), 2))
        return self.ERT

    def singleInversion(self, ifr=0, ipError=None, **kwargs):
        """Carry out single-frequency inversion with frequency (number).

        Parameters
        ----------
        ifr : int [0]
            frequency number
        ipError : float
            error of ip measurements [10% of median ip data]
        lamIP : float [100]
            regularization parameter for IP inversion
        **kwargs passed to ERT.invert:
            * lam : float [20]
                regularization parameter
            * zWeight : float [0.7]
                relative vertical weight
            * maxIter : int [20]
                maximum iteration number
            * robustData : bool [False]
                robust data reweighting using an L1 scheme (IRLS reweighting)
            * blockyModel : bool [False]
                blocky model constraint using L1 reweighting roughness vector
            * startModelIsReference : bool [False]
                startmodel is the reference model for the inversion

            forwarded to createMesh

            * depth
            * quality
            * paraDX
            * maxCellArea
        """
        if self.verbose:
            print("Resistivity inversion")
        lamIP = kwargs.pop('lamIP', kwargs.pop('lam', 100))

        if self.ERT is None:
            if self.verbose:
                print("Creating ERT manager.")
            self.createERTManager()

        if isinstance(ifr, float):  # choose closest frequency
            ifr = np.argmin(np.abs(self.freq - ifr))

        # hack until clearout
        if pg.core.haveInfNaN(self.RHOA[:, ifr].data):
            print(self.RHOA[:, ifr].data)
            print("Skipping calculation for freq", ifr,
                  "due to invalid resistivity values.")
            return
        # hack until clearout

        rhoa = self.RHOA[:, ifr].data
        if isinstance(self.RHOA, np.ma.masked_array):
            rhoa[self.RHOA[:, ifr].mask] = np.median(rhoa)
        self.data.set('rhoa', rhoa)
        if not self.data.allNonZero('k'):
            self.data.set('k', pg.core.geometricFactors(self.data))
        self.data.set('ip', self.PHIA[:, ifr].data)
        self.data.set('error', pg.Vector(self.data.size(), 0.03))
        error = ert.estimateError(self.data,
                                  absoluteUError=0.0001,
                                  relativeError=0.03).array()
        if isinstance(self.RHOA, np.ma.masked_array):
            error[self.RHOA[:, ifr].mask] = 1e8
        self.data.set('err', error)
        self.ERT.data = self.data

        self.res = self.ERT.invert(**kwargs)
        if self.verbose:
            print("Res:", min(self.res), max(self.res))

        self.pd = pg.Mesh(self.ERT.fop.regionManager().paraDomain())
        self.pd.setCellMarkers(pg.Vector(self.pd.cellCount(), 2))
        # CR why? TG: because in PD they are numbered but needed constant
        try:
            self.coverage = self.ERT.coverage()
        except Exception:  # for pg<=1.2
            self.coverage = self.ERT.coverageDC()

#        fIP = pg.LinearModelling(self.ERT.mesh, self.ERT.fop.jacobian())
#        if self.ERT.fop.regionManager().regionCount():
#            if self.ERT.fop.regionManager().regionExists(1):
#                self.ERT.fop.region(1).setBackground(True)

        fIP = pg.core.LinearModelling(self.pd, self.ERT.fop.jacobian())
        fIP.createRefinedForwardMesh(True)

        if self.verbose:
            print("IPData:", min(self.data('ip')), max(self.data('ip')))

        ipData = self.data('ip').array()
        if isinstance(self.RHOA, np.ma.masked_array):
            ipData[self.PHIA[:, ifr].mask] = np.median(ipData)
        if min(ipData) < 0:
            # check if ip is in radiant not mrad .. 1000!!!
            print("WARNING! found negative phases .. taking abs of ip data.")
            rhoai = self.data('rhoa') * pg.math.sin(pg.abs(ipData))
        else:
            # check if ip is in radiant not mrad .. 1000!!!
            rhoai = self.data('rhoa') * pg.math.sin(ipData)
        # TODO: switch to pg.Inversion
        iIP = pg.core.RInversion(rhoai, fIP, self.verbose)
        iIP.setRecalcJacobian(False)

        if ipError is None:
            ipError = np.median(ipData) * 0.1

        ipErrAbs = np.abs(rhoai/(np.abs(ipData)+1e-8) * ipError)
        if isinstance(self.RHOA, np.ma.masked_array):
            ipErrAbs[self.RHOA[:, ifr].mask] = 1e8
            ipErrAbs[self.PHIA[:, ifr].mask] = 1e8

        iIP.setAbsoluteError(ipErrAbs)
        tLog = pg.trans.TransLog()
        iIP.setTransModel(tLog)
        iIP.setLambda(lamIP)

        zWeight = kwargs.pop('zWeight', 0.3)
        if 'zweight' in kwargs:
            zWeight = kwargs.pop('zweight', 0.3)
            print("zweight option will be removed, Please use zWeight.")

        fIP.regionManager().setZWeight(zWeight)

        fIP.regionManager().setConstraintType(kwargs.pop('cType', 1))
        iIP.setModel(pg.Vector(self.res.size(), pg.median(rhoai)))

        if self.verbose:
            print("IP inversion")

        ipModel = iIP.run()
        self.phi = np.arctan2(ipModel, self.res)
        iIP.echoStatus()

    def singleMInversion(self, ifr=0, ipError=0.005, **kwargs):
        """Chargeability-based inversion."""
        if ifr >= len(self.freq):
            ifr = len(self.freq) - 1
        ma = pg.Vector(1 - self.RHOA[:, ifr] / self.RHOA[:, 0])
        iperr = pg.Vector(self.data.size(), ipError)
        mmin, mmax = 0.001, 1.0
        if kwargs.pop('verbose', True):
            print('discarding min/max', sum(ma < mmin), sum(ma > mmax))

        ma[ma < mmin] = mmin
        iperr[ma < mmin] = 1e5
        ma[ma > mmax] = mmax
        iperr[ma > mmax] = 1e5
        fIP = DCIPMModelling(self.ERT.fop, self.ERT.fop.mesh(), self.res)
        fIP.region(1).setBackground(True)
        fIP.region(2).setConstraintType(1)
        fIP.region(2).setZWeight(kwargs.pop('zWeight', 0.3))
        fIP.createRefinedForwardMesh(True)
        tD, tM = pg.trans.Trans(), pg.trans.TransLogLU(0.01, 1.0)
        # TODO: switch to pg.Inversion
        INV = pg.core.RInversion(ma, fIP, tD, tM, True, False)
        mstart = pg.Vector(len(self.res), 0.01)  # 10 mV/V
        INV.setModel(mstart)
        INV.setAbsoluteError(iperr)
        INV.setLambda(kwargs.pop('lam', 100))
        INV.setRobustData(True)
        self.m = INV.run()

    def individualInversion(self, verbose=False, **kwargs):
        """Carry out individual inversion for all frequencies ==> .RES."""
        nf = len(self.freq)
        for i in range(nf):
            if verbose:
                pg.info("Inverting frequency {:}".format(i))
            self.singleInversion(ifr=i, **kwargs)
            if i == 0:
                self.RES = np.zeros((self.pd.cellCount(), nf))
                self.PHI = np.zeros((self.pd.cellCount(), nf))

            self.RES[:, i] = self.res
            self.PHI[:, i] = self.phi

    def showSingleResult(self, res=None, phi=None, ax=None, nr=0, imin=0,
                         rmin=None, rmax=None, imax=None, save=None,
                         **kwargs):
        """Show resistivity and phase from single f inversion."""
        if isinstance(res, int):
            nr = int(res)
            phi = self.PHI[:, nr]
            res = self.RES[:, nr]
        if res is None:
            res = self.res
        if phi is None:
            phi = self.phi
        if ax is None:
            fig, ax = plt.subplots(nrows=2)  # , sharex=True, sharey=True)
        else:
            fig = ax[0].figure

        coverage = kwargs.pop('coverage', self.coverage)
        pg.show(self.pd, data=res, ax=ax[0], colorBar=True,
                logScale=True, cMax=rmax, cMin=rmin, cMap='Spectral_r',
                label=r"Resistivity in $\Omega$m",
                coverage=coverage, **kwargs)
        if phi is None:
            raise Exception("no valid phi values found.")
        pg.show(self.pd, data=phi*1000., ax=ax[1], colorBar=True,
                logScale=(imin > 0), cMax=imax, cMin=imin,
                label=r"-$\phi$ in mrad",
                coverage=coverage, **kwargs)

        showElecs = kwargs.pop('showElectrodes', False)
        if showElecs:
            drawSensors(ax[0], self.data.sensorPositions())
            drawSensors(ax[1], self.data.sensorPositions())

        if save is True:
            save = self.basename + '-Ind{:02d}'.format(nr) + '.pdf'

        if isinstance(save, str):
            fig.savefig(save, bbox_inches='tight')

    def simultaneousInversion(self, **kwargs):
        """Carry out both simultaneous resistivity and phase inversions."""
        self.simultaneousResistivityInversion(**kwargs)
        self.simultaneousPhaseInversion(**kwargs)

    def simultaneousResistivityInversion(self, **kwargs):
        """Carry out simultaneous resistivity inversion of all frequencies."""
        self.verbose = kwargs.get('verbose', self.verbose)

        if self.ERT is None:
            self.ERT = self.createERTManager(self.data)
            self.res = self.ERT.invert(**kwargs)

        nf = self.RHOA.shape[1]
        # mesh2d = pg.Mesh(self.ERT.fop.mesh())
        mesh2d = self.ERT.fop.regionManager().mesh()
        mesh2d.setCellMarkers(np.minimum(mesh2d.cellMarkers()+2, 2))
        sfwdR = ERTTLmod(nf=nf, data=self.data, mesh=mesh2d,
                         rotate=kwargs.pop('rotate', True))
#        self.sfwdR = ERTTLmod(nf=nf, fop=self.ERT.fop.mesh(), rotate=True)
        sfwdR.createRefinedForwardMesh(False)  # make sure things are ok
        self.pd = sfwdR.pd2d
        # self.pd.save(self.basename+'_pd.bms')
        if self.verbose:
            print(self.data.size()*nf, sfwdR.pd2d.cellCount()*nf)

        # there need to be a better preprocessing here to fight nan values
        alldata = self.RHOA.flatten(order='F')
        allerror = np.ones_like(alldata) * 0.03
        if isinstance(alldata, np.ma.masked_array):
            allerror[alldata.mask] = 1e8
            alldata = alldata.data

        startModel = pg.Vector(sfwdR.nc*nf, np.median(alldata))
        tLog = pg.trans.TransLog()
        self.sINV = pg.core.RInversion(alldata, sfwdR, tLog, tLog, True, False)
        self.sINV.setRelativeError(allerror)
        self.sINV.setModel(startModel)
        self.sINV.setLambda(kwargs.pop('lam', 10))
        self.sINV.setMaxIter(kwargs.pop('maxIter', 10))
        sfwdR.regionManager().setZWeight(kwargs.pop('zWeight', 0.3))
        res = self.sINV.run()
        self.RES = np.reshape(res, (nf, sfwdR.nc)).T

    def simultaneousPhaseInversion(self, **kwargs):
        """Carry out simultaneous phase inversion of all frequencies."""
        nf = self.RHOA.shape[1]
        if self.ERT is None:
            self.ERT = self.createERTManager()
            self.res = self.ERT.invert(**kwargs)

        fwd = ERTMultiPhimod(self.pd, self.ERT.fop.jacobian(), nf,
                             rotate=kwargs.pop('rotate', True))
        fwd.createRefinedForwardMesh(False)
        fwd.regionManager().setZWeight(kwargs.pop('zWeight', 0.3))

        alldata = pg.Vector(0)
        rhoa = self.ERT.inv.response  # pg 1.1
        if len(rhoa) == 0:
            rhoa = self.RHOA[:, 0]

        for i in range(nf):
            if min(self.PHIA[:, i]) < 0:
                print("WARNING! found negative phases "
                      ".. switch to abs of ip data.")
            rhoaii = rhoa * pg.math.sin(np.abs(np.array(self.PHIA[:, i])))
            alldata = pg.cat(alldata, rhoaii)

        error = 0.001 * max(alldata) / np.abs(alldata) + 0.03
        print(min(alldata), max(alldata))
        if kwargs.pop('onlyPositive', False):
            error[alldata <= 0] = 100
            alldata[alldata <= 0] = max(alldata) / 2

        tD = pg.trans.Trans()
        tM = pg.trans.TransLogLU(0, max(self.res))
        # TODO: switch to pg.Inversion
        INV = pg.core.RInversion(alldata, fwd, tD, tM, True, False)
        if hasattr(self, 'phiXX'):
            startModel = np.tile(self.phi, nf)
        else:
            startModel = pg.Vector(fwd.nc * nf, pg.median(alldata))

        INV.setModel(startModel)
        # INV.setReferenceModel(pg.Vector(fwd.nc * nf, pg.median(startModel)))
        INV.setRelativeError(error)
        INV.setRecalcJacobian(False)
        INV.setMaxIter(kwargs.pop('maxIter', 10))
        INV.setLambda(kwargs.pop('lam', 50))
        allresi = INV.run()
        np.savetxt('allresi.vec', allresi)
        self.PHI = np.arctan(np.reshape(allresi, (nf, fwd.nc)).T /
                             np.reshape(self.ERT.model, (-1, 1)))
        # np.reshape(self.ERT.resistivity, (-1, 1)))
        return INV

    def saveResults(self, basename=None, dirname=None):
        """Save inversion results to .rho and .phi file plus mesh."""
        if basename is None:
            basename = self.basename

        if dirname is not None:
            basename = os.path.join(dirname, basename)

        self.pd.save(basename+'_pd.bms')
        if hasattr(self, 'RES'):
            np.savetxt(basename+'.rho', self.RES)
        if hasattr(self, 'PHI'):
            np.savetxt(basename+'.phi', self.PHI)
        if hasattr(self, 'coverage'):
            np.savetxt(basename+'.coverage', self.coverage)

        self.saveFit(basename)

    def saveFit(self, basename=None):
        """Save fitted chargeability, time constant & exponent to file."""
        if basename is None:
            basename = self.basename

        if np.any(self.m) and np.any(self.tau) and np.any(self.c):
            np.savetxt(basename+'.rmtc', np.column_stack(
                (self.res, self.m, self.tau, self.c, self.fitChi2)))

    def loadFit(self, basename=None):
        """Load fitted chargeability, time constant & exponent from file."""
        if basename is None:
            basename = self.basename

        self.res, self.m, self.tau, self.c, self.fitChi2 = np.loadtxt(
            basename+'.rmtc', unpack=1)

    def loadResults(self, basename=None, take=0, loadFit=False, dirname=None):
        """Load inversion results from file into self.RES/PHI.

        Set also single-frequency result (self.res/phi) by index,
        maximum (take < 0) or sum (take > nfreq)
        """
        if basename is None:
            basename = self.basename

        if dirname is not None:
            basename = os.path.join(dirname, basename)

        self.pd = pg.Mesh(basename+'_pd.bms')
        self.RES = np.loadtxt(basename+'.rho')
        self.PHI = np.loadtxt(basename+'.phi')
        if os.path.isfile(basename+'.coverage'):
            self.coverage = np.loadtxt(basename+'.coverage')

        self.chooseResult(take=take)
        if loadFit:
            self.loadFit(basename)

    def chooseResult(self, take=0):
        """Choose single-frequency result (self.res/phi) from matrices.

        self.RES/PHI by index, maximum (take < 0) or sum (take > nfreq)
        """
        if take < 0:
            self.res = np.max(self.RES, axis=1)
            self.phi = np.max(self.PHI, axis=1)
        elif take > len(self.freq):
            self.res = np.sum(self.RES, axis=1)
            self.phi = np.sum(self.PHI, axis=1)
        else:
            self.res = self.RES[:, take]
            self.phi = self.PHI[:, take]

    def printColeColeParameters(self, point):
        """Print Cole-Cole parameters for point or id."""
        if isinstance(point, int):
            cid = point
        else:
            cid = self.getCellID(point)
            print("Detected ID={:d} for point ({:.1f}, {:.1f})".format(
                cid, point[0], point[1]))
        fstr = r"rho={:.1f}  Ohmm m={:.3f}  tau={:3e} s  c={:.2f}"
        vals = self.res[cid], self.m[cid], self.tau[cid], self.c[cid]
        print(fstr.format(*vals))

    def showAllPhases(self, imax=200, figsize=(10, 16), **kwargs):
        """Show all model phases in subplots using the same colorscale."""
        cMap = kwargs.pop('cMap', 'viridis')
        fig, ax = plt.subplots(nrows=len(self.freq)+1,
                               sharex=False, figsize=figsize)
        fig.subplots_adjust(hspace=0, wspace=0)
        self.figs['phases'] = fig
        for i in range(len(self.freq)):
            pg.show(self.pd, self.PHI[:, i] * 1e3, ax=ax[i], logScale=False,
                    cMin=0, cMax=imax, cMap=cMap, coverage=self.coverage,
                    colorBar=False, **kwargs)
            # if i:
            #     ax[i].set_xticks([])
            #     ax[i].set_xlabel('')

        icbar = ColorbarBase(ax[-1], norm=Normalize(vmin=0, vmax=imax),
                             orientation='horizontal')
        setCbarLevels(icbar, cMin=0, cMax=imax, nLevs=7)
        icbar.set_clim(0, imax)
        icbar.set_cmap(cMap)
        icbar.ax.set_title(r'$\phi$ in mrad')
        icbar.ax.set_aspect(1./25)
        return fig, ax

    def showAllResistivities(self, figsize=(10, 16), **kwargs):
        """Show model resistivities in subplots using the same colorscale."""
        cMap = kwargs.pop('cMap', 'Spectral_r')
        cMin = kwargs.pop('cMin', np.min(self.RES))
        cMax = kwargs.pop('cMax', np.max(self.RES))
        fig, ax = plt.subplots(nrows=len(self.freq)+1,
                               sharex=False, figsize=figsize)
        fig.subplots_adjust(hspace=0, wspace=0)
        self.figs['resistivities'] = fig
        for i in range(len(self.freq)):
            pg.show(self.pd, self.RES[:, i], ax=ax[i], logScale=True,
                    cMap=cMap, cMin=cMin, cMax=cMax, colorBar=0, **kwargs)
            # if i:
            #     ax[i].set_xticks([])

        cbar = ColorbarBase(ax[-1], norm=LogNorm(vmin=cMin, vmax=cMax),
                            orientation='horizontal')
        setCbarLevels(cbar, cMin=cMin, cMax=cMax, nLevs=7)
        cbar.set_clim(cMin, cMax)
        cbar.set_cmap(cMap)
        cbar.ax.set_title(r'$\rho$ in $\Omega$m')
        cbar.ax.set_aspect(1./25)
        return fig, ax

    def showAllResults(self, rmin=10, rmax=1000, imax=100, figsize=(10, 16),
                       **kwargs):
        """Show resistivities and phases next to each other in subplots."""
        cmap = kwargs.pop('cMap', 'viridis')
        fig, ax = plt.subplots(nrows=len(self.freq) + 1, ncols=2,
                               figsize=figsize)
        fig.subplots_adjust(hspace=0, wspace=0)
        self.figs['results'] = fig
        kwargs.setdefault('colorBar', False)

        showElecs = kwargs.pop('showElectrodes', False)

        for i, f in enumerate(self.freq):
            pg.show(self.pd, self.RES[:, i], ax=ax[i, 0],
                    cMin=rmin, cMax=rmax, cMap=cmap,
                    **kwargs)

            pg.show(self.pd, self.PHI[:, i]*1e3, ax=ax[i, 1],
                    cMin=0, cMax=imax, cMap=cmap, logScale=False,
                    **kwargs)
            if showElecs:
                drawSensors(ax[i, 0], self.data.sensorPositions())
                drawSensors(ax[i, 1], self.data.sensorPositions())

            ax[i, 0].text(ax[i, 0].get_xlim()[0],
                          ax[i, 0].get_ylim()[0],
                          'f=' + fstring(f))
            if i:
                ax[i, 0].set_xticks([])
                ax[i, 1].set_xticks([])

        cmap = pg.viewer.mpl.colorbar.cmapFromName(cmap)
        rcbar = ColorbarBase(ax[-1, 0], norm=LogNorm(vmin=rmin, vmax=rmax),
                             orientation='horizontal', cmap=cmap)
        setCbarLevels(rcbar, cMin=rmin, cMax=rmax, nLevs=7)
        icbar = ColorbarBase(ax[-1, 1], norm=Normalize(vmin=0, vmax=imax),
                             orientation='horizontal', cmap=cmap)
        setCbarLevels(icbar, cMin=0, cMax=imax, nLevs=7)

        #icbar.set_clim(0, imax)
        rcbar.ax.set_title(r'$\rho$ in $\Omega$m')
        icbar.ax.set_title(r'$\phi$ in mrad')
        rcbar.ax.set_aspect(1./25)
        icbar.ax.set_aspect(1./25)
        return fig, ax

    def generateResultPDF(self, rmin=10, rmax=1000, imax=200, figsize=(12, 12),
                          **kwargs):
        """Generate a multipage pdf with rho/phi for each frequency."""
        cmapRho = kwargs.pop('cMapRho', 'Spectral_r')
        cmapPhi = kwargs.pop('cMapPhi', 'viridis')
        basename = kwargs.pop("basename", self.basename)
        pdf = PdfPages(basename + '-result.pdf')
        fig, ax = plt.subplots(nrows=2, figsize=figsize, sharex=True)

        cb1 = True
        cb2 = True
        for i, fri in enumerate(self.freq):

            fstr = '(f={:d} Hz)'.format(int(fri))
            if fri < 1.:
                fstr = '(f={:d} mHz)'.format(int(fri*1e3))

            for axi in ax:
                axi.cla()

            cb1 = pg.show(self.pd, self.RES[:, i], ax=ax[0],
                          cMin=rmin, cMax=rmax, cMap=cmapRho, colorBar=cb1,
                          label=r'Resistivity in $\Omega$m ' + fstr,
                          **kwargs)[1]
            cb2 = pg.show(self.pd, self.PHI[:, i] * 1e3, ax=ax[1],
                          cMin=0, cMax=imax, cMap=cmapPhi, colorBar=cb2,
                          label=r'$\phi$ in mrad ' + fstr, logScale=False,
                          **kwargs)[1]

            fig.savefig(pdf, format='pdf')

        pdf.close()

    def getCellID(self, pos):
        """Return cell ID of nearest cell to position."""
        return self.pd.findCell(pg.Pos(*pos)).id()

    def getDataSpectrum(self, dataNo=None, abmn=None, verbose=False):
        """Return SIP spectrum class for single data number."""
        if hasattr(abmn, '__iter__'):
            bb = pg.abs(self.data('a') - abmn[0]-1) + \
                pg.abs(self.data('b') - abmn[1]-1) + \
                pg.abs(self.data('m') - abmn[2]-1) + \
                pg.abs(self.data('n') - abmn[3]-1)
            dataNo = np.argmin(bb)
            if verbose:
                print('Nr {}'.format(dataNo))

        return SIPSpectrum(f=self.freq, amp=self.RHOA[dataNo, :],
                           phi=self.PHIA[dataNo, :])

    def getModelSpectrum(self, cellID):
        """Return SIP spectrum for single cell (id or position)."""
        if hasattr(cellID, '__iter__'):  # tuple
            cellID = self.getCellID(cellID)

        return SIPSpectrum(f=self.freq, amp=self.RES[cellID, :],
                           phi=self.PHI[cellID, :])

    def showModelSpectrum(self, cellID, **kwargs):
        """Show SIP spectrum for single cell (id or position)."""
        spec = self.getModelSpectrum(cellID)
        return spec.showData(**kwargs)

    def showModelSpectra(self, positions, **kwargs):
        """Show model spectra for a number of positions or IDs."""
        fig, ax = plt.subplots(nrows=2, sharex=True)
        LABELS = []
        for pos in positions:
            label = 'x={:.1f} z={:.1f}'.format(*pos)
            LABELS.append(label)
#            kwargs['label'] = label
            self.showModelSpectrum(pos, ax=ax, **kwargs)

        for a in ax:
            a.set_ylim(auto=True)
        ax[0].set_xlim(min(self.freq), max(self.freq))
        ax[0].legend(LABELS, loc='best')

        return fig, ax

    def fitAllPhi(self, show=False, **kwargs):
        """Fit all phase spectra by Cole-Cole models."""
        mpar = kwargs.pop('mpar', [0.1, 0, 1])
        minf, maxf = min(self.freq), max(self.freq)
        taupar = kwargs.pop('taupar', [1./sqrt(minf*maxf), 0.1/maxf, 10/minf])
        cpar = kwargs.pop('cpar', [0.25, 0, 1])
        ePhi = kwargs.pop('ePhi', 0.001)

        nm = self.pd.cellCount()
        self.m = np.zeros(nm)
        self.tau = np.zeros(nm)
        self.c = np.zeros(nm)
        self.fitChi2 = np.zeros(nm)

        spec = SIPSpectrum(f=self.freq, amp=self.RES[0, :], phi=self.PHI[0, :])
        for i in range(nm):
            spec.amp = self.RES[i, :]
            spec.phi = self.PHI[i, :]
            spec.fitCCPhi(ePhi=ePhi, mpar=mpar, taupar=taupar, cpar=cpar)
            self.m[i] = spec.mCC[0]
            self.tau[i] = spec.mCC[1]
            self.c[i] = spec.mCC[2]
            self.fitChi2[i] = spec.chi2
        if show:
            self.showColeColeFit(**kwargs)

    def fitAllRhoPhi(self, show=False, **kwargs):
        """Fit all phase spectra by Cole-Cole models."""
        # mpar = kwargs.pop('mpar', [0.1, 0, 1])
        minf, maxf = min(self.freq), max(self.freq)
        taupar = kwargs.pop('taupar', [1./sqrt(minf*maxf), 0.1/maxf, 10/minf])
        cpar = kwargs.pop('cpar', [0.25, 0, 1])
        ePhi = kwargs.pop('ePhi', 0.001)
        eRho = kwargs.pop('eRho', 0.01)

        nm = self.pd.cellCount()
        self.m = np.zeros(nm)
        self.tau = np.zeros(nm)
        self.c = np.zeros(nm)
        self.fitChi2 = np.zeros(nm)

        spec = SIPSpectrum(f=self.freq, amp=self.RES[0, :], phi=self.PHI[0, :])
        for i in range(nm):
            spec.amp = self.RES[i, :]
            spec.phi = self.PHI[i, :]
            spec.fitColeCole(eRho=eRho, ePhi=ePhi, taupar=taupar,
                             cpar=cpar)
            # spec.fitCCC(ePhi=ePhi, mpar=mpar, taupar=taupar, cpar=cpar)
            self.res[i] = spec.mCC[0]
            self.m[i] = spec.mCC[1]
            self.tau[i] = spec.mCC[2]
            self.c[i] = spec.mCC[3]
            # self.fitChi2[i] = spec.chi2
        if show:
            return self.showColeColeFit(**kwargs)

    def showColeColeParameters(self, figsize=(8, 12), save=False,
                               rlim=(None, None), mlim=(None, None),
                               tlim=(None, None), clim=(0, 0.5),
                               mincov=0.05, **kwargs):
        """Show distribution of Cole-Cole parameters."""
        if 'coverage' in kwargs:
            coverage = kwargs.pop('coverage')
        else:
            coverage = 1 / np.sqrt(self.fitChi2)
            coverage[coverage > 1] = 1
            coverage[coverage < 0] = 0
            coverage *= (1 - mincov)
            coverage += mincov

        fig, ax = plt.subplots(nrows=4, sharex=True, sharey=True,
                               figsize=figsize)
        pg.show(self.pd, self.res, ax=ax[0], logScale=False, colorBar=True,
                coverage=coverage, cMin=rlim[0], cMax=rlim[1],
                label=r'resistivity $\rho$ [$\Omega$m]', cMap='Spectral_r',
                **kwargs)
        pg.show(self.pd, self.m, ax=ax[1], logScale=False, colorBar=True,
                coverage=coverage, cMin=mlim[0], cMax=mlim[1],
                label=r'chargeability $m$ [-]', cMap='plasma', **kwargs)
        pg.show(self.pd, self.tau, ax=ax[2], logScale=True, colorBar=True,
                coverage=coverage, cMin=tlim[0], cMax=tlim[1],
                label=r'time constant $\tau$ [s]', **kwargs)
        pg.show(self.pd, self.c, ax=ax[3], logScale=False, colorBar=True,
                cMin=clim[0], cMax=clim[1], coverage=coverage,
                label=r'relaxation exponent $c$ [-]', **kwargs)

        fig.tight_layout()
        if save:
            fig.savefig(self.basename+'-CCfit.pdf', bbox_inches='tight')

        self.figs['CC'] = fig
        return fig, ax

    def showColeColeFit(self, *args, **kwargs):
        """Redirecto to new name showColeColeParameters."""
        return self.showColeColeParameters(*args, **kwargs)

    def saveFigures(self, ext='.pdf', **kwargs):
        """Save all figures in .figs to disk."""
        kwargs.setdefault('bbox_inches', 'tight')
        for key in self.figs:
            self.figs[key].savefig(self.basename+'-'+key+ext, **kwargs)


def main(argv):
    """Main."""
    sip = FDIP(argv)
    print(sip)
    sip.generateSpectraPDF()
    sip.generateDataPDF()
    sip.writeAllData()
    sip.writeDataMat()
    sip.writeSingleFrequencyData()
    sip.filter(kmax=30000)
    sip.singleInversion()
    sip.showSingleResult()
    sip.simultaneousInversion()
    sip.showAllResults()


if __name__ == "__main__":
    main(sys.argv[1])
    pg.wait()
