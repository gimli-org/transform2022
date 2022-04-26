"""Time-Domain Induced Polarization (TDIP) data handling."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pygimli as pg
import pybert as pb
from . mipmodelling import (DCIPMModelling, DCIPSeigelModelling,
                            CCTDModelling, DCIPMSmoothModelling)
from pybert.importer import importTDIPdata


class TDIP():
    """Class managing time-domain induced polarisation (TDIP) field data."""

    def __init__(self, filename=None, **kwargs):
        """Initialize with optional data load.

        Parameters
        ----------
        filename : str
            name of file to read in, allowed formats are:
                * Syscal Pro export file (*.txt)
                * ABEM TXT export file (*.txt or raw time series)
                * Syscal Pro binary file (*.bin)
                * GDD format (*.gdd)
                * Ares II format (*.2dm)
                * Aarhus Workbench processed data (*.tx2 and *.dip)
                * res2dinv data

        **kwargs:
            * paraGeometry : PLC (pygimli mesh holding geometry)
                plc for the 2d inversion domain
            * paraMesh : pygimli Mesh instance
                plc for the 2d inversion domain
            * verbose : bool
                Be verbose.
        """
        self.verbose = kwargs.get('verbose', False)
        self.basename = 'newfile'  # for saving results and images
        self.figs = {}  # figure container
        self.header = {}  # header for supplemental information
        self.t = kwargs.pop('t', [])  # time vector
        self.data = kwargs.pop('data', None)  # data container
        self.rhoa = kwargs.pop('rhoa', None)  # app. resistivity matrix [Ohm m]
        self.MA = kwargs.pop('MA', None)  # app. chargeability matrix [mV/V]
        self.ERT = None  # Resistivity manager class instance
        self.sINV = None  # single inversion instance
        self.pd = None  # paraDomain
        self.res = None  # resistivity
        self.m = None  # single-time spectral chargeability
        self.M = None  # full-decay spectral chargeability
        self.m0 = None  # Cole-Cole chargeability
        self.tau = None  # Cole-Cole time constant
        self.c = None  # Cole-Cole exponent
        self.customParaGeometry = kwargs.pop('paraGeometry', None)
        self.customParaMesh = kwargs.pop('paraMesh', None)

        if filename is not None:
            self.loadData(filename, **kwargs)
            if not self.data.exists('k'):
                self.data['k'] = pg.core.geometricFactors(self.data, dim=2)

    def __repr__(self):  # for print function
        """Readable representation of the class."""
        out = ['TDIP data: ' + self.data.__str__() + ' nt=' + str(len(self.t))]
        out.append("MA shape = " + str(self.MA.shape))
        if hasattr(self, 'header'):
            for key in self.header:
                val = self.header[key]
                if isinstance(val, str):
                    out.append(val)
                elif isinstance(val, int) or isinstance(val, float):
                    out.append(key+' = '+str(val))
                else:
                    out.append(key+' = array('+str(val.shape)+')')
        return "\n".join(out)

    def loadData(self, filename=None, **kwargs):
        """Load data from any of the supported file types.

        Supported formats
        -----------------
        TXT - ABEM or Syscal Ascii (column) output
        BIN - Syscal binary format
        DAT - Res2dInv format
        GDD - GDD format (less tested)
        TX2 - Aarhus Workbench data
        DIP - AarhusInv (processed) data
        OHM - BERT format with ip1, ip2, ... fields
        OHM/MA - BERT format with scheme file and MA file
        """
        assert isinstance(filename, str), "Needs string to load files"
        self.basename = filename[:filename.rfind('.')]
        from os.path import isfile
        if isfile(self.basename+'.MA') and isfile(self.basename+'.shm'):
            self.data = pg.DataContainerERT(self.basename+'.shm')
            A = np.genfromtxt(self.basename+'.MA').T
            self.t = A[:, 0]
            self.MA = A[:, 1:]
        else:
            self.data, self.MA, self.t, self.header = importTDIPdata(filename)

        self.data.checkDataValidity(remove=False)
        print(self.data.size(), self.MA.shape)
        try:
            self.MA = self.MA[:, np.array(self.data('valid'), dtype=bool)]
        except Exception:
            print("no valid apparent chargeability")

        self.data.removeInvalid()
        self.ensureRhoa()

        return

    def load(self, *args, **kwargs):
        """Load data. Use loadData instead."""
        pg.deprecated("use loadData instead")
        self.loadData(*args, **kwargs)

    def ensureRhoa(self):
        """Make sure apparent resistivity is present in file."""
        if not self.data.allNonZero('k'):
            self.data.set('k', pb.geometricFactors(self.data, 2))  # check dim
        if not self.data.allNonZero('rhoa'):
            if not self.data.allNonZero('r'):
                self.data.set('r', self.data('u')/self.data('i'))

            self.data.set('rhoa', self.data('r') * self.data('k'))
            # self.filter(rmin=1e-8)

    def filter(self, tmin=0, tmax=1e9, kmax=1e6, electrode=None, forward=False,
               a=None, b=None, m=None, n=None, ab=None, mn=None, corrSID=1,
               rmin=0, rmax=1e9, emax=1e9, fitmax=1e9, m0max=None, m0min=None,
               taumin=-9e99, taumax=9e99, umin=0, umax=9e9, nr=[], mask=False):
        """Filter data with respect to frequencies and geometric factor.

        Parameters
        ----------
        tmin, tmax : double
            minimum/maximum time (gate center) in s
        rmin, rmax : double
            minimum/maximum apparent resistivity in Ohmm
        kmax : double
            maximum (absolute) geometric factor in m
        emax : double
            maximum error in percent
        m0min, m0max : double
            minimum/maximum (fitted) initial chargeability
        taumin, taumax : double
            minimum/maximum (fitted) time constant
        fitmax : double
            maximum exponential fit
        electrode : int
            electrode to be removed completely
        a/b/m/n : int
            delete data with specific current or potential electrode
        ab/mn : int
            delete data with specific current or potential dipole lengths
        corrSID: int [1]
            correct sensor index (like in data files)
        nr : iterable of ints []
            data indices to delete
        forward : bool
            keep only forward-directed measurements
        """
        print("filtering: nt={:d}, nd={:d}".format(*self.MA.shape))
        # time index
        ind = (self.t >= tmin) & (self.t <= tmax)
        self.MA = self.MA[ind, :]
        self.t = self.t[ind]
        if 'ipDT' in self.header:
            self.header['ipDT'] = self.header['ipDT'][ind]
        if 'ipGateT' in self.header:
            ind = np.append(ind, ind[-1])
            self.header['ipGateT'] = self.header['ipGateT'][ind]
        # data index
        ind = (np.abs(self.data('k')) <= kmax)  # maximum geometric factor
        ind[self.data['rhoa'] < rmin] = False
        ind[self.data['rhoa'] > rmax] = False
        if self.data.allNonZero('u'):
            ind[pg.abs(self.data['u']) >= umax] = False
            ind[pg.abs(self.data['u']) <= umin] = False

        ind[self.data['err'] > emax/100] = False
        if self.data.allNonZero('fit'):
            ind[self.data['fit'] > fitmax] = False
        if self.data.allNonZero('tau'):
            ind[self.data['tau'] > taumax] = False
            ind[self.data['tau'] < taumin] = False
        if self.data.allNonZero('m0'):
            if m0max is not None:
                ind[self.data['m0'] > m0max] = False
            if m0min is not None:
                ind[self.data['m0'] < m0min] = False

        ind[nr] = False  # individual numbers
        am = self.data("m") - self.data("a")
        if ab is not None:
            ind[np.isclose(np.abs(self.data["b"]-self.data["a"]), ab)] = False
        if mn is not None:
            ind[np.isclose(np.abs(self.data["n"]-self.data["m"]), mn)] = False
        if forward:
            ind[am < 0] = False  # reverse measurements
            # print(sum(ind))
        for name in ['a', 'b', 'm', 'n']:
            u = list(np.atleast_1d(eval(name)))
            if electrode is not None:
                u.extend(list(np.atleast_1d(electrode)))
            for uu in u:
                ind = ind & np.not_equal(self.data(name) + corrSID, uu)

        if mask:  # do not delete the data but mask only the IP values
            fi = pg.find(ind)
            if isinstance(self.MA, np.ma.masked_array):
                revind = np.setxor1d(np.arange(self.data.size()), fi)
                pg.debug(revind)
                self.MA.mask[:, revind] = 1
            else:
                idx = np.ones_like(self.MA)
                idx[:, fi] = 0
                self.MA = np.ma.masked_array(self.MA, idx)
        else:
            self.data.set('valid', pg.Vector(self.data.size()))
            self.data.markValid(pg.find(ind))
            self.data.removeInvalid()
            self.MA = self.MA[:, ind]

        if electrode is not None:
            self.data.removeUnusedSensors()

        print("filtered: nt={:d}, nd={:d}".format(*self.MA.shape))
        # return ind

    def mask(self, mamin=1e-6, mamax=10000, filter=False):
        """Mask out outliers.

        Parameters
        ----------
        The following filters will mask the whole decay (i.e. deactivate IP)

        m0min, m0max : double
            minimum/maximum (fitted) initial chargeability
        taumin, taumax : double
            minimum/maximum (fitted) time constant
        fitmax : double
            maximum exponential fit
        """
        self.MA = np.ma.masked_outside(self.MA, mamin, mamax)
        if filter:
            print("Filtering after masking")
            self.filter(nr=np.nonzero(
                ~np.ma.any(self.MA, axis=0).data)[0])

    def showData(self, *args, **kwargs):
        """Show apparent resistivity pseudosection.

        Parameters
        ----------
        cMin, cMax : float
            minimum/maximum colorbar range (otherwise min/max data)
        cMap : string or colormap ['Spectral_r']
            colormap to be used
        **kwargs : plotting arguments (see pb.showData)
        """
        kwargs.setdefault('cMap', 'Spectral_r')
        kwargs.setdefault('logScale', True)
        return pb.show(self.data, *args, **kwargs)

    def showRhoa(self, **kwargs):  # backward compatibility
        """Old function for showing apparent resistivity. Use showData."""
        pg.deprecated("Use showData.")
        return self.showData()

    def setGates(self, t=None, dt=None, delay=0.0):
        """Set time by specifying midpoints (t) or gate lengths dt & delay."""
        if t is None:
            assert dt is not None, "gate length and delay time must be set"
            if isinstance(dt, float):  # constant gate length
                self.dt = np.ones(len(self.t)) * dt
            else:
                self.dt = np.array(dt)

            t = np.cumsum(self.dt) - self.dt/2 + delay
            self.header['ipGateT'] = np.cumsum(np.hstack((0, self.dt))) + delay
            self.header['dt'] = self.dt
            self.header['delay'] = delay

        assert len(t) == self.MA.shape[0]
        self.t = t

    def integralChargeability(self, normalize=True, **kwargs):
        """Compute integral chargeability by summing up windows x dt.

        Parameters
        ----------
        normalize : bool [True]
            normalize such that mV/V is retrieved, otherwise msec
        start : int [0]
            first gate to take
        stop : int [self.MA.shape[1]]
            last gate to take

        Returns
        -------
            integral chargeability : numpy.array
        """
        start = kwargs.pop('start', 1)
        stop = kwargs.pop('stop', len(self.MA))
        mint = pg.Vector(self.data.size())
        for i in range(start-1, stop):
            mint += self.MA[i] * self.t[i]

        if normalize:
            mint /= sum(self.t[start-1:stop])

        self.data.set('mint', mint)

        return np.array(self.data('mint'))

    def showIntegralChargeability(self, **kwargs):
        """Show integral chargeability (kwargs forwarded to pb.show)."""
        if not self.data.haveData('mint'):
            self.data.set('mint', self.integralChargeability(**kwargs))

        kwargs.setdefault('cMap', 'plasma')
        kwargs.setdefault('logScale', True)

        return pb.show(self.data, self.data('mint'), **kwargs)

    def showMa(self, nr=0, **kwargs):
        """Show apparent chargeability (kwargs forwarded to pb.show).

        Parameters
        ----------
        nr : int [0]
            number of time gate to show
        **kwargs : any plotting arguments to be passed to pb.showData
        """
        kwargs.setdefault('cMap', 'plasma')
        kwargs.setdefault('logScale', True)
        return pb.show(self.data, self.MA[nr], **kwargs)

    def fitDataDecays(self, show=False, tmin=0):
        """Fit (data) decays by exponential function.

        Linear regression of log(M) over t and stores the result in data:
            m0 - zero-time chargeability
            tau - characteristic decay time
            fit - RMS of difference between measured and modelled (log) M
        """
        t = np.copy(self.t)
        fi = np.nonzero(t >= tmin)[0]
        G = np.ones((len(fi), 2))
        G[:, 1] = t[fi]
        Ginv = np.linalg.inv(G.T.dot(G)).dot(G.T)
        lma = np.log(np.abs(self.MA[fi, :].data)+0.0001)
        ab = Ginv.dot(lma)
        print(ab.shape)
        self.data.set('m0', np.exp(ab[0]))
        self.data.set('tau', -1./ab[1])
        fit = np.sqrt(np.mean((G.dot(ab)-lma)**2, axis=0))
        if isinstance(fit, np.ma.MaskedArray):
            self.data.set('fit', fit.data)
        else:
            self.data.set('fit', fit+1e-5)
        if show:
            pb.show(self.data, 'm0', cMin=0, logScale=False, label='M0 [mV/V]',
                    markOutside=True)
            pb.show(self.data, 'tau', label=r'$\tau$ [s]', markOutside=True,
                    cMin=min(np.abs(self.data('tau'))), logScale=True)
            pb.show(self.data, 'fit', label=r'fit rms [log]', logScale=False)

    def fitDecays(self, *args, **kwargs):
        """Fit data decays (old name). Use fitDataDecays instead."""
        pg.deprecated("Use fitDataDecays instead (or fitModelDecays)")
        self.fitDataDecays(*args, **kwargs)

    def generateDataPDF(self, rdict=None, mdict=None, **kwargs):
        """Generate multi-page pdf file with all data as pseudosections.

        Parameters
        ----------
        rdict : dict
            dictionary with plotting arguments for apparent resistivity
        mdict : dict
            dictionary with plotting arguments for apparent chargeability

        Any other keyword args are associated to mdict.
        """
        if rdict is None:
            rdict = dict(logScale=True)
        if mdict is None:
            posneg = kwargs.pop("posneg", None)
            if posneg:  #
                mdict = dict(cMin=-posneg, cMax=posneg, cMap='coolwarm')
            else:
                mdict = dict(cMin=1, cMax=np.max(self.MA), logScale=True)
        # set some default keywords
        mdict.update(**kwargs)
        rdict.setdefault('cMap', 'Spectral_r')  # default color scale
        rdict.setdefault('xlabel', 'x [m]')
        rdict.setdefault('label', r'$\rho_a$ [$\Omega$m]')
        mdict.setdefault('cMap', 'plasma')
        mdict.setdefault('xlabel', 'x [m]')
        mdict.setdefault('label', r'$m_a$ [mV/V]')
        fig, ax = plt.subplots()
        basename = kwargs.pop('basename', self.basename)
        with PdfPages(basename+'-alldata.pdf') as pdf:
            pb.show(self.data, 'rhoa', ax=ax, **rdict, **kwargs)
            ax.set_title('apparent resistivity')
            fig.savefig(pdf, format='pdf')
            if max(self.data('err')) > 0:
                fig.clf()
                ax = fig.add_subplot(111)
                pb.show(self.data, self.data('err')*100+0.01, ax=ax,
                        label=r'$\epsilon$ [%]', **kwargs)
                ax.set_title('error')
                fig.savefig(pdf, format='pdf')
            if self.data.allNonZero('stacks'):
                fig.clf()
                ax = fig.add_subplot(111)
                pb.show(self.data, 'stacks', ax=ax, label='stacks',
                        logScale=False, **kwargs)
                ax.set_title('stacks')
                fig.savefig(pdf, format='pdf')
            if self.data.allNonZero('i'):
                fig.clf()
                ax = fig.add_subplot(111)
                pb.show(self.data, self.data('i')*1000, ax=ax,
                        label=r'$I$ [mA]', **kwargs)
                ax.set_title('current')
                fig.savefig(pdf, format='pdf')
            if self.data.allNonZero('u'):
                fig.clf()
                ax = fig.add_subplot(111)
                pb.show(self.data, self.data('u')*1000, ax=ax,
                        label=r'$U$ [mV]', **kwargs)
                ax.set_title('voltage')
                fig.savefig(pdf, format='pdf')
            if self.data.allNonZero('tau'):
                fig.clf()
                ax = fig.add_subplot(111)
                tau = np.abs(self.data('tau').array())
                cMin, cMax = np.nanquantile(tau, [0.03, 0.97])
                pb.show(self.data, tau, ax=ax, label=r'$\tau$ [s]',
                        ind=(tau > 0), logScale=True, cMin=cMin, cMax=cMax,
                        **kwargs)
                ax.set_title('apparent relaxation time')
                fig.savefig(pdf, format='pdf')
            if self.data.allNonZero('fit'):
                fig.clf()
                ax = fig.add_subplot(111)
                pb.show(self.data, 'fit', ax=ax, label='fit (log10)', **kwargs)
                ax.set_title('fit')
                fig.savefig(pdf, format='pdf')
            if self.data.allNonZero('m0'):
                fig.clf()
                ax = fig.add_subplot(111)
                pb.show(self.data, 'm0', ax=ax, **mdict)
                ax.set_title('fitted (t=0) chargeability')
                fig.savefig(pdf, format='pdf')
            for i, ma in enumerate(self.MA):
                fig.clf()
                ax = fig.add_subplot(111)
                pb.show(self.data, ma, ax=ax, **mdict)
                tstr = " (t={:.3f}s)".format(self.t[i])
                if 'ipGateT' in self.header:
                    tstr = ' (t={:g}-{:g}s)'.format(
                        *(self.header['ipGateT'][i:i+2]))
                ax.set_title('apparent chargeability gate ' + str(i+1) + tstr)
                fig.savefig(pdf, format='pdf')

    def showApparentChargeability(self, ax=None, nr=0, **kwargs):
        """Show apparent chargeability of a single Windows."""
        kwargs.setdefault('cMap', 'plasma')
        kwargs.setdefault('cMin', 0.1)
        kwargs.setdefault('cMax', 1000)
        if ax is None:
            fig, ax = plt.subplots()
            self.figs['MA{:02d}'.format(nr)] = fig

        pb.show(self.data, self.MA[nr-1], ax=ax, **kwargs)

    def getDataIndex(self, abmn=None):
        """Return data index for given ABMN combination."""
        a = np.minimum(self.data('a'), self.data('b'))
        b = np.maximum(self.data('a'), self.data('b'))
        m = np.minimum(self.data('m'), self.data('n'))
        n = np.maximum(self.data('m'), self.data('n'))
        nr = np.nonzero(np.isclose(a, min(abmn[:2])-1) &
                        np.isclose(b, max(abmn[:2])-1) &
                        np.isclose(m, min(abmn[2:4])-1) &
                        np.isclose(n, max(abmn[2:4])-1))[0][0]
        return nr

    def getDataDecay(self, abmn=None):
        """Return apparent chargeability decay for given ABMN combination."""
        nr = self.getDataIndex(abmn)
        if isinstance(nr, np.int64):
            return self.MA[:, nr]
        else:
            print(abmn, nr, type(nr))
            raise Exception("No such abmn combination found.")

    def showDecay(self, nr=[], ax=None, ab=None, mn=None, verbose=True,
                  **kwargs):
        """Show decay curves for groups of data.

        Parameters
        ----------
        nr : iterable
            list of data indices to show, if not given ab/mn are analysed
        ab : [int, int]
            list of sensor numbers for current injection (counting from 1)
        mn : [int, int]
            list of sensor numbers for potential (counting from 1)

        Plotting Keywords
        -----------------
        showFit : bool [False]
            show fitted Debye or Cole-Cole curve
        label : str
            legend label for single data, otherwise generated from A,B,M,N
        basename : str
            string to prepend to automatically generated label
        marker : str ['x']
            marker to use for plotting
        xlim, ylim : [float, float]
            limits for x or y axis
        xscale : str ['linear']
            scaling of x axis
        yscale : str ['log']
            scaling of y axis
        xlabel : str [r'$t$ [s]']
            label for the x axis
        ylabel : str [r'$m_a$ [mV/V]']
            label for the y axis
        """
        data = self.data
        if "basename" in kwargs:
            bs = kwargs['basename']
        else:
            bs = 'abmn'
            if ab is not None:
                bs = "mn"
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
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        shFit = (kwargs.pop('showFit', False) and self.data.haveData('m0') and
                 self.data.haveData('tau'))
        ls = "" if shFit else "-"
        kwargs.setdefault('ls', ls)
        outkeys = ['xlim', 'ylim', 'xlog', 'ylog', 'xscale', 'yscale']
        if isinstance(nr, int):
            nr = [nr]
        if len(nr) > 0:
            if ax is None:
                self.figs['decay'], ax = plt.subplots()
            for nn in nr:
                abmn = [int(self.data(t)[nn]+1) for t in ['a', 'b', 'm', 'n']]
                kw = kwargs.copy()
                for key in outkeys:
                    if key in kw:
                        kw.pop(key)

                ma = self.MA[:, nn]
                if kw.pop('invalid', False):
                    ax.plot(self.t, ma.data, color='gray', ms=2,
                            marker=kwargs["marker"], ls=ls)
                    ax.plot(self.t, -np.array(ma.data), ls='--',
                            color='lightgray', ms=2, marker=kwargs["marker"])
                if ab is not None:
                    kw.setdefault('label', (bs+': '+'{:d} '*2).format(
                        *abmn[2:]))
                    kw.setdefault('color', "C"+str(abmn[2] % 9))
                else:
                    kw.setdefault('label', (bs+': '+'{:d} '*4).format(*abmn))
                li = ax.plot(self.t, ma, **kw)[0]
                if shFit and np.ma.any(ma):
                    ax.plot(self.t, self.data('m0')[nn] *
                            np.exp(-self.t/self.data('tau')[nn]),
                            color=li.get_color(), ls='--')
                    if verbose:
                        print(self.data['fit'][nn])

            ax.grid(True)
            ax.legend()
            if 'xlim' in kwargs:
                ax.set_xlim(kwargs['xlim'])
            if 'ylim' in kwargs:
                ax.set_ylim(kwargs['ylim'])
            if 'xscale' in kwargs:
                ax.set_xscale(kwargs['xscale'])
            if 'yscale' in kwargs:
                ax.set_yscale(kwargs['yscale'])
            ax.set_xlabel(kwargs.pop('xlabel', r'$t$ [s]'))
            ax.set_ylabel(kwargs.pop('ylabel', r'$m_a$ [mV/V]'))
            tit = ""
            if ab is not None:
                tit = "A-B = {:d}-{:d}".format(*abmn[:2])
            if mn is not None:
                tit += " , M-N = {:d}-{:d}".format(*abmn[2:])

            if len(tit) > 0 and len(ax.get_title()) == 0:
                ax.set_title(tit)

            return ax

    def generateDecayPDF(self, **kwargs):
        """Generate a pdf file with all decays sorted by current injections.

        Parameters
        ----------
        marker : str ['x']
            marker to use for plotting
        xlim, ylim : float [<automatic>]
            limits for x or y axis
        xscale : str ['linear']
            scaling of x axis
        yscale : str ['log']
            scaling of y axis
        """
        from matplotlib.backends.backend_pdf import PdfPages

        ab = (self.data('a')+1) * 1000 + self.data('b') + 1
        # ab = (self.data('a')) * 1000 + self.data('b')
        uab = np.array(np.unique(ab), dtype=int)
        # sort them according AB length (major) and A electrode (minor)
        dip = np.abs(uab // 1000 - (uab % 1000)) * 1000 + uab // 1000
        uab = uab[np.argsort(dip)]
        kwargs.setdefault("verbose", False)
        basename = kwargs.pop('basename', self.basename)
        with PdfPages(basename + '-decays.pdf') as pdf:
            fig, ax = plt.subplots()
            for u in uab:
                ab = [u // 1000, (u % 1000)]
                # ab = [u // 1000 + 1, (u % 1000) + 1]
                ax.cla()
                self.showDecay(ab=ab, ax=ax, **kwargs)
                fig.savefig(pdf, format='pdf')

    def saveData(self, **kwargs):
        """Save all data as shm and accompagnying .MA (like FDIP) files."""
        basename = kwargs.pop("basename", self.basename)
        self.data.save(basename+".shm", "a b m n k rhoa")
        MA = np.array(self.MA.data)
        if isinstance(self.MA, np.ma.masked_array):
            MA[self.MA.mask] = 999

        A = np.column_stack((self.t, MA))
        np.savetxt(basename+".MA", A.T, fmt="%6.3f", delimiter="\t")
        data = pg.DataContainerERT(self.data)
        tokens = "a b m n k rhoa"
        if self.data.haveData("m0"):
            tokens += " m0"

        for i in range(self.MA.shape[0]):
            tok = "ip"+str(i+1)
            data[tok] = MA[i, :]
            tokens += " " + tok

        data.save(basename+".ohm", tokens)

    def invertRhoa(self, **kwargs):
        """Invert apparent resistivity. kwargs forwarded to ERTManager."""
        if self.ERT is None:
            self.ERT = pg.physics.ert.ERTManager()

        self.ERT.fop.setData(self.data)
        if "mesh" in kwargs:
            self.ERT.fop.setMesh(kwargs["mesh"])

        show = kwargs.pop('show', False)
        if not self.data.haveData("err"):
            self.data["err"] = self.ERT.estimateError(self.data)

        self.res = self.ERT.invert(data=self.data, **kwargs)
        self.response = self.ERT.inv.response
        self.coverage = self.ERT.coverage()
        self.pd = self.ERT.paraDomain
        if show:
            return self.showResistivity()

    def invertMa(self, nr=0, ma=None, fop=None, mesh=None,
                 res=None, **kwargs):
        """Invert for chargeability.

        Directly invert apparent chargeability for intrinsic chargeability.
        As the Jacobian of the ERT forward operator is needed, the inversion of
        the DC data must be done before.

        Parameters
        ----------
        nr : int [0]
            gate to invert (counting from 1), 0 means fitted (zero-time) Ma
        error : float [0.002]
            error in apparent chargeability
        lam : float [100]
            regularization strength
        zWeight : float [1.0]
            vertical penalty
        robustData : bool [False]
            use L1 norm on data misfit
        blockyModel: bool [False]
            use L1 norm on model roughness
        regionFile : str
            load region file
        show : bool [False]
            show resulting chargeability distribution

        fop : pg.DC*MultiElectrodeModelling
            DC forward operator
        mesh : pg.Mesh
            inversion mesh
        res : iterable
            resistivity vector
        """
        show = kwargs.pop('show', False)
        if ma is None:
            if nr == 0 or nr == '0':
                if self.data.exists('m0'):
                    ma = self.data('m0') * 0.001
                else:
                    pg.info("fitted M0 not existing, taking first gate")
                    ma = self.MA[0] * 0.001
            else:
                ma = self.MA[nr-1] * 0.001  # should MA be in V/V instead?

        if isinstance(ma, np.ma.MaskedArray):
            ma = np.copy(ma.data)

        errLevel = kwargs.pop('error', 0.002)
        if hasattr(errLevel, '__iter__') and len(errLevel) == len(ma):
            maerr = errLevel
        else:
            maerr = np.ones_like(ma) * errLevel
        # restrict values above 1
        if 0:
            maerr[ma > 1] = 1e5
            ma[ma > 1] = 0.99
        # restrict values below 1
        if 0:
            maerr[ma < 0] = 1e5
            ma[ma < 0] = 0.001

        fIP = []
        verbose = kwargs.pop("verbose", False)
        if kwargs.pop("Seigel", False):
            fIP = DCIPSeigelModelling(self.ERT)
            res = self.res
        else:
            if fop is None:
                fop = self.ERT.fop
            if mesh is None:
                mesh = self.ERT.mesh
            if res is None:
                res = self.res  # self.ERT.model

            if hasattr(self, "response") and hasattr(self.response,
                                                     "__iter__"):
                fIP = DCIPMModelling(fop, mesh, res, verbose,
                                     response=self.response)
            else:
                fIP = DCIPMModelling(fop, mesh, res, verbose)

            if 'regionFile' in kwargs:
                fIP.regionManager().loadMap(kwargs.pop('regionFile'))
            else:
                if fop.regionManager().regionCount() > 1:
                    fIP.region(1).setBackground(True)
                    fIP.region(2).setConstraintType(1)

        fIP.regionManager().setZWeight(kwargs.pop('zWeight', 1.0))
        fIP.createRefinedForwardMesh(True)
        # tD, tM = pg.core.RTransLog(), pg.core.RTransLogLU(0, 0.99)
        INV = pg.core.RInversion(ma, fIP, True, kwargs.pop("debug", False))
        tD, tM = pg.core.RTrans(), pg.core.RTransLogLU(0.0001, 0.999)
        INV.setTransData(tD)
        INV.setTransModel(tM)
        mstart = pg.Vector(len(res), np.abs(pg.median(ma)))
        INV.setModel(mstart)
        INV.setAbsoluteError(maerr)
        INV.setLambda(kwargs.pop('lam', 100))
        INV.setRobustData(kwargs.pop('robustData', False))
        INV.setBlockyModel(kwargs.pop('blockyModel', False))
        self.m = INV.run()
        self.mafwd = INV.response()
        print("chi^2={:.1f} RMS={:.1f}mV/V".format(INV.chi2(),
                                                   INV.absrms()*1000))
        if show:
            return self.showChargeability()
        else:
            return INV

    def showResistivity(self, ax=None, **kwargs):
        """Show resistivity inversion result.

        Any kwargs (cMin, cMax, logScale) are forwarded to pg.show.
        """
        kwargs.setdefault('logScale', True)
        kwargs.setdefault('label', r'resistivity [$\Omega$m]')
        kwargs.setdefault('cMap', 'Spectral_r')
        if self.pd is None or len(self.res) != self.pd.cellCount():
            self.pd = self.ERT.paraDomain
        return pg.show(self.pd, self.res, ax=ax, **kwargs)

    def showChargeability(self, ax=None, **kwargs):
        """Show chargeability inversion result.

        Any kwargs (cMin, cMax, logScale) are forwarded to pg.show.
        """
        kwargs.setdefault('label', 'chargeability [mV/V]')
        kwargs.setdefault('cMap', 'plasma')
        return pg.show(self.ERT.paraDomain, self.m*1e3, ax=ax, **kwargs)

    def showResults(self, rkw={}, mkw={}):
        """Show result (resistivity and chargeability in two subfigures).

        Parameters
        ----------
        rkw : dict
            dictionary for being passed to showResistivity
        mkw : dict
            dictionary for being passed to showChargeability
        """
        self.figs['result'], ax = plt.subplots(nrows=2)
        self.showResistivity(ax=ax[0], **rkw)
        self.showChargeability(ax=ax[1], **mkw)
        return ax

    def individualInversion(self, **kwargs):
        """Carry out individual inversion for spectral chargeability.

        Creates numpy array self.M storing chargeabilities for all gates

        Parameters
        ----------
        error : float [0.002]
            assumed error of apparent chargeability
        **kwargs : dict
            passed to self.invertMa
        """
        errLevel = kwargs.pop('error', 0.002)
        error = np.ones(self.MA.shape[1]) * errLevel
        if self.ERT is None or self.res is None:
            self.invertRhoa(**kwargs)

        self.M = np.zeros((len(self.MA), len(self.res)))
        for i, ma in enumerate(self.MA):
            print('Inverting gate {}'.format(i+1))
            error[:] = errLevel
            if isinstance(ma, np.ma.MaskedArray):
                madata = np.copy(ma.data)
                error[ma.mask] = 1e8
                madata[ma.mask] = 0.1
                self.invertMa(ma=madata*0.001, error=error, **kwargs)
            else:
                self.invertMa(ma=ma*0.001, error=error, **kwargs)

            self.M[i] = self.m

    def simultaneousInversion(self, fop=None, res=None, **kwargs):
        """Carry out simultaneous inversion with smoothness along t axis."""
        errLevel = kwargs.pop('error', 0.002)
        if fop is None or res is None:
            if self.ERT is None:
                self.invertRhoa(**kwargs)  # rather check whether already done!

            fop = self.ERT.fop

        fop.setVerbose(False)
        error = np.ones_like(self.MA) * errLevel
        MA = np.copy(self.MA.data)  # make a copy as it will be changed
        if isinstance(self.MA, np.ma.MaskedArray):
            error[self.MA.mask] = 1e8
            MA[self.MA.mask] = 0.1

        fIP = DCIPMSmoothModelling(fop, self.pd, self.res,
                                   self.t)
        fIP.createRefinedForwardMesh(False)
        fIP.regionManager().setZWeight(kwargs.get('zWeight', 0.3))
        tD, tM = pg.core.RTrans(), pg.core.RTransLogLU(0, 0.99)
        INV = pg.core.RInversion(MA.ravel()*0.001, fIP, tD, tM, True, False)
        mstart = pg.Vector(fIP.nc*fIP.nt, 0.1)
        INV.setModel(mstart)
        INV.setAbsoluteError(error.ravel())
        print(kwargs)
        mm = INV.run()  # **kwargs)
        self.M = np.reshape(mm, (-1, self.pd.cellCount()))
        self.MAfwd = np.reshape(INV.response(), (len(self.t), -1))

    def getCellID(self, pos):
        """Return cell ID of nearest cell to position."""
        return self.pd.findCell(pos).id()

    def getModelDecay(self, cellID, return_index=False):
        """Return SIP spectrum for single cell (id or position)."""
        if hasattr(cellID, '__iter__'):  # tuple => position
            cellID = self.getCellID(cellID)

        if return_index:
            return self.M[:, cellID], cellID
        else:
            return self.M[:, cellID]

    def showModelDecay(self, cellID, **kwargs):
        """Show SIP spectrum for single cell (id or position)."""
        decay, idx = self.getModelDecay(cellID, return_index=True)
        shfit = kwargs.pop("showFit", False)
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            self.figs['modelDecay'], ax = plt.subplots()
            # kwargs.setdefault('xLabel)

        kwargs.setdefault("label", 'inverted')
        ax.loglog(self.t, decay*1000, 'x-', **kwargs)
        if shfit:
            if hasattr(self, 'FWR'):
                ax.loglog(self.t, self.FWR[:, idx]*1000, label='fitted')
            else:

                pg.warn("No fwd!")

        ax.legend()
        ax.grid(True)
        return ax

    def showModelDecays(self, positions=None, **kwargs):
        """Show model spectra for a number of positions or IDs."""
        kwargs.setdefault("showFit", False)
        self.figs['modelDecays'], ax = plt.subplots()
        LABELS = []
        if positions is None:  # not given: check for x and z vectors
            x = kwargs.pop("x", None)
            z = kwargs.pop("z", None)
            if z is None:
                z = kwargs.pop("y", None)
            if x is None and z is None:
                raise NameError("Specify either position vector or x and z")
            if isinstance(x, float) or isinstance(x, int):
                x = np.ones_like(z) * x
            if isinstance(z, float) or isinstance(z, int):
                z = np.ones_like(x) * z

            positions = [[xi, zi] for xi, zi in zip(x, z)]

        for pos in positions:
            label = 'x={:.1f} z={:.1f}'.format(*pos)
            LABELS.append(label)
            # kwargs['label'] = label
            self.showModelDecay(pos, ax=ax, **kwargs)

        ax.set_ylim(auto=True)
        # ax.set_xlim(min(self.freq), max(self.freq))
        ax.legend(LABELS, loc='best')

        return ax

    def fitModelDecays(self, show=False, useColeCole=False, **kwargs):
        """Fit model decays by exponential function or Cole-Cole model.

        Linear regression (Debye) of log(M) over t storing:
            m0 - zero-time (or Cole-Cole) chargeability
            tau - characteristic time constant
            c - Cole-Cole exponent (only if useColeCole=True)
            fit - RMS of difference between measured and modelled (log) M
        """
        G = np.ones((len(self.t), 2))
        G[:, 1] = self.t
        Ginv = np.linalg.inv(G.T.dot(G)).dot(G.T)
        lma = np.log(self.M)
        ab = Ginv.dot(lma)
        self.m0 = np.exp(ab[0])
        self.tau = np.minimum(np.maximum(-1./ab[1], 0.01), 100)
        self.c = np.ones(self.M.shape[1])
        self.fit = np.sqrt(np.mean((G.dot(ab)-lma)**2, axis=0))
        self.FWR = self.m0.reshape(1, -1) * np.exp(-self.t.reshape(-1, 1) *
                                                   self.tau.reshape(1, -1))
        mpar = kwargs.pop('mpar', (0.01, 0.0001, 1.))
        taupar = kwargs.pop('taupar', (1.0, 0.01, 10.0))
        cpar = kwargs.pop('cpar', (0.3, 0.0, 1.05))
        if useColeCole:
            f = CCTDModelling(self.t)
            f.region(0).setParameters(*mpar)  # M
            f.region(1).setParameters(*taupar)  # tau
            f.region(2).setParameters(*cpar)  # c
            INV = pg.core.RInversion(self.M[:, 0], f, False)
            INV.setRobustData(kwargs.pop('robustData', False))
            INV.setMarquardtScheme()
            INV.setRelativeError(kwargs.pop('error', 0.05))
            self.FWR = np.ones_like(self.M)
            # self.fit = np.ones(self.M.shape[1])
            print('Fitting model decays:')
            medmodel = [np.median(self.m0), np.median(self.tau), 0.5]
            for i in range(self.M.shape[1]):
                print('.', end='')
                INV.setData(self.M[:, i])
                # INV.setModel([self.m0[i], self.tau[i], 0.5])
                INV.setModel(medmodel)
                INV.setLambda(kwargs.pop('lam', 1000))
                model = INV.start()
                self.m0[i] = model[0]
                self.tau[i] = model[1]
                self.c[i] = model[2]
                self.FWR[:, i] = INV.response()
                self.fit[i] = INV.absrms()
            if show:
                pg.show(self.pd, self.fit, label='RMS fit')
                return self.showColeColeResults()
        else:  # only a Debye term
            if show:
                fig, ax = plt.subplots(nrows=2)
                pg.show(self.pd, self.m0*1000, ax=ax[0], label='M (mV/V)')
                pg.show(self.pd, self.tau, ax=ax[1], label=r'$\tau$ (s)')
                pg.show(self.pd, self.fit, label='RMS fit (log)', logScale=0)
                return ax

    def showColeColeResults(self, rlim=(None, None), clim=(0, 0.5),
                            mlim=(None, None), tlim=(None, None), shFit=0):
        """Show resulting Cole-Cole models."""
        isCC = int(min(self.c) < 1)
        fig, ax = plt.subplots(nrows=3+isCC+int(shFit), figsize=(8, 12))
        pg.show(self.pd, self.res, ax=ax[0], label='resistivity (Ohmm)',
                cMap='Spectral_r', cMin=rlim[0], cMax=rlim[1], logScale=True)
        pg.show(self.pd, self.m0*1000, ax=ax[1], cMin=mlim[0], cMax=mlim[1],
                label='M (mV/V)', cMap='plasma', logScale=False)
        pg.show(self.pd, self.tau, ax=ax[2], cMin=tlim[0], cMax=tlim[1],
                label=r'$\tau$ (s)')
        if isCC:
            pg.show(self.pd, self.c, cMin=clim[0], cMax=clim[1],
                    logScale=False, ax=ax[3], label=r'$c$ (-)')
        if shFit:
            pg.show(self.pd, self.fit, cMin=clim[0], cMax=clim[1],
                    logScale=False, ax=ax[-1], label=r'fit (-)')

        fig.tight_layout()
        self.figs['resultCC'] = fig
        return ax

    def generateModelPDF(self, rdict=None, mdict=None, **kwargs):
        """Generate a multi-page pdf file with all data as pseudosections.

        Parameters
        ----------
        rdict : dict
            dictionary with plotting arguments for apparent resistivity
        mdict : dict
            dictionary with plotting arguments for apparent chargeability
        """
        if rdict is None:
            rdict = dict(logScale=True, cMap='Spectral_r',
                         label=r'$\rho_a$ [$\Omega$m]', xlabel='x [m]')
        if mdict is None:
            mdict = dict(cMin=1, cMax=np.max(self.M)*1000, logScale=True,
                         cMap='plasma', label=r'$m$ [mV/V]', xlabel='x [m]')
        mdict.update(**kwargs)
        rdict.setdefault('cMap', 'Spectral_r')  # default color scale
        mdict.setdefault('cMap', 'plasma')
        fig, ax = plt.subplots()
        basename = kwargs.pop('basename', self.basename)
        with PdfPages(basename+'-allmodel.pdf') as pdf:
            self.showResistivity(ax=ax, **rdict)
            fig.savefig(pdf, format='pdf')
            for i, t in enumerate(self.t):
                fig.clf()
                ax = fig.add_subplot(111)
                pg.show(self.pd, self.M[i, :]*1000, ax=ax, **mdict)
                ax.set_title(r'$t({:d})$={:e}'.format(i, t))
                fig.savefig(pdf, format='pdf')

    def saveFigures(self, ext='.pdf', **kwargs):
        """Save all figures in .figs to disk."""
        kwargs.setdefault('bbox_inches', 'tight')
        basename = kwargs.pop('basename', self.basename)
        for key in self.figs:
            self.figs[key].savefig(basename+'-'+key+ext, **kwargs)

    def saveResults(self, **kwargs):
        """Save inversion results to .rho and .M file plus mesh."""
        basename = kwargs.pop("basename", self.basename)
        self.pd.save(basename+'_pd.bms')  # better .bms only?
        if hasattr(self, 'res'):
            np.savetxt(basename+'.rho', self.res)
        if hasattr(self, 'M'):
            np.savetxt(basename+'.M', self.M.T)

        self.saveFit()

    def loadResults(self, loadColeCole=False, **kwargs):
        """Load inversion results from file.

        Loads three (or four) files:
            - *_pd.bms : mesh file of inversion mesh
            - *.rho : resistivity vector (ascii column)
            - *.M : spectral chargeability (ascii columns for each gate)
            - *.rmtc : Cole-Cole/Debye results with rho, m, tau, c (and fit)
              (previously called .mtc without resistivity)

        Parameters
        ----------
        loadColeCole : bool [False]
            try loading Cole-Cole or Debye inversion results
        basename : str [self.basename]
            file base name (*) to load
        """
        basename = kwargs.pop("basename", self.basename)
        self.pd = pg.Mesh(basename+'_pd.bms')
        self.res = np.loadtxt(basename+'.rho')
        self.M = np.loadtxt(self.basename+'.M').T
        if self.M.shape[0] == self.pd.cellCount():  # old style
            self.M = self.M.T
        if loadColeCole:
            self.loadFit()

    def loadResult(self, *args, **kwargs):
        """Load results (old name). Use load results."""
        pg.deprecated("use loadResults instead")
        self.loadResults(*args, **kwargs)

    def saveFit(self, **kwargs):
        """Save fitted chargeability, time constant & exponent to file."""
        basename = kwargs.pop("basename", self.basename)
        if kwargs.pop("noRes", False):  # old style
            if np.any(self.m0) and np.any(self.tau) and np.any(self.c):
                np.savetxt(self.basename+'.mtc', np.column_stack(
                    (self.m0, self.tau, self.c, self.fit)))
        else:
            if (np.any(self.res) and np.any(self.m0) and
                    np.any(self.tau) and np.any(self.c)):
                np.savetxt(basename+'.rmtc', np.column_stack(
                    (self.res, self.m0, self.tau, self.c, self.fit)))

    def loadFit(self, **kwargs):
        """Load fitted chargeability, time constant & exponent from file."""
        basename = kwargs.pop("basename", self.basename)
        import os.path
        if os.path.isfile(basename+".rmtc"):
            self.res, self.m0, self.tau, self.c, self.fit = np.loadtxt(
                basename+'.rmtc', unpack=1)
        elif os.path.isfile(basename+".mtc"):
            self.m0, self.tau, self.c, self.fit = np.loadtxt(
                basename+'.mtc', unpack=1)
            pg.info("Loading deprecated fit result (mtc) instead of rmtc")
        else:
            pg.warn("Could not find fit result " + basename + ".(r)mtc")


if __name__ == "__main__":
    pass
