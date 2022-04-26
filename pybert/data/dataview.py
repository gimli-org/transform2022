# -*- coding: utf-8 -*-
"""Visualize ERT data."""

import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.pyplot as plt

import pygimli as pg
# from pygimli.viewer.mpl import cmapFromName
import pygimli.utils

try:
    from pygimli.viewer.mpl.dataview import generateMatrix, plotMatrix, patchMatrix
    from pygimli.viewer.mpl.dataview import patchValMap, showVecMatrix
except:
    from pygimli.mplviewer.dataview import generateMatrix, plotMatrix, patchMatrix
    from pygimli.mplviewer.dataview import patchValMap

import pybert as pb

from . datascheme import Pseudotype, DataSchemeManager


def showData(data, vals=None, **kwargs):
    """Utility one-liner to show a BERT datafile.

    Creates figure, axis and Show.

    Parameters
    ----------

    data : :bertapi:`BERT::DataContainerERT`

    **kwargs :

        * axes : matplotlib.axes
            Axes to plot into. Default is None and a new figure and
            axes are created.
        * vals : Array[nData]
            Values to be plotted. Default is data('rhoa').
        * schemeName : str
            Name for the scheme to be plotted with the old semiautomatic style.
            Default is 'auto' to draw the new plotting style.
        * : *
            is forwarded to plotERTData (old style) or drawData (new style)
    """
    # remove ax keyword global
    ax = kwargs.pop('axes', None)
    if ax is None:
        ax = kwargs.pop('ax', None)

    if ax is None:
        fig = plt.figure()
        ax = None
        axTopo = None
        if 'showTopo' in kwargs:
            ax = fig.add_subplot(1, 1, 1)
#            axs = fig.subplots(2, 1, sharex=True)
#            # Remove horizontal space between axes
#            fig.subplots_adjust(hspace=0)
#            ax = axs[1]
#            axTopo = axs[0]
        else:
            ax = fig.add_subplot(1, 1, 1)

    pg.checkAndFixLocaleDecimal_point(verbose=False)

#    vals = kwargs.pop('vals', data('rhoa'))
    if vals is None:
        vals = 'rhoa'
    if isinstance(vals, str):
        if data.haveData(vals):
            vals = data(vals)
        else:
            raise KeyError('field not in data container: ', vals)

    schemeName = kwargs.pop('schemeName', 'auto')
    if kwargs.pop("am", False):
        ex = pg.x(data)
        cx = ex[data["a"]]
        px = ex[data["m"]]
        return showVecMatrix(px, cx, vals, ax=ax, **kwargs)

    if schemeName == 'auto':
        ax, cbar = plotERTData(data, vals=vals, ax=ax, **kwargs)
    else:
        # print("data: ", min(vals), max(vals))
        drawData(ax, data, vals, schemeName=schemeName, **kwargs)

    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])

    if 'showTopo' in kwargs:
        # if axTopo is not None:
        print(ax.get_position())
        axTopo = plt.axes([ax.get_position().x0,
                           ax.get_position().y0,
                           ax.get_position().x0+0.2,
                           ax.get_position().y0+0.2])

        x = pg.x(data)
        x *= (ax.get_xlim()[1] - ax.get_xlim()[0]) / (max(x)-min(x))
        x += ax.get_xlim()[0]
        axTopo.plot(x, pg.z(data), '-o', markersize=4)
        axTopo.set_ylim(min(pg.z(data)), max(pg.z(data)))
        axTopo.set_aspect(1)

    # ax.set_aspect('equal')
    # plt.pause(0.1)
    pg.viewer.mpl.updateAxes(ax)

    if schemeName == 'auto':
        return ax, cbar
    else:
        return ax


def plotERTData(data, **kwargs):
    """Plot ERT data as pseudosection matrix (position over separation).

    Parameters
    ----------
    data : pybert.DataContainerERT
        data container with sensorPositions and a/b/m/n fields
    vals : iterable of data.size() [data('rhoa')]
        vector containing the vals to show
    ax : mpl.axis
        axis to plot, if not given a new figure is created
    cMin/cMax : float
        minimum/maximum color vals
    logScale : bool
        logarithmic colour scale [min(A)>0]
    label : string
        colorbar label

    **kwargs:
        * dx : float
            x-width of individual rectangles
        * ind : integer iterable or IVector
            indices to limit display
        * var : int [0]
            historical plotting styles (1, 2)
        * circular : bool
            Plot in polar coordinates when plotting via patchValMap
    Returns
    -------
    ax:
        The used Axes
    cbar:
        The used Colorbar or None
    """
    vals = kwargs.pop('vals', data('rhoa'))
    valid = data.get("valid").array().astype("bool")
    vals = ma.array(vals, mask=~valid)

    ind = kwargs.pop('ind', None)

    if ind is not None:
        vals = vals[ind]
        mid, sep = midconfERT(data, ind)
    else:
        if data.allNonZero('mid') and data.allNonZero('sep'):
            print('taking existing mid/sep')
            mid, sep = data('mid'), data('sep')
        else:
            mid, sep = midconfERT(data, circular=kwargs.get('circular', False))

    var = kwargs.pop('var', 0)
    ax = None
    cbar = None

    if var == 0:  # default style
        dx = kwargs.pop('dx', np.median(np.diff(np.unique(mid))))*2
        ax, cbar, ymap = patchValMap(vals, mid, sep, dx=dx, **kwargs)
    else:  # only here for special cases
        A, xmap, ymap = generateMatrix(mid, sep, vals, **kwargs)
        if var == 1:
            la = A.shape[1]
            B = np.zeros((A.shape[0], la*2+2))
#            for i in range(2):
#                B[:, i:la*2+i-1:2] = A
            for i in range(4):
                B[:, i:la*2+i-1:2] += A

            xmap2 = {}
            for k in xmap:
                xmap2[k * 2] = xmap[k]
                xmap2[k*2 + 1] = xmap[k]

            kwargs.setdefault('aspect', 2)
            ax, cbar = plotMatrix(B, xmap2, ymap, showally=False, **kwargs)
        elif var == 2:
            ax, cbar = plotMatrix(A, xmap, ymap, showally=False, **kwargs)
        else:
            ax, cbar = patchMatrix(A, xmap, ymap, **kwargs)

    if kwargs.get('circular', False):
        a = np.array([np.arctan2(x[1], x[0]) for x in data.sensors()])
        p = list(range(len(a)))
        # p.append(0)
        ax.plot(np.cos(a)[p], np.sin(a)[p], 'o', color='black')

        for i in range(len(a)):
            ax.text(1.15 * np.cos(a[i]),
                    1.15 * np.sin(a[i]), str(i+1),
                    horizontalalignment='center',
                    verticalalignment='center')
        ax.set_axis_off()
        ax.set_aspect(1)

    else:
        ytl = generateConfStr(np.sort([int(k) for k in ymap]))
        if 'DD1' in ytl and 'WB2' in ytl and 'DD2'not in ytl:
            ytl[ytl.index('DD1')] = 'WB1'
        if 'WA1' in ytl and 'SL2' in ytl and 'WA2'not in ytl:
            ytl[ytl.index('WA1')] = 'SL1'

        yt = ax.get_yticks()
        yt = np.unique(yt.clip(0, len(ytl)-1))
#        if yt[0] == yt[1]:
#            yt = yt[1:]
        dyt = np.diff(yt)
        if len(dyt) > 2:
            if dyt[-1] < dyt[-2]:
                yt = yt[:-1]

        ax.set_yticks(yt)
        ax.set_yticklabels([ytl[int(yti)] for yti in yt])
    return ax, cbar


def drawData(axes, data, vals, schemeName='A_M',
             patchView=False,
             colorBar=True,
             cMin=None, cMax=None, linear=False, label="",
             **kwargs):
    """TODO DOCUMENTME."""
    scheme = None
    dm = DataSchemeManager()

    scheme = dm.scheme(schemeName)

    if scheme is None:
        scheme = dm.scheme('unknown')

    gci = None

    if np.min(vals) == np.max(vals):
        drawDataAsMarker(axes, data, scheme=scheme)

    elif patchView:
        gci = pb.data.drawDataAsPatches(axes, data, vals,
                                        scheme=scheme,
                                        **kwargs)
    else:
        gci = drawDataAsMatrix(axes, data, vals, scheme=scheme,
                               logScale=not linear,
                               **kwargs)

        if colorBar:
            pg.viewer.mpl.createColorbar(gci,
                                        cMin=cMin, cMax=cMax,
                                        nLevs=5, label=label,
                                        **kwargs)

    if gci is not None:
        if cMin != cMax:
            gci.set_clim((cMin, cMax))

        cmap = kwargs.pop('cMap', None)
        if cmap is not None:
            if type(cmap) is str:
                gci.set_cmap(pg.viewer.mpl.cmapFromName(cmap))
            else:
                gci.set_cmap(cmap)

    return gci


def midconfERT(data, ind=None, rnum=1, circular=False, **kwargs):
    """Return the midpoint and configuration key for ERT data.

    Return the midpoint and configuration key for ERT data.

    Parameters
    ----------
    data : pybert.DataContainerERT
        data container with sensorPositions and a/b/m/n fields

    ind : []
        Documentme

    rnum : []
        Documentme

    circular : bool
        Return midpoint in degree (rad) instead if meter.

    Returns
    -------
    mid : np.array of float
        representative midpoint (middle of MN, AM depending on array)
    conf : np.array of float
        configuration/array key consisting of
        1) array type (Wenner-alpha/beta, Schlumberger, PP, PD, DD, MG)
            00000: pole-pole
            10000: pole-dipole or dipole-pole
            30000: Wenner-alpha
            40000: Schlumberger or Gradient
            50000: dipole-dipole or Wenner-beta
        2) potential dipole length (in electrode spacings)
        3) separation factor (current dipole length or (di)pole separation)
    """
#    xe = np.hstack((pg.x(data.sensorPositions()), np.nan))  # not used anymore
    x0 = data.sensorPosition(0).x()
    xe = pg.x(data.sensorPositions()) - x0
    ux = pg.unique(xe)

     # and not(kwargs.pop('flat', True)):
    if len(ux) * 2 > data.sensorCount():  # check for 2D with topography case
        dx = np.array(pg.utils.diff(pg.utils.cumDist(data.sensorPositions())))
        dxM = pg.mean(dx)
        if min(pg.y(data)) != max(pg.y(data)) or \
           min(pg.z(data)) != max(pg.z(data)):
            # Topography case
            if (pg.max(abs(dx-dxM)) < dxM*0.9):
                # if the maximum spacing < meanSpacing/2 we assume equidistant
                # spacing and no missing electrodes
                dx = np.ones(len(dx)) * dxM
            else:
                # topography with probably missing electrodes
                dx = np.floor(dx/np.round(dxM))*dxM
                pass
        if max(dx) < 0.5:
            print("Detecting small distances, using mm accuracy")
            rnum = 3
        xe = np.hstack((0., np.cumsum(np.round(dx, rnum)), np.nan))

        de = np.median(np.diff(xe[:-1])).round(rnum)
        ne = np.round(xe/de)
    else:  # 3D (without topo) case => take positions directly
        de = np.median(np.diff(ux)).round(1)
        ne = np.array(xe/de, dtype=int)

    # a, b, m, n = data('a'), data('b'), data('m'), data('n')
    # check if xe[a]/a is better suited (has similar size)
    if circular:
        # for circle geometry
        center = np.mean(data.sensorPositions(), axis=0)

        x = pg.x(data)-center[0]
        y = pg.y(data)-center[1]

        a = np.array([np.arctan2(y[i], x[i]) for i in data('a')])
        b = np.array([np.arctan2(y[i], x[i]) for i in data('b')])
        m = np.array([np.arctan2(y[i], x[i]) for i in data('m')])
        n = np.array([np.arctan2(y[i], x[i]) for i in data('n')])

        a = np.unwrap(a) % (np.pi*2)
        b = np.unwrap(b) % (np.pi*2)
        m = np.unwrap(m) % (np.pi*2)
        n = np.unwrap(n) % (np.pi*2)

    else:
        a = np.array([ne[int(i)] for i in data('a')])
        b = np.array([ne[int(i)] for i in data('b')])
        m = np.array([ne[int(i)] for i in data('m')])
        n = np.array([ne[int(i)] for i in data('n')])

    if ind is not None:
        a = a[ind]
        b = b[ind]
        m = m[ind]
        n = n[ind]

    anan = np.isnan(a)
    if np.any(anan):
        a[anan] = b[anan]
        b[anan] = np.nan

    ab, am, an = np.abs(a-b), np.abs(a-m), np.abs(a-n)
    bm, bn, mn = np.abs(b-m), np.abs(b-n), np.abs(m-n)

    if circular:
        for v in [ab, mn, bm, an]:
            v[v > np.pi] = 2*np.pi - v[v > np.pi]

    # 2-point (default) 00000
    sep = np.abs(a-m)
    mid = (a+m) / 2

    # 3-point (PD, DP) (now only b==-1 or n==-<1, check also for a and m)
    imn = np.isfinite(n)*np.isnan(b)
    mid[imn] = (m[imn]+n[imn]) / 2
    sep[imn] = np.minimum(am[imn], an[imn]) + 10000 + 100 * (mn[imn]-1) + \
        (np.sign(a[imn]-m[imn])/2+0.5) * 10000
    iab = np.isfinite(b)*np.isnan(n)
    mid[iab] = (a[iab]+b[iab]) / 2  # better 20000 or -10000?
    sep[iab] = np.minimum(am[iab], bm[iab]) + 10000 + 100 * (ab[iab]-1) + \
        (np.sign(a[iab]-n[iab])/2+0.5) * 10000
    #  + 10000*(a-m)

    # 4-point alpha: 30000 (WE) or 4000 (SL)
    iabmn = np.isfinite(a) & np.isfinite(b) & np.isfinite(m) & np.isfinite(n)
    ialfa = np.copy(iabmn)
    ialfa[iabmn] = (ab[iabmn] >= mn[iabmn]+2)  # old
    mnmid = (m[iabmn] + n[iabmn]) / 2
    ialfa[iabmn] = np.sign((a[iabmn]-mnmid)*(b[iabmn]-mnmid)) < 0

    mid[ialfa] = (m[ialfa] + n[ialfa]) / 2
    spac = np.minimum(bn[ialfa], bm[ialfa])
    abmn3 = np.round((3*mn[ialfa]-ab[ialfa])*10000)/10000
    sep[ialfa] = spac + (mn[ialfa]-1)*100*(abmn3 != 0) + \
        30000 + (abmn3 < 0)*10000
    # gradient

    # 4-point beta
    ibeta = np.copy(iabmn)
    ibeta[iabmn] = (bm[iabmn] >= 0.5*mn[iabmn]) & (~ialfa[iabmn])

    if circular:
        # print(ab[ibeta])
        ibeta = np.copy(iabmn)

        def averageAngle(vs):
            sumsin = 0
            sumcos = 0

            for v in vs:
                sumsin += np.sin(v)
                sumcos += np.cos(v)

            return np.arctan2(sumsin, sumcos)

        abC = averageAngle([a[ibeta], b[ibeta]])
        mnC = averageAngle([m[ibeta], n[ibeta]])

        mid[ibeta] = averageAngle([abC, mnC])

        # speccial case when dipoles  are completly opposite
        iOpp = abs(abs((mnC - abC)) - np.pi) < 1e-3
        mid[iOpp] = averageAngle([b[iOpp], m[iOpp]])

        minAb = min(ab[ibeta])
        sep[ibeta] = 50000 + (np.round(ab[ibeta]/minAb)) * 100 + \
            np.round(np.minimum(np.minimum(am[ibeta], an[ibeta]),
                                np.minimum(bm[ibeta], bn[ibeta])) / minAb)
    else:
        mid[ibeta] = (a[ibeta] + b[ibeta] + m[ibeta] + n[ibeta]) / 4

        sep[ibeta] = 50000 + (ab[ibeta]-1) * 100 + np.minimum(
            np.minimum(am[ibeta], an[ibeta]), np.minimum(bm[ibeta], bn[ibeta]))

    # 4-point gamma
    # multiply with electrode distance and add first position
    if not circular:
        mid *= de
        mid += x0
    return mid, sep


def generateConfStr(yy):
    """Generate configuration string to characterize array."""
    types = ['PP', 'PD', 'DP', 'WA', 'SL', 'DD']  # base types
    spac = yy % 100  # source-receiver distance
    dip = np.round(yy//100) % 100  # MN dipole length
    typ = np.round(yy//10000)
    # check if SL is actually GR (multi-gradient)

    # check if DD-n-n should be renamed
    rendd = (np.mean(spac / (dip+1)) < 2.1)
    keys = []
    for s, d, t in zip(spac, dip, typ):
        key = types[t]
        if d > 0:
            if rendd and d+1 == s and t == 5:
                key = 'WB'
            else:
                key = key + str(d+1) + '-'
        key = key + "{:2d}".format(s)  # str(s)
        keys.append(key)

    return keys


def createPseudoPosition(data, scheme, scaleX=False):
    """Create pseudo x position and separation for the dataset.

    ScaleX: scales the x positions regarding the real electrode positions.
    """
    nElecs = data.sensorCount()

    if scheme.typ == Pseudotype.DipoleDipole:
        x = (data('a') + data('b') + data('m') + data('n')) / 4.0
        sep = pg.abs((data('n') + data('m'))/2. - (data('b') + data('a'))/2.)

    elif scheme.typ == Pseudotype.WennerBeta:
        x = (data('a') + data('b') + data('m') + data('n')) / 4.0
        sep = pg.abs((data('n')+data('m'))/2. - (data('b')+data('a'))/2.)/2.
        sep = sep + 1.

    elif scheme.typ == Pseudotype.WennerAlpha:
        x = (data('a') + data('b') + data('m') + data('n')) / 4.0
        sep = pg.abs((data('b')-data('a'))/2 + (data('n')-data('m'))/2.)/2.0
        sep = sep + 1.

    elif scheme.typ == Pseudotype.Schlumberger:
        x = (data('a') + data('b') + data('m') + data('n')) / 4.0
        sep = pg.abs((data('b')-data('a')) / 2.0 + (data('n')-data('m')) / 2.)

    elif scheme.typ == Pseudotype.PoleDipole:
        x = data('m')
        sep = pg.abs(data('a') - data('m'))
        sep = sep + 1.

    elif scheme.typ == Pseudotype.HalfWenner:
        x = data('m')
        sep = data('a') - data('m')
        sep = sep + 1.

    elif scheme.typ == Pseudotype.Gradient:
        x = (data('m') + data('n')) / 2.0

        def psmin(vec1, vec2):
            ret = pg.Vector(vec1.size(), 0.0)
            for i in range(len(vec1)):
                ret[i] = min(vec1[i], vec2[i])
            return ret
        sep = psmin((x - data('a')), (data('b') - x)) / 3.0

    elif scheme.typ == Pseudotype.AB_MN:
        x = data('a') * float(nElecs) + data('b')
        sep = data('m') * float(nElecs) + data('n')
    elif scheme.typ == Pseudotype.AB_M:
        x = data('a') * float(nElecs) + data('b')
        sep = data('m')
    elif scheme.typ == Pseudotype.AB_N:
        x = data('a') * float(nElecs) + data('b')
        sep = data('n')
    elif scheme.typ == Pseudotype.Test:
        x = data('a') * float(nElecs) + data('b')
        sep = pg.abs(data('m') - data('n')) * float(nElecs) * float(nElecs) + \
            data('m') * float(nElecs) + data('n')
    else:
        x = pg.Vector(data('a'))
        sep = pg.Vector(data('m'))

    # need a copy here , so we do not change the original data !!!
    x = pg.Vector(x)  # I do not get this point!
    sep = pg.Vector(sep)

    if scaleX:
        x += data.sensorPositions()[0][0]
        x *= data.sensorPositions()[0].distance(data.sensorPositions()[1])
        sep -= 1.
        sep *= -1.

    return x, sep


def createDataMatrix(data, vals, scheme):
    """Create a matrix that represents the ERT data."""
    nElecs = data.sensorCount()
    nData = data.size()

    # create horizontal (separation) and vertical (x)'pseudopositions'
    x, sep = createPseudoPosition(data, scheme)

    # unique separations
    Sidx = pg.unique(pg.sort(sep))

    # unique x-position
    Xidx = pg.unique(pg.sort(x))

    # scale parameter
    dataWidthInMatrix = 1
    xOffset = 0
    xLength = len(Xidx)

#    print(min(pygimli.utils.diff(Xidx)))
    if scheme.typ > 2 and len(Xidx) > 1:

        if pg.min(pygimli.utils.diff(Xidx)) < 1.0:

            dataWidthInMatrix = int(1.0 / pg.min(pygimli.utils.diff(Xidx)))
            if dataWidthInMatrix > 1:
                xOffset = int(Xidx[0] * dataWidthInMatrix) - 1
                xLength = (nElecs - 1) * dataWidthInMatrix

#    print("xLength: ", xLength)

    mat = np.ndarray(shape=(len(Sidx), xLength,), dtype=float, order='F')

#    mat = arange(0.0, len(Sidx) * xLength)
    mat[:] = 0.0
    mat = ma.masked_where(mat == 0.0, mat)
#    mat = mat.reshape(len(Sidx), xLength)

    xMin = 1e99
    xMax = -1e99
    for i in range(0, nData):
        if data.get('valid')[i]:
            xPos = pg.find(Xidx == x[i])[0]

            mat[pg.find(Sidx == sep[i]), xPos + xOffset] = vals[i]

            for j in range(1, dataWidthInMatrix):
                mat[pg.find(Sidx == sep[i]), xPos + xOffset + j] = vals[i]

            xMin = min(xMin, xPos)
            xMax = max(xMax, xPos + xOffset + dataWidthInMatrix)

#    print("datasize:", data.size(), "shown: ",
#          len(mat[~mat.mask]) / dataWidthInMatrix)
    notShown = (data.size() - len(mat[~mat.mask]) / dataWidthInMatrix)

    if notShown > 0:
        print("data not shown: ", notShown)
    return mat, Xidx, Sidx, dataWidthInMatrix, xMin, xMax


def drawDataAsMatrix(ax, data, vals, scheme,
                     mat=None, logScale=True, **kwargs):
    """Draw data as matrix image in axes ax."""
    norm = None

    if vals is None:
        if mat is not None:
            if isinstance(mat, pg.Matrix):
                # t = []
                # for i in mat: t.append(pg.RVectorToList(i))
                # m = array(t)
                # m.reshape(len(mat), len(mat[0]))
                t = np.zeros((len(mat), len(mat[0])))
                for i, row in enumerate(mat):
                    t[i] = row
                mat = t
            elif isinstance(mat, list):
                t = []
                for i in mat:
                    t.append(i)
                m = np.array(t)
                m.reshape(len(mat), len(mat[0]))
                mat = m

        else:
            raise BaseException('drawDataAsMatrix(...) No vals/matrix given.')
    else:
        cmin = np.min(vals)
        cmax = np.max(vals)

        if cmin <= 0:
            logScale = False

        if logScale:
            vals, cmin, cmax = pg.viewer.mpl.colorbar.findAndMaskBestClim(
                vals, cMin=None, cMax=None, dropColLimitsPerc=5)
            norm = mpl.colors.LogNorm()

    matSpacing = None

    if mat is None:
        if data:
            mat, matXidx, matSidx, matSpacing, xMin, xMax = createDataMatrix(
                data, vals, scheme)
        else:
            raise Exception(('no data or matrix given'))

    mat = ma.masked_where(mat == 0.0, mat)

    if min(mat.flat) < 0:
        norm = mpl.colors.Normalize()
    else:
        norm = mpl.colors.LogNorm()

    image = ax.imshow(mat, interpolation='nearest', norm=norm)

    image.get_cmap().set_bad([1.0, 1.0, 1.0, 0.0])

#    print(mat.shape)
#    print(matXidx)
#    print(matSidx)
#    print(min(matXidx), max(matXidx), matSpacing)
#    print(min(matSidx), max(matSidx), matSpacing)

    ax.set_xlim(xMin - matSpacing,
                xMax + matSpacing)
#    ax.set_xlim(data.sensorPositions()[0][0]*matSpacing - matSpacing,
#                data.sensorPositions()[-1][0]*matSpacing + matSpacing)

    annotateSeparationAxis(ax, scheme, grid=True)

    return image


def annotateSeparationAxis(ax, scheme, grid=False):
    """Draw y-axes tick labels corresponding to the separation."""
    prefix = scheme.prefix

    def sepName(sep):
        suffix = ""

        if sep == 0:
            return ''
        elif sep > 0:
            suffix = "'"

        if grid:
            ax.plot(ax.get_xlim(), [sep, sep], color='black', linewidth=1,
                    linestyle='dotted')

        return prefix + ' $' + str(abs(int(sep))) + suffix + '$'

    ax.yaxis.set_ticklabels([sepName(l) for l in ax.yaxis.get_ticklocs()])


def drawElectrodesAsMarker(ax, data):
    """Draw electrode marker, these marker are pickable."""
    elecsX = []
    elecsY = []

    for i in range(len(data.sensorPositions())):
        elecsX.append(data.sensorPositions()[i][0])
        elecsY.append(data.sensorPositions()[i][1])

    electrodeMarker, =  ax.plot(elecsX, elecsY, 'x', color='black', picker=5.)

    ax.set_xlim([data.sensorPositions()[0][0]-1.,
                 data.sensorPositions()[data.sensorCount() - 1][0] + 1.])
    return electrodeMarker


def drawDataAsMarker(ax, data, scheme, **kwargs):
    """Draw pseudosection scheme for the data using marker only."""
    # first draw the electrodes
    electrodeMarker = drawElectrodesAsMarker(ax, data)

    # now draw the data Marker
    x, sep = createPseudoPosition(data, scheme, scaleX=True)

#    print((max(sep)))
#    print(sep)

    maxSepView = max(sep) + 2

    if max(sep) > 0:
        maxSepView = maxSepView - 1
    ax.set_ylim([min(sep) - 1, maxSepView])

    dataMarker, = ax.plot(x, sep, '.', color='black', picker=5., **kwargs)

    annotateSeparationAxis(ax, scheme, grid=True)

    return electrodeMarker, dataMarker


def createDataPatches(ax, data, scheme, **kwargs):
    """Create patches for a pseudosection."""
#    swatch = pg.Stopwatch(True)
    x, sep = createPseudoPosition(data, scheme, scaleX=True)

    ax.set_ylim([min(sep)-1, max(sep)+1])

    # dx2 = (data.sensorPositions()[1][0] - data.sensorPositions()[0][0])/4.
    dx2 = (x[1]-x[0])/2.
    dSep2 = 0.5

    polys = []
    for i, xv in enumerate(x):
        s = sep[i]
        polys.append(list(zip([xv-dx2, xv+dx2, xv+dx2, xv-dx2],
                              [s - dSep2, s - dSep2, s + dSep2, s + dSep2])))

    patches = mpl.collections.PolyCollection(polys, antialiaseds=False,
                                             lod=True, **kwargs)
    patches.set_edgecolor('face')
    # patches.set_linewidth(0.001)
    ax.add_collection(patches)

#    print("Create data patches takes t = ", swatch.duration(True))
    return patches


def drawDataAsPatches(ax, data, vals, scheme,
                      writeValues=False, logScale=True, **kwargs):
    """Draw pseudosection as patch graphic."""
    # first draw the electrodes
#    electrodeMarker = drawElectrodesAsMarker(ax, data)  # never used!

    # now draw the data Marker
    gci = createDataPatches(ax, data, scheme, **kwargs)

    vals = ma.masked_where(vals == 0, vals * data.get('valid'))

    pg.viewer.mpl.setMappableData(gci, vals, logScale=logScale)

#    if min(vals) < 0 :
#        writeValues = True

    if writeValues:
        x, sep = createPseudoPosition(data, scheme, scaleX=True)

        for i, xv in enumerate(x):
            ax.text(xv, sep[i], str(round(vals[i], 2)), fontsize='8',
                    horizontalalignment='center', verticalalignment='center')

    annotateSeparationAxis(ax, scheme, grid=True)
    return gci
