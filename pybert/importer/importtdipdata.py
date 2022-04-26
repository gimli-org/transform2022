import numpy as np
import pygimli as pg
from .importData import (importAsciiColumns, importRes2dInv, importDIP,
                         importAres2, importABEM, importData)


def importTDIPdata(filename, verbose=False):
    """Read in TDIP data.

    Supported formats
    -----------------
    TXT - ABEM LS or Syscal Pro Ascii output
    BIN - Syscal binary format
    GDD - GDD format
    2DM - Ares II Ascii format
    AMP - ABEM SAS4000 format
    TX2 - Aarhus Workbench data
    DIP - AarhusInv (processed) data
    DAT - Res2dInv format

    Returns
    -------
    data : DataContainerERT
        data container
    MA : numpy.array (ngates, ndata)
        spectral chargeability matrix
    t : iterable
        time vector
    header : dict
        dictionary with supporting information
    """
    header = {}
    if isinstance(filename, str):
        ext = filename[filename.rfind('.')+1:]
        if ext.lower() in ['txt', 'tx2', 'gdd']:
            data, header = importAsciiColumns(filename, verbose=verbose,
                                              return_header=True)
        elif ext.lower() in ['dat', 'res2dinv']:
            data, header = importRes2dInv(filename, verbose=verbose,
                                          return_header=True)
        elif ext.lower() == 'amp':
            return importABEM(filename, return_all=True)
        elif ext.lower() == 'dip':
            return importDIP(filename, return_all=True)
        elif ext.lower() == '2dm':
            return importAres2(filename, return_all=True)
        else:
            data = importData(filename)
    elif isinstance(filename, pg.DataContainer):
        data = filename
    else:
        raise TypeError("Cannot use this type:"+type(filename))

    ipkey = ''
    testkeys = ['IP_#{}(mV/V)', 'M{}', 'ip{}']
    for key in testkeys:
        if data.exists(key.format(1)):
            ipkey = key
    MA = []
    i = 1
    while data.exists(ipkey.format(i)):
        ma = data(ipkey.format(i))
        if max(ma) <= 0:
            break

        MA.append(ma)
        i += 1

    MA = np.array(MA)
    t = np.arange(MA.shape[0]) + 1  # default: t is gate number
    if 'ipGateT' in header:  # Syscal
        t = header['ipGateT'][:-1] + np.diff(header['ipGateT'])/2
    elif data.exists('Gate1') and data.exists('Ngates'):
        ngates = int(data('Ngates')[0])
        dt = np.array([data('Gate'+str(i+1))[0] for i in range(ngates)]) * 1e-3
        delay = 0.001  # hard-coded as lost in import
        t = np.cumsum(dt) - dt/2 + delay
        header['ipGateT'] = np.cumsum(np.hstack((0, dt))) + delay

    testkeys = ['TM{}']
    for key in testkeys:
        if data.exists(key.format(1)):
            tt = [data(key.format(i+1))[0] for i in range(
                MA.shape[0])]
            dt = np.array(tt)
            if sum(dt) > 100:
                dt *= 0.001

            delay = 0.0
            if data.exists(key.format(0)):
                delay = data(key.format(0))[0]
                if delay > 1:
                    delay *= 0.001

            t = np.cumsum(dt) - dt/2 + delay
            header['ipGateT'] = np.cumsum(np.hstack((delay, dt)))

    return data, MA, t, header
