# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import re

import codecs

import numpy as np
from datetime import datetime

import pygimli as pg
from pygimli.utils.utils import uniqueRows

try:
    import pybert as pb
except ImportError:
    sys.stderr.write('''ERROR: cannot import the library 'pybert'. ''' +
                     '''Ensure that pybert is in your PYTHONPATH ''')
    sys.exit(1)

# self.registerOpenFileSuffix(suffix, plugin.MainOpenWildcard[i],
#     plugin.PluginApplication, plugin.MainOpenFileSlot)
# self.fileSuffixes[suffix] = [wildcard, cls, callback]
# MainOpenFileSuffix = ['.dat', '.ohm', '.shm', '.edt', '.data']
# MainOpenFileSlot = BertApp.openDataFile
# MainOpenWildcard = ["BERT unified data file (*.dat)"

bertImportDataFileSuffixesDict = dict()
bertImportDataFileSuffixesDict['.dat'] = ["Unified data file (*.dat)", 'Gimli']
bertImportDataFileSuffixesDict['.ohm'] = ["Unified data file (*.ohm)", 'Gimli']
bertImportDataFileSuffixesDict['.shm'] = ["Unified data file (*.shm)", 'Gimli']
bertImportDataFileSuffixesDict['.udf'] = ["Unified data file (*.udf)", 'Gimli']
bertImportDataFileSuffixesDict['.bin'] = ["Syscal Pro (*.bin)", 'SyscalPro']
bertImportDataFileSuffixesDict['.tx0'] = ["LGM 4-point light (*.tx0)",
                                          '4PointLight']
bertImportDataFileSuffixesDict['.txt'] = ["ASCII (*.txt)", 'AsciiColumns']
bertImportDataFileSuffixesDict['.txt'] += ["LGM 4-point light (*.txt)",
                                           '4PointLight']
bertImportDataFileSuffixesDict['.txt'] += ["GeoSys (*.txt)",
                                           'GeoSys']
bertImportDataFileSuffixesDict['.txt'] += ["ASCII (*.txt)",
                                           'ABEMAscii']
bertImportDataFileSuffixesDict['.txt'] += ["Resecs-ASCII (*.txt)",
                                           'ResecsAscii']
bertImportDataFileSuffixesDict['.tx2'] = ["WB ASCII (*.tx2)",
                                          'AsciiColumns']
bertImportDataFileSuffixesDict['.abem'] = ["ABEM-ASCII (*.txt)",
                                           'ABEMAscii']
bertImportDataFileSuffixesDict['.resecs'] = ["Resecs-ASCII (*.txt)",
                                             'ResecsAscii']
bertImportDataFileSuffixesDict['.flw'] = ["Geotom FLW (*.flw)", 'FLW']
bertImportDataFileSuffixesDict['.slm'] = ["Geotom Schlumberger (*.slm)",
                                          'Geotom']
bertImportDataFileSuffixesDict['.wen'] = ["Geotom Wenner (*.wen)", 'Geotom']
bertImportDataFileSuffixesDict['.dd'] = ["Geotom dipole dipole (*.dd)", 'Geotom']
bertImportDataFileSuffixesDict['.pd'] = ["Geotom pole dipole (*.pd)", 'Geotom']
bertImportDataFileSuffixesDict['.pp'] = ["Geotom pole pole (*.pp)", 'Geotom']
bertImportDataFileSuffixesDict['.stg'] = ["SuperSting (*.stg)", 'SuperSting']
bertImportDataFileSuffixesDict['.2dm'] = ["Ares II (*.2dm)", 'Ares2']
bertImportDataFileSuffixesDict['.dip'] = ["Aarhus (*.dip)", 'DIP']
bertImportDataFileSuffixesDict['.amp'] = ["ABEM AMP (*.AMP)", 'ABEM']
bertImportDataFileSuffixesDict['.res2dinv'] = ["Res2dInv (*.res2dinv)",
                                               'Res2dInv']


def importData(filename, format='auto', verbose=False, debug=False):
    """
    Import datafile into GIMLi unified data format

    Parameters
    ----------
    format : string, optional [auto]
        *Gimli - gimli-unified data format
        *Res2dInv - res2dinv (partly)
        *Geotom - Geotom wen/slm/dd/pd/pp files
        *SyscalPro - IRIS SYSCAL pro
        *SuperSting - AGI SuperSting
        *Lippmann - Lippmann 4-point light
        *Iris - IRIS test unknown
        *ABEM - ABEM test unknown
        *ABEMTerrameterSAS - ABEM test unknown
        *GeoSys - GeoSYS test unknown

    verbose : boolean, optional [False]
        Be verbose during import.

    debug : boolean, optional [False]
        DEPRECATED
        Gives some more debug information.
    """

    if debug:
        pg.warn("debug kwarg is deprecated. Use -d or setDebug(True)")

    def tryImport(filename, funct):
        pg.debug("Try to import ", filename, " by ", funct)
        d = None
        try:
            d = funct(filename, verbose)
        except Exception as e:
            # traceback.print_exc()
            pg.debug(e)  # , withTrace=True)  # TODO!!!
        return d
    # def tryImport(...)

    d = None
    if not os.path.exists(filename):
        raise Exception('File does not exist.: ' + filename)

    if format.lower() == 'auto':
        # first try the one associated with the file extension
        ext = filename[filename.rfind('.'):].lower()

        if ext in bertImportDataFileSuffixesDict:
            for fun in bertImportDataFileSuffixesDict[ext][1::2]:
                importFunction = 'import' + fun
                if importFunction in dir(sys.modules[__name__]):
                    d = tryImport(filename, eval(importFunction))

            if importFunction in dir(sys.modules[__name__]):
                d = tryImport(filename, eval(importFunction))
            else:
                pg.debug("Error: Import function does not exist: " +
                         importFunction)

        if d is None:
            d = tryImport(filename, importGimli)
        if d is None:
            d = tryImport(filename, importRes2dInv)
        if d is None:
            d = tryImport(filename, import4PointLight)
        if d is None:
            d = tryImport(filename, importPolePoleGeotom)
        if d is None:
            d = tryImport(filename, importSyscalPro)
        if d is None:
            d = tryImport(filename, importIris)
        if d is None:
            d = tryImport(filename, importABEMAscii)
        if d is None:
            d = tryImport(filename, importABEM)
        if d is None:
            d = tryImport(filename, importABEMTerrameterSAS)
        if d is None:
            d = tryImport(filename, importSuperSting)
        if d is None:
            d = tryImport(filename, importResecsAscii)
        if d is None:
            d = tryImport(filename, importGeoSys)
        if d is None:
            d = tryImport(filename, importFLW)

    else:
        if 'import' + format in dir(sys.modules[__name__]):
            d = tryImport(filename,
                          getattr(sys.modules[__name__], 'import' + format))
        else:
            raise Exception('There is no import filter for format:', format)

    pg.info("imported: ",  d)
    return d


def importGimli(filename, verbose=False):
    data = pg.DataContainerERT(filename)
    return data
# def importGimli(...)


def importPolePoleGeotom(filename, verbose=False):
    '''
        Pole-Pole-from geotom?? EA- LOS_A1.dat
        "B_X"   "B_Y"   "N_X"   "N_Y"   "Rhos"
        13  0   13  0.5 110.17
        ....
        13  0   13  1   90.71
    '''
    data = pg.DataContainerERT()
    with open(filename, 'r') as fi:
        content = fi.readlines()

        header = content[0].split('\r\n')[0].split()
        if len(header) == 5:
            if header[0] == '"B_X"' and header[1] == '"B_Y"' and \
               header[2] == '"N_X"' and header[3] == '"N_Y"' and \
               header[4] == '"Rhos"':

                data.resize(len(content) - 1)

                for i, row in enumerate(content[1:]):
                    vals = row.split()
                    eaID = -1
                    ebID = data.createSensor(pg.Pos(float(vals[0]),
                                                    float(vals[1]),
                                                    0.0)).id()
                    emID = -1
                    enID = data.createSensor(pg.Pos(float(vals[2]),
                                                    float(vals[3]),
                                                    0.0)).id()

                    data.createFourPointData(i, eaID, ebID, emID, enID)
                    data('rhoa')[i] = float(vals[4])

                data.sortSensorsX()
            else:
                raise Exception("Probably no Geotom-Pole-Pole file: " +
                                filename + " header: " + str(header))
        else:
            raise Exception("Probably no Geotom-Pole-Pole file: " + filename +
                            " 5 != " + str("len(header)"))

    return data
# def importPolePoleGeotom(...)


def importRes2dInv(filename, verbose=False, return_header=False):
    """Read res2dinv format

    Parameters
    ----------
    filename : str
    verbose : bool [False]
    return_header : bool [False]

    Returns
    -------
    pg.DataContainerERT and (in case of return_header=True)
    header dictionary

    Format
    ------
        str - title
        float - unit spacing [m]
        int - Array Number (1-Wenner, 3-Dipole-dipole atm only)
        int - Number of Datapoints
        float - x-location given in terms of first electrode
                use 1 if mid-point location is given
        int - 0 for no IP, use 1 if IP present
        str - Phase Angle  if IP present
        str - mrad if IP present
        0,90.0 - if IP present
        dataBody
    """

    def getNonEmptyRow(i, comment='#'):
        s = next(i)
        while s[0] is comment:
            s = next(i)
        return s.split('\r\n')[0]
    # def getNonEmptyRow(...)

    with open(filename, 'r') as fi:
        content = fi.readlines()

    it = iter(content)
    header = {}
    header['name'] = getNonEmptyRow(it, comment=';')
    header['spacing'] = float(getNonEmptyRow(it, comment=';'))
    typrow = getNonEmptyRow(it, comment=';')
    typ = int(typrow.rstrip('\n').rstrip('R').rstrip('L'))

    if typ == 11:
        # independent electrode positions
        header['subtype'] = int(getNonEmptyRow(it, comment=';'))
        header['dummy'] = getNonEmptyRow(it, comment=';')
        isR = int(getNonEmptyRow(it, comment=';'))

    nData = int(getNonEmptyRow(it, comment=';'))
    xLoc = float(getNonEmptyRow(it, comment=';'))
    hasIP = int(getNonEmptyRow(it, comment=';'))

    if hasIP:
        header['ipQuantity'] = getNonEmptyRow(it, comment=';')
        header['ipUnit'] = getNonEmptyRow(it, comment=';')
        header['ipData'] = getNonEmptyRow(it, comment=';')
        ipline = header['ipData'].rstrip('\n').rstrip('\r').split(' ')
        if len(ipline) > 2:  # obviously spectral data?
            header['ipNumGates'] = int(ipline[0])
            header['ipDelay'] = float(ipline[1])
            header['onTime'] = float(ipline[-2])
            header['offTime'] = float(ipline[-1])
            header['ipDT'] = np.array(ipline[2:-2], dtype=float)
            header['ipGateT'] = np.cumsum(np.hstack((header['ipDelay'],
                                                     header['ipDT'])))

    data = pg.DataContainerERT()
    data.resize(nData)

    if typ == 9 or typ == 10:
        raise Exception("Don't know how to read:" + str(typ))

    if typ == 11 or typ == 12 or typ == 13:  # mixed array

        res = pg.Vector(nData, 0.0)
        ip = pg.Vector(nData, 0.0)
        specIP = []

        for i in range(nData):
            vals = getNonEmptyRow(it, comment=';').replace(',', ' ').split()

            # row starts with 4
            if int(vals[0]) == 4:
                eaID = data.createSensor(pg.Pos(float(vals[1]),
                                                float(vals[2])))
                ebID = data.createSensor(pg.Pos(float(vals[3]),
                                                float(vals[4])))
                emID = data.createSensor(pg.Pos(float(vals[5]),
                                                float(vals[6])))
                enID = data.createSensor(pg.Pos(float(vals[7]),
                                                float(vals[8])))
            elif int(vals[0]) == 3:
                eaID = data.createSensor(pg.Pos(float(vals[1]),
                                                float(vals[2])))
                ebID = -1
                emID = data.createSensor(pg.Pos(float(vals[3]),
                                                float(vals[4])))
                enID = data.createSensor(pg.Pos(float(vals[5]),
                                                float(vals[6])))
            elif int(vals[0]) == 2:
                eaID = data.createSensor(pg.Pos(float(vals[1]),
                                                float(vals[2])))
                ebID = -1
                emID = data.createSensor(pg.Pos(float(vals[3]),
                                                float(vals[4])))
                enID = -1
            else:
                raise Exception('dont know how to handle row', vals[0])
            res[i] = float(vals[int(vals[0])*2+1])
            if hasIP:
                # ip[i] = float(vals[int(vals[0])*2+2])
                ipCol = int(vals[0])*2+2
                ip[i] = float(vals[ipCol])
                if 'ipNumGates' in header:
                    specIP.append(vals[ipCol:])

            data.createFourPointData(i, eaID, ebID, emID, enID)

        if isR:
            data.set('r', res)
        else:
            data.set('rhoa', res)

        if hasIP:
            data.set('ip', ip)
            if 'ipNumGates' in header:
                A = np.array(specIP, dtype=float)
                A[A > 1000] = -999
                A[A < -1000] = -999
                for i in range(header['ipNumGates']):
                    data.set('ip'+str(i+1), A[:, i])

        data.sortSensorsX()
        data.sortSensorsIndex()
        if return_header:
            return data, header
        else:
            return data

    # amount of values per collumn per typ
    nntyp = [0, 3, 3, 4, 3, 3, 4, 4, 3, 0, 0, 8, 10]

    nn = nntyp[typ] + hasIP

    # dataBody = pg.Matrix(nn, nData)
    dataBody = np.zeros((nn, nData))

    for i in range(nData):
        vals = getNonEmptyRow(it, comment=';').replace(',', ' ').split()
        dataBody[:, i] = np.array(vals, dtype=float)
#        for j in range(nn):
#            dataBody[j][i] = float(vals[j])

    XX = dataBody[0]
    EL = dataBody[1]
    SP = pg.Vector(nData, 1.0)

    if nn - hasIP == 4:
        SP = dataBody[2]

    AA = None
    BB = None
    NN = None
    MM = None

    if typ == 1:  # Wenner
        AA = XX - xLoc * EL * 1.5
        MM = AA + EL
        NN = MM + EL
        BB = NN + EL
    elif typ == 2:  # Pole-Pole
        AA = XX - xLoc * EL * 0.5
        MM = AA + EL
    elif typ == 3:  # Dipole-Dipole
        AA = XX - xLoc * EL * (SP / 2. + 1.)
        BB = AA + EL
        MM = BB + SP * EL
        NN = MM + EL
        pass
    elif typ == 3:  # Dipole-Dipole
        AA = XX - xLoc * EL * (SP / 2. + 1.)
        BB = AA + EL
        MM = BB + SP * EL
        NN = MM + EL
    elif typ == 4:  # WENNER-BETA
        AA = XX - xLoc * EL * 1.5
        BB = AA + EL
        MM = BB + EL
        NN = MM + EL
    elif typ == 5:  # WENNER-GAMMA
        AA = XX - xLoc * EL * 1.5
        MM = AA + EL
        BB = MM + EL
        NN = BB + EL
    elif typ == 6:  # POLE-DIPOLE
        AA = XX - xLoc * SP * EL - (SP - 1.) * (SP < 0.) * EL
        MM = AA + SP * EL
        NN = MM + pg.sign(SP) * EL
    elif typ == 7:  # SCHLUMBERGER
        AA = XX - xLoc * EL * (SP + 0.5)
        MM = AA + SP * EL
        NN = MM + EL
        BB = NN + SP * EL
    else:
        raise Exception('Datatype ' + str(typ) + ' not yet suppoted')

    for i in range(len(AA)):

        if AA is not None:
            eaID = data.createSensor(pg.Pos(AA[i], 0.0))
        else:
            eaID = -1

        if BB is not None:
            ebID = data.createSensor(pg.Pos(BB[i], 0.0))
        else:
            ebID = -1

        if MM is not None:
            emID = data.createSensor(pg.Pos(MM[i], 0.0))
        else:
            emID = -1

        if NN is not None:
            enID = data.createSensor(pg.Pos(NN[i], 0.0))
        else:
            enID = -1

        data.createFourPointData(i, eaID, ebID, emID, enID)

    data.set('rhoa', dataBody[nn - hasIP - 1])
    if hasIP:
        data.set('ip', dataBody[nn - 1])

    data.sortSensorsX()
    if return_header:
        return data, header
    else:
        return data
# def importRes2dInv(...)


def importGeotom(filename, verbose=False):
    """Read data from Geotom instrument data (*.flw/wen/dd etc.)

        str - title
        float - unit spacing [m]
        int - Array Number (1-Wenner, 3-Dipole-dipole atm only)
        int - Number of Datapoints
        float - x-location given in terms of first electrode
                use 1 if mid-point location is given
        int - 0 for no IP, use 1 if IP present
        str - Phase Angle  if IP present
        str - mrad if IP present
        0,90.0 - if IP present
        dataBody
    """

    def getNonEmptyRow(i, comment='#'):
        s = next(i)
        while s[0] is comment:
            s = next(i)
        return s.split('\r\n')[0]
    # def getNonEmptyRow(...)

    with open(filename, 'r') as fi:
        content = fi.readlines()

    it = iter(content)
    s = next(it)
    while s[0] == '/':
        s = next(it)
    header = {}
    header['name'] = s.split('\r\n')[0]
    header['spacing'] = float(getNonEmptyRow(it, comment=';'))
    typ = int(re.search('[0-9]+', getNonEmptyRow(it, comment=';'))[0])

    if typ == 11:
        # independent electrode positions
        header['subtype'] = int(getNonEmptyRow(it, comment=';'))
        header['dummy'] = getNonEmptyRow(it, comment=';')
        isR = int(getNonEmptyRow(it, comment=';'))

    nData = int(getNonEmptyRow(it, comment=';'))
    xLoc = float(getNonEmptyRow(it, comment=';'))
    hasIP = int(getNonEmptyRow(it, comment=';'))

    if hasIP:
        header['ipStr1'] = getNonEmptyRow(it, comment=';')
        header['ipStr2'] = getNonEmptyRow(it, comment=';')
        header['ipStr3'] = getNonEmptyRow(it, comment=';')

    data = pg.DataContainerERT()
    data.resize(nData)

    if typ == 9 or typ == 10:
        raise Exception('Cannot yet handle datatype:' + str(typ))

    if typ == 11 or typ == 12 or typ == 13:  # mixed array

        res = pg.Vector(nData, 0.0)

        for i in range(nData):
            vals = getNonEmptyRow(it, comment=';').split()

            # row starts with 4
            if int(vals[0]) == 4:
                eaID = data.createSensor(pg.Pos(float(vals[1]), 0.0))
                ebID = data.createSensor(pg.Pos(float(vals[3]), 0.0))
                emID = data.createSensor(pg.Pos(float(vals[5]), 0.0))
                enID = data.createSensor(pg.Pos(float(vals[7]), 0.0))
                res[i] = float(vals[9])
            else:
                raise Exception('dont know how to handle row', vals[0])

            data.createFourPointData(i, eaID, ebID, emID, enID)

        if isR:
            data.set('r', res)
        else:
            data.set('rhoa', res)

        data.sortSensorsX()
        data.sortSensorsIndex()

        return data

    # ammount of values per collumn per typ
    nntyp = [0, 3, 3, 4, 3, 3, 4, 4, 3, 0, 0, 8, 10]

    nn = nntyp[typ] + hasIP + 3  # current, voltage, error

    dataBody = pg.Matrix(nn, nData)

    for i in range(nData):
        vals = getNonEmptyRow(it, comment=';').replace(';', '').split()
        if vals[-1] != '!':
            for j in range(min(nn, len(vals))):
                dataBody[j, i] = float(vals[j])

    XX = dataBody[0]
    EL = dataBody[1]
    SP = pg.Vector(nData, 1.0)

#    if nn - hasIP == 4+3:
    if nntyp[typ] == 4:
        SP = dataBody[2]

    AA = None
    BB = None
    NN = None
    MM = None

    if typ == 1:  # Wenner
        AA = XX - xLoc * EL * 1.5
        MM = AA + EL
        NN = MM + EL
        BB = NN + EL
    elif typ == 2:  # Pole-Pole
        AA = XX - xLoc * EL * 0.5
        MM = AA + EL
    elif typ == 3:  # Dipole-Dipole
        AA = XX - xLoc * EL * (SP / 2. + 1.)
        BB = AA + EL
        MM = BB + SP * EL
        NN = MM + EL
        pass
    elif typ == 3:  # Dipole-Dipole
        AA = XX - xLoc * EL * (SP / 2. + 1.)
        BB = AA + EL
        MM = BB + SP * EL
        NN = MM + EL
    elif typ == 4:  # WENNER-BETA
        AA = XX - xLoc * EL * 1.5
        BB = AA + EL
        MM = BB + EL
        NN = MM + EL
    elif typ == 5:  # WENNER-GAMMA
        AA = XX - xLoc * EL * 1.5
        MM = AA + EL
        BB = MM + EL
        NN = BB + EL
    elif typ == 6:  # POLE-DIPOLE
        AA = XX - xLoc * SP * EL - (SP - 1.) * (SP < 0.) * EL
        MM = AA + SP * EL
        NN = MM + pg.sign(SP) * EL
    elif typ == 7:  # SCHLUMBERGER
        AA = XX - xLoc * EL * (SP + 0.5)
        MM = AA + SP * EL
        NN = MM + EL
        BB = NN + SP * EL
    else:
        raise Exception('Datatype ' + str(typ) + ' not yet suppoted')

    for i in range(len(AA)):

        if AA is not None:
            eaID = data.createSensor(pg.Pos(AA[i], 0.0))
        else:
            eaID = -1

        if BB is not None:
            ebID = data.createSensor(pg.Pos(BB[i], 0.0))
        else:
            ebID = -1

        if MM is not None:
            emID = data.createSensor(pg.Pos(MM[i], 0.0))
        else:
            emID = -1

        if NN is not None:
            enID = data.createSensor(pg.Pos(NN[i], 0.0))
        else:
            enID = -1

        if eaID != ebID:
            data.createFourPointData(i, eaID, ebID, emID, enID)

    data.set('rhoa', dataBody[nntyp[typ]-1])
    data.set('i', dataBody[nntyp[typ]+hasIP]*1e-3)
    data.set('u', dataBody[nntyp[typ]+hasIP+1]*1e-3)
    data.set('err', dataBody[nntyp[typ]+hasIP+2]*1e-2)
    if hasIP:
        data.set('ip', dataBody[nntyp[nn] + 1])

    data.sortSensorsX()
    return data
# def importGeotom(...)


def importSyscalPro(filename, verbose=False):  # , return_header=False):
    """READ IRIS Syscal Pro or Elrec Pro file (binary *.bin)

       Ported from matlab version (from Tobias Pfaff, Uni Heidelberg)
    """
    import struct

    data = pg.DataContainerERT()

    with open(filename, 'rb') as fi:
        readData = fi.read()

        # Filesize: 1029 (bytes header) + nBlocks * 304 (bytes per block)
        nBlocks = (len(readData) - 1029) / 304.0
        if nBlocks > 0 and (nBlocks == round(nBlocks)):
            nBlocks = round(nBlocks)
            if verbose:
                pg.debug(nBlocks, "blocks")
            headerIdentification = str(readData[0:20])
            if verbose:
                pg.debug(headerIdentification)
            if 'Pro' not in headerIdentification and False:
                raise Exception('This is probably no SYSCAL Pro data file: ' +
                                filename + " : " + headerIdentification)

            measureingTime = readData[20:40]
            if verbose:
                pg.debug(measureingTime)

            startBlock = 1029  # hex 404
        else:
            raise Exception('Size of the SYSCAL Pro data file is not valid: ' +
                            filename + " : " + str(len(readData)) + " ; " +
                            str(nBlocks))

        # the file size and header seems to be ok. start parsing.
        nBlocks = int(nBlocks)
        data.resize(nBlocks)
        # Main data
        sp = pg.Vector(nBlocks)  # self potential
        vp = pg.Vector(nBlocks)  # voltage difference
        curr = pg.Vector(nBlocks)  # injected current
        gm = pg.Vector(nBlocks)  # global chargeability
        dev = pg.Vector(nBlocks)  # std. deviation
        # Auxiliary data
        stacks = pg.Vector(nBlocks)  # number of stacks measured
        rs_check = pg.Vector(nBlocks)  # rs_check reception dipole
        vab = pg.Vector(nBlocks)  # absolute injected voltage
        bat_tx = pg.Vector(nBlocks)  # tx battery voltage
        bat_rx = pg.Vector(nBlocks)  # rx battery voltage
        temp = pg.Vector(nBlocks)  # temperature
        TM = np.zeros((nBlocks, 20))
        delay = pg.Vector(nBlocks)
        MA = np.zeros((nBlocks, 20))
        valid = pg.Vector(nBlocks)  # store visibility prefilterd in prosys

        for i in range(nBlocks):
            block = readData[startBlock:startBlock+304]
            # short(max cycles) , short(min cycles), float32(measurement time),
            # float32(delay for measurement)
            [maxCyles, minCycles, measTime, delayTime, visible] = \
                struct.unpack_from('hhffi', block, offset=0)  # 16 byte

            valid[i] = (visible == 1)
            delay[i] = delayTime

            # Read electrode positions for each data
            # (C1_x C2_x P1_x P2_x C1_y C2_y P1_y P2_y C1_z C2_z P1_z P2_z)
            ePos = struct.unpack_from('ffff ffff ffff', block, offset=16)
            eaID, ebID, emID, enID = -1, -1, -1, -1
            if (ePos[0] < 99999.99):
                eaID = data.createSensor(pg.Pos(ePos[0], ePos[4], ePos[8]))
            if (ePos[1] < 99999.99):
                ebID = data.createSensor(pg.Pos(ePos[1], ePos[5], ePos[9]))
            if (ePos[2] < 99999.99):
                emID = data.createSensor(pg.Pos(ePos[2], ePos[6], ePos[10]))
            if (ePos[3] < 99999.99):
                enID = data.createSensor(pg.Pos(ePos[3], ePos[7], ePos[11]))

            data.createFourPointData(i, eaID, ebID, emID, enID)

            # Read data float32 (sp, vp, in, rho, gm, dev)
            [sp[i], vp[i], curr[i], rhoDummy, gm[i], dev[i]] = \
                struct.unpack_from('fff fff', block, offset=64)

            # Time domain IP chargeability time window lengths
            TM[i, :] = struct.unpack_from('20f', block, offset=88)
            # Associated partial chargeabilities
            MA[i, :] = struct.unpack_from('20f', block, offset=168)
            # print(times, ma)

            # status bits (int16)
            # (80:multichannel(lower bits=channel #) 16:single)
            # number of measurements (int16) (starting with 0)
            # [status, nMeas] = struct.unpack_from('hh', block, offset = 248)

            # The name for the sequence used by Syscal-Pro
            # name = block[252 : 272]

            # Read auxiliary information
            [stacks[i], rs_check[i], vab[i], bat_tx[i], bat_rx[i], temp[i]] = \
                struct.unpack_from('ffffff', block, offset=272)

    #         date & time in some strange format
    #         gtime = (dtime(2)*(2^32/fact)+dtime(1)/fact - d0)/24/3600 -
    #             datenum(2004,0,0); % date in day-since-04 format
    #        dtime = struct.unpack_from('II', block, offset=296)
    #        print (dtime[1] * (2**32/fact) + dtime[0] / fact - d0) /24./3600.
    #        print dtime
            # 296 + 8 = 304

            startBlock += 304
        # END for each data block

        data.set('valid', valid)
        data.add('sp', sp, 'Self potential (Please check this).|V')
        data.add('u', vp * 1e-3, 'Measured voltage difference|V')
        data.add('i', curr * 1e-3, 'Injected current|A')
        data.add('gm', gm, 'Chargeability|mV/V')
        data.add('ip', gm, 'Induced polarisation|mV/V')
        data.add('err', dev, 'Standard deviation')
        data.add('stacks', stacks)
        data.add('TM0', delay)
        data.add('rs_check', rs_check)
        data.add('vab', vab, 'Injected Voltage|V')
        data.add('bat_tx', bat_tx, 'Battery Voltage|V')
        data.add('bat_rx', bat_rx, 'Battery Voltage|V')
        data.add('temp', temp, 'Temperature|C')

        for i in range(20):
            data.set('TM' + str(i+1), TM[:, i])
            data.set('M' + str(i+1), MA[:, i])

        data.sortSensorsX()

#    if return_header:
#       return data, header
#    else:
    return data
# def importSyscalPro(filename):


def importIris(filename, verbose=False):
    """Import IRIS Instruments Ascii output file

    El-array Spa.1 Spa.2 Spa.3 Spa.4 Rho  Dev.  M   Sp   Vp   In
    Schlum. VES 0.00 43.00 21.00 22.00 25.08 0.0 0.00 -56.0 17.713 1025.036
    """
    with open(filename, 'r') as fi:
        content = fi.readlines()
    fi.close()

    # inconsistent dataformat  we need to add leading dummy token
    content[0] = content[0].replace('-', ' ')

    d = readAsDictionary(content)
    if pg.debug():
        pg.debug(d.keys())

    nData = len(d['array'])

    data = pg.DataContainerERT()
    data.resize(nData)

    for i in range(nData):
        eaID = data.createSensor([d['Spa.1'][i], 0.0, 0.0])
        ebID = data.createSensor([d['Spa.2'][i], 0.0, 0.0])
        emID = data.createSensor([d['Spa.3'][i], 0.0, 0.0])
        enID = data.createSensor([d['Spa.4'][i], 0.0, 0.0])
        data.createFourPointData(i, eaID, ebID, emID, enID)

    data.set('i', np.array(d['In']) / 1000.0)
    data.set('u', np.array(d['Vp']) / 1000.0)
    data.set('sp', d['Sp'])
    data.set('err', d['Dev.'])

    data.sortSensorsX()
    return data


# def importIris(...)

def importABEMAscii(filename, verbose=False, return_header=False):
    """Import DataContainer from ABEM or Resecs Ascii (txt) export."""

    return importABEMTerrameterSAS(filename, verbose=verbose)

    with open(filename, 'r', encoding='iso-8859-15') as fid:
        lines = fid.readlines()
        header = {}
        indata = False
        nstop = 0
        for n, line in enumerate(lines):
            li = line.split('\t')
            if len(li) > 8 and not indata:
                tokenline = line.rstrip().replace('#', '_')
                nheader = n
                if verbose:
                    pg.debug('header', nheader)
                indata = True
            fdp = line.find(': ')
            if not indata and fdp >= 0 and nstop == 0:
                tok = line[line[:fdp].rfind(' ')+1:fdp]
                val = line[fdp+2:].rstrip()
                if val.isnumeric():
                    val = float(val)
                    if val.is_integer():
                        val = int(val)

                header[tok] = val

            if indata and len(li) < 8 and nstop == 0:  # no data anymore
                nstop = n
                if verbose:
                    pg.debug('stop', nstop)
                indata = False
        if nstop == 0:
            nstop = len(lines)-3
        if verbose:
            pg.debug(nheader, len(lines), nstop, len(lines)-nstop-3)
            pg.debug(len(tokenline.split('\t')))
        str2date = lambda x: datetime.strptime(x.decode("utf-8"),
                                               '%Y-%m-%d %H:%M:%S').timestamp()
        Data = np.genfromtxt(filename, names=tokenline.split('\t'),
                             delimiter='\t', converters={"Time": str2date},
                             skip_header=nheader+1,
                             skip_footer=len(lines)-nstop-3)
        fields = Data.dtype.names
        if verbose:
            pg.debug("Fields", fields)
            pg.debug(header)
        elpos = []
        for el in ['A', 'B', 'M', 'N']:  # the ABEM variant
            alle = [Data[el+i] for i in ['x', 'y', 'z'] if el+i in fields]
            if len(alle) > 0:
                elpos.append(np.column_stack(alle))

        if len(elpos) == 0:  # try the Resecs variant
            for el in ['C1', 'C2', 'P1', 'P2']:
                alle = [Data[el+i] for i in ['x', 'y', 'z'] if el+i in fields]
                if len(alle) > 0:
                    elpos.append(np.column_stack(alle))

        eln = [(elp[:, 0]*9990+elp[:, 1])*9990+elp[:, 2]*10 for elp in elpos]
        nall = np.unique(eln)
        data = pg.DataContainerERT()
        for ni in nall:
            if np.isfinite(ni):
                ze = np.mod(ni, 999) / 10
                ye = np.mod((ni-ze) / 9990., 999) / 10
                xe = ((ni-ze)/999.-ye)/9990.0 / 10
                data.createSensor([xe, ye, ze])

        data.resize(len(Data))
        for i in range(len(Data)):
            abmn = np.array([np.nonzero(nall == eli[i])[0] for eli in eln])
            abmn[abmn >= data.sensorCount()] = -1  # infinite electrodes
            data.createFourPointData(i, *[int(a) for a in abmn])
        # translate data tokens with possible unit defaults
        tokenmap = {'ImA': 'i', 'VoltageV': 'u', 'ROhm': 'r', 'Rho': 'rhoa',
                    'AppROhmm': 'rhoa', 'Var': 'err', 'I': 'i', 'U': 'u',
                    'M': 'ma', 'P': 'ip', 'D': 'err', 'UV': 'u', 'RO': 'r',
                    'RhoaOm': 'rhoa', 'IP_sum_window_11': 'ip', 'Time': 't'}
        unitmap = {'ImA': 1e-3, 'Var': 0.01, 'U': 1e-3, 'I': 1e-3, 'D': 0.01}
        for fi in fields:
            if fi in tokenmap:
                data.set(tokenmap[fi], Data[fi] * unitmap.get(fi, 1.0))

        if return_header:
            return data, header

        return data


def importResecsAscii(filename, verbose=False):
    """Import Geoserve Resecs Ascii Export file."""
    nhead = 0

    fid = open(filename)
    for i, line in enumerate(fid):
        if line[0:4] == 'Type':
            nhead = i
            tokens = line.split("\t")
        if line[0:3] == 'GND':
            nhead += 1

    fid.close()
    nhead += 1
    ndata = i - nhead + 1
    if verbose:
        pg.debug(nhead, " header lines, ", ndata, " data")

    # Resecs ASCII format output tokens
    data_tokens = ['U', 'I', 'Rho', 'P', 'D']
    mult = [1e-3, 1e-3, 1., 1., 1e-2]
    datnr, toknr = [], []
    for nr, tok in enumerate(tokens):
        if (data_tokens.count(tok)):
            datnr.append(nr)
            toknr.append(data_tokens.index(tok))

    if verbose:
        pg.debug("found ", [data_tokens[nr] for nr in toknr], " at ", datnr)

    # load all data into big matrix
    DATA = np.loadtxt(filename, usecols=datnr, skiprows=nhead).T

    # position tokens set by Resecs instrument
    pos_tokens = ['C1(x)', 'C1(y)', 'C1(z)', 'C2(x)', 'C2(y)', 'C2(z)',
                  'P1(x)', 'P1(y)', 'P1(z)', 'P2(x)', 'P2(y)', 'P2(z)']
    posnr = np.zeros((len(pos_tokens),), dtype=int)
    for nr, tok in enumerate(pos_tokens):
        if (tokens.count(tok)):
            posnr[nr] = tokens.index(tok)

    # read positions from file
    POS = np.loadtxt(filename, usecols=posnr, skiprows=nhead)
    # extract positions for A/B/M/N and generate ids from it
    c1, c2, p1, p2 = POS[:, 0:3], POS[:, 3:6], POS[:, 6:9], POS[:, 9:12]
    nc1 = (c1[:, 0] * 9990 + c1[:, 1])*9990 + c1[:, 2] * 10
    nc2 = (c2[:, 0] * 9990 + c2[:, 1])*9990 + c2[:, 2] * 10
    np1 = (p1[:, 0] * 9990 + p1[:, 1])*9990 + p1[:, 2] * 10
    np2 = (p2[:, 0] * 9990 + p2[:, 1])*9990 + p2[:, 2] * 10
    # generate unique id
    nall = np.unique(np.hstack((nc1, nc2, np1, np2)))
    data = pg.DataContainerERT()
    for i in range(len(nall)):
        ze = np.mod(nall[i], 999) / 10
        ye = np.mod((nall[i]-ze) / 9990., 999) / 10
        xe = ((nall[i]-ze)/999.-ye)/9990.0 / 10
        pos = pg.Pos(xe, ye, ze)
        data.createSensor(pos)

    # rename tokens to match gimli tokens
    data_tokens[2:5] = ['rhoa', 'ip', 'err']
    data.resize(ndata)
    for i in range(ndata):
        data.createFourPointData(i, int(np.nonzero(nall == nc1[i])[0]),
                                 int(np.nonzero(nall == nc2[i])[0]),
                                 int(np.nonzero(nall == np1[i])[0]),
                                 int(np.nonzero(nall == np2[i])[0]))

        for j in range(len(toknr)):
            itok = data_tokens[toknr[j]].lower()
            data(itok)[i] = DATA[j, i] * mult[toknr[j]]

    if verbose:
        pg.debug(data)

    savestr = 'a b m n valid '
    for i in range(len(toknr)):
        savestr += data_tokens[toknr[i]].lower() + ' '

    data.checkDataValidity(False)
    data.setInputFormatString(savestr)

    if data.size() == 0:
        raise Exception('No Data found in importResecsAscii')

    return data


def import4PointLightNative(filename, verbose=False, return_header=False):
    """Import Lippmann 4-point light hp 10w native instrument data (*.txt)
    """
    with codecs.open(filename, 'r', errors='replace') as fi:
        content = fi.readlines()

    if content[0][0] != 'S' or content[-1][0] != 'E':
        raise Exception("no native 4-point-light file format")

    header = {}
    header['version'] = content[1]
    header['nr'] = content[2]
    header['comment'] = content[3]
    header['data'] = content[4]
    header['freq'] = content[5]
    header['minU'] = content[6]
    header['maxAv'] = content[7]
    header['maxEr'] = content[8]
    header['meaType'] = content[9]
    dx = float(content[10])
    x0 = float(content[11])
    header['eStart'] = float(content[12].split()[0])
    header['eEnd'] = float(content[12].split()[1])
    header['actEle'] = content[13]

    pg.debug('Reading (comment):', header['comment'])
    data = pg.DataContainerERT()
    for line in content[14:]:
        vals = line.split()
        if len(vals) > 7:
            data.addFourPointData(int(vals[0])-1, int(vals[1])-1,
                                  int(vals[2])-1, int(vals[3])-1,
                                  indexAsSensors=True,
                                  u=float(vals[4]), u90=float(vals[5]),
                                  i=float(vals[6]))

    data.scale([dx, 1, 1])
    data.translate([x0, 0, 0])
    if return_header:
        return data, header
    else:
        return data


def import4PointLight(filename, verbose=False):
    """Import Lippmann 4-point light instrument data (*.tx0)"""
    known_tokens = ['A', 'B', 'M', 'N', 'U', 'I', 'rho', 'phi', 'dU', 'dU90']
    data = pg.DataContainerERT()
    nel = 0
    datnr, toknr = [], []

#    with open(filename, 'r') as fi:
    with codecs.open(filename, 'r', encoding='utf8', errors='replace') as fi:
        content = fi.readlines()

    if content[0][0] == 'S' and content[-1][0] == 'E':
        return import4PointLightNative(filename, verbose=verbose)
    dataSect = -3

#    tok_units = {'err': 0.01, }  # to be later used

    for i, line in enumerate(content):
        if verbose:
            pass
#            print(i, line)

        line = line.rstrip('\n').replace(',', '.')
        if (line.find('* Electrode last num') >= 0):
            nel = int(line.split()[-1])  # number of electrodes
            if verbose:
                pg.debug(nel, "electrodes")
            for n in range(nel):
                # create dummy sensors in case of Roll-along files where first
                # sensors are missing
                data.createSensor(pg.Pos(-1000.+n, 0., 0.))

        if line.find('* Count') == 0:
            nData = int(line.split()[-1])  # number of data

        if (line.find('* Electrode [') == 0):
            sxyz = line.split()[-3:]
            if sxyz[-1] == 'X':
                sxyz = line.split()[-4:-1]
            pos = pg.Pos(float(sxyz[0]), float(sxyz[1]), float(sxyz[2]))
            data.setSensorPosition(int(line[13:16])-1, pos)

        if (line.find('* num') == 0):
            if 1:
                d = readAsDictionary(content[i+1:], content[i].split()[1:])
                if verbose:
                    pg.debug("Token:", d.keys())

#                 usual token are
# ['num', 'A', 'B', 'M', 'N', 'I', 'U', 'dU', 'U90', 'dU90', 'rho', 'phi', 'f',
# 'n', 'nAB', 'Profile', 'Spread', 'PseudoZ', 'X', 'Y', 'Z', 'Date', 'Time',
# 'U(Tx)']
                if d['num'][-1] != len(d['num']):
                    pg.debug(d['num'][-1], len(d['num']))
                    raise Exception('Insufficient data found!')

                data.resize(len(d['num']))
                for i in range(data.size()):
                    data.createFourPointData(
                        i, int(d['A'][i]-1), int(d['B'][i]-1),
                        int(d['M'][i]-1), int(d['N'][i]-1))

                data.set('i', np.array(d['I'])/1000.0)
                data.set('u', np.array(d['U'])/1000.0)
                data.set('ip', np.array(d['phi']))
                data.set('rhoa', np.array(d['rho']))
                data.set('err', np.array(d['dU']))

            else:

                # evaluate tokens (physical fields) in file
                for nr, tok in enumerate(line[1:].split()):
                    if (known_tokens.count(tok)):
                        datnr.append(nr)
                        toknr.append(known_tokens.index(tok))

                known_tokens[6:10] = ['rhoa', 'ip', 'err', 'iperr']  # to BERT
                if verbose:
                    pg.debug(datnr, toknr, [known_tokens[t] for t in toknr])
                dataSect = -2  # prepare to be in data section 2 lines later
                # unit line should be evaluated to gain multiplicator
                data.resize(nData)  # initialize vectors with appropriate size

            if dataSect >= 0:  # read actual data if in data section
                sabmn = line.split()

                if dataSect >= nData:
                    data.resize(dataSect+1)
                # create electrode array
                data.createFourPointData(dataSect,
                                         int(sabmn[1])-1, int(sabmn[2])-1,
                                         int(sabmn[3])-1, int(sabmn[4])-1)

                for i in range(4, len(toknr)):  # all tokens except ABMN
                    itok = known_tokens[toknr[i]].lower()
                    try:
                        ff = float(sabmn[datnr[i]])
                        if itok.lower() == 'err':
                            ff = ff / 100
                        if itok.lower() == 'i':
                            ff = ff / 1000.
                        if itok.lower() == 'u':
                            ff = ff / 1000.

                        data(itok)[dataSect] = ff
                    except IndexError:
                        pass

            if dataSect >= -2:
                dataSect += 1  # so that unit line will not be read

#    savestr = ''
#    for i in range(len(toknr)):
#        savestr += known_tokens[toknr[i]].lower() + ' '

#    data.setInputFormatString(savestr)

    if data.size() == 0:
        raise Exception('No Data found in import4PointLight')

    # data.set('rhoa', pg.abs(data('rhoa')))
    # data.checkDataValidity()

    return data


def importRollAlong4PointLight(basename, style='1', corI=0, start=1,
                               verbose=False):
    """Import several 4-point light roll along data files (*.tx0)

    Parameters
    ----------
    basename : str
        the base file name (.tx0 extension will be stripped)
    style : naming style ['1']
        the roll-along files are called
        '1': basename+'_1.tx0', basename+'_2.tx0' etc.
        'A': basename+'_A.tx0', basename+'_B' etc.
        'a': basename+'_a.tx0', basename+'_b.tx0'
    """
    basename = basename.rstrip('.tx0')
    data = import4PointLight(basename + '.tx0')
    if verbose:
        pg.debug(data)
    for n in range(start, 100):
        if style == 'A':
            fname1 = basename + '_'+chr(64+n)+'.tx0'
        elif style == 'a':
            fname1 = basename + '_'+chr(96+n)+'.tx0'
        else:
            fname1 = basename + '_'+str(n)+'.tx0'
        if not os.path.exists(fname1):
            break
        if verbose:
            pg.debug(fname1)
        data1 = import4PointLight(fname1)
        for i in range(n*20):
            data1.setSensorPosition(i, data.sensorPosition(i))
        data.add(data1)
        if verbose:
            pg.debug(data)

    return data


def import4PointLightOld(filename, verbose=False):
    """import Lippmann 4-point light instrument data (*.tx0)
    DEPRECATED will be removed soon
    """
    known_tokens = ['A', 'B', 'M', 'N', 'U', 'I', 'rho', 'phi', 'dU', 'dU90']
    data = pg.DataContainerERT()
    nel, ndata = 0, 0
    datnr, toknr = [], []
    fid = open(filename)
    for i, line in enumerate(fid):
        line = line.rstrip('\n')
        if (line.find('* Electrode last num') == 0):
            nel = int(line.split()[-1])
        if (line.find('* Count') == 0):
            ndata = int(line.split()[-1])
        if verbose:
            pg.debug("{} electrodes, {} data".format(nel, ndata))
        if (line.find('* Electrode [') == 0):
            sxyz = line.split()[-3:]
            pos = pg.Pos(float(sxyz[0]), float(sxyz[1]), float(sxyz[2]))
            data.createSensor(pos)
        if (line.find('* num') == 0):
            for nr, tok in enumerate(line[1:].split()):
                if (known_tokens.count(tok)):
                    datnr.append(nr)
                    toknr.append(known_tokens.index(tok))

            if verbose:
                pg.debug(datnr, toknr, [known_tokens[t] for t in toknr])

            break

    fid.close()

    # load complete data matrix according to detected columns
    datmat = np.loadtxt(filename, skiprows=i+2, usecols=datnr).T
    # redefine tokens to match BERT definitions
    known_tokens[6:10] = ['rhoa', 'ip', 'err', 'iperr']
    savestr = ''
    data.resize(len(datmat[0]))
    # set data for known tokens
    for i in range(len(datmat)):
        itok = known_tokens[toknr[i]].lower()
        offset, mult = 0., 1.
        if (toknr[i] < 4):
            offset = -1.  # A B M N
        if ((4, 5).count(toknr[i])):
            mult = 1e-3  # U/mV I/mA
        if ((8, 9).count(toknr[i])):
            mult = 1e-2  # dU/% dU90/%
        data.set(itok, pg.asvector(datmat[i] * mult + offset))
        savestr += itok + ' '

    data.setInputFormatString(savestr)
    for n in range(len(datmat[0])):
        data.markValid(n)

    if verbose:
        pg.debug(data, data.tokenList())

    return data


def importABEM(filename, spacing=None, xcoords=None,
                         verbose=False, return_all=False):
    """ Import old ABEM (AMP) format

        Filename:
        Instrument ID:
        Date & Time:
        Base station:                   0.00    0.00    0.00    0.00       0.00
        Rows header/data/topography:   27           5949         0
        Acquisition mode:               2
        Measurement method:             Section
        Electrode layout:               11              Freeform        GN4
        Co-ordinate type:               XYZ:1
        Smallest electrode spacing:     1.00
        Marine survey (R,h,a,b):        -               -         -          -
        Protocol #1:                    GRAD2XA
        Protocol #2:                    -
        Protocol #3:                    -
        Protocol #4:                    -
        Protocol #5:                    -
        Protocol #6:                    -
        Protocol #7:                    -
        Protocol #8:                    -
        Operator:
        Client:
        Comment #1:
        Comment #2:
        Comment #3:
        Comment #4:

        No. Time A(x) A(y) A(z) B(x) B(y) B(z) M(x) M(y) M(z) N(x) N(x) N(z) I(mA)
        Voltage(V) Res.(ohmm) Error(%) T(on) T(0) T(N):01 App.Char (ms) Error(ms)
        *

    """
    with open(filename, 'r') as fi:
        content = fi.readlines()

    nData = 0
    header = {}
    for nHeader, line in enumerate(content):
        if line.rfind("A(x)") >= 0:
            break
        if line.rfind(":") > 0:
            sp = line.rstrip("\n").split(":")
            header[sp[0]] = sp[-1].replace("\t", "").replace(" ", "")

    print(line)
    if spacing is None and "Smallest electrode spacing" in header:
        spacing = float(header["Smallest electrode spacing"])
        print("Spacing", spacing)
    if xcoords is None and "Co-ordinate type" in header:
        xcoords = int(header["Co-ordinate type"])

    nHeader += 1
    nData = len(content) - nHeader
    nbtimegates = 0
    while "T(N):{:02d}".format(nbtimegates+1) in line:
        nbtimegates += 1

    if verbose:
        print("ABEM file format assuming data", nData)

    count = 0
    MA = []
    MAMV = []
    if len(content) >= nHeader + nData:
        data = pg.DataContainerERT()
        data.resize(nData)

        for i, row in enumerate(content[nHeader:nHeader + nData]):
            ma=[]
            mamv=[]
            t=[]
            vals = row.split()
            if xcoords:
                eaID = data.createSensor(pg.Pos(float(vals[2]), 0.0, 0.0))
                ebID = data.createSensor(pg.Pos(float(vals[3]), 0.0, 0.0))
                emID = data.createSensor(pg.Pos(float(vals[4]), 0.0, 0.0))
                enID = data.createSensor(pg.Pos(float(vals[5]), 0.0, 0.0))
                dvals = vals[6:]
            else:
                eaID = data.createSensor(pg.Pos(float(vals[2])*spacing-spacing, 0.0, 0.0))
                ebID = data.createSensor(pg.Pos(float(vals[5])*spacing-spacing, 0.0, 0.0))
                emID = data.createSensor(pg.Pos(float(vals[8])*spacing-spacing, 0.0, 0.0))
                enID = data.createSensor(pg.Pos(float(vals[11])*spacing-spacing, 0.0, 0.0))
                dvals = vals[14:]

            to = float(dvals[5])  # vals[11]
            dt = float(dvals[6])
            time = to + dt/2
            data.createFourPointData(count, eaID, ebID, emID, enID)
            data('i')[count] = float(dvals[0]) * 1e-3
            data('u')[count] = float(dvals[1])
            data('r')[count] = float(dvals[2])
            data('err')[count] = float(dvals[3])
            for l in range(nbtimegates):
                t.append(float(time))
                ma.append(float(dvals[7+l*3]))  # msec
                dt = float(dvals[6+l*3])
                mamv.append(float(dvals[7+l*3])/dt)  # mV/V ?
                time += dt

            data('ip')[count] = float(dvals[7])/float(dvals[6])
            data('iperr')[count] = float(dvals[8])/float(dvals[6])
            count += 1

            MA.append(ma)
            MAMV.append(mamv)

        data.sortSensorsX()
    else:
        raise Exception("Read ABEM .. file content to small " +
                        str(len(content)) + " expected: " + str(nHeader+nData))

    if verbose:
        print(data, data.tokenList())

    if return_all:
        return data, np.transpose(np.array(MAMV)), t, header
    else:
        return data

def importABEMold(filename, verbose=False):
    """ Import old ABEM (AMP) format

        Filename:
        Instrument ID:
        Date & Time:
        Base station:                   0.00    0.00    0.00    0.00       0.00
        Rows header/data/topography:   27           5949         0
        Acquisition mode:               2
        Measurement method:             Section
        Electrode layout:               11              Freeform        GN4
        Co-ordinate type:               XYZ:1
        Smallest electrode spacing:     1.00
        Marine survey (R,h,a,b):        -               -         -          -
        Protocol #1:                    GRAD2XA
        Protocol #2:                    -
        Protocol #3:                    -
        Protocol #4:                    -
        Protocol #5:                    -
        Protocol #6:                    -
        Protocol #7:                    -
        Protocol #8:                    -
        Operator:
        Client:
        Comment #1:
        Comment #2:
        Comment #3:
        Comment #4:

        No. Time A(x) B(x) M(x) N(x) I(mA)  Voltage(V) App.R.(ohmm) Error(%)
        *

    """
    with open(filename, 'r') as fi:
        content = fi.readlines()
    fi.close()

    nData = 0
    nHeader = 0
    if len(content) > 3:

        sizes = content[4].split('\r\n')[0].split()

        if sizes[0] == "Rows":
            # Rows header/data/topography:   27           5949         0
            nHeader = int(sizes[2])
            nData = int(sizes[3])
        else:
            raise Exception("Read ABEM .. size format unknows" + sizes)
    else:
        raise Exception("Read ABEM .. file content too small " +
                        str(len(content)))

    if verbose:
        pg.debug("ABEM file format assuming data", nData)

    count = 0
    if len(content) >= nHeader + nData:
        data = pg.DataContainerERT()
        data.resize(nData)

        for i, row in enumerate(content[nHeader:nHeader + nData]):
            vals = row.split()
            if len(vals) == 10:
                eaID = data.createSensor(pg.Pos(float(vals[2]), 0.0, 0.0))
                ebID = data.createSensor(pg.Pos(float(vals[3]), 0.0, 0.0))
                emID = data.createSensor(pg.Pos(float(vals[4]), 0.0, 0.0))
                enID = data.createSensor(pg.Pos(float(vals[5]), 0.0, 0.0))

                data.createFourPointData(count, eaID, ebID, emID, enID)
                data('i')[count] = float(vals[6]) * 1e-3
                data('u')[count] = float(vals[7])
                data('err')[count] = float(vals[9]) / 100.
                count += 1
            else:
                raise Exception("Read ABEM .. cannot interpret data tokens " +
                                str(len(vals)), row)

        data.sortSensorsX()

    else:
        raise Exception("Read ABEM .. file content to small " +
                        str(len(content)) + " expected: " + str(nHeader+nData))

    if verbose:
        pg.debug(data, data.tokenList())

    return data


def importAsciiColumns(filename, verbose=False, return_header=False):
    """Import any ERT data file organized in columns with column header

    Input can be:
        * Terrameter LS or SAS Ascii Export format, e.g.
    Time MeasID DPID Channel A(x) A(y) A(z) B(x) B(y) B(z) M(x) M(y) M(z) \
    N(x) N(y) N(z) F(x) F(y) F(z) Note I(mA) Uout(V) U(V)        SP(V) R(O) \
    Var(%)         Rhoa Cycles Pint Pext(V) T(°C) Lat Long
    2016-09-14 07:01:56 73 7 1 8 1 1 20 1 1 12 1 1 \
    16 1 1 14 1 2.076  99.8757 107.892 0.0920761 0 0.921907 \
    0.196302 23.17 1 12.1679 12.425 42.1962 0 0
        * Resecs Output format

    """
    data = pg.DataContainerERT()
    header = {}
    with open(filename, 'r', encoding='iso-8859-15') as fi:
        content = fi.readlines()
        if content[0].startswith('Injection'):  # Resecs lead-in
            n = 0
            for n in range(20):
                if len(content[n]) < 2:
                    break

            content = content[n+1:]

        d = readAsDictionary(content, sep='\t')
        if len(d) < 2:
            d = readAsDictionary(content)

        # if 'col1' in d:
        #     raise Exception("file seems to be no valid table")

        nData = len(next(iter(d.values())))
        data.resize(nData)
        print(d.keys())
        if 'Spa.1' in d:  # Syscal Pro
            abmn = ['Spa.1', 'Spa.2', 'Spa.3', 'Spa.4']
            if verbose:
                pg.debug("detected Syscalfile format")
        elif 'A(x)' in d:  # ABEM Terrameter
            abmn = ['A', 'B', 'M', 'N']
            if verbose:
                pg.debug("detected ABEM file format")
        elif 'Tx1' in d:  # GDD
            abmn = ['Tx1', 'Tx2', 'Rx1', 'Rx2']
            if verbose:
                pg.debug("detected GDD file format")
        elif 'xA' in d:  # Workbench TX2 processed data
            abmn = ['xA', 'xB', 'xM', 'xN']
            if verbose:
                pg.debug("detected Workbench file format")
        elif 'C1(x)' in d or 'C1(xm)' in d:  # Resecs
            abmn = ['C1', 'C2', 'P1', 'P2']
            if verbose:
                pg.debug("detected RESECS file format")
        else:
            pg.debug("no electrode positions found!")
            pg.debug("Keys are:", d.keys())
            raise Exception("No electrode positions found!")

        for i in range(nData):
            if abmn[0]+'(z)' in d:
                eID = [data.createSensor([d[se+'(x)'][i], d[se+'(y)'][i],
                                          d[se+'(z)'][i]]) for se in abmn]
            elif abmn[0]+'(zm)' in d:
                eID = [data.createSensor([d[se+'(xm)'][i], d[se+'(ym)'][i],
                                          d[se+'(zm)'][i]]) for se in abmn]
            elif abmn[0]+'(y)' in d:
                eID = [data.createSensor([d[se+'(x)'][i], d[se+'(y)'][i],
                                          0.]) for se in abmn]
            elif abmn[0]+'(ym)' in d:
                eID = [data.createSensor([d[se+'(xm)'][i], d[se+'(ym)'][i],
                                          0.]) for se in abmn]
            elif abmn[0]+'(x)' in d:
                eID = [data.createSensor([d[se+'(x)'][i], 0.,
                                          0.]) for se in abmn]
            elif abmn[0]+'(xm)' in d:
                eID = [data.createSensor([d[se+'(xm)'][i], 0.,
                                          0.]) for se in abmn]
            else:
                eID = [data.createSensor([d[se][i], 0., 0.]) for se in abmn]

            data.createFourPointData(i, *eID)

        # data.save('tmp.shm', 'a b m n')
        tokenmap = {'I(mA)': 'i', 'I': 'i', 'In': 'i', 'Vp': 'u',
                    'VoltageV': 'u', 'U': 'u', 'U(V)': 'u', 'UV': 'u',
                    'R(Ohm)': 'r',  'RO': 'r', 'R(O)': 'r', 'Res': 'r',
                    'Rho': 'rhoa', 'AppROhmm': 'rhoa', 'Rho-a(Ohm-m)': 'rhoa',
                    'Rho-a(Om)': 'rhoa',
                    'Var(%)': 'err', 'D': 'err', 'Dev.': 'err', 'Dev': 'err',
                    'M': 'ma', 'P': 'ip', 'IP sum window': 'ip',
                    'Time': 't'}
        # Unit conversions (mA,mV,%), partly automatically assumed
        unitmap = {'I(mA)': 1e-3, 'Var(%)': 0.01,  # ABEM
                   'U': 1e-3, 'I': 1e-3, 'D': 0.01,  # Resecs
                   'Dev.': 0.01, 'In': 1e-3, 'Vp': 1e-3}  # Syscal
        abmn = ['a', 'b', 'm', 'n']
        if 'Cycles' in d:
            d['stacks'] = d['Cycles']
        for key in d.keys():
            vals = np.asarray(d[key])
            if key.startswith('IP sum window'):  # there is a trailing number
                key = 'IP sum window'  # apparently not working
            if np.issubdtype(vals.dtype, np.floating,  # 'float'  'int'
                             ) or np.issubdtype(vals.dtype, np.signedinteger):
                if key in tokenmap:  # use the standard (i, u, rhoa) key
                    if key not in abmn:
                        if verbose:
                            pg.debug("Setting", tokenmap[key], "from", key)
                        data.set(tokenmap[key],
                                 vals * unitmap.get(key, 1.0))
                else:  # use the original key if not XX(x) etc.
                    if not re.search('([x-z])', key) and key not in abmn:
                        data.set(key.replace(' ', '_'), d[key])

        r = data('u') / data('i')
        if hasattr(d, 'R(0)'):
            if np.linalg.norm(r-d['R(O)']) < 1e4:  # no idea what's that for
                data.set('r', r)
            else:
                pg.debug("Warning! File inconsistent")

    data.sortSensorsX()
    if return_header:
        return data, header
    else:
        return data


def importABEMTerrameterSAS(filename, verbose=False):
    """Import Terrameter SAS Ascii Export format.

    Time MeasID DPID Channel A(x) A(y) A(z) B(x) B(y) B(z) M(x) M(y) M(z) \
    N(x) N(y) N(z) F(x) F(y) F(z) Note I(mA) Uout(V) U(V)        SP(V) R(O) \
    Var(%)         Rhoa Cycles Pint Pext(V) T(°C) Lat Long
    2016-09-14 07:01:56 73 7 1 8 1 1 20 1 1 12 1 1 \
    16 1 1 14 1 2.076  99.8757 107.892 0.0920761 0 0.921907 \
    0.196302 23.17 1 12.1679 12.425 42.1962 0 0
    """
    with open(filename, 'r', encoding='iso-8859-15') as fi:
        content = fi.readlines()
    fi.close()

    d = readAsDictionary(content, sep='\t')
    nData = len(d['I(mA)'])

    data = pg.DataContainerERT()
    data.resize(nData)

    for i in range(nData):
        eaID = data.createSensor([d['A(x)'][i], d['A(y)'][i], d['A(z)'][i]])
        ebID = data.createSensor([d['B(x)'][i], d['B(y)'][i], d['B(z)'][i]])
        emID = data.createSensor([d['M(x)'][i], d['M(y)'][i], d['M(z)'][i]])
        enID = data.createSensor([d['N(x)'][i], d['N(y)'][i], d['N(z)'][i]])
        data.createFourPointData(i, eaID, ebID, emID, enID)

    data.set('i', np.array(d['I(mA)'])/1000.0)
    data.set('u', d['U(V)'])
    if 'Uout(V)' in d:
        data.set('uout', d['Uout(V)'])
    if 'SP(V)' in d:
        data.set('sp', d['SP(V)'])
    if 'Var(%)' in 'd':
        data.set('var', d['Var(%)'])

    if 'R(Ohm)' in d:
        data.set('r', d['R(Ohm)'])
    else:
        r = data('u') / data('i')

    if hasattr(d, 'R(0)'):
        if np.linalg.norm(r-d['R(O)']) < 1e4:
            data.set('r', r)
        else:
            pg.debug("Warning! File inconsistent")

    data.sortSensorsX()
    return data


def importGeoSys(filename, verbose=False):
    """import GeoSys format

        Example format
        --------------
        Messgebiet   :Test 1b
        Profilname   :Kopf
        Datum        :11.11.2014
        Registrierer :FG
        Geraet       :GMS150
        Richtung     :
        Wetter       :gut
        Anordnung    :S
        Konfiguration:1
        Kabelbaum    :G050-050
        Steuerdatei  :A26W0508.ESD

        A   M   N   B     zeit  f   mn/2  ab/2    KF    I/mA  U/mV  R/OhmM  Q

        *

    """
    with open(filename, 'r') as fi:
        content = fi.readlines()

    nHeader = 13
    count = 0

    tokenLineStr = content[12]
    tokenLine = tokenLineStr.split()

    if len(content) >= nHeader:
        data = pg.DataContainerERT()

        data.resize(len(content))

        for i, row in enumerate(content[nHeader:-1]):
            vals = row.split()
            if len(vals) == 0:
                continue
            if len(vals) > 11:
                if tokenLine[0] == 'A' and tokenLine[1] == 'M' and \
                   tokenLine[2] == 'N' and tokenLine[3] == 'B' and \
                   tokenLine[9] == 'I/mA' and tokenLine[10] == 'U/mV':

                    vv = [float(vali) for vali in vals]
                    eaID = data.createSensor(pg.Pos(vv[0], 0.0, 0.0))
                    emID = data.createSensor(pg.Pos(vv[1], 0.0, 0.0))
                    enID = data.createSensor(pg.Pos(vv[2], 0.0, 0.0))
                    ebID = data.createSensor(pg.Pos(vv[3], 0.0, 0.0))

                    data.createFourPointData(count, eaID, ebID, emID, enID)
                    data('i')[count] = float(vals[9]) * 1e-3
                    data('u')[count] = float(vals[10]) * 1e-3
                    count += 1
                else:
                    raise Exception("Cannot interpret tokenLine. " +
                                    tokenLineStr)
            else:
                raise Exception("Read GeoSYS - cannot interpret data tokens " +
                                str(len(vals)), row)

        data.sortSensorsX()
        data.resize(count)
    else:
        raise Exception("Read ABEM .. file content to small " +
                        str(len(content)) + " expected: " + str(nHeader))

    if verbose:
        pg.debug(data, data.tokenList())

    return data


def importFLW(filename, verbose=False):
    """Import Geotom free FLW format.

        //FILENAME
        //- Kabelrichtung:
        //  Kabel I: revers
        //  Kabel II: normal
        //  Kabel III: normal
        //  Kabel IV: revers
        //- Kabelanordnung: parallel (2-2)
        //- Optimierung: letzte Messung
        //- Sortierung: nach Level
        //- Stromstufen: 5.0 mA .. 50.0 mA
        Type:         Flow
        Name:         Name
        Comment:
        Comment:
        Comment:
        Spacing:      x
        First El:     x
        Nr of points: Nr
        IP present:   0
          1   4   2   3      5.000    51.0139    128.21   0.6
    """
    with open(filename, 'r') as fi:
        content = fi.readlines()
    fi.close()

    dataStart = 0
    x0 = 0
    spacing = 1.0
    for i, c in enumerate(content):
        vals = c.split()
        if c[0] != '/' and len(vals) > 0:
            if len(vals) > 6 and vals[0] != 'Comment:':
                dataStart = i-1
                break
            else:
                if vals[0] == 'Spacing:':
                    v = vals[1].split(',')
                    if len(v) > 1:
                        spacing = float(v[1])
                    else:
                        spacing = float(v[0])
                elif vals[0] == 'First El:':
                    v = vals[1].split(',')
                    if len(v) > 1:
                        x0 = float(v[1])
                    else:
                        x0 = float(v[0])

    d = readAsDictionary(content[dataStart:], token=[])

    data = pg.DataContainerERT()
    nData = len(d['col0'])
#  A   B   M   N  ??    I/mA    U/mV        ??       ??      ??      ??
#  date     time
#  1   2   3   4  $4    0.100   -24.4087    920.18   0.0    -28.59   0.9
#  27.05.2015  14:06:10
#
#  A   B   M   N  ??    I/mA    U/mV         ??      ??   date        time
#  1   2   3   4  $8    1.000   -41.5541     15.67   0.0  24.06.2015  14:52:54
    for i in range(nData):
        eaID = data.createSensor([(d['col0'][i]-1)*spacing + x0, 0.0, 0.0])
        ebID = data.createSensor([(d['col1'][i]-1)*spacing + x0, 0.0, 0.0])
        emID = data.createSensor([(d['col2'][i]-1)*spacing + x0, 0.0, 0.0])
        enID = data.createSensor([(d['col3'][i]-1)*spacing + x0, 0.0, 0.0])
        data.createFourPointData(i, eaID, ebID, emID, enID)

    if '$' in d['col4'][0]:
        data.set('i', np.array(d['col5'])*1e-3)
        data.set('u', np.array(d['col6'])*1e-3)
    else:
        data.set('i', np.array(d['col4'])*1e-3)
        data.set('u', np.array(d['col5'])*1e-3)

    data.sortSensorsX()

    return data


def importSuperSting(datafile, verbose=True):
    """Import ERT data from AGI SuperSting instrument (*.stg file)."""
    ALL = np.genfromtxt(datafile, delimiter=',', skip_header=3)
    Apos = ALL[:, 9:12]
    Bpos = ALL[:, 12:15]
    Mpos = ALL[:, 15:18]
    Npos = ALL[:, 18:21]
    # what about infinite electrodes?
    pos = np.vstack((Apos, Bpos, Mpos, Npos))
    upos, ifwd, irev = uniqueRows(pos)
    data = pg.DataContainerERT()
    for ipos in upos:
        data.createSensor(ipos)

    data.resize(len(ALL))
    ABMN = irev.reshape(4, -1).T
    for i, abmn in enumerate(ABMN):
        ind = [int(ii) for ii in abmn]
        data.createFourPointData(i, *ind)  # ind[1], ind[0], ind[2], ind[3])

    data.set('i', ALL[:, 6] * 1e-3)
    data.set('u', ALL[:, 4] * data('i'))  # U=R*I
    data.set('err', ALL[:, 5] * 0.001)
    data.set('rhoa', ALL[:, 7])
    if ALL.shape[1] > 30:
        data.set('ip', ALL[:, 30]*1000)  # M integrated in msec
        for i in range(6):
            data.set('ip'+str(i+1), ALL[:, 24+i])

    data.markValid(data('rhoa') > 0)
    data.checkDataValidity()
    data.sortSensorsX()
    return data


def importAres2(filename, verbose=True, return_header=False, return_all=False):
    """Import ERT data from Ares II (GF Brno) multielectrode instrument."""
    header = {}
    data = pg.DataContainerERT()
    with open(filename) as fid:
        lines = fid.readlines()
        for i, line in enumerate(lines):
            lines[i] = re.sub("\*[0-9]", "", line.rstrip())

        for i, line in enumerate(lines):
            if line.startswith('C1'):
                break
            if ':' in line and len(line) <= 80:
                sp = line.split(':')
                header[sp[0]] = sp[1]

        # first remove *1 in elecrode spacings & find out what it means
        cols = np.genfromtxt(lines[i:], names=True, delimiter='\t',
                             autostrip=True)

        if verbose:
            print(header)
        if 'Electrode distance' in header:
            dx = float(header['Electrode distance'].split('m')[0])
        elif 'Distance' in header:
            dx = float(header['Distance'].split('m')[0])
        else:
            raise Exception('Electrode distance cannot be determined.')

        if 'Profile length' in header:
            plen = float(header['Profile length'].split('m')[0])
        elif 'Length' in header:
            plen = float(header['Length'].split('m')[0])

        nel = int(plen/dx)+1
        for i in range(nel):
            data.createSensor([i*dx, 0])

        data.resize(len(cols))
        tokIn = ['C1', 'C2', 'P1', 'P2']
        tokOut = ['a', 'b', 'm', 'n']
        names = cols.dtype.names
        for i in range(4):
            tok = tokIn[i]+'el'
            if tok not in names:
                tok = tokIn[i]+'_el'
            col = cols[tok]
            col[np.isinf(col)] = -1
            data.set(tokOut[i], col)

        if 'ImA' in names:
            data.set('i', cols['ImA'] / 1000)
        elif 'I_mA' in names:
            data.set('i', cols['I_mA'] / 1000)
        if 'UmV' in names:
            data.set('u', cols['UmV'] / 1000)
        elif 'V_mV' in names:
            data.set('u', cols['V_mV'] / 1000)
        if 'Stdev' in names:
            data.set('err', cols['Stdev'] / 100)
        elif 'Stdev_' in names:
            data.set('err', cols['Stdev_'] / 100)
        if 'AppResOhmm' in names:
            data.set('rhoa', cols['AppResOhmm'])
        elif 'AppRes_Ohmm' in names:
            data.set('rhoa', cols['AppRes_Ohmm'])

        MA = []
        for i in range(33):
            sti = "IP" + str(i+1)
            if sti in cols.dtype.names:
               MA.append(cols[sti] * 10)  # in % instead of mV/V

        t = []
        delay = 0.005  # fixed in instrument
        if "IP windows" in header:
            dt = np.fromstring(header["IP windows"][:-2], dtype=float,
                               sep="\t") * 0.001  # ms
            tGate = delay + np.hstack((0, np.cumsum(dt)))
            header["ipGateT"] = tGate
            t = tGate[:-1] + dt/2

        data.markValid(data('rhoa') > 0)
        if return_all:  # the variant needed for the TDIP class
            return data, np.array(MA), t, header
        elif return_header:
            return data, header
        else:
            return data


def importDIP(filename, verbose=True, return_header=False, return_all=False):
    """Import ERT data from Ares II (GF Brno) multielectrode instrument."""
    header = {}
    data = pg.DataContainerERT()
    with open(filename) as fid:
        lines = fid.readlines()
        tokens = None
        for i, line in enumerate(lines):
            if line.find('NumElect') > 0 or line.find('rho') > 0:
                tokens = line[1:].replace('IP ', 'IP_').split()
            if tokens is not None and line[0] != '/':
                break

        A = np.genfromtxt(lines[i:-1], names=tokens)

        data = pg.DataContainerERT()
        ndata = len(A)
        ux = np.unique(np.hstack((A['XA'], A['XB'], A['XM'], A['XN'])))
        for i, xi in enumerate(ux):
            data.createSensor([xi, 0, 0])

        for i in range(ndata):
            nn = [int(np.nonzero(ux == A['X'+s][i])[0]) for s in
                  ['A', 'B', 'M', 'N']]
            data.createFourPointData(i, *nn)

        data.set('valid', A['InUse'])
        # FW: Avoid f-strings to keep Py 3.5 compatibility
        nst = ['IP_{}'.format(i) for i in range(1, 10)]
        i = 10
        while 'IP{}_data'.format(i) in tokens:
            nst.append('IP{}'.format(i))
            i += 1

        MAall = np.column_stack([A[nsi+'_data'] for nsi in nst]).T
        VA = np.column_stack([A[nsi+'_inUse'] for nsi in nst]).T
        MA = np.ma.MaskedArray(MAall, np.isclose(VA, 0))

        t = np.array([A[nsi+'_center'][0] for nsi in nst])
        dt = np.array([A[nsi+'_width'][0] for nsi in nst])
        header['ipDT'] = dt
        header['delay'] = t[0] - dt[0]/2
        header['ipGateT'] = np.hstack((t-dt/2, t[-1]+dt[-1]/2))
        data.set('rhoa', A['Rho'])

        if return_all:  # the variant needed for the TDIP class
            return data, MA, t, header
        elif return_header:  # might become the default for all
            return data, header
        else:  # just the data container as currently used by importData
            return data  # unless all importers support returning a header


def readAsDictionary(content, token=None, sep=None):
    """Read list of strings from a file as column separated dictionary.

        e.g.
        token1 token2 token3 token4
        va1    va2    val3   val4
        va1    va2    val3   val4
        va1    va2    val3   val4

    Parameters
    ----------
    content: [string]
        List of strings read from file:
        e.g.
        with open(filename, 'r') as fi:
            content = fi.readlines()
        fi.close()
    token: [string]
        If given the tokens will be the keys of the resulting dictionary.
        When token is None, tokens will be the first row values.
        When token is a empty list, the tokens will be autonamed to
        'col' + str(ColNumber)
    ret: dictionary
        Dictionary of all data
    """
    data = dict()

    if token is None:
        header = content[0].splitlines()[0].split(sep)
        token = []

        for i, tok in enumerate(header):
            tok = tok.lstrip()
            token.append(tok)

    for i, row in enumerate(content[1:]):
        vals = row.splitlines()[0].split(sep)
        for j, v in enumerate(vals):
            v = v.replace(',', '.')

            if len(token) < j+1:
                token.append('col' + str(j))
            if token[j] not in data:
                data[token[j]] = [None] * (len(content)-1)
            try:
                data[token[j]][i] = float(v)
            except:
                if len(v) == 1 and v[0] == '-':
                    v = 0.0
                data[token[j]][i] = v

    return data


def readABEMProtocolFile(xmlfile, verbose=False):
    """Read ABEM protocol file (*.xml) as DataContainerERT."""
    # import xml.etree.ElementTree as ET
    ET = pg.optImport("xml.etree.ElementTree",
                      "import ABEM protocol files (*.xml)")
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    if verbose:
        for child in root:
            pg.debug(child.tag, child.text)

    seq = root.find('Sequence')
    A, B, M, N, C = [], [], [], [], []
    for mea in seq.findall('Measure'):
        tx = mea.find('Tx')
        a, b = [int(k) for k in tx.text.split()]
        recs = mea.findall('Rx')
        C.append(len(recs))
        for rx in recs:
            m, n = [int(k) for k in rx.text.split()]
            A.append(a)
            B.append(b)
            M.append(m)
            N.append(n)

    ABMN = np.column_stack((A, B, M, N))
    nel = np.max(ABMN)
    dx = 1
    data = pg.DataContainerERT()
    for i in range(nel):
        data.createSensor([i*dx, 0])

    data.resize(ABMN.shape[0])
    for i, abmn in enumerate(ABMN):
        data.createFourPointData(i, *[int(el)-1 for el in abmn])

    if verbose:
        pg.debug(data)
        pg.debug("{:d} injections (mean c={:.1f})".format(len(C),
                                                          data.size()/len(C)))
    return data


def importSIP256Test(fileName, verbose=False):
    """Read SIP256 file (RES format) and return a DataContainer.

    Experimental to be a little bit more flexible
    Read SIP256 file (RES format)  and return a DataContainer.

    Supported: SIP256D

    TODO: UNICODE problems with ° sign
    TODO: find BEGIN END frequencies bug in fileformat
    TODO: read older versions

    Parameters
    ----------
    fileName: str
        *.RES file (SIP256 raw output file)

    verbose: bool
        Do some output [False].

    Returns
    -------
        data : pg.DataContainer

    Examples
    --------
        data = importSIP256('myfile.res', True)
    """

    def readSIP256Freqs_(content, start, endStr):
        freqs = []
        for i, line in enumerate(content[start:]):
            if endStr in line:
                break

            vals = line.split()
#             20000.00000000 48000.00000000   96  170    1     0.0    1
            if int(vals[6]) == 1:
                freqs.append(float(vals[0]))
        return i+1, freqs

    def readSIP256Layout_(content, start, endStr):
        sensors = []
        for i, line in enumerate(content[start:]):
            if endStr in line:
                break
            vals = line.split()
            sensors.append(float(vals[2]))

        return i+1, sensors

    def readReading_(content, nFreq, start, endStr):
        readings = []
#        Reading:    1 / RU-A:    1  RU-B:    2
        vals = content[start].split('\n')[0].split()
        eA = int(vals[4])
        eB = int(vals[6])

        for i in range(start+1, len(content)):
            line = content[i].split('\n')[0]
            if endStr in line:
                break

            if "Remote Unit:" in line:
                # ru = line.split()[2]  # not used
                # handle awful file format inconsistencies here
                content[i+1] = content[i+1].replace('Frequency /Hz',
                                                    'Frequency/Hz')

                mat = readAsDictionary(content[i+1:i+2+nFreq])
                i += nFreq+1

                readings.append(mat)
#                print(ru, mat.keys(), mat['K.-F./m'])

#            vals = line.split()
#             #20000.00000000 48000.00000000   96  170    1     0.0    1
#            if int(vals[6]) == 1:
#                freqs.append(float(vals[0]))

        return i+1, [eA, eB], readings

    with open(fileName, 'r') as fi:
        content = fi.readlines()

    data = pg.DataContainerERT()

    version = content[0].split()[0]

    if version != 'SIP256D':
        print("Warning  .. maybe wrong format .. until now this importer" +
              "only supports SIP256D. Pls. contact Carsten.", version)

    nReadings = 0
    readings = []
    freqs = []
    injections = []
    for i in range(len(content)):
        line = content[i].split('\n')[0]
        if '[Number of Readings]' in line:
            nReadings = int(line.split(']')[1])
            for r in range(nReadings):
                readings.append(content[i + 2 + r].split())
            i += nReadings + 2

        if '[Begin Layout]' in line:
            i, sensors = readSIP256Layout_(content, start=i+1,
                                           endStr='[End Layout]')
            for i, s in enumerate(sensors):
                data.createSensor([s, 0, 0])

        if '[Begin FrequencyParameter]' in line:
            i, freqs = readSIP256Freqs_(content, start=i+1,
                                        endStr='[End FrequencyParameter]')

        if 'Reading:' in line:
            i, [eA, eB], inj = readReading_(content, len(freqs), start=i,
                                            endStr='Reading')
            if eA != eB:
                injections.append([[eA, eB], inj])

    print(freqs)
    nRU = len(readings[0]) - 3
    nElecs = nRU + 1

    count = 0
    data.add('Ki', pg.Vector(0))

    def k(a, b, m, n):
        try:
            a *= 4
            b *= 4
            m *= 4
            n *= 4
            return abs(1./(1./(2*np.pi) * (1./abs(m-a) - 1./abs(m-b) -
                                           1./abs(n-a) + 1./abs(n-b))))
        except:
            return np.nan

    for i, reading in enumerate(injections):
        eA = reading[0][0]
        eB = reading[0][1]
        AB = int((eB-eA))

        config = np.array(readings[i], int)
        print(i+1, config)

        for j, meas in enumerate(reading[1]):
            eM = j + 1
            eN = j + 2

            if eN in config[3:]:
                eM = (eN-1)

                if eA == 9 and eB == 13 and eN == 3:
                    print("File Format unknown .. hack here! ask Tino!")
                    eN = 5
                elif eA == 9 and eB == 13 and eN == 4:
                    print("File Format unknown .. hack here! ask Tino!")
                    eN = 5
                else:
                    eN = min(nElecs-1, eM + AB)

#                print("\t", i+1, eA, eB, eM, eN)
#                #eN = j + 1
#                print(eA, eB, eM, eN, meas['K.-F./m'][0], readings[i][j+1])
#
#            print(reading['K.-F./m'])
#            if not np.isnan(meas['K.-F./m'][0]):
            if (abs(k(eA, eB, eM, eN) - meas['K.-F./m'][0]) > 1.3):

                print(eA, eB, eM, eN, k(eA, eB, eM, eN), meas['K.-F./m'][0],
                      readings[i][j+1])

            if not np.isnan(meas['K.-F./m'][0]):
                data.createFourPointData(count, eA-1, eB-1, eM-1, eN-1)
                data('Ki')[count] = meas['K.-F./m'][0]

                for f in range(len(meas['Frequency/Hz'])):
                    freq = meas['Frequency/Hz'][f]

                    rhoaName = 'rhoa:'+str(freq)
                    if not data.exists(rhoaName):
                        data.add(rhoaName, pg.Vector(2))

                    phaseName = 'pa:'+str(freq)
                    if not data.exists(phaseName):
                        data.add(phaseName, pg.Vector(2))

                    iName = 'i:'+str(freq)
                    if not data.exists(iName):
                        data.add(iName, pg.Vector(2))

                    data(phaseName)[count] = meas['PA/'][f]
                    data(rhoaName)[count] = meas['RA/Ohmm'][f]
                    data(iName)[count] = meas['IA/mA'][f]

                count += 1
    data.set("k", pb.geometricFactors(data))
    data.resize(count)
    return data


if __name__ == "__main__":
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        print(filename)
        ext = filename[filename.rfind('.'):].lower()
        fun = bertImportDataFileSuffixesDict[ext][1]
        importFunction = 'import' + fun
        data = eval(importFunction)(filename)
#        data = importData(datafile)
        print(data)
        pb.show(data)
