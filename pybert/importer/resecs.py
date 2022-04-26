# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np
import pygimli as pg


def readResecsAsciiStreamingData(filename):
    """Read continuous (streaming) data from Resecs instrument."""
    debug = True

    swatch = pg.Stopwatch(True)

    fi = open(filename, 'r')

    # firstIsHeader = False  # not used
    data = list()
    token = []
    # search for header line
    line = fi.readline()
    line = line.replace('"', '').replace(' N', 'N').replace(' S', 'S').replace(
        ' W', 'W').replace(' E', 'E').replace(' UTC', '')
    line = line.split()
    if len(line) > 0:
        # if first character of the first line is alpha, we found a header line
        if line[0][0].isalpha():
            token = line
            # firstIsHeader = True  # not used!
        else:  # we assume defaults:
            if len(line) == 8:
                token = ['Lat', 'Lon', 'Alt', 'I', 'U', 'D', 'Channel', 'Time']
            elif len(line) == 20:
                token = ['C1(x)', 'C1(y)', 'C1(z)', 'C2(x)', 'C2(y)', 'C2(z)',
                         'P1(x)', 'P1(y)', 'P1(z)', 'P2(x)', 'P2(y)', 'P2(z)',
                         'Lat', 'Lon', 'Alt', 'I', 'U', 'D', 'Channel', 'Time']
            else:
                raise Exception('Cannot interpret line: ', len(line), line)
            fi.seek(0)

        for t in token:
            data.append(list())
    else:
        raise Exception('The given datafile is not valid: ', filename)

    if debug:
        print("t=", swatch.duration(True))

#    m = g.RMatrix()
#    g.loadMatrixCol( m, filename )
#    print m
#    #data=np.genfromtxt( fi )
#    print data
    for line in fi:
        if 'na' in line:
            continue

        line = line.replace(' N', 'N').replace(' S', 'S').replace(
            ' W', 'W').replace(' E', 'E').replace('"', '').replace(' UTC', '')
        vals = line.split('\n')[0].split('\r')[0].split('\t')
        if len(vals) == len(token):
            for i, v in enumerate(vals):
                if token[i] == 'U':
                    data[i].append(float(v))
                elif token[i] == 'I':
                    data[i].append(float(v))
                elif token[i] == 'D':
                    data[i].append(float(v))
                elif token[i] == 'Channel':
                    data[i].append(int(v))
                elif token[i] == 'C1(x)':
                    data[i].append(float(v))
                elif token[i] == 'Time':
                    try:
                        t = datetime.strptime(v, '%H:%M:%S')
                        data[i].append(t.hour*3600 + t.minute*60 + t.second)
                    except Exception as e:
                        print(e, v, t)
                else:
                    data[i].append(v)
        else:
            raise Exception('input format unknown ', len(vals), vals, token,
                            len(token), line)

    fi.close()

    if debug:
        print("t=", swatch.duration(True))

    d = dict()
    for i, t in enumerate(token):
        if t == 'U' or t == 'I' or t == 'Channel' or t == 'C1(x)' or t == 'D':
            d[t] = pg.asvector(data[i])
        else:
            d[t] = data[i]

    if 'C1(x)' not in d:
        d['C1(x)'] = pg.Vector(len(data[0]), 0.0)

    if debug:
        print("t=", swatch.duration(True))

    return d


def convGraeberLonLat(lo0, la0):
    """convert 5411.97569 N    1339.50120 E  >> 51.770517       6.311200"""
    e = 1.0
    n = 1.0

#    if la0 < 180.0: ## !! format = 51.770517       6.311200
#    #x0,y0=float(sp[ilon]),float(sp[ilat])
#    #fx=np.floor(x0/100)
#    #fy=np.floor(y0/100)
#    x2, y2 = proj( lo0, la0 )

    if type(lo0) == str:
        if lo0.find('W') > -1:
            e = -1.0
        lo0 = float(lo0.replace('E', '').replace('W', ''))

    if type(la0) == str:
        if la0.find('S') > -1:
            n = -1.0
        la0 = float(la0.replace('N', '').replace('S', ''))

    if la0 < 180.0:  # !! format = 51.770517       6.311200
        return lo0, la0

    lod = np.floor(lo0 / 100.)
    lad = np.floor(la0 / 100.)
    lo = (lo0 - lod * 100.) / 60. + lod
    la = (la0 - lad * 100.) / 60. + lad

    return lo * e, la * n


def resecsLatLonToUtm(lat, lon, proj):
    """WARNING PLEASE CHECK LAT vs.LON"""

    if len(lat) != len(lon):
        raise Exception('latLonToUtm( lat, lon) sizes differ: ',
                        len(lat), len(lon))

    x = pg.Vector(len(lon))
    y = pg.Vector(len(lat))

    for i, v in enumerate(lat):
        lonlat = convGraeberLonLat(lon[i], lat[i])
        x2, y2 = proj(lonlat[0], lonlat[1])

        x[i] = x2
        y[i] = y2

    return x, y
