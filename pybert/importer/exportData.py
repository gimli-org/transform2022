# -*- coding: utf-8 -*-
import numpy as np


def exportData(data, filename, fmt='Gimli', verbose=False):
    """ Export datafile:

        supported: syscal -- syscal sequence
    """
    if fmt.lower() == 'gimli':
        data.save(filename)
    elif 'resecs' in fmt.lower():
        exportResecsAsciiFile(data, filename, verbose)
    elif 'res2dinv' in fmt.lower():
        exportRes2dInv(data, filename, verbose)
    elif 'syscal' in fmt.lower():
        writeSyscalSequence(data, filename, verbose)
    else:
        raise Exception("exportData format " + str(fmt) +
                        " not yet implemented. Please contact support.")
# def exportData(...)


def exportResecsAsciiFile(data, filename, verbose=False):
    """ export data as column-baed Ascii File using Resecs format """
    raise Exception("Not Implemented yet!")
# def exportResecsAsciiFile


def writeSyscalSequence(data, filename, verbose=False):
    """ export data container as syscal sequence (array definition) file """
    if verbose:
        print("writing syscal sequence file",  filename, " ...")

    fi = open(filename, 'w')
    fi.write('#\tX\tY\tZ\n')
    for i, e in enumerate(data.sensorPositions()):
        fi.write('%d\t%g\t%g\t%g\n' % (i+1, e[0], e[1], e[2]))

    fi.write('#\tA\tB\tM\tN\n')
    for i in range(data.size()):
        fi.write('%d\t%d\t%d\t%d\t%d\n' % (i+1, data('a')[i]+1, data('b')[i]+1,
                                           data('m')[i]+1, data('n')[i]+1))
    fi.close()
# def writeSyscalSequence(...)


def exportRes2dInv(data, filename="out.res2dinv", ar_idfy=11, sep='\t',
                   arrayName='mixed array', rhoa=False, verbose=False):
    """Save data file under res2dinv general array format."""
    x = [np.round(ii[0], decimals=2) for ii in data.sensorPositions()]
    y = [np.round(ii[1], decimals=2) for ii in data.sensorPositions()]
    if not np.any(y):
        y = [np.round(ii[2], decimals=2) for ii in data.sensorPositions()]

    dist = [np.sqrt((x[ii]-x[ii-1])**2 + (y[ii]-y[ii-1])**2)
            for ii in np.arange(1, len(y))]
    dist2 = [x[ii]-x[ii-1] for ii in np.arange(1, len(y))]
    print(min(dist), min(dist2))
    # %% check for resistance or resistivity
    if (data.allNonZero('r') or data.allNonZero('R')) and not rhoa:
        datType = '1'
        res = data('r')
    elif data.allNonZero('rhoa'):
        datType = '0'
        res = data('rhoa')
    else:
        raise BaseException("No valid apparent resistivity data!")
    # %% check for ip
    # %% write res2Dinv file
    with open(filename, 'w') as fi:
        fi.write(arrayName+'\n')
        fi.write(str(np.round(min(dist2), decimals=2))+'\n')
        fi.write(str(ar_idfy)+'\n')
        fi.write('0\n')
        fi.write('Type of resistivity data (1=resistance,0=resistivity) \n')
        fi.write(datType+'\n')
        fi.write(str(len(data('r')))+'\n')
        fi.write('2'+'\n')
        if data.allNonZero('ip'):
            fi.write('1\n')
            fi.write('Chargeability\n')
            fi.write('mV/V\n')
            fi.write('0.01,3.9\n')  # delay/integration time
            ip = data('ip')
            if data.allNonZero('r'):
                lines = ['4' + sep +
                         str(x[int(data('a')[oo])]) + sep +
                         str(y[int(data('a')[oo])]) + sep +
                         str(x[int(data('b')[oo])]) + sep +
                         str(y[int(data('b')[oo])]) + sep +
                         str(x[int(data('m')[oo])]) + sep +
                         str(y[int(data('m')[oo])]) + sep +
                         str(x[int(data('n')[oo])]) + sep +
                         str(y[int(data('n')[oo])]) + sep +
                         str(res[oo]) + sep + str(ip[oo])
                         for oo in range(len(data('a')))]
            else:
                lines = ['4' + sep +
                         str(x[int(data('a')[oo])]) + sep +
                         str(y[int(data('a')[oo])]) + sep +
                         str(x[int(data('b')[oo])]) + sep +
                         str(y[int(data('b')[oo])]) + sep +
                         str(x[int(data('m')[oo])]) + sep +
                         str(y[int(data('m')[oo])]) + sep +
                         str(x[int(data('n')[oo])]) + sep +
                         str(y[int(data('n')[oo])]) + sep + str(ip[oo])
                         for oo in range(len(data('a')))]
        else:
            fi.write('0\n')
            lines = ['4' + sep +
                     str(x[int(data('a')[oo])]) + sep +
                     str(y[int(data('a')[oo])]) + sep +
                     str(x[int(data('b')[oo])]) + sep +
                     str(y[int(data('b')[oo])]) + sep +
                     str(x[int(data('m')[oo])]) + sep +
                     str(y[int(data('m')[oo])]) + sep +
                     str(x[int(data('n')[oo])]) + sep +
                     str(y[int(data('n')[oo])]) + sep + str(res[oo])
                     for oo in range(len(data('a')))]

        fi.writelines("%s\n" % l for l in lines)
# def exportRes2dInv(...)
