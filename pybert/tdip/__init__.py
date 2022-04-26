#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Time-domain induced polarization (TDIP) Data Manager."""

from .tdipdata import TDIP
from .mipmodelling import (DCIPMModelling, DCIPSeigelModelling,
                           ColeColeTD, DCIPMSmoothModelling,
                           CCTDModelling)

TDIPdata = TDIP  # backward compatibility

__all__ = ['TDIP', 'TDIPdata', 'HIRIP', 'DCIPMModelling',
           'DCIPSeigelModelling', 'ColeColeTD']
