#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:13:01 2020

@author: ricardo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:40:12 2020

@author: ricardo
"""
import pandas as pd, numpy as np

filename = (
    "/home/ricardo/Documents/fisica/ic/2020/catalogues/originais_mais_que não uso/"
    + "os_dr1_splus_gaia_dr2_sdss_dr9.fits"
)

from astropy.table import Table as tb

"""Pegar WDs já classificadas e os índices dos objetos sem classificação.
Deles, pegar mjd, plate e fiber e então pegar as imagens dos espectros
para confirmar a classificação em WD ou não.
"""
""" 
t = tb.read(filename).to_pandas()
t.subClass = t.subClass.str.decode("utf-8")
ind = t.subClass.str.rstrip()
index = np.array([], dtype = int)
white_index = index.copy()

#pegar as que foram classificadas como WD
for i,j in enumerate(ind):

    if "QSO" in j:

        index = np.append(index, i)
"""
