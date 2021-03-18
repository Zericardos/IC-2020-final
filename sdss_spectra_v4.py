#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:00:29 2020

@author: ricardo
"""


import pandas as pd
from func import sdss_spec

savein = True

# Bastava criar uma função em que baixasse uma lista de espectros com base na
# lista de arquivos pelos quais obtivemos plate, fiberID, MJD

tfp_tfn_splus_r = pd.read_csv('tfp_tfn_splus_r_v3.csv')
lista_pmf_splus_r = sdss_spec(tfp_tfn_splus_r, 'specs S-PLUS r', save=savein)

tfp_tfn_gaia = pd.read_csv('tfp_tfn_gaia_v3.csv')
lista_pmf_gaia = sdss_spec(tfp_tfn_gaia, 'specs GAIA', save=savein)

tfp_tfn_gaia_splus = pd.read_csv('tfp_tfn_gaia_splus_r_v3.csv')
lista_pmf_gaia_splus = sdss_spec(tfp_tfn_gaia_splus, 'specs GAIA + S-PLUS r',
                                 save=savein)
