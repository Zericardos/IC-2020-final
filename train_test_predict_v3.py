#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:10:49 2020

@author: ricardo
"""
import pandas as pd
from func import prize_random, rec_all


wd_table = pd.read_csv('wd_table_final.csv', dtype = {'source_id' : str})
os_table = pd.read_csv('os_table_final.csv', dtype = {'source_id' : str})
#Mesma coisa, mas para a classificação binária
targ = pd.read_csv('targ.csv', dtype = {'source_id' : str})

'''
Features S-PLUS. Os 12 primeiros filtros do S-PLUS e depois diferença entre eles.
Banda larga com banda estreita no mesmo intervalo e depois diferença entre filtros
de banda larga 
'''

ft_gaia = pd.read_csv('ft_gaia_v3.csv')
targ_set_gaia, pred_gaia = prize_random(ft_gaia, targ, title = \
                                          'GAIA', save = True)

tfp_tfn_gaia = rec_all(wd_table, os_table, targ_set_gaia, pred_gaia)
tfp_tfn_gaia.to_csv('tfp_tfn_gaia_v3.csv')


#DataFrame com as features do Gaia e S-PLUS
feat_gaia_splus = pd.read_csv('ft_gaia_splus_r_v3.csv')
targ_set_gaia_splus, pred_gaia_splus = prize_random(feat_gaia_splus, targ, title = \
                                          'GAIA S-PLUS', save = True)
tfp_tfn_gaia_splus = rec_all(wd_table, os_table, targ_set_gaia_splus, pred_gaia_splus)
tfp_tfn_gaia_splus.to_csv('tfp_tfn_gaia_splus_r_v3.csv')

#DataFrame só com as features diferenças de magnitude em relação a banda r do S-PLUS
ft_splus_r = pd.read_csv('ft_splus_r_v3.csv')
targ_set_dif_splus_r, pred_dif_splus_r = prize_random(ft_splus_r, targ, title = \
                                          'S-PLUS r', save = True)
tfp_tfn_splus_r = rec_all(wd_table, os_table, targ_set_dif_splus_r, pred_dif_splus_r)
tfp_tfn_splus_r.to_csv('tfp_tfn_splus_r_v3.csv')

