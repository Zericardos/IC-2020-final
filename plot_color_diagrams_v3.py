#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:09:50 2020

@author: ricardo
"""
import pandas as pd
from func import plot_color_diagrams, genft_df_gaia

MAGin = False
HEADER_X = ['ujava_aper', 'f378_aper', 'f395_aper', 'f410_aper', 'f430_aper',\
                       'g_aper', 'f515_aper', 'r_aper', 'f660_aper', 'i_aper',\
                           'f861_aper', 'z_aper', 'phot_bp_mean_mag',\
                               'phot_rp_mean_mag']

HEADER_Y = ['MAG G']

tfp_tfn_gaia_splus_r = pd.read_csv('tfp_tfn_gaia_splus_r_v3.csv')
tfp_tfn_gaia_splus_r  = pd.concat([genft_df_gaia(tfp_tfn_gaia_splus_r,\
                                                     normal = True, Index = False),\
                                       tfp_tfn_gaia_splus_r], axis = 1)
plot_color_diagrams(tfp_tfn_gaia_splus_r, 'GAIA + S-PLUS r', head_x = HEADER_X,\
                    head_y = HEADER_Y, MAG = MAGin)
         
tfp_tfn_gaia = pd.read_csv('tfp_tfn_gaia_v3.csv')
tfp_tfn_gaia  = pd.concat([genft_df_gaia(tfp_tfn_gaia, normal = True, Index = False),\
                                       tfp_tfn_gaia], axis = 1)

plot_color_diagrams(tfp_tfn_gaia, 'GAIA', head_x = HEADER_X, head_y = HEADER_Y,\
                    MAG = MAGin)
                   
tfp_tfn_splus_r = pd.read_csv('tfp_tfn_splus_r_v3.csv')
tfp_tfn_splus_r  = pd.concat([genft_df_gaia(tfp_tfn_splus_r, normal =\
                                                 True, Index = False),\
                                       tfp_tfn_splus_r], axis = 1)
plot_color_diagrams(tfp_tfn_splus_r, 'S-PLUS r', head_x = HEADER_X, head_y = HEADER_Y,\
                    MAG = MAGin)
