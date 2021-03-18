#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:10:10 2020

@author: ricardo
"""

import numpy as np
import pandas as pd
import time
from func import gen_df_features_targets
from func import genft_df_gaia
from func import concat
from func import gen_ft_dif_splus_x

start_time = time.perf_counter()

wd_table = pd.read_csv("wd_table_final.csv")
os_table = pd.read_csv("os_table_final.csv")

wd_table = wd_table.iloc[np.unique(wd_table["source_id"],
                                   return_index=True)[1]]
os_table = os_table.iloc[np.unique(os_table["source_id"],
                                   return_index=True)[1]]

wd_table = wd_table.loc[wd_table.ndet_aper == 12]
os_table = os_table.loc[os_table.ndet_aper == 12]

wd_table.to_csv("wd_table_final.csv", index=False)
os_table.to_csv("os_table_final.csv", index=False)

# Pegar features do S-PLUS, diferenças de magnitude em relação aos filtros
# largos e estreitos de suas respectivas bandas. Targets tem classificação
# binária, se é WD ou Other Stars e sua respectiva source_id


features_splus1, targets_data1 = gen_df_features_targets(wd_table, splus=True)
features_splus2, targets_data2 = gen_df_features_targets(os_table, splus=True)

# só pega as 3 features do gaia, rp,bp e paralaxe e os índices de objetos que
# não tem nan ou paralaxes positivas naquelas features

features_gaia1, non_null_index1 = genft_df_gaia(wd_table, normal=True)
features_gaia2, non_null_index2 = genft_df_gaia(os_table, normal=True)

# Atualiza os DataFrames com aqueles índices

features_splus1, targets_data1 = (
    features_splus1.iloc[non_null_index1],
    targets_data1.iloc[non_null_index1],
)

# Mesmo tratamento
features_splus2, targets_data2 = (
    features_splus2.iloc[non_null_index2],
    targets_data2.iloc[non_null_index2],
)
# Mesmo tratamento
features_gaia1 = features_gaia1.iloc[non_null_index1]
features_gaia2 = features_gaia2.iloc[non_null_index2]

# Junta DataFrame de features S-PLUS: de WD com Other Stars e salva
feat_splus = concat(features_splus1, features_splus2)
# Mesmo tratamento, mas para features do GAIA
ft_gaia = concat(features_gaia1, features_gaia2)

# Mesmo tratamento, mas para a classificação binária
targ = concat(targets_data1, targets_data2)

# DataFrame com as features do Gaia e S-PLUS
ft_gaia_splus = pd.concat([ft_gaia, feat_splus], axis=1)
# DataFrame só com as features diferenças de magnitude
# em relação a banda r do S-PLUS

ft_splus_r = gen_ft_dif_splus_x(ft_gaia_splus)
ft_gaia_splus_r = pd.concat([ft_gaia, ft_splus_r], axis=1)

targ.to_csv("targ.csv", index=False)
ft_gaia_splus_r.to_csv("ft_gaia_splus_r_v3.csv", index=False)
ft_splus_r.to_csv("ft_splus_r_v3.csv", index=False)
ft_gaia.to_csv("ft_gaia_v3.csv", index=False)

# DataFrame acima com as features do GAIA

# DataFrame com as magnitudes absolutas do S-PLUS, usada paralaxe do GAIA

time_spent = time.perf_counter() - start_time

print("Time spent = ", time_spent)
