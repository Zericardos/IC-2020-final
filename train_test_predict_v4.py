#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:10:49 2020

@author: ricardo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from func import prize_random
from func import true_falses_distribution

wd_table = pd.read_csv("wd_table_final.csv", dtype={"source_id": str})
os_table = pd.read_csv("os_table_final.csv", dtype={"source_id": str})
targ = pd.read_csv("targ.csv", dtype={"source_id": str})

# Features S-PLUS. Os 12 primeiros filtros do S-PLUS e depois diferença entre
# eles. Banda larga com banda estreita no mesmo intervalo e depois diferença
# entre filtros de banda larga

nrange = [1000]
# nrange = [4]
for ntreinamentos in nrange:

    ft_gaia = pd.read_csv("ft_gaia_v4.csv")

    feat_set_gaia, targ_set_gaia, rfc_gaia, pred_gaia, accuracy_gaia =\
        prize_random(
            ft_gaia,
            targ,
            prize=ntreinamentos,
            title="GAIA " + str(ntreinamentos),
            save=True,
            return_all="yes",
        )

    tf_gaia = true_falses_distribution(
        wd_table, os_table, targ_set_gaia, pred_gaia)

    tf_gaia.hist()
    plt.savefig("GAIA " + str(ntreinamentos) + " histogram.pdf")
    plt.close("all")

    indexes = ["mean", "median", "std"]
    ms_gaia = pd.DataFrame(
        [tf_gaia.mean(axis=0), tf_gaia.median(axis=0), tf_gaia.std(axis=0)],
        index=indexes,
    )

    feat_splus_r = pd.read_csv("ft_splus_r_v4.csv")

    (
        feat_set_splus_r,
        targ_set_splus_r,
        rfc_splus_r,
        pred_splus_r,
        accuracy_splus_r,
    ) = prize_random(
        feat_splus_r,
        targ,
        prize=ntreinamentos,
        title="S-PLUS r " + str(ntreinamentos),
        save=True,
        return_all="yes",
    )

    tf_splus_r = true_falses_distribution(
        wd_table, os_table, targ_set_splus_r, pred_splus_r
    )

    tf_splus_r.hist()
    plt.savefig("S-PLUS r " + str(ntreinamentos) + " histogram.pdf")
    plt.close("all")
    ms_splus_r = pd.DataFrame(
        [tf_splus_r.mean(axis=0), tf_splus_r.median(
            axis=0), tf_splus_r.std(axis=0)],
        index=indexes,
    )

    feat_gaia_splus_r = pd.read_csv("ft_gaia_splus_r_v4.csv")

    (
        feat_set_gaia_splus_r,
        targ_set_gaia_splus_r,
        rfc_gaia_splus_r,
        pred_gaia_splus_r,
        accuracy_gaia_splus_r,
    ) = prize_random(
        feat_gaia_splus_r,
        targ,
        prize=ntreinamentos,
        title="GAIA S-PLUS r " + str(ntreinamentos),
        save=True,
        return_all="yes",
    )

    tf_gaia_splus_r = true_falses_distribution(
        wd_table, os_table, targ_set_gaia_splus_r, pred_gaia_splus_r
    )

    tf_gaia_splus_r.hist()
    plt.savefig("GAIA S-PLUS r " + str(ntreinamentos) + " histogram.pdf")
    plt.close("all")

    ms_gaia_splus_r = pd.DataFrame(
        [
            tf_gaia_splus_r.mean(axis=0),
            tf_gaia_splus_r.median(axis=0),
            tf_gaia_splus_r.std(axis=0),
        ],
        index=indexes,
    )

    tps = pd.DataFrame(
        data=(ms_gaia.tp, ms_splus_r.tp, ms_gaia_splus_r.tp),
        index=["GAIA", "S-PLUS r", "GAIA + S-PLUS r"],
    )

    tns = pd.DataFrame(
        data=(ms_gaia.tn, ms_splus_r.tn, ms_gaia_splus_r.tn),
        index=["GAIA", "S-PLUS r", "GAIA + S-PLUS r"],
    )

    fns = pd.DataFrame(
        data=(ms_gaia.fn, ms_splus_r.fn, ms_gaia_splus_r.fn),
        index=["GAIA", "S-PLUS r", "GAIA + S-PLUS r"],
    )

    fps = pd.DataFrame(
        data=(ms_gaia.fp, ms_splus_r.fp, ms_gaia_splus_r.fp),
        index=["GAIA", "S-PLUS r", "GAIA + S-PLUS r"],
    )

    legenda = ["GAIA", "S-PLUS r", "GAIA + S-PLUS r"]
    xlabel = "set"
    tfs = [tps, tns, fns, fps]
    title = ["True Positive", "True Negative",
             "False Negative", "False Positive"]

    nfig = 3

    for i, l in enumerate(tfs):

        fig, ax = plt.subplots(len(indexes), 1, dpi=300, num=nfig, sharex=True)

        st = fig.suptitle(title[i])

        for j, k in enumerate(indexes):

            ax[j].scatter(np.arange(len(l)), l.iloc[:, j])
            ax[j].set_yticks(l.iloc[:, j].values)
            ax[j].set_yticklabels(l.iloc[:, j].values.round(3), fontsize=4)
            ax[j].set_ylabel(k)

        # plt.yticks(fontsize = 6)
        plt.xticks([0, 1, 2], legenda)
        fig.tight_layout()
        st.set_y(0.95)
        fig.subplots_adjust(top=0.9)
        nfig += 1
        fig.savefig(title[i] + " " + str(ntreinamentos) + " stats.png")
        plt.close("all")

# Para exportar as tabelas, desabilitar as aspas. Era melhor ter implementado
# função

"""
from func import rec_all

tfp_tfn_gaia = rec_all(wd_table, os_table, targ_set_gaia, pred_gaia)
tfp_tfn_gaia.to_csv('tfp_tfn_gaia_v4.csv')


# DataFrame com as features do Gaia e S-PLUS
feat_gaia_splus = pd.read_csv('ft_gaia_splus_r_v4.csv')
targ_set_gaia_splus, pred_gaia_splus = prize_random(
    feat_gaia_splus, targ, title='GAIA S-PLUS', save=True)
tfp_tfn_gaia_splus = rec_all(
    wd_table, os_table, targ_set_gaia_splus, pred_gaia_splus)
tfp_tfn_gaia_splus.to_csv('tfp_tfn_gaia_splus_r_v4.csv')

# DataFrame só com as features diferenças de magnitude em relação
# a banda r do S-PLUS

ft_splus_r = pd.read_csv('ft_splus_r_v4.csv')
targ_set_dif_splus_r, pred_dif_splus_r = prize_random(
    ft_splus_r, targ, title='S-PLUS r', save=True)
tfp_tfn_splus_r = rec_all(
    wd_table, os_table, targ_set_dif_splus_r, pred_dif_splus_r)
tfp_tfn_splus_r.to_csv('tfp_tfn_splus_r_v4.csv')
"""
