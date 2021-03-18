#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:37:49 2020

@author: ricardo
"""


from astropy.table import vstack, Table as tb

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Make tables


def read_all_measures(table_name, seed=0):
    """Read table from S-PLUS with astropy.Table module

    Then, take all objects with all aper-type magnitudes measures

    Parameters:
    ----------
    table_name : str
        file name with it's adress and extension

    Returns
    -------
    astropy table

    """
    # read
    table = tb.read(table_name)

    return table[table['ndet_aper'] == 12]


def read_shuffle_cut(table_name: str, newtable_size: int, seed=0):
    """Read the astropy table given a name/address with all magnitude measures.

    Then pick objects randomly, given a random seed, to a new table with
    newtable_size


    Parameters
    ----------
    table_name : str
        Name/address the original table to read
    newtable_size : int
        Size of new table

    Returns
    -------
    astropy.table
        New table of size newtable_size with objects from read table.
    """
    def unique_random(table, ind_range, subset_size: int):
        """
        Choose randomly subset_size elements from table

        Parameters
        ----------
        table : astropy.table
            Original table to get objects
        newtable : astropy.table
            Table to stack objects vertically downward
        ind_range : ndarray
            minimum and maximum indices to catch in table
        subset_size : int
            number of objects to catch in table between minimum and maximum
            indices
        seed: int
            set random consistently
        Returns
        -------
        astropy.table
            modified table with subset_size
        """
        np.random.seed(seed)
        # choose indices
        rand_ind = np.random.choice(
            np.arange(ind_range[0], ind_range[1]), subset_size, replace=False)
        # assist table
        return table[rand_ind]

    def ntble(table, table_size: int, newtable_size: int, N: int, n: int, r=0,
              seed=0):
        """Generate a newtable with table_size elements from table.

        Table is divided into N smaller tables and unique_random function
        takes randomly n elements from each and stack vertically to it

        Parameters
        ----------
        table : astropy.table
            original table to catch objects
        table_size : int
            size of the table. len(table)
        newtable_size : TYPE
            size of the newtable. len(newtable)
        N : int
            Number of small tables to catch objects from them
        n : int
            Number of objects to catch from each small table
        seed: int
            set random consistently
        r : int, optional
            Size of residual small table, if any. The default value is 0.

        Returns
        -------
        newtable : astropy.table
            Table with table_size
        """
        # assist small table
        tab_aux = table[:newtable_size]
        # seed to set random consistently
        np.random.seed(seed)
        # pick up indices, ndarray of size n
        rand_ind = np.random.choice(np.arange(newtable_size), n, replace=False)
        # first step of newtable is complete
        newtable = tab_aux[rand_ind]

        # for other small tables, just stack vertically downward in a loop
        for i in range(N - 1):
            i += 1
            ind_range = (i * newtable_size, (i + 1) * newtable_size)
            tab_aux = unique_random(table, ind_range, n)
            newtable = vstack([newtable, tab_aux])
            # add assist table to newtable
        # if there a residual small table, to complete the task
        if r != 0:
            i += 1
            # note the size of small table and number objects are different
            ind_range = (i * newtable_size + 1, T)
            tab_aux = unique_random(table, ind_range, r)
            newtable = vstack([newtable, tab_aux])
        # newtable is complete
        return newtable

    # read table
    table = read_all_measures(table_name)
    # to simplify, compress variable name
    s = newtable_size
    # size of table
    T = len(table)
    # condition to take new table with size s
    if (s < len(table)) and s != 0:

        # number of small tables to pick objects from each
        N = T // s
        # if number of small tables are smaller than size of every small table
        # (subset)
        if N < s:
            # size of subset to cut and shuffle
            n = s // N
            # residual subset given a s
            r = s % N
            # with key parameters, create a new table
            newtable = ntble(table, T, s, N, n, r=r)
        # if number of small tables are greater than size of every small table
        # (subset)
        else:
            # resize the number of subset to size of every subset
            N = s
            # from each subset, take only one object
            n = 1
            # resize subset size
            sub_s = T // N
            # create newtable
            newtable = ntble(table, T, sub_s, N, n)

    else:  # if newtable_size == 0 or it's greater than table size
        newtable = table
    # random the newtable
    S = len(newtable)
    ind_range = (0, S)
    newtable = unique_random(newtable, ind_range, S)

    return newtable

# Filter the tables


def gen_df_features_targets(data, splus=False, normal=True, targets=True):
    """Pega as features do S-PLUS.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        Pode ser cor: splus = True, só as mags:
            normal = True ou as duas splus e normal = True.
            Colocar splus = False e normal = False implica por normal = True.
            Retoma também outro DataFrame com classificação binária e source id
            do GAIA.

    splus : bool, optional
        DESCRIPTION. Retoma as magnitudes aper. The default is False.

    normal : TYPE, optional
        Se True, retoma um DataFrame com as diferenças de magnitude dos filtros
        largos com os filtros de bandas estreitas que operam na mesma banda
        daquele filtro largo. Por fim, há diferenças de magnitude em relação ao
        filtro r. The default is True.

    Returns
    -------
    features_df : pandas.core.frame.DataFrame
        DataFrame com as features desejadas
    targets_df : pandas.core.frame.DataFrame
        DataFrame com duas colunas: Se é ou não WD. Se não é é 'Other Stars'.
        Source id do GAIA

    """
    size_df = len(data)  # df : dataframe, tb : table

    if normal is False and splus is False:
        normal = True

    if normal is True:

        header = ['f378 - ujava', 'f395 - g', 'f410- g', 'f430 - g',
                  'f515 - g', 'r - f660', 'f861 - z', 'ujava - r', 'g - r',
                  'r - i', 'r - z']
        # header = pd.DataFrame(header)

        features_df = pd.DataFrame(columns=header)

        # features, color index
        features_df.iloc[:, 0] = data['f378_aper'] - data['ujava_aper']
        features_df.iloc[:, 1] = data['f395_aper'] - data['g_aper']
        features_df.iloc[:, 2] = data['f410_aper'] - data['g_aper']
        features_df.iloc[:, 3] = data['f430_aper'] - data['g_aper']
        features_df.iloc[:, 4] = data['f515_aper'] - data['g_aper']
        features_df.iloc[:, 5] = data['r_aper'] - data['f660_aper']
        features_df.iloc[:, 6] = data['f861_aper'] - data['z_aper']
        features_df.iloc[:, 7] = data['ujava_aper'] - data['r_aper']
        features_df.iloc[:, 8] = data['g_aper'] - data['r_aper']
        features_df.iloc[:, 9] = data['r_aper'] - data['i_aper']
        features_df.iloc[:, 10] = data['r_aper'] - data['z_aper']

    # create targets
    targets_df = pd.DataFrame(columns=['Type', 'source_id'])

    if splus is True:

        header2 = ['ujava_aper', 'f378_aper', 'f395_aper', 'f410_aper',
                   'f430_aper', 'g_aper', 'f515_aper', 'r_aper', 'f660_aper',
                   'i_aper', 'f861_aper', 'z_aper']

        features2 = pd.DataFrame(columns=header2)

        for i, j in enumerate(header2):

            features2.iloc[:, i] = data[j]

        try:
            features_df = pd.concat([features2, features_df], axis=1)
        # Exceção deve ser específica, nunca geral
        except:
            features_df = features2

    if targets is False:

        return features_df
    else:
        try:  # if they are wd
            # check
            data['Type']
            targets_df.iloc[:, 0] = np.repeat('WD', size_df)
            targets_df.iloc[:, 1] = data['source_id'].astype(str)
        # Exceção deve ser específica, nunca geral
        except:
            targets_df.iloc[:, 0] = np.repeat('Other Stars', size_df)
            targets_df.iloc[:, 1] = data['source_id'].astype(str)

        return features_df, targets_df


def genft_df_gaia(data, null=True, positive_parallax_only=True,
                  normal=False, Index=True):
    """Retorna um DataFrame com as features do GAIA

    Pode retornar outro DataFrame com os índices dos objetos que se encaixam
    nas condições pedidas, Index = True.

    Pode pegar todos os objetos com paralaxe ou tomar só os com valores
    positivos, positive_parallax_only = True

    Pega também só objetos que tenham valores não nulos(nan) nas medidas
    pedidas, null = True

    Também pode dar só valores bp, rp, parallax, normal = False ou bp - rp, e
    MAG G normal = True


    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame com medidas do GAIA.
    null : bool, optional
        Pega também só objetos que tenham valores não nulos(nan) nas medidas
        pedidas,. The default is True.
    positive_parallax_only : bool, optional
        DESCRIPTION. The default is True.
    normal : bool, optional
        Dar só valores bp, rp, parallax, normal = False ou bp - rp, e MAG G.
        The default is False.
    Index : bool, optional
        Retorna outro DataFrame com os indices dos objetos que se encaixam nas
        condições pedidas. The default is True.

    Returns
    -------
    Pandas DataFrame
        Pode retornar dois, um com as features pedidas e outro
        com os índices originais dos objetos selecionados

    """
    def gaia_mag_error(flux_error, flux_mean):

        relative_error = 2.5 * flux_error / (np.log(10) * flux_mean)
        return relative_error

    header = ['phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax']

    features_df = pd.DataFrame(columns=header)

    if normal is False:
        # features
        for i, j in enumerate(header):

            features_df.iloc[:, i] = data[j]

    else:
        features_df = pd.DataFrame(columns=['bp - rp'])
        # features
        features_df = pd.concat([features_df, ABS_MAG(data,
                                                      ['phot_g_mean_mag'],
                                                      data['parallax'])],
                                axis=1)

        features_df.rename(columns={'PHOT_G_MEAN_MAG': 'MAG G'}, inplace=True)

        features_df['bp - rp'] = data['phot_bp_mean_mag'] - \
            data['phot_rp_mean_mag']

    if null is True:

        indexn = ~ pd.isnull(features_df).any(1)
        indexn = np.where(indexn)

        if positive_parallax_only is True:

            indexp = data['parallax'] > 0
            indexp = np.where(indexp)

            index = np.intersect1d(indexn, indexp)
            if Index is True:
                return features_df, index
            else:
                return features_df

        else:
            if Index is True:
                return features_df, indexn
            else:
                return features_df

    else:

        if positive_parallax_only is True:

            indexp = features_df > 0

            if Index is True:
                return features_df, indexp

            else:
                return features_df

        else:
            return features_df


def concat(data1, data2, seed_concat=0):
    """Concatena DataFrames ao outro e os embaralha.

    Os dois DataFrames devem ter o mesmo número de colunas.
    Parameters
    ----------
    data1 : Pandas DataFrame

    data2 : Pandas DataFrame

    seed_concat : int, optional
        Seed para embaralhar os dois DataFrames. The default is 0.

    Returns
    -------
    data : Pandas DataFrame
        DataFrame concatenado

    """
    data = pd.concat([data1, data2])
    data = data.sample(frac=1, random_state=seed_concat).reset_index(drop=True)

    return data


def gen_ft_dif_splus_x(data, dif='r'):
    """Retorna um DataFrame com as cores em relação ao filtro r do S-PLUS.

    Queria por para qualquer filtro, mas acho que tinha que por tantos ifs, que
    melhor deixar assim

    Parameters
    ----------
    data : Pandas DataFrame
        DESCRIPTION.
    dif : TYPE, optional
        DESCRIPTION. The default is 'r'.

    Returns
    -------
    features_dif : Pandas DataFrame
        Retorna as 11 features de cores em relação ao filtro r

    """
    header_r = ['u - r', 'f378 - r', 'f395 - r', 'f410 - r', 'f430 - r',
                'g - r', 'f515 - r', 'r - f660', 'r - i', 'r - f861', 'r - z']

    features_dif = pd.DataFrame(columns=header_r)

    x = data[dif + '_aper']

    # features, difference color index
    features_dif.iloc[:, 0] = data['ujava_aper'] - x
    features_dif.iloc[:, 1] = data['f378_aper'] - x
    features_dif.iloc[:, 2] = data['f395_aper'] - x
    features_dif.iloc[:, 3] = data['f410_aper'] - x
    features_dif.iloc[:, 4] = data['f430_aper'] - x
    features_dif.iloc[:, 5] = data['g_aper'] - x
    features_dif.iloc[:, 6] = data['f515_aper'] - x
    features_dif.iloc[:, 7] = x - data['f660_aper']
    features_dif.iloc[:, 8] = x - data['i_aper']
    features_dif.iloc[:, 9] = x - data['f861_aper']
    features_dif.iloc[:, 10] = x - data['z_aper']

    return features_dif


def ABS_MAG(DataFrame, header: list, parallax, new=True, bottom=False):
    """Retorna Dataframe com as magnitudes absolutas do DataFrame de entrada.

    Precisa especificar quais colunas vai querer transformar por meio do header
    Pode ou não retornar o mesmo DataFrame acrescido das magnitudes absolutas
    com parâmetro new e colocar ou não no final do DataFrame com bottom.
    A ordem das colunas do novo DataFrame será a ordem do header dado.

    Parameters
    ----------
    DataFrame : Pandas DataFrame
        As colunas de magnitude precisam ter nomes da lista do header.

    header : list
        Nome das magnitudes aparentes que serão convertidas em absolutas. Cada
        nome deve estar em string normal

    parallax : Pandas Core DataFrame
        Uma só coluna com o mesmo número de objetos do DataFrame. Deve estar na
        mesma ordem do Dataframe, i.e, a primeira paralaxe deve corresponder ao
        primeiro objeto do DataFrame.

    new : bool, optional
        Se True, retorna um novo DataFrame só com as magnitudes absolutas.
        Se False, retorna o input DataFrame acrescido das magnitudes absolutas.
        The default is True.

    bottom : bool, optional
        Só vale se new = False.
        Se bottom = False, retorna magnitudes absolutas nas primeiras colunas.
        Se bottom = True, retorna magnitudes absolutas nas últimas colunas.
        The default is False.

    Returns
    Pandas DataFrame com a magnitude absoluta pedida

    """
    MAGs = pd.DataFrame(columns=header)
    MAGs.columns = map(str.upper, header)
    parallax = DataFrame.parallax
    d = 1e3 / parallax
    D = - 5 * np.log10(d) + 5

    for i, j in zip(MAGs.columns, header):
        MAGs[i] = DataFrame[j] + D

    if new is True:
        return MAGs

    else:
        if bottom is False:
            MAGs = pd.concat([MAGs, DataFrame], axis=1)

        else:
            MAGs = pd.concat([DataFrame, MAGs], axis=1)

    return MAGs


def MAG(mags, parallax):
    """Cria um DataFrame com magnitudes absolutas

    Parameters
    ----------
    mags : Pandas DataFrame
        Magnitude aparentes
    parallax : Pandas DataFrame ou series


    Returns
    -------
    M : Pandas DataFrame

    """
    M = pd.DataFrame(columns=mags.columns)
    M.columns = map(str.upper, mags.columns)

    d = 1e3 / parallax
    D = - 5 * np.log10(d) + 5

    for i, j in zip(M.columns, mags.columns):
        M[i] = mags[j] + D

    return M


# Train test predict


def prize_random(features, targets, testsize=.2, prize=100,
                 title='Confusion matrix', plot=False, save=False,
                 return_all=False):
    """Pega os objetos com features e seus respectivos targets.

    Divide em conjuntos de treinamentos e testes. O tamanho padrão é 80% de
    treinamentos e 20% de testes, pode-se mudar o tamanho do conjutno de teste
    e treinamento será 1 - x%. Salva ou plota a Confusion Matrix
    Para cada treinamento/teste, é gerado uma seed que sorteará os conjuntos de
    treinamento/teste. O sorteio está vinculado a essa seed. Assim, se for
    comparar diferentes features, o sorteio será o mesmo para cada um.

    Parameters
    ----------
    features : Pandas DataFrame
        Conjunto com as features
    targets : Pandas DataFrame
        Só com as categorias, cada objeto deverá corresponder a cada objeto da
        feature.
    testsize : float, optional
        Tamanho em porcentagem do conjunto de teste, o conjunto de treinamento
        será seu complementar. Tem que ser entre 0 e 1!!! The default is .2.
    prize : int, optional
        Número de sorteios em que os treinamentos/testes são realizados.
        Quanto maior o núemro, mais demora. No final, pega-se o treinamento com
        maior acurácia. Em caso de empate, pega-se o último. É bom mudar essa
        parte pois a acurácia é feita no conjunto inteiro e as vezes se está
        interessado apenas em um subconjunto, e.g, só de WD. The default is 100
    title : str, optional
        Título da Confusion Matrix. The default is 'Confusion matrix'.
    plot : bool, optional
        Plotar ou não a Confusion Matrix. The default is False.
    save : bool, optional
        Salvar ou não a Confusion Matrix, se True, não plota.
        The default is False.
    return_all : True, optional
        Retorna todos as variáveis internas geradas na função como cada subcon-
        -junto de treinamento/teste, cada previsão. Cada conjunto classificador
        . The default is False.

    Returns por padrão, subconjuntos do conjunto que obteve melhor previsão
    -------
    features_train_set : Pandas DataFrame
        Features do conjunto de treinamento
    features_test_set : Pandas DataFrame
        Features do conjunto de teste
    targ_train_set : Pandas DataFrame
        Alvos do conjunto de treinamento.
    targ_test_set : Pandas DataFrame
        Alvos do conjunto de teste
    targ_pred_set : Pandas DataFrame
        Alvos que o algoritmo tentou prever

    """
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    trainsize = 1 - testsize
    feat_set = []
    targ_set = []
    rfc = []
    pred = []
    accuracy = []

    for train_seed in range(prize):

        # fset: features set, tset: target set, fgs: features gaia(all) + splus
        train_fset_fgs, test_fset_fgs, train_tset_fgs, test_tset_fgs =\
            train_test_split(features, targets, test_size=testsize,
                             train_size=trainsize, random_state=train_seed)

        rfc_i = RandomForestClassifier()

        x_train, x_test = train_fset_fgs, test_fset_fgs
        y_train, y_test = train_tset_fgs, test_tset_fgs

        rfc_i.fit(X=x_train, y=y_train.iloc[:, 0])

        pred_i = rfc_i.predict(x_test)

        accuracy_i = accuracy_score(y_test.iloc[:, 0], pred_i)

        feat_set.append([x_train, x_test]),\
            targ_set.append([y_train, y_test]),\
            rfc.append(rfc_i), pred.append(pred_i), accuracy.append(accuracy_i)

    max_accuracy_array = np.where(accuracy == max(accuracy))[0][-1]

    if save is True and plot is True:
        plot is False
        print('not plotted, figure saved')

    j = max_accuracy_array

    features_train_set = feat_set[j][0]
    features_test_set = feat_set[j][1]

    targ_train_set = targ_set[j][0]
    targ_test_set = targ_set[j][1]

    targ_pred_set = pd.DataFrame({'predicted': pred[j]})

    class_labels = list(set(targets.iloc[:, 0]))

    model_cm = confusion_matrix(
        y_true=targ_set[j][1].iloc[:, 0], y_pred=pred[j], labels=class_labels)

    fig = plt.figure(figsize=(21, 10))
    plot_confusion_matrix(model_cm, classes=class_labels,
                          normalize=False, title=title)

    if plot is True:
        plt.show()

    if save is True:

        plt.savefig(title, dpi=300)
        plt.clf()
        plt.close()

    elif plot is False:

        plt.clf()
        plt.close()

    if return_all is True:

        return features_train_set, features_test_set, targ_train_set,\
            targ_test_set, targ_pred_set

    elif return_all == 'yes':
        return feat_set, targ_set, rfc, pred, accuracy

    else:
        return targ_test_set, targ_pred_set


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Print and plot the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')


def rec_all(table1, table2, target_test, predicted, classe='WD'):
    """Recupera os objetos de previsão dada as tabelas originais.

    Divide um DataFrame em quatro classes, VP, FP, FN, VN

    Parameters
    ----------
    table1 : Pandas DataFrame de um duas classes de objetos, e.g WD ou OS

    table2 : Pandas DataFrame
        Da outra classe dos objetos previstos
    target_test : Pandas DataFrame
        Conjunto alvo de teste
    predicted : Pandas DataFrame
        Conjunto alvo de previsão
    classe : str, optional
        Classe que se quer recuperar. The default is 'WD'.
        Com base nela que serão classificadas VP, FP, FN e VN no cabeçalho
        (header) CRF
    Returns
    -------
    Pandas DataFrame
        Dos objetos testados classificados como VP, FP, FN e VN dada a classe
        como referência.

    """
    def recover(tive_in, table):

        rec_ind = np.array([])

        for i in tive_in['source_id']:
            rec_ind = np.append(rec_ind, np.where(table['source_id'] == i)[0])

        recovered = table.iloc[rec_ind]

        return recovered

    index_classe_positive = np.where(predicted == classe)[0]
    positive = target_test.iloc[index_classe_positive]

    index_true_positive = np.where(positive['Type'] == classe)[0]
    true_positive_in = positive.iloc[index_true_positive]

    index_false_positive = np.where(positive['Type'] != classe)[0]
    false_positive_in = positive.iloc[index_false_positive]

    index_classe_negative = np.where(predicted != classe)[0]
    negative = target_test.iloc[index_classe_negative]

    index_true_negative = np.where(negative['Type'] != classe)[0]
    true_negative_in = negative.iloc[index_true_negative]

    index_false_negative = np.where(negative['Type'] == classe)[0]
    false_negative_in = negative.iloc[index_false_negative]

    true_positive = recover(true_positive_in, table1)
    true_positive['Unnamed: 0'] = 'True_positive'
    true_positive.rename(columns={'Unnamed: 0': 'CRF'}, inplace=True)

    false_positive = recover(false_positive_in, table2)
    false_positive['Unnamed: 0'] = 'False_positive'
    false_positive.rename(columns={'Unnamed: 0': 'CRF'}, inplace=True)

    true_negative = recover(true_negative_in, table2)
    true_negative['Unnamed: 0'] = 'True_negative'
    true_negative.rename(columns={'Unnamed: 0': 'CRF'}, inplace=True)

    false_negative = recover(false_negative_in, table1)
    false_negative['Unnamed: 0'] = 'False_negative'
    false_negative.rename(columns={'Unnamed: 0': 'CRF'}, inplace=True)

    tfp_tfn = pd.concat([true_positive, false_positive, true_negative,
                         false_negative], ignore_index=True)

    return tfp_tfn


def read(tablename: str()):
    """Read a table

    This function must be deleted!
    Parameters
    ----------
    tablename : str()
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return pd.read_csv(tablename, dtype={'source_id': str})


# Plots

# Constants
# header_x = ['ujava_aper', 'f378_aper', 'f395_aper', 'f410_aper',\
#    'f430_aper', 'g_aper', 'f515_aper', 'r_aper', 'f660_aper', 'i_aper',\
#                           'f861_aper', 'z_aper']
# header_xlabel = ['ujava', 'f378', 'f395', 'f410', 'f430','g', 'f515', 'r',\
#                 'f660', 'i', 'f861', 'z']
# header_ylabel = ['bp', 'rp', 'gp', 'parallax']
# HEADER_Y = ['phot_bp_mean_mag', 'phot_rp_mean_mag', 'phot_g_mean_mag',
# 'parallax']


def plot_color_diagrams(df_true_false_positive_tf_negative, Title: str,
                        header_x, header_y, header_xlabel, header_ylabel,
                        MAG=False):
    """Salva os diagramas de cores com base nos headers fornecidos.

     Ele pega os objetos da esquerda para a direita (header_x). Ele plota no
     eixo X todas as diferenças em relação a cada filtro, no eixo Y plota as
     medidas do GAIA (sem magnitude absoluta por padrão).
    Há enumeração das falsas na classe CRF do DataFrame (vindo da função
                                                         rec_all)
    Parameters
    ----------
    df_true_false_positive_tf_negative : Pandas DataFrame
        Retornado por rec_all.
    Title : str
    header_x : list, optional
        Com os headers do Pandas DataFrame. Calcula a diferença de magnitude
        entre todos os dados. The default is header_x.
    header_y : list, optional
        Com os headers do Pandas DataFrame. Calcula a diferença de magnitude
        entre todos os dados.
        Pode ter só um elemento. The default is HEADER_Y.
    MAG : bool, optional
        Calcula as magnitudes absolutas usadas no eixo Y. The default is False.

    Returns
    -------
    None. Só salva

    """
    # constants
    mk = 2
    figure_size = (21, 10)
    f_dpi = 300

    X = df_true_false_positive_tf_negative

    ind_tp = np.where(X['CRF'] == 'True_positive')[0]
    tp = X.iloc[ind_tp]

    ind_fp = np.where(X['CRF'] == 'False_positive')[0]
    fp = X.iloc[ind_fp]

    ind_tn = np.where(X['CRF'] == 'True_negative')[0]
    tn = X.iloc[ind_tn]

    ind_fn = np.where(X['CRF'] == 'False_negative')[0]
    fn = X.iloc[ind_fn]

    if len(header_y) == 1:
        M = X[header_y]

    else:
        M = X[header_y[:-1]]

    if MAG is True:
        str_phot = 'phot_'
        str_mean = '_mean_'

        d = 1e3 / X['parallax']
        D = - 5 * np.log10(d) + 5

        for i in M.columns:
            M[i] = M[i] + D

        M.columns = [y.upper() for y in M.columns]
        str_phot = 'PHOT_'
        str_mean = '_MEAN_'

    # M = pd.concat([M, X['parallax']], axis = 1)
    figure_number = 0
    pdf = PdfPages(Title + '.pdf')

    for a in M.columns:
        # a = 0:2, a:g,a,r; gaia header
        y1 = M[a].iloc[ind_tp]
        y2 = M[a].iloc[ind_fn]
        y3 = M[a].iloc[ind_tn]
        y4 = M[a].iloc[ind_fp]

        if len(header_y) == 1:
            ylabel = a

        else:
            ylabel = a.replace(str_phot, '').replace(str_mean, ' ') + ' GAIA'

        for (b, c), d in zip(enumerate(header_x[:-1]), header_xlabel):
            # b = 0:11, c = ujava_aper, f378_aper,..,f861_aper
            x1_1 = tp[c]
            x2_1 = fn[c]
            x3_1 = tn[c]
            x4_1 = fp[c]

            for f, g in zip(header_x[(b + 1):], header_xlabel[(b + 1):]):
                # f depende de b, cada vez menor. f é PELO MENOS um mag header
                # a frente que c

                x1_2 = tp[f]
                x2_2 = fn[f]
                x3_2 = tn[f]
                x4_2 = fp[f]

                figure_number += 1

                x1 = x1_1 - x1_2
                x2 = x2_1 - x2_2
                x3 = x3_1 - x3_2
                x4 = x4_1 - x4_2

                xlabel = d + ' - ' + g

                title = Title

                fig, ax = plt.subplots(num=figure_number, figsize=figure_size,
                                       dpi=f_dpi, clear=True)

                rows = ['True Positive', 'False Negative',
                        'True Negative', 'False Positive']
                n_rows = len(rows)
                colors_l = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
                # pos : [left, bottom, width, height]
                ax.set_position([0.04, 0.025, 1.1, 0.9])

                # Plot bars and create text labels for the table
                data = [[len(tp)], [len(fn)], [len(tn)], [len(fp)]]
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                n_columns = ('T')
                col_label = ['Total']

                n_rows = len(data)

                # Initialize the vertical-offset for the stacked bar chart.
                y_offset = np.array([0.0] * len(n_columns))

                cell_text = []

                for row in range(n_rows):
                    y_offset = np.array([0.0] * len(n_columns))
                    y_offset = y_offset + data[row]
                    cell_text.append(['%1.1d' % (x) for x in y_offset])

                ax.plot(x1, y1, 'o', marker='o', markersize=mk + .1,
                        alpha=.8, color=colors_l[0], zorder=5,
                        label='True Positive')

                ax.plot(x2, y2, 's', markersize=mk + .2, alpha=.6,
                        color=colors_l[1],
                        zorder=20, label='False Negative')

                for n, x2n in enumerate(x2):
                    ax.annotate(n + 1, (x2n, y2.iloc[n]), color=colors_l[1])

                ax.plot(x3, y3, 'o', marker='.', markersize=mk - .1, alpha=.9,
                        color=colors_l[2], zorder=0, label='True Negative')

                ax.plot(x4, y4, 'x', markersize=mk + .4, alpha=.7,
                        color=colors_l[3], zorder=10, label='False Positive')

                for n, x4n in enumerate(x4):
                    ax.annotate(n + 1, (x4n, y4.iloc[n]), color=colors_l[3])
                table = ax.table(cellText=cell_text, colLabels=col_label,
                                 rowLabels=rows, rowColours=colors_l,
                                 in_layout=True, bbox=(1.0835, 0, .025, .4))
                # bbox parameters (xi, yi, width, height)
                ax.invert_yaxis()
                table.scale(1, 4)
                table.set_fontsize(9)

                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', pad_inches=1.2)
                plt.close('all')

    pdf.close()


def sdss_spec(DataFrame, Title, save=True,
              local_save='/home/ricardo/Documents/fisica/ic/2019/python/wd_fits/spec/',
              allset=True):
    """Baixa os espectros de objetos de um DataFrame.

    Os espectros são do SDSS e deve ter no cabeçalho do DataFrame: Plate, MJD e
    Fiber. Os espectros são salvos no local_save.
    Pode-se optar por não salvá-los na máquina. Nesse caso deve-se mexer nas
    configurações padrões do fetch_sdss_spectrum.

    download_if_missing: boolean (default = True). Download the fits file if it
    is not cached locally

    cache_to_disk: boolean (default = True). Cache downloaded file to data_home

    Parameters
    ----------
    DataFrame : Pandas DataFrame
        DataFrame com Plate, MJD e Fiber em seu header

    save: bool, optional.
        Cria os pdf com os espectros, se disponível. The default is True.

    local_save : str, optional. Esse optional tem que ser mudado primeiro para
    cada máquina!
        The default is '/home/ricardo/Documents/fisica/ic/2019/python/wd_fits/spec/'.

    Returns
    -------
    sp : list
        Lista de spectros do sdss, extensão .fits
    i_nosp : list
        lista dos índices dos objetos que não foram encontrados objetos

    """
    from astroML.datasets import fetch_sdss_spectrum
    # plot or not
    figure_size = (21, 10)
    f_dpi = 300
    if save is True:
        pdfsp = PdfPages(Title + '.pdf')

    not_found = []
    # por enquanto só as falsas, n: vermelho, p:laranja

    fn = DataFrame.iloc[np.where(DataFrame['CRF'] == 'False_negative')[0]]
    fp = DataFrame.iloc[np.where(DataFrame['CRF'] == 'False_positive')[0]]

    if allset is True:
        tn = DataFrame.iloc[np.where(DataFrame['CRF'] == 'True_negative')[0]]
        tp = DataFrame.iloc[np.where(DataFrame['CRF'] == 'True_positive')[0]]
        lista2 = ['fn', 'fp', 'tp', 'tn']
        colors_l = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
        lista = [fn, fp, tp, tn]
    else:
        lista2 = ['fn', 'fp']
        colors_l = ['tab:red', 'tab:orange']
        lista = [fn, fp]

    figure_number = 0

    print(Title)

    for i, j in enumerate(lista):

        for k, row in enumerate(j.itertuples()):
            plate = row.plate
            mjd = row.mjd
            fiber = row.fiberid
            figure_number += 1

            try:
                spec = fetch_sdss_spectrum(
                    plate, mjd, fiber, data_home=local_save)
                if save is True:
                    fig, ax = plt.subplots(num=figure_number,
                                           figsize=figure_size, dpi=f_dpi,
                                           clear=True)

                    ax.plot(spec.wavelength(), spec.spectrum,
                            '-k', label='spectrum')
                    ax.plot(spec.wavelength(), spec.error,
                            '-', color='gray', label='error')
                    ax.set_title(Title + ' ' + row.CRF + ' ' + str(k + 1) + ' '
                                 'Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals(
                    ),
                        color=colors_l[i])

                    # ax.set_title('Plate = %(plate)i, MJD = %(mjd)i,
                    # Fiber = %(fiber)i' % locals())
                    ax.legend(loc=4)

                    ax.text(0.05, 0.95, 'z = %.2f' % spec.z, size=16,
                            ha='left', va='top', transform=ax.transAxes)

                    ax.set_xlabel(r'$\lambda (\AA)$')
                    ax.set_ylabel(
                        '$Flux \hspace{.6} (10^{-17} ergs \hspace{.6} cm^{-2} \hspace{.6} s^{-1 } \hspace{.6} \AA)$')
                    #ax.set_ylim(-10, 300)
                    pdfsp.savefig(fig, bbox_inches='tight', pad_inches=1.2)
                    plt.close('all')
            # Exceção deve ser específica, nunca geral
            except:

                print('not found: ', lista2[i] + ' ' + str(k + 1) + ', plate: ',
                      plate, ', mjd: ', mjd, ', fiber: ', fiber)
                not_found.append([Title, lista2[i], k, plate, mjd, fiber])
    if save is True:
        pdfsp.close()

    return not_found


def true_falses_distribution(wdtable, ostable, targ_set, pred, r_all=False):

    TP = 'True_positive'
    FP = 'False_positive'
    FN = 'False_negative'
    TN = 'True_negative'

    tp = np.array([])
    fp = tp.copy()
    fn = tp.copy()
    tn = tp.copy()

    targ_set_i = []
    pred_set_i = []
    tfp_tfn_i = []

    ntreinamentos = len(targ_set)

    for i in range(ntreinamentos):

        targ_set_i.append(targ_set[i][1])
        pred_set_i.append(pred[i])
        tfp_tfn_i.append(
            rec_all(wdtable, ostable, targ_set_i[i], pred_set_i[i]))

        tp = np.append(tp, len(np.where(tfp_tfn_i[i].CRF == TP)[0]))
        tn = np.append(tn, len(np.where(tfp_tfn_i[i].CRF == TN)[0]))
        fn = np.append(fn, len(np.where(tfp_tfn_i[i].CRF == FN)[0]))
        fp = np.append(fp, len(np.where(tfp_tfn_i[i].CRF == FP)[0]))

    tf = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}
    tf = pd.DataFrame(data=tf)

    if r_all is False:
        return tf
    else:
        return [tf, tfp_tfn_i]


def sdss_spec_jpeg(DataFrame, Title, save=True,
                   local_save='/home/ricardo/Documents/fisica/ic/2019/python/wd_fits/spec/',
                   allset=True, Type='.jpg'):
    """Baixa os espectros de objetos de um DataFrame.

    Os espectros são do SDSS e deve ter no cabeçalho do DataFrame: Plate, MJD e
    Fiber. Os espectros são salvos no local_save.
    Pode-se optar por não salvá-los na máquina. Nesse caso deve-se mexer nas
    configurações padrões do fetch_sdss_spectrum.

    download_if_missing: boolean (default = True). Download the fits file if it
    is not cached locally

    cache_to_disk: boolean (default = True). Cache downloaded file to data_home

    Parameters
    ----------
    DataFrame : Pandas DataFrame
        DataFrame com Plate, MJD e Fiber em seu header

    save: bool, optional.
        Cria os pdf com os espectros, se disponível. The default is True.

    local_save : str, optional. Esse optional tem que ser mudado primeiro para
    cada máquina!
        The default is '/home/ricardo/Documents/fisica/ic/2019/python/wd_fits/
        spec/'.

    Returns
    -------
    sp : list
        Lista de spectros do sdss, extensão .fits
    i_nosp : list
        lista dos índices dos objetos que não foram encontrados objetos

    """
    from astroML.datasets import fetch_sdss_spectrum
    # plot or not
    figure_size = (21, 10)
    f_dpi = 300

    if save is True and Type == '.pdf':
        sp_file = PdfPages(Title + Type)
    not_found = []
    # por enquanto só as falsas, n: vermelho, p:laranja

    fn = DataFrame.iloc[np.where(DataFrame['CRF'] == 'False_negative')[0]]
    fp = DataFrame.iloc[np.where(DataFrame['CRF'] == 'False_positive')[0]]

    if allset is True:
        tn = DataFrame.iloc[np.where(DataFrame['CRF'] == 'True_negative')[0]]
        tp = DataFrame.iloc[np.where(DataFrame['CRF'] == 'True_positive')[0]]
        lista2 = ['fn', 'fp', 'tp', 'tn']
        colors_l = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
        lista = [fn, fp, tp, tn]

    else:
        lista2 = ['fn', 'fp']
        colors_l = ['tab:red', 'tab:orange']
        lista = [fn, fp]

    figure_number = 0

    print(Title)

    for i, j in enumerate(lista):

        for k, row in enumerate(j.itertuples()):

            plate = row.plate
            mjd = row.mjd
            fiber = row.fiberid
            figure_number += 1

            if save is True and Type == '.jpg':
                sp_file = PdfPages(Title + Type)

            try:
                spec = fetch_sdss_spectrum(
                    plate, mjd, fiber, data_home=local_save)
                if save is True:
                    fig, ax = plt.subplots(num=figure_number,
                                           figsize=figure_size, dpi=f_dpi,
                                           clear=True)

                    ax.plot(spec.wavelength(), spec.spectrum,
                            '-k', label='spectrum')
                    ax.plot(spec.wavelength(), spec.error,
                            '-', color='gray', label='error')
                    ax.set_title(Title + ' ' + row.CRF + ' ' + str(k + 1)
                                 + ' Plate = %(plate)i, MJD = %(mjd)i, Fiber = %\
                                 (fiber)i' % locals(),
                                 color=colors_l[i])

                    # ax.set_title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %
                    # (fiber)i' % locals())
                    ax.legend(loc=4)

                    ax.text(0.05, 0.95, 'z = %.2f' % spec.z, size=16,
                            ha='left', va='top', transform=ax.transAxes)

                    ax.set_xlabel(r'$\lambda (\AA)$')
                    ax.set_ylabel(
                        '$Flux \hspace{.6} (10^{-17} ergs \hspace{.6} cm^{-2}\
                            \hspace{.6} s^{-1 } \hspace{.6} \AA)$')
                    #ax.set_ylim(-10, 300)
                    sp_file.savefig(fig, bbox_inches='tight', pad_inches=1.2)
                    plt.close('all')
                    if save is True and Type == '.jpg':
                        sp_file.close()
            # Exceção deve ser específica, nunca geral
            except:

                print('not found: ', lista2[i] + ' ' + str(k + 1) + ', plate: ',
                      plate, ', mjd: ', mjd, ', fiber: ', fiber)
                not_found.append([Title, lista2[i], k, plate, mjd, fiber])

    if save is True and Type == '.pdf':
        sp_file.close()

    return not_found
