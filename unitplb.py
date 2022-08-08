import numpy as np
import pandas as pd
import re
from unittest.mock import MagicMock, patch
import sklearn.naive_bayes

CSVTEST = 'data/testinfo.csv'


class AssertTest(object):
    def __init__(self, params):
        self.assert_param_message = '\n'.join(
            [str(k) + ': ' + str(v) + '' for k, v in params.items()])

    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + \
            '\n\nUnit Test params\n' + self.assert_param_message


def successmsg():
    print('Test cases got passed!')


def tstnumdf(numdf):
    tdf = numdf(CSVTEST)
    assert isinstance(
        tdf, pd.DataFrame), 'Returned type is {}.'.format(type(tdf))
    namecols = list(tdf)
    assert 'Cattype' in namecols, 'No Cattype col got'
    assert 'ClassVal' in namecols, 'No ClassVal col got'
    assert 'FileName' in namecols, 'No FileName col got'
    assert 'Assctask' in namecols, 'No Assctask col got'
    assert tdf.loc[0,
                       'Cattype'] == 1, '`majorplg`failed.'
    assert tdf.loc[2,
                       'Cattype'] == 0, '`notplg` failed.'
    assert tdf.loc[30,
                       'Cattype'] == 3, '`nearplg` failed.'
    assert tdf.loc[5,
                       'Cattype'] == 2, '`lightplg` failed.'
    assert tdf.loc[37, 'Cattype'] == - \
        1, 'orginal file mapping test, failed; should have a Cattype = -1.'
    assert tdf.loc[41, 'Cattype'] == - \
        1, 'orginal file mapping test, failed; should have a Cattype = -1.'
    successmsg()


def testctmt(comp_df, cmntfn):
    tval = cmntfn(comp_df, 1, '0Ae.txt')
    assert isinstance(tval, float), 'Returned type is {}.'.format(
        type(tval))
    assert tval <= 1.0, 'Value unnormalized, value should be <=1, got: ' + \
        str(tval)
    filenames = ['0Aa.txt', '0Ab.txt', '0Ac.txt', '0Ad.txt']
    ng1 = [0.39814814814814814, 1.0,
               0.86936936936936937, 0.5935828877005348]
    ng3 = [0.0093457943925233638, 0.96410256410256412,
               0.61363636363636365, 0.15675675675675677]
    rsng1= []
    rsng3 = []
    for i in range(4):
        val_1 = cmntfn(comp_df, 1, filenames[i])
        val_3 = cmntfn(comp_df, 3, filenames[i])
        rsng1.append(val_1)
        rsng3.append(val_3)
    assert all(np.isclose(rsng1, ng1, rtol=1e-04)), \
        'n=1 intersection calculation failed'
    assert all(np.isclose(rsng3, ng3, rtol=1e-04)), \
        'n=3 intersection calculation failed'
    successmsg()


def test_lcs(df, lcs_word):
    tstidx = 10
    atxt = df.loc[tstidx, 'TextVal']
    task = df.loc[tstidx, 'Assctask']
    orgfile_rows = df[(df['ClassVal'] == -1)]
    orgfile_row = orgfile_rows[(orgfile_rows['Assctask'] == task)]
    stxt = orgfile_row['TextVal'].values[0]
    tval = lcs_word(atxt, stxt)
    assert isinstance(tval, float), 'Returned type is {}.'.format(
        type(tval))
    assert tval <= 1.0, 'Value should <=1, got: '+str(tval)
    lcs_vals = [0.1917808219178082, 0.8207547169811321,
                0.8464912280701754, 0.3160621761658031, 0.24257425742574257]
    results = []
    for i in range(5):
        atxt = df.loc[i, 'TextVal']
        task = df.loc[i, 'Assctask']
        orgfile_rows = df[(df['ClassVal'] == -1)]
        orgfile_row = orgfile_rows[(orgfile_rows['Assctask'] == task)]
        stxt = orgfile_row['TextVal'].values[0]
        val = lcs_word(atxt, stxt)
        results.append(val)
    assert all(np.isclose(results, lcs_vals, rtol=1e-05)
               ), 'LCS calculations failed'
    successmsg()


def test_data_split(train_x, train_y, xtst, ytst):
    assert isinstance(train_x, np.ndarray),\
        'train_x not array, instead got type: {}'.format(type(train_x))
    assert isinstance(train_y, np.ndarray),\
        'train_y not array, instead got type: {}'.format(type(train_y))
    assert isinstance(xtst, np.ndarray),\
        'xtst not array, instead got type: {}'.format(type(xtst))
    assert isinstance(ytst, np.ndarray),\
        'ytst not array, instead got type: {}'.format(type(ytst))
    assert len(train_x) + len(xtst) == 95, \
        'Unexpected amount of train + test data. Should be 95 filedata, got ' + \
        str(len(train_x) + len(xtst))
    assert len(xtst) > 1, \
        'Unexpected amount of test data. Should be multiple test files.'
    assert train_x.shape[1] == 2, \
        'train_x should have as many columns as selected features, got: {}'.format(
            train_x.shape[1])
    assert len(train_y.shape) == 1, \
        'train_y must one dimension, got shape: {}'.format(train_y.shape)
    successmsg()
