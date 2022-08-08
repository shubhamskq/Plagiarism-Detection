import re
import operator
import pandas as pd


def crtdtype(df, train_value, tvalue, datatype_var, cmpdfcol, opcmp, vcmp,
             sampleno, sampleseed):
    dfsbset = df[opcmp(df[cmpdfcol], vcmp)]
    dfsbset = dfsbset.drop(columns=[datatype_var])
    dfsbset.loc[:, datatype_var] = train_value
    df_sampled = dfsbset.groupby(['Assctask', cmpdfcol], group_keys=False).apply(
        lambda x: x.sample(min(len(x), sampleno), random_state=sampleseed))
    df_sampled = df_sampled.drop(columns=[datatype_var])
    df_sampled.loc[:, datatype_var] = tvalue
    for index in df_sampled.index:
        dfsbset.loc[index, datatype_var] = tvalue
    for index in dfsbset.index:
        df.loc[index, datatype_var] = dfsbset.loc[index, datatype_var]


def trtsdf(clean_df, random_seed=100):
   ndf = clean_df.copy()
   ndf.loc[:, 'Datatype'] = 0
   crtdtype(ndf, 1, 2, 'Datatype', 'Cattype',
             operator.gt, 0, 1, random_seed)
   crtdtype(ndf, 1, 2, 'Datatype', 'Cattype',
             operator.eq, 0, 2, random_seed)
   mapping = {0: 'orgfile', 1: 'train', 2: 'test'}
   ndf.Datatype = [mapping[item] for item in ndf.Datatype]
   return ndf


def processfile(file):
    txtall = file.read().lower()
    txtall = re.sub(r"[^a-zA-Z0-9]", " ", txtall)
    txtall = re.sub(r"\t", " ", txtall)
    txtall = re.sub(r"\n", " ", txtall)
    txtall = re.sub("  ", " ", txtall)
    txtall = re.sub("   ", " ", txtall)
    return txtall


def crtxtcol(df, file_directory='data/'):
    textdf = df.copy()
    text = []
    for row_i in df.index:
        filename = df.iloc[row_i]['FileName']
        file_path = file_directory + filename
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            file_text = processfile(file)
            text.append(file_text)
    textdf['TextVal'] = text
    return textdf


def getanssrctxt(df, ansfile):
    srcfname = 'org' + ansfile[-5:]
    sidxno = df[df['FileName'] == srcfname].index.values.astype(int)[
        0]
    aidxno = df[df['FileName'] == ansfile].index.values.astype(int)[0]
    stxt = df.loc[sidxno, 'TextVal']
    atxt = df.loc[aidxno, 'TextVal']
    return stxt, atxt
