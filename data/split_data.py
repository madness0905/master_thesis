import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'


def split_stratified_into_train_test(
    df_input,
    stratify_colname='y',
    frac_train=0.80,
    frac_test=0.20,
    random_state=None,
):

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' %
                         (stratify_colname))

    X = df_input  # Contains all columns.
    y = df_input[[stratify_colname
                  ]]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=(1.0 - frac_train),
        random_state=random_state)

    return df_train, df_test


def generate_random_split(
    df,
    numb=1,
    frac_train=0.8,
    frac_test=0.2,
):

    df['publisher+label'] = df['publisher'].astype(str) + df['label'].astype(
        str
    )  # create publisher + label split for case that labels are bnot perfectly correlated with publisher
    train, test = split_stratified_into_train_test(df, 'publisher+label',
                                                   frac_train, frac_test)
    train.loc[:, 'split{}'.format(numb)] = 'train'
    test.loc[:, 'split{}'.format(numb)] = 'test'

    return pd.concat([train, test])


def generate_publisher_split(
    df,
    frac_train=0.80,
    frac_test=0.20,
    random_state=None,
    numb=1,
):
    """
    Splitting dataframe so that test and validation set contain publishers not present in trainingset
    """

    publisher = df['publisher'].unique()
    i = 0
    while i != 1:
        M = int(np.round(np.size(df['publisher'].unique()) * frac_train))
        publisher_training = np.random.choice(publisher, M, replace=False)
        publisher_testing = np.setdiff1d(publisher, publisher_training)
        print(set(publisher_training).intersection(publisher_testing))
        print(publisher_training, publisher_testing)

        df_training = df.loc[df['publisher'].isin(publisher_training)]
        df_testing = df.loc[df['publisher'].isin(publisher_testing)]
        try:
            n_fake_training = df_training['label'].value_counts()['fake']
            n_real_training = df_training['label'].value_counts()['real']
            n_fake_testing = df_testing['label'].value_counts()['fake']
            n_real_testing = df_testing['label'].value_counts()['real']
        except:
            continue
        print(
            abs(n_fake_testing / (n_fake_testing + n_real_testing) -
                n_fake_training / (n_fake_training + n_real_training)),
            abs(n_real_testing / (n_fake_testing + n_real_testing) -
                n_real_training / (n_fake_training + n_real_training)))
        if abs(n_fake_testing /
               (n_fake_testing + n_real_testing) - n_fake_training /
               (n_fake_training + n_real_training)) < 0.10 and abs(
                   n_real_testing /
                   (n_fake_testing + n_real_testing) - n_real_training /
                   (n_fake_training + n_real_training)) < 0.10:
            i = 1
        else:
            i = 0
    df_train = df_training
    df_test = df_testing
    print(df_train['publisher'].value_counts())
    print(df_test['publisher'].value_counts())
    df_train.loc[:, 'split{}'.format(numb)] = 'train'
    df_test.loc[:, 'split{}'.format(numb)] = 'test'
    __import__("pdb").set_trace()
    __import__("pdb").set_trace()

    print("Train size: {} , Test size: {}".format(len(df_train), len(df_test)))
    return pd.concat([df_train, df_test])
