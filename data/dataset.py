from features.features import *
from data.make_dataset import load_FANGCOVID
from data.read_write_tfrecords import write_tfrecords, read_tfrecords
from typing import Literal
import glob
import pandas as pd
import os
import spacy
import tensorflow as tf

pd.options.mode.chained_assignment = None  # default='warn'
spDE = spacy.load('de_core_news_sm',
                  disable=['attribute_rule', 'lemmatizer', 'ner'])


class Dataset:

    def __init__(
        self,
        dataset: Literal['FANG-COVID', 'Veritas'],
        kind: Literal['preprocessed', 'not_preprocessed', 'training', None],
        split: Literal['publisher_split', 'random_split'],
    ):
        self.dataset = dataset
        self.kind = kind
        self.split = split
        self.data_folder_path = os.path.join(
            os.path.split(os.path.dirname(__file__))[0],
            "data/{}/{}".format(dataset, kind
                                or ''))  # empty string in case kind=None
        self.tf_record_path = os.path.join(self.data_folder_path, self.split)
        self.feature_names = [
            'ttr', 'avg_word_length', 'num_of_char', 'num_of_words',
            'bag_of_words', 'upper_case_ratio', 'POS_ratios',
            'first_pers_pron_ratio', 'comp_ratio', 'superl_ratio', 'citation',
            'cardinal', 'tfidf'
        ]

    def get_articles_labels_publishers(self):
        if self.dataset == 'FANG-COVID':
            df = load_FANGCOVID(self.data_folder_path,
                                split=self.split,
                                kind=self.kind)
            df['label'] = df['label'].map({'fake': 0, 'real': 1})
        elif self.dataset == 'Veritas':
            raise NotImplementedError

        else:
            raise ValueError('{} is not in [FANG-COVID, Veritas]'.format(
                self.dataset))

        return df['article'].to_list(), df['label'].to_list(
        ), df['publisher'].to_list()

    def _compute_features(self, df: pd.DataFrame):
        print("computing features ...")
        __import__("pdb").set_trace()

        article = df["article"]

        df['ttr'] = article.apply(lambda x: ttr(x)).to_numpy()

        df['num_of_char'] = article.apply(lambda x: num_of_char(x))
        bow = bag_of_words(article)
        tfidf_feat = tfidf(article)

        df['tfidf'] = [tfidf_feat[i] for i in range(tfidf_feat.shape[0])]

        df['bag_of_words'] = [bow[i] for i in range(bow.shape[0])]
        df['upper_case_ratio'] = article.apply(lambda x: upper_case_ratio(x))
        df['citation'] = article.apply(lambda x: citation(x))
        df['cardinal'] = article.apply(lambda x: cardinal(x))
        df['index'] = np.arange(1, df.shape[0])
        pos_ratios_list = []
        first_pers_pron_ratio_list = []
        num_of_words_list = []
        comp_ratio_list = []
        superl_ratio_list = []
        avg_word_length_list = []
        # tokenized_list = []

        for doc in spDE.pipe(article.to_list(), batch_size=20, n_process=4):

            # tokenized_list.append([token.pos_ for token in doc])
            avg_word_length_list.append(avg_word_length(doc))
            pos_ratios_list.append(POS_ratios(doc))
            comp_ratio, superl_ratio = comp_superl_ratio(doc)
            comp_ratio_list.append(comp_ratio)
            superl_ratio_list.append(superl_ratio)
            first_pers_pron_ratio_list.append(first_pers_pron_ratio(doc))
            num_of_words_list.append(num_of_words(doc))

        df['avg_word_length'] = avg_word_length_list
        # df['tokenized'] = tokenized_list
        df['label'] = df['label'].map({'fake': 0, 'real': 1})
        df['label_publisher_cardinal'] = df['publisher'].astype(
            'category').cat.codes
        df['POS_ratios'] = pos_ratios_list
        df['num_of_words'] = num_of_words_list
        df['first_pers_pron_ratio'] = first_pers_pron_ratio_list
        df['comp_ratio'] = comp_ratio_list
        df['superl_ratio'] = superl_ratio_list

        self.features = df

        print('Done.')

    def write_tfrecords(self, overwrite, compression_type):

        if not os.path.exists(self.data_folder_path):
            os.mkdir(self.data_folder_path)

        if not os.path.exists(self.tf_record_path):
            os.mkdir(self.tf_record_path)

        if not glob.glob(
                os.path.join(self.tf_record_path, compression_type
                             or '')) or overwrite:

            if self.dataset == 'FANG-COVID':
                df_articles = load_FANGCOVID(
                    self.data_folder_path,
                    split=self.split,
                    kind=self.kind,
                    numb=10)  # default number of different splits
            elif self.dataset == 'Veritas':
                raise NotImplementedError

            else:
                raise ValueError('{} is not in [FANG-COVID, Veritas]'.format(
                    self.dataset))

            self._compute_features(df_articles)

            if overwrite:
                existing_tfrecord = glob.glob(
                    os.path.join(self.tf_record_path, compression_type or '*'))

                for zip_file in existing_tfrecord:
                    os.remove(zip_file)

            print("Writing data to tfrecords...")
            write_tfrecords(self.features, self.tf_record_path)
            print("Done.")

    def get_data_iterators(self, numb: int):
        """numb indicates which trainig split to select and can be in the range of 200"""
        dataset, parser = read_tfrecords(self.tf_record_path)
        dataset = dataset.shuffle(600)
        dataset_map = dataset.map(lambda x: parser(x, numb))

        dataset_train = dataset_map.filter(
            lambda x: x['split{}'.format(numb)] == 'train'
        )  # for some reason tf can't convert to autograph :(
        dataset_test = dataset_map.filter(
            lambda x: x['split{}'.format(numb)] == 'test')
        return {
            'train': dataset_train,
            'test': dataset_test,
        }


def main():
    kind = 'preprocessed'  # which dataset to select, will be None for the FANGCOVID data
    dataset_name = 'FANG-COVID'
    split = 'random_split'  # split by publisher or random
    data = Dataset(dataset_name, kind, split)  # initialize data class
    __import__("pdb").set_trace()

    data.write_tfrecords(  # creates tfrecord named either: publisher_split.tfrecord or random_split.tfrecord depending on which split the class is initialized on
        overwrite=True,
        compression_type=
        None  # overwrite doesn't work as expected, even when no tfrecord file exists, none is created
    )  # takes about 30 minutes, computes features, if only articles and labels are wanted try get_articles_labels
    iterators = data.get_data_iterators(0)  # get 0ths split

    publisher_split_train = iterators['train']
    publisher_split_test = iterators['test']

    article, labels = data.get_articles_labels(
    )  # get plain articles and labels in python string format stored in a list


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    with tf.device('/CPU:0'):
        main()
