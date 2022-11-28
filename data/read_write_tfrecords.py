import os
import numpy as np
import tensorflow as tf
import scipy
import pandas as pd
import glob


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    if isinstance(value, str):
        value = value.encode('utf-8')

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):

    array = tf.io.serialize_tensor(np.float32(array))
    return array


def serialize_sparse_array(array):
    array = convert_sparse_matrix_to_sparse_tensor(array)

    return tf.io.serialize_sparse(array).numpy()


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def parse_single_ndarray(mat: np.ndarray, name: str) -> dict:
    shape = mat.shape
    data_encode = {}
    name = '{}/raw'.format(name)
    for i in range(len(shape)):
        data_encode['{}/{}-dim'.format(name, i)] = _int64_feature(mat.shape[i])
    data_encode['{}/ndim'.format(name)] = _int64_feature(len(shape))
    data_encode[name] = _bytes_feature(serialize_array(mat))
    return data_encode


def parse_single_sparse_ndarray(mat: scipy.sparse._csr.csr_matrix,
                                name: str) -> dict:

    mats = serialize_sparse_array(mat.astype(
        np.float32))  # get 3 vectorse describing sparse matrix
    return tf.train.FeatureList(feature=[_bytes_feature(m) for m in mats])


def create_example(row):
    feature_dict = {}  # only sinlge feature examples
    feature_list = {}
    for name, value in row.items():
        if type(value) == np.ndarray:
            data = parse_single_ndarray(value, name)
            feature_dict.update(data)
        elif type(
                value
        ) == scipy.sparse._csr.csr_matrix:  # serializing converts to sequential feature
            feature_list[name +
                         '/sparse_tensor'] = parse_single_sparse_ndarray(
                             value, name)

        elif isinstance(value, (bytes, str)):
            feature_dict[name] = _bytes_feature(value)
        elif isinstance(value, (int, np.integer, bool, np.bool)):
            feature_dict[name] = _int64_feature(value)
        elif isinstance(value, (float, np.floating)):
            feature_dict[name] = _float_feature(value)
        else:
            raise Exception(f'Unsupported type {type(value)!r}')
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    context = tf.train.Features(feature=feature_dict)
    example = tf.train.SequenceExample(context=context,
                                       feature_lists=feature_lists)

    return example


def write_tfrecords(df: pd.DataFrame, path: str):
    filename = path + ".tfrecord"
    writer = tf.io.TFRecordWriter(
        filename)  # create a writer that'll store our data to disk
    count = 0
    list_of_features = df.to_dict('records')
    # iterate over each row and create example
    for row in list_of_features:
        example = create_example(row)
        writer.write(example.SerializeToString())
        count += 1
    print(f"Wrote {count} elements to TFRecord")


def read_tfrecords(path):
    path = path + '.tfrecord'
    dataset = tf.data.TFRecordDataset(path)
    return dataset, parse_fn


def parse_fn(element, numb):

    # unfortunately there is no way to automate this vor varying feature types (inlcuding sparse and numpy) so this need to be adjusted whenever new features are added or removed
    sparse_data = {
        'bag_of_words/sparse_tensor': tf.io.FixedLenSequenceFeature([],
                                                                    tf.string),
        'tfidf/sparse_tensor': tf.io.FixedLenSequenceFeature([], tf.string)
    }

    data = {
        'url': tf.io.FixedLenFeature([], tf.string),
        'article': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_publisher_cardinal': tf.io.FixedLenFeature([], tf.int64),
        'publisher': tf.io.FixedLenFeature([], tf.string),
        'split{}'.format(numb): tf.io.FixedLenFeature([], tf.string),
        'ttr': tf.io.FixedLenFeature([], tf.float32),
        'avg_word_length': tf.io.FixedLenFeature([], tf.float32),
        'num_of_char': tf.io.FixedLenFeature([], tf.int64),
        'num_of_words': tf.io.FixedLenFeature([], tf.int64),
        'upper_case_ratio': tf.io.FixedLenFeature([], tf.float32),
        'POS_ratios/raw': tf.io.FixedLenFeature([], tf.string),
        'POS_ratios/raw/ndim': tf.io.FixedLenFeature([], tf.int64),
        'POS_ratios/raw/0-dim': tf.io.FixedLenFeature([], tf.int64),
        'first_pers_pron_ratio': tf.io.FixedLenFeature([], tf.float32),
        'comp_ratio': tf.io.FixedLenFeature([], tf.float32),
        'superl_ratio': tf.io.FixedLenFeature([], tf.float32),
        'citation': tf.io.FixedLenFeature([], tf.int64),
        'cardinal': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed, parsed_sparse = tf.io.parse_single_sequence_example(
        element, sequence_features=sparse_data, context_features=data)
    # tf.io.deserialize_many_sparse() requires the dimensions to be [N,3] so we add one dimension with expand_dims
    for key in ['tfidf/sparse_tensor', 'bag_of_words/sparse_tensor']:

        sparse_tensor = tf.expand_dims(parsed_sparse[key], axis=0)
        # deserialize sparse tensor
        if 'bag_of_words' in key:

            sparse_tensor = tf.cast(
                tf.io.deserialize_many_sparse(sparse_tensor, dtype=tf.float32),
                tf.int64)
        else:
            sparse_tensor = tf.io.deserialize_many_sparse(sparse_tensor,
                                                          dtype=tf.float32)
        # convert from sparse to dense
        dense_tensor = tf.sparse.to_dense(sparse_tensor)
        # remove extra dimenson [1, 3] -> [3]
        feature = tf.squeeze(dense_tensor)
        parsed[key.split('/')[0]] = feature

    pos_ratio = tf.io.parse_tensor(parsed['POS_ratios/raw'],
                                   out_type=tf.float32)

    parsed['POS_ratios'] = pos_ratio
    for key in [
            'POS_ratios/raw', 'POS_ratios/raw/ndim', 'POS_ratios/raw/0-dim'
    ]:
        parsed.pop(key)

    return parsed
