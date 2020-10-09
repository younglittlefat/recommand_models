"""Download and clean the Census Income Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app as absl_app
from absl import flags
from six.moves import urllib
from six.moves import zip
import tensorflow as tf

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

# _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
#                         [0], [0], [0], [''], ['']]
_CSV_COLUMN_DEFAULTS = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
                        [0], [0], [0], [0], [0]]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'],
            hash_bucket_size=_HASH_BUCKET_SIZE),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


def feature_segment_function(feature_name, feature_value):
    if feature_name == "age":
        feature_value = int(feature_value)
        if feature_value < 18:
            return "age_0_18"
        elif 18 <= feature_value < 25:
            return "age_18_25"
        elif 25 <= feature_value < 30:
            return "age_25_30"
        elif 30 <= feature_value < 35:
            return "age_30_35"
        elif 35 <= feature_value < 40:
            return "age_35_40"
        elif 40 <= feature_value < 45:
            return "age_40_45"
        elif 45 <= feature_value < 50:
            return "age_45_50"
        elif 50 <= feature_value < 55:
            return "age_50_55"
        elif 55 <= feature_value < 60:
            return "age_55_60"
        elif 60 <= feature_value < 65:
            return "age_60_65"
        else:
            return "age_65"

    elif feature_name == "capital_gain":
        feature_value = int(feature_value)
        if feature_value == 0:
            return "capital_gain_0"
        elif 0 < feature_value < 10000:
            return "capital_gain_%s" % (int(feature_value / 1000) + 1)
        else:
            return "capital_gain_larger_than_10000"
    elif feature_name == "capital_loss":
        feature_value = int(feature_value)
        if feature_value == 0:
            return "capital_loss_0"
        elif 0 < feature_value < 3000:
            return "capital_loss_%s" % (int(feature_value / 100) + 1)
        else:
            return "capital_loss_larger_than_3000"
    elif feature_name == "fnlwgt":
        feature_value = int(int(feature_value) / 100)
        if 0 <= feature_value < 3000:
            return "fnlwgt_%s" % (int(feature_value / 200) * 200)
        elif 3000 <= feature_value < 5000:
            return "fnlwgt_%s" % (int(feature_value / 500) * 500)
        else:
            return "fnlwgt_larger_than_5000"
    else:
        return feature_value


def encode_raw_features(train_file_path, test_file_path, mapping_path):
    id_mapping = {}
    # if os.path.exists(mapping_path):
    #     with open(mapping_path, "r") as f:
    #         for line in f:
    #             key, idx = line.strip().split(",")
    #             try:
    #                 idx = int(idx)
    #             except:
    #                 tf.logging.error("Error in decoding line=%s" % line)
    #                 continue
    #             id_mapping[key] = idx
    #     print("key num = %s" % len(id_mapping))
    #     return id_mapping

    # if mapping_path does not exist
    id_mapping["?"] = 0
    mapping_idx = 1

    def decode_file(file_path, mapping, start_idx):

        # output the encoded feature and label
        encoded_feat_file_path = "%s.encoded" % file_path
        fw = open(encoded_feat_file_path, "w")

        with open(file_path, "r") as f:
            for line in f:
                word_list = list(map(lambda x: x.replace(" ", ""), line.strip().split(",")))
                raw_label = word_list[-1]
                label = None
                if raw_label.strip(".") == "<=50K":
                    label = 0
                elif raw_label.strip(".") == ">50K":
                    label = 1
                else:
                    tf.logging.error("Error label=%s, now continue" % raw_label)
                    continue
                raw_feature_list = word_list[:-1]
                if len(raw_feature_list) != 14:
                    continue

                # Get the segmented feature
                feat_list = list(map(lambda x: feature_segment_function(_CSV_COLUMNS[x[0]], x[1]), enumerate(raw_feature_list)))
                for feat in feat_list:
                    # Get the encoded value of this feature
                    if feat not in mapping:
                        mapping[feat] = start_idx
                        fw.write("%s," % start_idx)
                        start_idx += 1
                    else:
                        fw.write("%s," % mapping[feat])
                # write label
                fw.write("%s\n" % label)
        fw.close()
        return mapping, start_idx

    id_mapping, mapping_idx = decode_file(train_file_path, id_mapping, mapping_idx)
    id_mapping, mapping_idx = decode_file(test_file_path, id_mapping, mapping_idx)
    print("key num = %s" % len(id_mapping))
    with open(mapping_path, "w") as fw:
        for key in id_mapping:
            fw.write("%s,%s\n" % (key, id_mapping[key]))
    return id_mapping


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have run census_dataset.py and '
            'set the --data_dir argument to the correct path.' % data_file)

#    def parse_csv(value):
#        tf.logging.info('Parsing {}'.format(data_file))
#        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
#        features = dict(list(zip(_CSV_COLUMNS, columns)))
#        labels = features.pop('income_bracket')
#        classes = tf.equal(labels, '>50K')  # binary classification
#        return features, classes

    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = tf.stack(columns[:-1], axis=0)
        classes = columns[-1]
        return features, classes

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def define_data_download_flags():
    """Add flags specifying data download arguments."""
    flags.DEFINE_string(
        name="data_dir", default="../data/census_data/",
        help="Directory to download and extract data.")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # encode_raw_features("../data/census_data/adult.data", "../data/census_data/adult.test", "../data/census_data/id_mapping")
    dataset = input_fn("../data/census_data/adult.test.encoded", 1, False, 100)
    print(dataset)
    dataset_iter = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        while True:
            try:
                feat, label = sess.run(dataset_iter.get_next())
                print(feat)
                print(label)
                # break
            except Exception as e:
                tf.logging.error(e)
                break

