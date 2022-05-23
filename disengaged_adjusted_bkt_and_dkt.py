# -*- coding: utf-8 -*-
'''
Packages

'''
# Install pyBKT from pip!
#pip install pyBKT

# Import all required packages including pyBKT.models.Model!
import numpy as np
import pandas as pd
from pyBKT.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

#%%
'''

Data Preparation

'''
##load the AssistMENT data 
###the data is available here: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
df = pd.read_csv('skill_builder_data.csv', encoding = 'latin', low_memory=False)
df['rt'] = df['ms_first_response'].div(1000).round(2)
df['hint'] = df['hint_count'] / df['hint_total'] *100
df_new = df[(df['rt'] >0) & df['skill_id'] != ""]

df_new['eng'] = df_new['hint'] / df_new['rt']
sns.pairplot(df_new, hue = 'correct')



#%%
'''

BAYESIAN KNOWLEDGE TRACING (BKT)

'''

###load the test and train data 
###we have baseline data and disengaged adjusted data
df_base_train = pd.read_csv('data_base_train.csv', encoding = 'latin', low_memory=False)
df_base_test = pd.read_csv('data_base_test.csv', encoding = 'latin', low_memory=False)

df_diseng_train = pd.read_csv('data_diseng_train.csv', encoding = 'latin', low_memory=False)
df_diseng_test = pd.read_csv('data_diseng_test.csv', encoding = 'latin', low_memory=False)

###we will train and evaluate BKT the baseline data 
np.seterr(divide='ignore', invalid='ignore')

model = Model(seed = 0, num_fits = 20)
model.fit(data_path = 'df_base_train.csv')
print("Standard BKT:", model.evaluate(data_path = 'df_base_test.csv', metric="precision_score"))
print("Standard BKT:", model.evaluate(data_path = 'df_base_test.csv', metric="auc"))
print("Standard BKT:", model.evaluate(data_path = 'df_base_test.csv', metric="recall_score"))
print("Standard BKT:", model.evaluate(data_path = 'df_base_test.csv', metric="accuracy"))

###we will train and evaluate BKT the disengaged-adjusted data 
np.seterr(divide='ignore', invalid='ignore')

model = Model(seed = 0, num_fits = 20)
model.fit(data_path = 'df_diseng_train.csv')
print("Standard BKT:", model.evaluate(data_path = 'df_diseng_test.csv', metric="precision_score"))
print("Standard BKT:", model.evaluate(data_path = 'df_diseng_test.csv', metric="auc"))
print("Standard BKT:", model.evaluate(data_path = 'df_diseng_test.csv', metric="recall_score"))
print("Standard BKT:", model.evaluate(data_path = 'df_diseng_test.csv', metric="accuracy"))

'''

DEEP KNOWLEDGE TRACING (DKT)

'''
###the code is adapted from https://github.com/lccasagrande/Deep-Knowledge-Tracing
###load the test and train data 
###we have baseline data and disengaged adjusted data


MASK_VALUE = -1.  # The masking value cannot be zero.


def load_dataset(fn, batch_size=32, shuffle=True):
    df = pd.read_csv(fn , encoding = 'unicode_escape', engine ='c')

    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {fn}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {fn}")

    if not (df['correct'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)

    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('user_id').apply(
        lambda r: (
            r['skill_with_answer'].values[:-1],
            r['skill'].values[1:],
            r['correct'].values[1:],
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142
    features_depth = df['skill_with_answer'].max() + 1
    skill_depth = df['skill'].max() + 1

    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(
                values=[
                    tf.one_hot(skill, depth=skill_depth),
                    tf.expand_dims(label, -1)
                ],
                axis=-1
            )
        )
    )

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(MASK_VALUE, MASK_VALUE),
        padded_shapes=([None, None], [None, None]),
        drop_remainder=True
    )

    length = nb_users // batch_size
    return dataset, length, features_depth, skill_depth


def split_dataset(dataset, total_size, test_fraction, val_fraction=None):
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between (0, 1)")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between (0, 1)")

    test_size = np.ceil(test_fraction * total_size)
    train_size = total_size - test_size

    if test_size == 0 or train_size == 0:
        raise ValueError(
            "The train and test datasets must have at least 1 element. Reduce the split fraction or get more data.")

    train_set, test_set = split(dataset, test_size)

    val_set = None
    if val_fraction:
        val_size = np.ceil(train_size * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set


def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred




class DKTModel(tf.keras.Model):
    """ The Deep Knowledge Tracing model.
    Arguments in __init__:
        nb_features: The number of features in the input.
        nb_skills: The number of skills in the dataset.
        hidden_units: Positive integer. The number of units of the LSTM layer.
        dropout_rate: Float between 0 and 1. Fraction of the units to drop.
    Raises:
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
    """

    def __init__(self, nb_features, nb_skills, hidden_units=100, dropout_rate=0.2):
        inputs = tf.keras.Input(shape=(None, nb_features), name='inputs')

        x = tf.keras.layers.Masking(mask_value=MASK_VALUE)(inputs)

        x = tf.keras.layers.LSTM(hidden_units,
                                 return_sequences=True,
                                 dropout=dropout_rate)(x)

        dense = tf.keras.layers.Dense(nb_skills, activation='sigmoid')
        outputs = tf.keras.layers.TimeDistributed(dense, name='outputs')(x)

        super(DKTModel, self).__init__(inputs=inputs,
                                       outputs=outputs,
                                       name="DKTModel")

    def compile(self, optimizer, metrics=None):
        """Configures the model for training.
        Arguments:
            optimizer: String (name of optimizer) or optimizer instance.
                See `tf.keras.optimizers`.
            metrics: List of metrics to be evaluated by the model during training
                and testing. Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary, such as
                `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
                You can also pass a list (len = len(outputs)) of lists of metrics
                such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
                `metrics=['accuracy', ['accuracy', 'mse']]`.
        Raises:
            ValueError: In case of invalid arguments for
                `optimizer` or `metrics`.
        """

        def custom_loss(y_true, y_pred):
            y_true, y_pred = get_target(y_true, y_pred)
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)

        super(DKTModel, self).compile(
            loss=custom_loss,
            optimizer=optimizer,
            metrics=metrics,
            experimental_run_tf_function=False)

    def fit(self,
            dataset,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_data=None,
            shuffle=True,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1):
        """Trains the model for a fixed number of epochs (iterations on a dataset).
        Arguments:
            dataset: A `tf.data` dataset. Should return a tuple
                of `(inputs, (skills, targets))`.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                Note that the progress bar is not particularly useful when
                logged to a file, so verbose=2 is recommended when not running
                interactively (eg, in a production environment).
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch)
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. The default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If x is a
                `tf.data` dataset, and 'steps_per_epoch'
                is None, the epoch will run until the input dataset is exhausted.
            validation_steps: Only relevant if `validation_data` is provided.
                Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If
                'validation_steps' is None, validation
                will run until the `validation_data` dataset is exhausted.
            validation_freq: Only relevant if validation data is provided. Integer
                or `collections_abc.Container` instance (e.g. list, tuple, etc.).
                If an integer, specifies how many training epochs to run before a
                new validation run is performed, e.g. `validation_freq=2` runs
                validation every 2 epochs. If a Container, specifies the epochs on
                which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
                validation at the end of the 1st, 2nd, and 10th epochs.
        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        Raises:
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        return super(DKTModel, self).fit(x=dataset,
                                         epochs=epochs,
                                         verbose=verbose,
                                         callbacks=callbacks,
                                         validation_data=validation_data,
                                         shuffle=shuffle,
                                         initial_epoch=initial_epoch,
                                         steps_per_epoch=steps_per_epoch,
                                         validation_steps=validation_steps,
                                         validation_freq=validation_freq)

    def eval(self,
                 dataset,
                 verbose=1,
                 steps=None,
                 callbacks=None):
        """Returns the loss value & metrics values for the model in test mode.
        Computation is done in batches.
        Arguments:
            dataset: `tf.data` dataset. Should return a
            tuple of `(inputs, (skills, targets))`.
            verbose: 0 or 1. Verbosity mode.
                0 = silent, 1 = progress bar.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.
                If x is a `tf.data` dataset and `steps` is
                None, 'evaluate' will run until the dataset is exhausted.
                This argument is not supported with array inputs.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
                See [callbacks](/api_docs/python/tf/keras/callbacks).
        Returns:
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        Raises:
            ValueError: in case of invalid arguments.
        """
        return super(DKTModel, self).evaluate(dataset,
                                              verbose=verbose,
                                              steps=steps,
                                              callbacks=callbacks)

    def evaluate_generator(self, *args, **kwargs):
        raise SyntaxError("Not supported")

    def fit_generator(self, *args, **kwargs):
        raise SyntaxError("Not supported")
        

class BinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(BinaryAccuracy, self).update_state(y_true=true,
                                                 y_pred=pred,
                                                 sample_weight=sample_weight)


class AUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(AUC, self).update_state(y_true=true,
                                                 y_pred=pred,
                                                 sample_weight=sample_weight)


class Precision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(Precision, self).update_state(y_true=true,
                                            y_pred=pred,
                                            sample_weight=sample_weight)


class Recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(Recall, self).update_state(y_true=true,
                                         y_pred=pred,
                                         sample_weight=sample_weight)


class SensitivityAtSpecificity(tf.keras.metrics.SensitivityAtSpecificity):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(SensitivityAtSpecificity, self).update_state(y_true=true,
                                                           y_pred=pred,
                                                           sample_weight=sample_weight)


class SpecificityAtSensitivity(tf.keras.metrics.SpecificityAtSensitivity):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(SpecificityAtSensitivity, self).update_state(y_true=true,
                                                           y_pred=pred,
                                                           sample_weight=sample_weight)


class FalseNegatives(tf.keras.metrics.FalseNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(FalseNegatives, self).update_state(y_true=true,
                                                 y_pred=pred,
                                                 sample_weight=sample_weight)


class FalsePositives(tf.keras.metrics.FalsePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(FalsePositives, self).update_state(y_true=true,
                                                 y_pred=pred,
                                                 sample_weight=sample_weight)


class TrueNegatives(tf.keras.metrics.TrueNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(TrueNegatives, self).update_state(y_true=true,
                                                y_pred=pred,
                                                sample_weight=sample_weight)


class TruePositives(tf.keras.metrics.TruePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        true, pred = get_target(y_true, y_pred)
        super(TrueNegatives, self).update_state(y_true=true,
                                                y_pred=pred,
                                                sample_weight=sample_weight)

####load the baseline data
fn = "df_base.csv" # Dataset path
verbose = 1 # Verbose = {0,1,2}
best_model_weights = "weights/bestmodel" # File to save the model.
log_dir = "logs" # Path to save the logs.
optimizer = "adam" # Optimizer to use
lstm_units = 100 # Number of LSTM units
batch_size = 32 # Batch size
epochs = 10 # Number of epochs to train
dropout_rate = 0.3 # Dropout rate
test_fraction = 0.3 # Portion of data to be used for testing
validation_fraction = 0.2 # Portion of training data to be used for validation


dataset, length, nb_features, nb_skills = load_dataset(fn=fn, batch_size=batch_size, shuffle = True)

train_set, test_set, val_set = split_dataset(dataset=dataset,
                                                       total_size=length,
                                                       test_fraction=test_fraction,
                                                       val_fraction=validation_fraction)


set_sz = length * batch_size
test_set_sz = (set_sz * test_fraction)
val_set_sz = (set_sz - test_set_sz) * validation_fraction
train_set_sz = set_sz - test_set_sz - val_set_sz
print("============= Data Summary =============")
print("Total number of students: %d" % set_sz)
print("Training set size: %d" % train_set_sz)
print("Validation set size: %d" % val_set_sz)
print("Testing set size: %d" % test_set_sz)
print("Number of skills: %d" % nb_skills)
print("Number of features in the input: %d" % nb_features)
print("========================================")

student_model = DKTModel(
                        nb_features=nb_features,
                        nb_skills=nb_skills,
                        hidden_units=lstm_units,
                        dropout_rate=dropout_rate)

student_model.compile(
        optimizer=optimizer,
        metrics=[
            BinaryAccuracy(),
            AUC(),
            Precision(),
            Recall()
        ])

student_model.summary()


history = student_model.fit(dataset=train_set,
                            epochs=epochs,
                            verbose=verbose,
                            validation_data=val_set,
                            callbacks=[ 
                                tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
                                tf.keras.callbacks.ModelCheckpoint(best_model_weights,
                                                                   save_best_only=True,
                                                                   save_weights_only=True),
                                tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ])


result = student_model.evaluate(test_set, verbose=verbose)

####load the disengaged-adjusted data
fn = "df_diseng.csv" # Dataset path
verbose = 1 # Verbose = {0,1,2}
best_model_weights = "weights/bestmodel" # File to save the model.
log_dir = "logs" # Path to save the logs.
optimizer = "adam" # Optimizer to use
lstm_units = 100 # Number of LSTM units
batch_size = 32 # Batch size
epochs = 10 # Number of epochs to train
dropout_rate = 0.3 # Dropout rate
test_fraction = 0.3 # Portion of data to be used for testing
validation_fraction = 0.2 # Portion of training data to be used for validation



dataset, length, nb_features, nb_skills = load_dataset(fn=fn, batch_size=batch_size, shuffle = True)

train_set, test_set, val_set = split_dataset(dataset=dataset,
                                                       total_size=length,
                                                       test_fraction=test_fraction,
                                                       val_fraction=validation_fraction)


set_sz = length * batch_size
test_set_sz = (set_sz * test_fraction)
val_set_sz = (set_sz - test_set_sz) * validation_fraction
train_set_sz = set_sz - test_set_sz - val_set_sz
print("============= Data Summary =============")
print("Total number of students: %d" % set_sz)
print("Training set size: %d" % train_set_sz)
print("Validation set size: %d" % val_set_sz)
print("Testing set size: %d" % test_set_sz)
print("Number of skills: %d" % nb_skills)
print("Number of features in the input: %d" % nb_features)
print("========================================")

student_model = DKTModel(
                        nb_features=nb_features,
                        nb_skills=nb_skills,
                        hidden_units=lstm_units,
                        dropout_rate=dropout_rate)

student_model.compile(
        optimizer=optimizer,
        metrics=[
            BinaryAccuracy(),
            AUC(),
            Precision(),
            Recall()
        ])

student_model.summary()


history = student_model.fit(dataset=train_set,
                            epochs=epochs,
                            verbose=verbose,
                            validation_data=val_set,
                            callbacks=[ 
                                tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
                                tf.keras.callbacks.ModelCheckpoint(best_model_weights,
                                                                   save_best_only=True,
                                                                   save_weights_only=True),
                                tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ])


result = student_model.evaluate(test_set, verbose=verbose)







