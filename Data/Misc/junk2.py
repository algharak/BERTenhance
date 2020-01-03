from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
def extendList(val, list=[]):
    list.append(val)
    return list

list1 = extendList(10)
list2 = extendList(123,[])
list3 = extendList('a')

print ('list1 = %s' % list1)
print ('list2 = %s' % list2)
print ("list3 = %s" % list3)

'''



BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

for i in range(100):
    print (i)


tags = set()
a=1
a +=1

print(a)
print('2')
print('3')


is_training = True

if is_training:
    tags.add("train")
    bert_module = hub.Module(BERT_MODEL_HUB,trainable=True)
print ('hello')
print ('hello')
print ('hello')
print ('hello')
exit()




class Date(object):

    def __init__(self, day=0, month=0, year=0):
        self.day = day
        self.month = month
        self.year = year

    @classmethod
    def from_string(cls, date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        date1 = cls(day, month, year)
        return date1

    @staticmethod
    def is_date_valid(date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        return day <= 31 and month <= 12 and year <= 3999

date2 = Date.from_string('11-09-2012')
is_date = Date.is_date_valid('11-09-2012')
a = Date(20,10,1960)

b = Date.from_string('11-06-2199')

print('done')


'''


'''
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
TRAIN_DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)



# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.
# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    # Separate the label from the features
    label = features.pop('Species')
    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    # Parse each line.
    dataset = dataset.map(_parse_line)
   # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset

def read_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        reader_np = np.array(list(reader))[1:,:].astype(np.float)
        return reader_np


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    #dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)) never worked, use the next line
    dataset = (dict(features), labels)
    xx = (dict(features), labels)
    # Shuffle, repeat, and batch the examples.
    #dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    #outclass = dataset.output_classes
    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    # Convert the inputs to a Dataset.
    #dataset = tf.data.Dataset.from_tensor_slices(inputs) did not work.  see next line
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    #dataset = dataset.batch(batch_size)
    dataset = inputs
    # Return the dataset.
    return dataset


def my_model_fn(features, labels, mode, params):
    # Build the network
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    # Create 2 hidden layers with 10 units each
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    ### Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    # Return if it is in prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    ### Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    ### Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # Return if in evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    ### Create optimizer and trainer
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Create the optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

(train_x,train_y), (test_x,test_y)= load_data()
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [20, 20],
        # The model must choose between 3 classes.
        'n_classes': 3,},
    model_dir='./mod_metrics')

#classifier.train(input_fn=lambda:train_input_fn(train_x, train_y,1),steps=10000)
#classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y,1),steps=10000)
classifier.predict(input_fn=lambda:eval_input_fn(test_x, test_y,1))
print('done')


if mode == tf.estimator.ModeKeys.TRAIN:
            # init poly optimizer
            optimizer = PolyOptimizer(params)
            # define train op
            train_op = optimizer.optimize(loss, training, params["total_steps"])

            # if params["output_train_images"] is true output images during training
            if params["output_train_images"]:
                tf.summary.image("training", features["image"])
            scaffold = tf.train.Scaffold(init_op=None, init_fn=tools.fine_tune.init_weights("squeezenext",params["fine_tune_ckpt"]))
            # create estimator training spec, which also outputs the model_stats of the model to params["model_dir"]
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[
                tools.stats._ModelStats("squeezenext", params["model_dir"],
                                        features["image"].get_shape().as_list()[0])
            ],scaffold=scaffold)


'''


