import Config
import numpy as np
from tensorflow.contrib import learn
import tensorflow as tf
import ReadData as read
import os
import csv
import time
# CHANGE THIS: Load data. Load your own data here
if Config.eval_train:
    # x_raw, y_test = data_helpers.load_data_and_labels(Config.positive_data_file, Config.negative_data_file)
    x_raw, y_test = read.load_data('data/test.csv')
    y_test = np.argmax(y_test, axis=1)
    # y_test = np.array(y_test)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(Config.checkpoint_dir, "runs","1525113068", "vocab")
print vocab_path
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
# checkpoint_file = tf.train.latest_checkpoint(os.path.join(Config.checkpoint_dir,"runs"))
checkpoint_file = "/home/rasmalai/PycharmProjects/NLPProject/runs/1525113068/checkpoints/model-100"
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=Config.allow_soft_placement,
      log_device_placement=Config.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        batch_size = graph.get_operation_by_name("batch_size").outputs[0]
        pad = graph.get_operation_by_name("pad").outputs[0]
        real_len = graph.get_operation_by_name("real_len").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = read.batch_iteration(list(x_test), Config.batch_size, 1)

        # Collect the predictions here
        all_predictions = []


        def real_len1(batches):
            return [np.ceil(np.argmin(batch + [0]) * 1.0 / Config.max_pool_size) for batch in batches]


        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, feed_dict={input_x: x_test_batch, dropout_keep_prob: 1.0,real_len: real_len1(x_test_batch), batch_size: len(x_test_batch),
                    pad: np.zeros([len(x_test_batch), 1, Config.embedding_dim, 1])})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(Config.checkpoint_dir, "..", "predictioncnnrnnmultiadam.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)