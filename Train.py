from tensorflow.contrib import learn
import numpy as np
import ReadData as read
import tensorflow as tf
import Config
import CNN
import time
import os
import datetime
print("Loading data...")
train_data ,train_labels = read.load_data("data/train.csv")
dev_data,dev_labels = read.load_data("data/dev.csv")
train_labels=np.array(train_labels);
# train_labels=train_labels.T;

max_document_length = max([len(x.split(" ")) for x in train_data])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(train_data)))
x_dev = np.array(list(vocab_processor.fit_transform(dev_data)))
print len(x_train)
print "done loading data"

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=Config.allow_soft_placement,
      log_device_placement=Config.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        print "starting cnn sess"

        cnn = CNN.CNN(
            sequence_length=x_train.shape[1],
            num_classes=train_labels.shape[0],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=Config.embedding_dim,
            filter_sizes=list(map(int, Config.filter_sizes.split(","))),
            num_filters=Config.num_filters,
            l2_reg_lambda=Config.l2_reg_lambda)

        print "grads and optimizer part"
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        print "this is losssssssss"
        # print sess.run(loss_summary)
        print "this is accuracyyyyyy"
        # print sess.run(acc_summary)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=Config.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                print "starting a single train step"

                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: Config.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, writer=None):
                print "starting eval dev step"

                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            print "generating batches with labels"
            # Generate batches
            # batches = read.batch_iteration(
            #     list(zip(x_train, train_labels)), Config.batch_size, Config.num_epochs)
            batches = read.batch_iteration(x_train, train_labels, Config.batch_size, Config.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = batch[0],batch[1]
                print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                print len(y_batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % Config.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, dev_labels, writer=dev_summary_writer)
                    print("")
                if current_step % Config.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))


