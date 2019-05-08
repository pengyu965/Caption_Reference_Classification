import model.cnn_model as cnn_model
import model.lstm_model as lstm_model
import os
import json
import argparse
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default = "CNN", 
                        help = "Choose the one model from LSTM and CNN")
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--predict', action='store_true',
                        help='Get prediction result')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    # parser.add_argument('--load', type=int, default=99,
                        # help='Epoch id of pre-trained model')
    parser.add_argument("--data", type= str, help='training_data')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--epoch', type=int, default = 10, 
                        help = "Max Epoch")
    parser.add_argument('--bsize', type=int, default=60,
                        help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.4,
                        help='Keep probability for dropout')
    parser.add_argument('--class_num', type=int, default = 3, 
                        help='class number')
    # parser.add_argument('--maxepoch', type=int, default=100,
    #                     help='Max number of epochs for training')

    # parser.add_argument('--im_name', type=str, default='.png',
    #                     help='Part of image name')

    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = get_args()
    
    if os.path.exists("./log/") == False:
        os.mkdir("./log/")
    if os.path.exists("./weight/") == False:
        os.mkdir("./weight")

    if FLAGS.model == "CNN":
        if FLAGS.train:
            model = cnn_model.Model(FLAGS.bsize, FLAGS.lr, FLAGS.keep_prob, FLAGS.class_num, is_training=True)
            trainer = cnn_model.Trainer(FLAGS.data, model, FLAGS.epoch, FLAGS.class_num)
            
            writer = tf.summary.FileWriter("./log/cnn_log")
            saver = tf.train.Saver()

            with tf.Session() as sess:
                # print(tf.trainable_variables())
                sess.run(tf.global_variables_initializer())
                try:
                    saver.restore(sess, "./weight/cnn/cnn_model.ckpt")
                    print("Checkpoint found\nModel Restored")
                except:
                    print("No Checkpoint found")

                writer.add_graph(sess.graph)
                trainer.train(sess, writer)

                save_path = saver.save(sess, "./weight/cnn/cnn_model.ckpt")
            
            writer.close()

    if FLAGS.model == "LSTM":
        if FLAGS.train:
            model = lstm_model.Model(FLAGS.bsize, FLAGS.lr, FLAGS.keep_prob, FLAGS.class_num, is_training=True)
            trainer = lstm_model.Trainer(FLAGS.data, model, FLAGS.epoch, FLAGS.class_num)
            
            writer = tf.summary.FileWriter("./log/lstm_log/")
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                try:
                    saver.restore(sess, "./weight/lstm/lstm_model.ckpt")
                    print("Checkpoint found\nModel Restored")
                except:
                    print("No Checkpoint found")

                writer.add_graph(sess.graph)
                trainer.train(sess, writer)
                
                save_path = saver.save(sess, "./weight/lstm/lstm_model.ckpt")
            
            writer.close()

