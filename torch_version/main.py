import torch 

import model.cnn_model as cnn_model

import os 
import json
import argparse 
import sys

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
    parser.add_argument("--data", type= str, help='input_data')

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
    
    # if os.path.exists("./log/") == False:
    #     os.mkdir("./log/")
    if os.path.exists("./weight/") == False:
        os.mkdir("./weight/")

    if FLAGS.model == "CNN":
        if os.path.exists("./weight/CNN/") == False:
            os.mkdir("./weight/CNN/")

        if FLAGS.train:
            model = cnn_model.Model(FLAGS.keep_prob, FLAGS.class_num)
            print(model)

            try:
                model.load_state_dict(torch.load("./weight/CNN/weight.pt"))
                print("\n***\nCheckpoint found\nModel Restored\n***\n")
            except:
                print("\n***\nNo Checkpoint found\nTraining from begining\n***\n")

            trainer = cnn_model.Trainer(FLAGS.data, model, FLAGS.bsize, FLAGS.lr, FLAGS.epoch)
            trainer.train()

            torch.save(model.state_dict(), "./weight/CNN/weight.pt")

        if FLAGS.predict:
            model = cnn_model.Model(FLAGS.keep_prob, FLAGS.class_num)
            print(model)

            try:
                model.load_state_dict(torch.load("./weight/CNN/weight.pt"))
                print("\n***\nCheckpoint found\nModel Restored\n***\n")
            except:
                print("\n***\nNo Checkpoint found\nPrediction Abort, train the model first.\n***\n")
                sys.exit()
            
            predictor = cnn_model.Predictor(FLAGS.data, model)
            predictor.predict()