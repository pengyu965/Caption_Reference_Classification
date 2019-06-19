import torch 

import os 
import json
import argparse 
import sys
from model import trainer

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

        import model.cnn_model as cnn_model

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

            trainer = trainer.Trainer(FLAGS.data, model, FLAGS.bsize, FLAGS.lr, FLAGS.epoch)
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
            
            predictor = trainer.Predictor(FLAGS.data, model)
            predictor.predict()
    
    if FLAGS.model == "googlenet":
        import model.googlenet_model as googlenet_model

        if os.path.exists("./weight/GoogLeNet/") == False:
            os.mkdir("./weight/GoogLeNet/")
        
        if FLAGS.train: 
            model = googlenet_model.GoogLeNet(num_classes=3)
            print(model)

            try:
                model.load_state_dict(torch.load("./weight/GoogLeNet/weight.pt"))
                print("\n***\nCheckpoint found\nModel Restored\n***\n")
            except:
                print("\n***\nNo Checkpoint found\nTraining from begining\n***\n")
            
            trainer = trainer.Trainer(FLAGS.data, model, FLAGS.bsize, FLAGS.lr, FLAGS.epoch)
            trainer.train()

            torch.save(model.state_dict(), "./weight/GoogLeNet/weight.pt")

        if FLAGS.predict:
            model = googlenet_model.GoogLeNet(num_classes=3, aux_logits=False, init_weights=False)
            model.load_state_dict(torch.load("./weight/GoogLeNet/weight.pt"))
            try:
                model.load_state_dict(torch.load("./weight/GoogLeNet/weight.pt"))
                print("\n***\nCheckpoint found\nModel Restored\n***\n")
            except:
                print("\n***\nNo Checkpoint found\nPrediction Abort, train the model first.\n***\n")
                sys.exit()
            
            predictor = trainer.Predictor(FLAGS.data, model)
            predictor.predict()
