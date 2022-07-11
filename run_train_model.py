import argparse
from src.models.model_train import ModelTrain


if __name__ == "__main__":

        # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",required=True,
                    help="path to the targets input file")
    ap.add_argument("-b", "--bestmodel", required=True ,
                    help="path to bestmodel dictionary")
    ap.add_argument("-s", "--scale", type=str,default='yes',
                    choices=["yes","no"],
                     help="choose to normalize data (default=yes)")
    ap.add_argument("-t", "--testset", type=float,default=0.1,
	                help="test set size")
    ap.add_argument("-r", "--random", type=str,default='no',
                    choices=["yes","no"],
	                help="set split test set to random 'yes' or fixed seed 'no' ")
    ap.add_argument("-c", "--cpus", type=int,default=4,
	                help="number of cpus to use in the models (default=4)")
    ap.add_argument("-o", "--output",required=True,
                    help="path to the model selection output")

    args = vars(ap.parse_args())
    kwargs = vars(ap.parse_args())

    ModelTrain(**kwargs)