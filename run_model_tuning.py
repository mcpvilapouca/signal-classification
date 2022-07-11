import argparse
from src.models.model_tuning import ModelTuning


if __name__ == "__main__":

        # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--target",required=True,
                    help="path to the targets input file")
    ap.add_argument("-s", "--scale", type=str,default='yes',
                    choices=["yes", "no"],
                    help="normalize the data (default: yes)")
    ap.add_argument("-m", "--model", type=str,required=True,
                    choices=["knn","svm","rf","gb","xgb"],
                    help="choose model to tune")
    ap.add_argument("-n", "--nsplits", type=int,default=5,
	                help="nsplits in stratified kfold (default=5)")
    ap.add_argument("-i", "--niters", type=int,default=1000,
	                help="niters RandomizedSearchCV (default=1000)")
    ap.add_argument("-c", "--cpus", type=int,default=10,
	                help="number of cpus to use in RandomizedSearchCV (default=10)")
    ap.add_argument("-o", "--output",required=True,
                    help="path to the model selection output")

    args = vars(ap.parse_args())
    kwargs = vars(ap.parse_args())

    ModelTuning(**kwargs)