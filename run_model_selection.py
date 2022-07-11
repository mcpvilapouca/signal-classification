import argparse
from src.models.model_selection import ModelSelection


if __name__ == "__main__":

        # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--target",required=True,
                    help="path to the targets input file")
    ap.add_argument("-s", "--scale", type=str,
                    choices=["yes", "no"],
                    help="normalize the data (default: yes)")
    ap.add_argument("-n", "--nsplits", type=int,
	                help="nsplits in repeated stratified kfold (default=5)")
    ap.add_argument("-r", "--nrepeats", type=int,
	                help="nrepeats in repeated stratified kfold (default=10)")
    ap.add_argument("-c", "--cpus", type=int,
	                help="number of cpus to use in each available model and stratifiedkfold (default=4)")
    ap.add_argument("-o", "--output",required=True,
                    help="path to the model selection output")

    args = vars(ap.parse_args())
    kwargs = vars(ap.parse_args())

    ModelSelection(**kwargs)