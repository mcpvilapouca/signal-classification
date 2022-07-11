import argparse
from src.features.feature_selection import FeatureSelection


if __name__ == "__main__":

        # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--target",required=True,
                    help="path to the targets input file")
    ap.add_argument("-u", "--uppercorrelation", type=float,default=0.85,
                    help="threshold to remove features with higher correlation (default=0.85)")
    ap.add_argument("-l", "--lowercorrelation", type=float,default=0.05,
	                help="threshold to remove features with small correlation with target (default=0.05)")
    ap.add_argument("-o", "--output",required=True,
                    help="path to the FeatureExtraction output")

    args = vars(ap.parse_args())
    kwargs = vars(ap.parse_args())

    FeatureSelection(**kwargs)