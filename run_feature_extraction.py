import argparse
from src.features.feature_extraction import FeatureExtraction


if __name__ == "__main__":

        # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--target",required=True,
                    help="path to the targets input file")
    ap.add_argument("-n", "--nsegments", type=int, default=10,
                    help="number of signal segments for feature extraction (default=10)")
    ap.add_argument("-s", "--signal", type=str, default='raw',
	                choices=["raw", "resample"],help="choose between raw or resampled signals (default=raw)")
    ap.add_argument("-", "--remove", type=str, default='no',
	                choices=["no", "yes"],help="choose yes to remove outliers (default=no)")
    ap.add_argument("-o", "--output",required=True,
                    help="path to the FeatureExtraction output")

    args = vars(ap.parse_args())
    kwargs = vars(ap.parse_args())

    FeatureExtraction(**kwargs)