import argparse
from src.data.generate_dataframe import GenerateDataFrame

if __name__ == "__main__":

        # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--target",required=True,
                    help="path to the targets file")
    ap.add_argument("-l", "--ltargets",
                    help="list of targets ")
    ap.add_argument("-f", "--features", required=True,
                    help="path to the features file")
    ap.add_argument("-o", "--output",
                    help="generated dataframe (output)")
    ap.add_argument("-n", "--nodes", type=int,
                    help="number of points for interpolation (default=average)")

    args = vars(ap.parse_args())
    kwargs = vars(ap.parse_args())

    GenerateDataFrame(**kwargs)
