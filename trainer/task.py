from trainer import model
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_path",
        help="Path to a single csv file with training data",
        required=True
    )
    parser.add_argument(
        "--output_path",
        help="Path to a bucket where the trained model will be stored",
        required=True
    )
    parser.add_argument(
        "--regularization",
        help="Type of regularization used. One of l1, l2, or none",
        default="none"
    )
    
    args = parser.parse_args().__dict__

    model.train(args)
