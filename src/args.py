
import argparse


parser = argparse.ArgumentParser()
# parser.add_argument("-m", "--model", help="Model Name")


parser.add_argument("-n", "--name", 
                    default="final_fr", help="Run Name")

parser.add_argument("-b", "--batch_size", type=int,
                    default=8, help="Batch Size")
parser.add_argument("-s", "--size", type=int,
                    default=600, help="size of input (has to divide 500)")
parser.add_argument("-nc", "--num_curves", type=int,
                    default=9, help="Number of curves to regress")
parser.add_argument("-ne", "--num_epochs", type=int,
                    default=60, help="Number of epochs")


parser.add_argument("-fe", "--feat_ext", 
                    default="fcn_resnet50", help="Feature Extractor")
parser.add_argument("-rh", "--reg_head", 
                    default="LinearHead", help="Regression Head")


parser.add_argument("-opt", "--optimizer", 
                    default="adam", help="optimizer")



parser.add_argument("-lr", "--learning_rate", type=float,
                    default=0.01, help="Learning Rate")

parser.add_argument("-mom", "--momentum", type=float,
                    default=0.9, help="Momentum")

