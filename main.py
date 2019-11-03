import os
import pandas as pd


def get_dataset_directory():
    curr_dir = os.getcwd()  # gets the current working directory
    dataset_dir = os.path.join(curr_dir, "dataset")  # concatenates
    return dataset_dir


def initialise():
    dataset_directory = get_dataset_directory()


def main():
    utility_matrix = initialise()


main()

