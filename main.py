import os
import pandas as pd


def get_dataset_directory():
    curr_dir = os.getcwd()  # gets the current working directory
    dataset_dir = os.path.join(curr_dir, "dataset")  # concatenates
    return dataset_dir


def initialise():
    dataset_directory = get_dataset_directory()
    dataset = os.path.join(dataset_directory, "test_rating.dat")
    file = open(dataset, "r")
    dict = {}  # dictionary of {movie:{user:rating}}
    dict_mean = {}  # dictionary of {movieid : user ratings mean}

    for line in file:
        fields = line.split("::")
        userid = fields[0]
        movieid = fields[1]
        rating = fields[2]

        if movieid in dict.keys():
            dict[movieid][userid] = rating
            dict_mean[movieid] += int(rating)
        else:
            dict[movieid] = {}
            dict[movieid][userid] = rating
            dict_mean[movieid] = int(rating)

    for key in dict_mean.keys():
        dict_mean[key] = dict_mean[key]/len((dict[key]))

    print(dict_mean)
    return dict, dict_mean


def main():
    utility_matrix, util_mean = initialise()


main()


# get dataset in dataframe
# find similarity between users
# find row mean
# find baseline estimate bx + bi + myu
