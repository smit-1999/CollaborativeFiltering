import os
import pandas as pd
import math

def get_dataset_directory():
    curr_dir = os.getcwd()  # gets the current working directory
    dataset_dir = os.path.join(curr_dir, "dataset")  # concatenates
    return dataset_dir


def initialise():
    dataset_directory = get_dataset_directory()
    dataset = os.path.join(dataset_directory, "test_ratings.dat")
    file = open(dataset, "r")
    dict = {}  # dictionary of {movie:{user:rating}}
    dict_mean = {}  # dictionary of {movieid : mean(of user ratings)}

    for line in file:
        fields = line.split("::")
        userid = fields[0]
        movieid = fields[1]
        rating = fields[2]

        if movieid in dict.keys():
            dict[movieid][userid] = int(rating)
            dict_mean[movieid] += int(rating)
        else:
            dict[movieid] = {}
            dict[movieid][userid] = int(rating)
            dict_mean[movieid] = int(rating)

    for key in dict_mean.keys():
        dict_mean[key] = dict_mean[key]/len((dict[key]))


    return dict, dict_mean


def mod(dict_a):
    """
    This function calculates the magnitude of a vector(here, dictionary)
    :param dict_a: {userid : rating}
    :return: Magnitude of a vector
    """
    val = 0
    for key in dict_a.keys():
        val = val + dict_a[key]**2
    return math.sqrt(val)


def similarity(dict_a, dict_b):
    """
    This function calculates the similarity between two movies
    :param dict_a: Contains the ratings given to movie A (userid:rating)
    :param dict_b: Contains the ratings given to movie B (userid:rating)
    :return: Similarity between two movies
    """

    min_dict_length = min(len(dict_a), len(dict_b))
    score = 0
    if min_dict_length == len(dict_a):
        for key in dict_a.keys():
            if key in dict_b.keys():
                score += dict_a[key]*dict_b[key]
    else:
        for key in dict_b.keys():
            if key in dict_a.keys():
                score += dict_a[key] * dict_b[key]

    # print(score, mod(dict_a), mod(dict_b))
    score = score/((mod(dict_a))*(mod(dict_b)))
    return score


def normalize(dict, dict_mean):
    for movie, user_rating in  dict.items():
        mean_val = dict_mean[movie]
        for user, rating in user_rating.items():
            # print(type(rating))
            dict[movie][user] = rating - mean_val
#    print(dict)
    return dict, dict_mean


def pairwise_sim(movie, user, utility_matrix):
    similarity_dict = {}
    for movie2 in utility_matrix:
        similarity_dict[movie2] = similarity(utility_matrix[str(movie)], utility_matrix[str(movie2)])
#    print(similarity_dict)
    return similarity_dict


def predicted_rating(similarity_matrix, userid, movieid, utility_matrix, util_mean):
    num = 0
    den = 1

    for movie in similarity_matrix.keys():

        if similarity_matrix[movie] > 0 and similarity_matrix[movie] != 1 and (userid in utility_matrix[movie].keys()):
            print('similarity val=', similarity_matrix[movie], 'for movie:', movie)
            print((utility_matrix[movie].keys()))
            print(utility_matrix[movie][userid])
            num += (similarity_matrix[movie]*(utility_matrix[movie][userid])+util_mean[movie])
            den += similarity_matrix[movie]
    print('num=', num, 'den:', den)
    return num/den


def main():
    utility_matrix, util_mean = initialise()


    """"
    print(utility_matrix)
    score = similarity(utility_matrix['1'], utility_matrix['3'])
    print(score)
    """

    utility_matrix, util_mean = normalize(utility_matrix, util_mean)
    similarity_matrix = pairwise_sim(1, 5, utility_matrix)
    pred_val = predicted_rating(similarity_matrix, '5', '1', utility_matrix, util_mean)
    print(pred_val)


main()


# get dataset in dataframe
# find similarity between users
# find row mean
# find baseline estimate bx + bi + myu
