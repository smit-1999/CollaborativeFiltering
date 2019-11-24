import os
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import time


def get_dataset_directory():
    """
    This method fetches the directory of the folder where our current
    dataset resides using relative path.
    :return: It returns the directory of the dataset folder.
    """
    curr_dir = os.getcwd()  # gets the current working directory
    dataset_dir = os.path.join(curr_dir, "dataset")  # concatenates
    return dataset_dir


def initialise():
    """
    This method initialises all required matrices for prediction.
    It creates a list of lists data which contains tuples of the form
    (userid, movieid, rating).df is the dataframe obtained using this data
    df is then split into train and test parts.

    :return: It returns utility matrix and the mean movie matrix,
    mean user matrix and util_user matrix.
    """
    dataset_directory = get_dataset_directory()
    dataset = os.path.join(dataset_directory, "ratings.dat")
    file = open(dataset, "r")
    dict = {}       # dictionary of {movie:{user:rating}}
    dict_mean = {}  # dictionary of {movieid : mean(of user ratings)}

    dict_user = {}  # dictionary of {user:{movie:rating}} for train dataframe
    user_mean = {}  # dictionary of {userid : mean of movie ratings awtched}
    global_sum = 0
    cnt = 0
    data = []
    for line in file:
        fields = line.split("::")
        userid = fields[0]
        movieid = fields[1]
        rating = fields[2]
        global_sum += int(rating)
        cnt += 1
        data.append([userid, movieid, rating])

    df = pd.DataFrame(data, columns=['userid', 'movieid', 'rating'])
    train_df, test_df = train_test_split(df, test_size=0.00001)
    # print(train_df)
    print('size of test_df:', len(test_df))
    # print(test_df)
    for tuple in train_df.itertuples():
        userid = tuple[1]
        movieid = tuple[2]
        rating = tuple[3]

        if movieid in dict.keys():
            dict[movieid][userid] = int(rating)
            dict_mean[movieid] += int(rating)
        else:
            dict[movieid] = {}
            dict[movieid][userid] = int(rating)
            dict_mean[movieid] = int(rating)
        
        # for dict_user and user_mean dictionary
        if userid in user_mean.keys():
            dict_user[userid][movieid] = int(rating)
            user_mean[userid] += int(rating)
        else:
            dict_user[userid] = {}
            dict_user[userid][movieid] = int(rating)
            user_mean[userid] = int(rating)
            
    for key in dict_mean.keys():
        dict_mean[key] = dict_mean[key]/len((dict[key]))
        
    for user in user_mean.keys():
        user_mean[user] = user_mean[user]/len(dict_user[user])

    return dict, dict_mean, user_mean, (global_sum/cnt), test_df, dict_user


def mod(dict_a):
    """
    This function calculates the magnitude of a vector(here, dictionary)
    :param dict_a: {userid : rating}
    :return: Magnitude of a vector
    """
    val = 0
    for key in dict_a.keys():
        val = val + (dict_a[key]*dict_a[key])
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

    if mod(dict_a) == 0:
        return 0

    if mod(dict_b) == 0:
        return 0

    score = score/((mod(dict_a))*(mod(dict_b)))
    return score


def normalize(dict, dict_mean):
    """
    This method normalizes the dict dictionary by subtracting the mean
    from each nonzero value.
    :param dict:It is a dictionary of dictionaries which contains
        {movieid:{userid:rating}}
    :param dict_mean:dictionary of the form {movieid:(mean of ratings
    given to this movie by users who have watched it)}
    :return:It updates the dictionaries given in argument and returns them
    """
    for movie, user_rating in dict.items():
        mean_val = dict_mean[movie]
        for user, rating in user_rating.items():
            # print(type(rating))
            dict[movie][user] = rating - mean_val
#    print(dict)
    return dict, dict_mean


def pairwise_sim(movie, user, utility_matrix):
    """
    Ths method iterates through each movie and calculates similarity
    dictionary which stores {movie:(sim(movieid, movie))}
    :param movie: movieid, the movie whose similarity is calculated
    wrt other movies
    :param user:the user whose rating is to be predicted
    :param utility_matrix: dictionary of dictionaries {movieid:{userid:rating}}
    :return:It returns a dictionary which stores similarity of each movie wrt
    the movieid
    """
    similarity_dict = {}
    for movie2 in utility_matrix:
        similarity_dict[movie2] = similarity(utility_matrix[movie], utility_matrix[movie2])

    return similarity_dict


def predicted_rating(similarity_matrix, movieid, userid, utility_matrix, util_mean, global_mean, util_user_mean):
    """
    This method is the core method which predicts the rating based on
    similarity values
    :param similarity_matrix: the output of pairwise_sim
    :param movieid: the movie for which we need to predict the rating
    :param userid: the userid for which rating has to be predicted
    :param utility_matrix: dictionary of dictionary which stores {
    movie:{user:rating}}
    :param util_mean: dictionary of every movie wih its mean rating
    :param global_mean:Global mean of all movie ratings
    :param util_user_mean:dictionary of every user wih its mean rating
    :return:Predicted rating for userid,movieid
    """
    num = 0
    num_baseline = 0
    den = 1

    for movie in similarity_matrix.keys():
        if similarity_matrix[movie] > 0 and similarity_matrix[movie] != 1 and (userid in utility_matrix[movie].keys()):
            # print('similarity val=', similarity_matrix[movie], 'for movie:', movie)
            # print((utility_matrix[movie].keys()))
            # print(utility_matrix[movie][userid])
            num += (similarity_matrix[movie]*(utility_matrix[movie][userid]+util_mean[movie]))
            num_baseline += (similarity_matrix[movie]*((utility_matrix[movie][userid]+util_mean[movie]) -
                             (global_mean + util_mean[movie]-global_mean + util_user_mean[userid] - global_mean)))
            den += similarity_matrix[movie]
    return (num/den), (num_baseline/den)


def pairwise_sim_user(userid, util_user):
    similarity_dict = {}
    for user in util_user:
        similarity_dict[user] = similarity(util_user[userid], util_user[user])

    return similarity_dict


def user_user(util_user, userid, movieid,  user_mean):
    """
    This is a method to predict value using user-user filtering
    :param util_user: Dictionary of dictionaries
    :param userid: Target user
    :param movieid: Target movie
    :param user_mean: dictionary of user and its mean ratings
    :return:Predicted value
    """
    sim_user_dict = pairwise_sim_user(userid, util_user)
    num = 0
    den = 1

    for user in sim_user_dict.keys():
        if sim_user_dict[user] > 0 and sim_user_dict[user] != 1 and (movieid in util_user[user].keys()):
            num += sim_user_dict[user]*(util_user[user][movieid]+user_mean[user])
            den += sim_user_dict[user]
    return num/den


def main():
    start_time = time.time()
    utility_matrix,  util_mean, user_mean, global_mean, test_df, util_user = initialise()

    # print('utility matrix:', utility_matrix)
    # print('test_df', test_df)
    utility_matrix, util_mean = normalize(utility_matrix, util_mean)
    util_user, user_mean = normalize(util_user, user_mean)
    rmse = 0
    rmse_baseline = 0
    rmse_user = 0
    mae = mae_user = mae_baseline = 0
    elapsed_time = time.time() - start_time
    print('Time for initialising:', elapsed_time)

    start_time_run = time.time()
    for tuple in test_df.itertuples():
        user = tuple[1]
        movie = tuple[2]
        rating = int(tuple[3])
        print('In test_df user:', user, ' movie:', movie, 'actual rating:', rating)
        # print('prinitng train data for movie:', utility_matrix[movie])
        similarity_matrix = pairwise_sim((tuple[2]), (tuple[1]), utility_matrix)
        pred_val, pred_baseline = predicted_rating(similarity_matrix, tuple[2], tuple[1], utility_matrix, util_mean,
                                                   global_mean, user_mean)
        pred_user = user_user(util_user, user, movie, user_mean)
        bx = 0
        bi = 0
        if user in user_mean.keys():
            bx = user_mean[user] - global_mean
        if movie in util_mean.keys():
            bi = util_mean[movie] - global_mean
        pred_baseline += (global_mean + bx + bi)
        mae += (rating-pred_val)
        rmse += (pred_val - rating)**2
        mae_baseline += (rating-pred_baseline)
        rmse_baseline += (pred_baseline - rating)**2
        mae_user += (rating-pred_user)
        rmse_user += (pred_user-rating)**2
        print('pred_val:', pred_val, 'pred_baseline:', pred_baseline)
        print('pred val using user user:', pred_user)
        print('############')
    end_time_run = time.time()-start_time_run
    print('Time for predicting test dataset:', end_time_run)
    print("RMSE:", math.sqrt(rmse/len(test_df)), 'mae:', mae/len(test_df))
    print("RMSE Baseline: ", math.sqrt(rmse_baseline/len(test_df)), 'mae_baseline:', mae_baseline/len(test_df))
    print('RMSE_user:', rmse_user, 'mae_user:', mae_user/len(test_df))


main()
