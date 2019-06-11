import numpy as np
import pandas as pd
import pyltr
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from BPRGRU import BPRGRU, identity_loss, init_normal
import argparse
from time import time
from datetime import datetime

#################### Arguments ####################
from ContextRNN import ContextRNN
from DRCF import DRCF
from DeepSessionInterestNetwork import AttentionModel
from MultiRanker import MultiRanker
from utils import indexing_items, indexing_item_context


def parse_args():

    parser = argparse.ArgumentParser(description="Run RecSys Challenge 2019 Experiments")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: bprgru', default="crnnt")

    parser.add_argument('--d', type=int, default=10,
                        help='Dimension')

    parser.add_argument('--ml', type=int, default=10,
                        help='Maximum lenght of user sequence')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Epoch number')

    parser.add_argument('--small', type=int, default=1,
                        help='Run on small dataset')

    parser.add_argument('--ns', type=str, default="city",
                        help='Negative Sample Mode : ')

    parser.add_argument('--mode', type=int, default="1",
                        help='Model Mode : ')

    return parser.parse_args()

if __name__ == '__main__':
    start = time()

    args = parse_args()

    path = args.path
    modelName = args.model
    dim = args.d
    maxlen = args.ml
    epochs = args.epochs
    small = True if args.small == 1 else False
    negSampleMode = args.ns
    mode = args.mode

    fullData = True if modelName in ["am", "crnn", "crnnt"] else False
    # small = False
    # negSampleMode = "nce"

    print("Reading......")
    if not fullData:

        cols=["user_id","session_id","timestamp","step","action_type","reference","platform","city","device","current_filters","impressions","prices","interactions"]
        df = pd.read_csv(path+"data/train.groupby.csv", sep="\t", names=cols) if not small else pd.read_csv(path+"data/train.groupby.csv", sep="\t", names=cols, nrows=10000)
        df_val = pd.read_csv(path+"data/val.groupby.csv", sep="\t", names=cols) if not small else pd.read_csv(path+"data/val.groupby.csv", sep="\t", names=cols, nrows=10000)
        df_test = pd.read_csv(path+"data/test.groupby.csv", sep="\t", names=cols) if not small else pd.read_csv(path+"data/test.groupby.csv", sep="\t", names=cols, nrows=10000)
        # metadata = pd.read_csv("data/item_metadata.csv")

        # Indexing all items
        item_index = indexing_items(df, df_val, df_test)
    else:
        df = pd.read_csv(path+"data/train.csv") if not small else pd.read_csv(path+"data/train.csv", nrows=1000)
        df = df[~df["reference"].isin(['unknown'])]
        df = df[df['reference'].str.isnumeric()]
        df_test = pd.read_csv(path + "data/test.csv") if not small else pd.read_csv(path + "data/test.csv", nrows=1000)
        df_test = df_test[~df_test["reference"].isin(['unknown'])]

        remove_actions = ['filter selection', 'change of sort order', 'search for poi', 'search for destination']
        df = df[~df.action_type.isin(remove_actions)]
        df_test = df_test[~df_test.action_type.isin(remove_actions)]
        if not small:
            # drop error rows where clickout == None are not the last action
            df_test = df_test.drop([794770, 2255685])
        df_val = df_test[df_test.groupby("session_id").cumcount(ascending=False) > 0]

        indexes = indexing_item_context(df, df_val, df_test)

        # maxlen = 16
        # save time
        # maxlen = int(np.mean(df.groupby("session_id").apply(lambda x: len(x))))






    if modelName == "bprgru":
        ranker = BPRGRU(dim, maxlen, item_index, negSampleMode)
    elif modelName == "drcf2":
        ranker = DRCF(dim, maxlen, item_index, negSampleMode)
    elif modelName == "mr":
        ranker = MultiRanker(dim, maxlen, item_index)

    elif modelName == "am":
        ranker = AttentionModel(dim, maxlen, indexes)

    elif modelName == "crnn":
        item_index, action_index = indexes
        ranker = ContextRNN(dim, maxlen, item_index, action_index, mode, 1)

    elif modelName == "crnnt":
        item_index, action_index = indexes
        ranker = ContextRNN(dim, maxlen, item_index, action_index, mode, 2)



    runName = "%s_d%d_ml%d_%s_m%d_%s" % (modelName, dim, maxlen, negSampleMode, mode, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    # Start Training

    metric = pyltr.metrics.NDCG(k=25)
    bestNDCG = 0

    if modelName in ["am", "crnn", "crnnt"]:

        valSession, x_val, y_val = ranker.generate_data(df_val, "val")
        testSession, testItemId, x_test = ranker.generate_data(df_test, "test")



        for epoch in range(epochs):

            t1 = time()
            x_train, y_train = ranker.generate_data(df, "train")
            t2 = time()
            hist = ranker.model.fit(x_train, y_train, batch_size=256, verbose=1, epochs=1, shuffle=True)
            loss = hist.history['loss'][0]
            t3 = time()
            if modelName == "am":
                pred = ranker.model.predict(x_val).flatten().tolist()
            else:
                pred = ranker.get_score(x_val)[0].flatten()

            ndcg = metric.calc_mean(valSession, y_val, pred)

            output = 'Iteration %d, data[%.1f s], train[%.1f s], loss = %.4f, NDCG = %.4f, test[%.1f s]' % (
                epoch, t2 - t1, t3-t2, loss, ndcg, time() - t3)

            with open(path+"out/%s.out" % runName, "a") as myfile:
                myfile.write(output+"\n")

            print(output)

            if bestNDCG <= ndcg:
                bestNDCG = ndcg
                ranker.model.save(path+'h5/%s.h5' % runName)
            else:
                print("Early Stopping")
                break

        if modelName == "am":
            pred = ranker.model.predict(x_test).flatten().tolist()
        else:
            pred = ranker.get_score(x_test)[0].flatten().tolist()

        df_rank = pd.DataFrame.from_dict({"session_id":testSession, "item": testItemId, "score":pred})
        _df_rank = df_rank.groupby("session_id").tail(1)
        df_rank = df_rank.groupby("session_id").apply(lambda x: x.sort_values(["score"], ascending = False)).reset_index(drop=True)
        df_rank = df_rank.groupby(['session_id'])['item'].apply(list)
        df_rank = df_rank.apply(lambda x : ' '.join(x))
        _df_rank["item_recommendations"] = df_rank.values
        _df_test = df_test.groupby("session_id").tail(1)[df_test.action_type == "clickout item"][df_test.reference.isna()]
        _df_rank = pd.merge(_df_rank, _df_test, on="session_id")
        _df_rank[['user_id','session_id', 'timestamp', 'step', 'item_recommendations']].to_csv(path+'res/submission_%s.csv' % runName, sep=',', header=True, index=False)


    else:
        valSession, x_val, y_val = ranker.generate_test_data(df_val)
        testSession, testItemId, x_test = ranker.generate_test_data(df_test, isValidate=False)



        for epoch in range(epochs):
            t1 = time()
            x_train, y_train = ranker.generate_train_data(df)
            t2 = time()
            hist = ranker.model.fit(x_train, y_train, batch_size=256, epochs=1, verbose=0, shuffle=True)
            loss = hist.history['loss'][0]
            t3 = time()
            pred = ranker.get_score(x_val)[0].flatten()
            ndcg = metric.calc_mean(valSession, y_val, pred)
            output = 'Iteration %d, data[%.1f s], train[%.1f s], loss = %.4f, NDCG = %.4f, test[%.1f s]' % (
                epoch, t2 - t1, t3-t2, loss, ndcg, time() - t3)

            with open(path+"out/%s.out" % runName, "a") as myfile:
                myfile.write(output+"\n")

            print(output)

            if bestNDCG <= ndcg:
                bestNDCG = ndcg
                ranker.model.save(path+'h5/%s.h5' % runName)
            else:
                print("Early Stopping")
                break

        # load the best model
        # ranker.model = load_model(path+'h5/%s.h5' % runName, custom_objects={'identity_loss': identity_loss, 'init_normal': init_normal})
        if modelName == "mr":
            ranker.model = load_model(path+'h5/%s.h5' % runName, custom_objects = {'identity_loss': ranker.identity_loss, 'getDotDifference': ranker.getDotDifference, 'getDotDifferenceShape': ranker.getDotDifferenceShape})

        else:
            ranker.model = load_model(path+'h5/%s.h5' % runName)


        pred = ranker.get_score(x_test)[0].flatten()

        # Generate submission file

        df_rank = pd.DataFrame.from_dict({"id":testSession, "item": testItemId, "score":pred})
        df_rank = df_rank.groupby("id").apply(lambda x: x.sort_values(["score"], ascending = False)).reset_index(drop=True)
        df_rank = df_rank.groupby(['id'])['item'].apply(list)
        df_rank = df_rank.apply(lambda x : ' '.join(x))
        df_test["item_recommendations"] = df_rank.values
        df_test[['user_id','session_id', 'timestamp', 'step', 'item_recommendations']].to_csv(path+'res/submission_%s.csv' % runName, sep=',', header=True, index=False)


    total_time = (time() - start ) / 3600
    with open(path + "out/%s.out" % runName, "a") as myfile:
        myfile.write("Total time: %.2f h" % total_time + "\n")
