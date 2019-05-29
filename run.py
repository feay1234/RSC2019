import numpy as np
import pandas as pd
import pyltr
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from BPRGRU import BPRGRU, identity_loss


# Read files
cols=["user_id","session_id","timestamp","step","action_type","reference","platform","city","device","current_filters","impressions","prices","interactions"]
df = pd.read_csv("data/train.groupby.csv", sep="\t", names=cols, nrows=100 )
df_val = pd.read_csv("data/val.groupby.csv", sep="\t", names=cols, nrows=100)
df_test = pd.read_csv("data/test.groupby.csv", sep="\t", names=cols, nrows=100)

# metadata = pd.read_csv("data/item_metadata.csv")


# Indexing all items

allItems = set(df['reference'].unique())
for i in df['impressions']:
    allItems.update([int(i) for i in i.split("|")])
for i in df['interactions']:
    if type(i) == str:
        allItems.update([int(i) for i in i.split("|")])

for i in df_val['impressions']:
    allItems.update([int(i) for i in i.split("|")])
for i in df_val['interactions']:
    if type(i) == str:
        allItems.update([int(i) for i in i.split("|")])

for i in df_test['impressions']:
    allItems.update([int(i) for i in i.split("|")])
for i in df_test['interactions']:
    if type(i) == str:
        allItems.update([int(i) for i in i.split("|")])

item_index = {int(i): idx + 1 for idx, i in enumerate(allItems)}


# Initialise params

modelName = "bprgru"
dim = 50
maxlen = 10
epochs = 50

if modelName == "bprgru":
    ranker = BPRGRU(dim, maxlen, item_index)


runName = "%s_d%d_ml%d" % (modelName, dim, maxlen)



# Start Training

metric = pyltr.metrics.NDCG(k=25)

valSession, x_val, y_val = ranker.generate_test_data(df_val)
testSession, testItemId, x_test = ranker.generate_test_data(df_test, isValidate=False)

bestNDCG = 0

for i in range(epochs):
    x_train, y_train = ranker.generate_train_data(df)
    ranker.model.fit(x_train, y_train, batch_size=256, nb_epoch=1)
    pred = ranker.get_score(x_val)[0].flatten()
    ndcg = metric.calc_mean(valSession, y_val, pred)
    print('Epoch:', i, ' Our model:',  ndcg)
    if bestNDCG < ndcg:
        bestNDCG = ndcg
        ranker.model.save('h5/%s.h5' % runName)
    else:
        print("Early Stopping")
        break

# load the best model
ranker.model = load_model('h5/%s.h5' % runName, custom_objects={'identity_loss': identity_loss})


# Generate submission file

pred = ranker.get_score(x_test)[0].flatten()

df_rank = pd.DataFrame.from_dict({"id":testSession, "item": testItemId, "score":pred})
df_rank = df_rank.groupby("id").apply(lambda x: x.sort_values(["score"], ascending = False)).reset_index(drop=True)
df_rank = df_rank.groupby(['id'])['item'].apply(list)
df_rank = df_rank.apply(lambda x : ' '.join(x))
df_test["item_recommendations"] = df_rank.values
df_test[['user_id','session_id', 'timestamp', 'step', 'item_recommendations']].to_csv('res/submission_%s.csv' % runName, sep=',', header=True, index=False)



