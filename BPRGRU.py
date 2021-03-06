import sys;

from keras.layers import Input, Embedding, GRU, Dot, Subtract, Activation, SimpleRNN
from keras.models import Model
from tqdm.autonotebook import tqdm
from keras import backend as K
import numpy as np
from keras_preprocessing.sequence import pad_sequences

from utils import negative_sample

def init_normal(shape, name=None):
    return K.random_normal_variable(shape, mean = 0.0, scale=0.01, name=name)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def bpr_triplet_loss(X):
    positive_item_latent, negative_item_latent = X

    loss = 1 - K.log(K.sigmoid(positive_item_latent - negative_item_latent))

    return loss

class BPRGRU():

    def __init__(self, dim, maxlen, item_index, negSampleMode="normal"):
        self.dim = dim
        self.maxlen = maxlen
        self.item_index = item_index
        self.negSampleMode = negSampleMode


        posInput = Input(shape = (1, ))
        negInput = Input(shape = (1, ))
        seqInput = Input(shape = (self.maxlen,))

        # posPositionInput = Input(shape = (1, ))
        # negPositionInput = Input(shape = (1, ))
        #
        # posPriceInput = Input(shape = (1, ))
        # negPriceInput = Input(shape = (1, ))

        itemEmbedding = Embedding(len(item_index)+1, self.dim, mask_zero=True)

        # rnn = Sequential()
        # rnn.add(itemsEmbedding)
        # rnn.add(SimpleRNN(self.dim, unroll=True))
        # gru = SimpleRNN(self.dim, unroll=True)
        gru = SimpleRNN(self.dim, unroll=True)
        # featureDense = Dense(100, activation='relu')

        uEmb = gru(itemEmbedding(seqInput))
        # uEmb = rnn(seqInput)
        # pEmb = Flatten()(itemEmbedding(posInput))
        # nEmb = Flatten()(itemEmbedding(negInput))
        pEmb = itemEmbedding(posInput)
        nEmb = itemEmbedding(negInput)

        # pPosition = featureDense(posPositionInput)
        # nPosition = featureDense(negPositionInput)
        #
        # pPrice = featureDense(posPriceInput)
        # nPrice = featureDense(negPriceInput)

        pDot = Dot(axes = -1)([uEmb, pEmb])
        nDot = Dot(axes = -1)([uEmb, nEmb])
        # pDot = Dot(([uEmb, pEmb], mode='dot')
        # nDot = merge([uEmb, nEmb], mode='dot')

        # pDot = merge([pDot, pPosition, pPrice], mode="concat")
        # nDot = merge([nDot, nPosition, nPrice], mode="concat")

        # dense = Dense(1, init='lecun_uniform')

        rel_score = pDot
        irr_score = nDot

        # Subtract scores.
        diff = Subtract()([rel_score, irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        # out = merge([rel_score, irr_score], mode=bpr_triplet_loss, output_shape=(1,))

        # model = Model(input = [posInput, negInput, seqInput, posPositionInput, negPositionInput, posPriceInput, negPriceInput], output = out)
        self.model = Model(inputs = [posInput, negInput, seqInput], outputs = prob)
        self.model.compile(optimizer = "adam", loss = "binary_crossentropy")
        self.get_score = K.function([posInput, seqInput], [rel_score])

    def predict(self, inp):
        # input = [testItem, testSeq, testPosition, testPrice]
        return self.get_score(inp)[0].flatten()

    def generate_train_data(self, df):

        pos, neg, seq, pos_feature, neg_feature, seq_feature, pos_price, neg_price, pos_position, neg_position = [], [], [], [], [], [], [], [], [], []
        for city, rows in tqdm(df.groupby("city")):
            for idx, row in rows.iterrows():
                impressions = [self.item_index[int(i)] for i in row['impressions'].split("|")]
                # prices = [int(i) for i in row['prices'].split("|")]

                gtItem = self.item_index[int(row['reference'])]
                pos.append(gtItem)
                #             pos_feature.append(get_item_feature(gtItem))
                # pos_price.append(prices[impressions.index(gtItem)] if gtItem in impressions else max(prices))
                # pos_position.append(impressions.index(gtItem) + 1 if gtItem in impressions else 26)

                interactions = [self.item_index[int(i)] for i in row['interactions'].split("|")] if type(
                    row['interactions']) == str else []

                if self.negSampleMode == "city":


                    if rows['reference'].nunique() > 1:
                        pool = [self.item_index[i] for i in rows['reference'].unique().tolist()]
                    else:
                        pool = impressions
                        if len(pool) == 1:
                            pool = np.arange(len(self.item_index)).tolist()

                elif self.negSampleMode == "imp":
                    pool = impressions
                    if len(pool) == 1:
                        pool = np.arange(len(self.item_index)).tolist()

                elif self.negSampleMode == "nce":
                    pool = impressions
                    if len(pool) == 1:
                        pool = np.arange(len(self.item_index)).tolist()
                    else:
                        if gtItem in pool:
                            pool.remove(gtItem)
                        tmpSeq = [interactions] * len(pool)
                        tmpSeq = pad_sequences(tmpSeq, maxlen=self.maxlen)
                        # print(pool)

                        pool = np.expand_dims(pool, axis=-1)
                        pred = self.predict([np.array(pool), tmpSeq])
                        pool = [[x for _, x in sorted(zip(pred, pool))][-1]]

                sample = negative_sample(pool, gtItem)
                neg.append(sample)
                #             neg_feature.append(get_item_feature(sample))
                # neg_price.append(prices[impressions.index(sample)] if sample in impressions else max(prices))
                # neg_position.append(impressions.index(sample) + 1 if sample in impressions else 26)

                seq.append(interactions)
                #             feature_interactions = [get_item_feature(item_index[int(i)]) for i in row['interactions'].split("|")] if type(row['interactions']) == str else []
                #             seq_feature.append(feature_interactions)

        pos = np.array(pos)
        neg = np.array(neg)
        seq = pad_sequences(seq, maxlen=self.maxlen)
        # pos_feature = np.array(pos_feature)
        # neg_feature = np.array(neg_feature)
        # seq_feature = pad_sequences(seq_feature, maxlen=self.maxlen)
        # pos_price = np.array(pos_price)
        # neg_price = np.array(neg_price)
        # pos_position = np.array(pos_position)
        # neg_position = np.array(neg_position)
        labels = np.ones(len(pos))

        return [pos, neg, seq], [labels]

    def generate_test_data(self, df, isValidate=True):
        sessions, items, seq, labels, itemIds, positions, prices = [], [], [], [], [], [], []
        for city, rows in df.groupby("city"):
            for idx, row in rows.iterrows():
                impressions = [self.item_index[int(i)] for i in row['impressions'].split("|")]
                price = [int(i) for i in row['prices'].split("|")]
                interactions = [self.item_index[int(i)] for i in row['interactions'].split("|")] if type(
                    row['interactions']) == str else []

                if isValidate:
                    try:
                        gtItem = self.item_index[int(row['reference'])]
                        tmp = np.zeros(len(impressions))
                        tmp[impressions.index(gtItem)] = 1
                    except Exception as e:
                        continue
                    labels.extend(tmp)
                    sessions.extend([idx] * len(impressions))
                else:
                    sessions.extend([row['session_id']] * len(impressions))
                    itemIds.extend(row['impressions'].split("|"))
                items.extend(impressions)
                # positions.extend([i + 1 for i in range(len(impressions))])
                # prices.extend(price)
                seq.extend([interactions for i in range(len(impressions))])

        items = np.array(items)
        items = items.reshape(len(items), 1)
        seq = pad_sequences(seq, maxlen=self.maxlen)
        sessions = np.array(sessions)
        labels = np.array(labels)
        # prices = np.array(prices).reshape(len(items), 1)
        # positions = np.array(positions).reshape(len(items), 1)

        # if isValidate:
        #     return sessions, items, seq, positions, prices, labels
        # return sessions, itemIds, items, seq, positions, prices
        if isValidate:
            return sessions, [items, seq], labels
        return sessions, itemIds, [items, seq]
