import sys;

from keras.layers import Input, Embedding, GRU, Dot, Subtract, Activation, SimpleRNN
from keras.models import Model
from tqdm.autonotebook import tqdm
from keras import backend as K
import numpy as np
from keras_preprocessing.sequence import pad_sequences

from BPRGRU import BPRGRU
from utils import negative_sample

class MultiRanker(BPRGRU):

    def __init__(self, dim, maxlen, item_index, negSampleMode="normal"):
        self.dim = dim
        self.maxlen = maxlen
        self.item_index = item_index
        self.negSampleMode = negSampleMode


        posInput = Input(shape = (1, ))
        negCityInput = Input(shape = (1, ))
        negInteractInput = Input(shape = (1, ))
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
        rnn = SimpleRNN(self.dim, unroll=True)
        # featureDense = Dense(100, activation='relu')

        uEmb = rnn(itemEmbedding(seqInput))
        # uEmb = rnn(seqInput)
        # pEmb = Flatten()(itemEmbedding(posInput))
        # nEmb = Flatten()(itemEmbedding(negInput))
        pEmb = itemEmbedding(posInput)
        ncEmb = itemEmbedding(negCityInput)
        niEmb = itemEmbedding(negInteractInput)
        nEmb = itemEmbedding(negInput)

        # pPosition = featureDense(posPositionInput)
        # nPosition = featureDense(negPositionInput)
        #
        # pPrice = featureDense(posPriceInput)
        # nPrice = featureDense(negPriceInput)

        pDot = Dot(axes = -1)([uEmb, pEmb])
        ncDot = Dot(axes = -1)([uEmb, ncEmb])
        niDot = Dot(axes = -1)([uEmb, niEmb])
        nDot = Dot(axes = -1)([uEmb, nEmb])

        # pDot = Dot(([uEmb, pEmb], mode='dot')
        # nDot = merge([uEmb, nEmb], mode='dot')

        # pDot = merge([pDot, pPosition, pPrice], mode="concat")
        # nDot = merge([nDot, nPosition, nPrice], mode="concat")

        # dense = Dense(1, init='lecun_uniform')

        pScore = pDot
        ncScore = ncDot
        niScore = niDot
        nScore = nDot

        # Subtract scores.
        diff1 = Subtract()([pScore, niScore])
        diff2 = Subtract()([niScore, ncScore])
        diff3 = Subtract()([ncScore, nScore])

        # Pass difference through sigmoid function.
        prob = Subtract()([Subtract()([Activation("sigmoid")(diff1), Activation("sigmoid")(diff2)]), Activation("sigmoid")(diff3)])

        # model = Model(input = [posInput, negInput, seqInput, posPositionInput, negPositionInput, posPriceInput, negPriceInput], output = out)
        self.model = Model(inputs = [posInput, negInteractInput, negCityInput, negInput, seqInput], outputs = prob)
        self.model.compile(optimizer = "adam", loss = "binary_crossentropy")
        self.get_score = K.function([posInput, seqInput], [pScore])

    def predict(self, inp):
        # input = [testItem, testSeq, testPosition, testPrice]
        return self.get_score(inp)[0].flatten()

    def generate_train_data(self, df):

        pos, neg, negInt, negCity, seq = [], [], [], [], []

        for city, rows in tqdm(df.groupby("city")):
            for idx, row in rows.iterrows():
                impressions = [self.item_index[int(i)] for i in row['impressions'].split("|")]

                gtItem = self.item_index[int(row['reference'])]
                pos.append(gtItem)

                interactions = [self.item_index[int(i)] for i in row['interactions'].split("|")] if type(
                    row['interactions']) == str else []


                impPool = impressions if len(impressions) > 1 else np.arange(len(self.item_index)).tolist()

                intPool = interactions if len(interactions) > 1 else impPool
                cityPool = [self.item_index[i] for i in rows['reference'].unique().tolist()] if rows['reference'].nunique() > 1 else impPool


                negInt.append(negative_sample(intPool, gtItem))
                negCity.append(negative_sample(cityPool, gtItem))
                neg.append(negative_sample(impPool, gtItem))

                #             neg_feature.append(get_item_feature(sample))
                # neg_price.append(prices[impressions.index(sample)] if sample in impressions else max(prices))
                # neg_position.append(impressions.index(sample) + 1 if sample in impressions else 26)

                seq.append(interactions)
                #             feature_interactions = [get_item_feature(item_index[int(i)]) for i in row['interactions'].split("|")] if type(row['interactions']) == str else []
                #             seq_feature.append(feature_interactions)

        pos = np.array(pos)
        negInt = np.array(negInt)
        negCity = np.array(negCity)
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

        return [pos, negInt, negCity, neg, seq], [labels]
