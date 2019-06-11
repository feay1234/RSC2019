import numpy as np
from keras import Model

from keras.layers import Input, Embedding, Subtract, Activation, Concatenate, Dense, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from utils import negative_sample


class ContextRNN():
    def __init__(self, dim, maxlen, item_index, action_index, positionMode=1):
        self.dim = dim
        self.maxlen = maxlen
        self.item_index = item_index
        self.action_index = action_index
        self.positionMode = positionMode

        p_itemInput = Input(shape=(self.maxlen,))
        p_actionInput = Input(shape=(self.maxlen,))
        p_poisitionInput = Input(shape=(self.maxlen,1,)) if self.positionMode == 1 else Input(shape=(self.maxlen,))
        p_priceInput = Input(shape=(self.maxlen,1,))

        n_itemInput = Input(shape=(self.maxlen,))
        n_actionInput = Input(shape=(self.maxlen,))
        n_poisitionInput = Input(shape=(self.maxlen,1,)) if self.positionMode == 1 else Input(shape=(self.maxlen,))
        n_priceInput = Input(shape=(self.maxlen,1,))

        itemEmbedding = Embedding(len(item_index) + 1, self.dim, mask_zero=True)
        actionEmbedding = Embedding(len(action_index) + 1, self.dim, mask_zero=True)
        positionEmbedding = Embedding(26 + 1, self.dim, mask_zero=True)

        piEmb = itemEmbedding(p_itemInput)
        niEmb = itemEmbedding(n_itemInput)
        paEmb = actionEmbedding(p_actionInput)
        naEmb = actionEmbedding(n_actionInput)
        # ppoEmb = K.expand_dims(p_poisitionInput, axis=-1)
        # npoEmb = K.expand_dims(n_poisitionInput, axis=-1)
        # pprEmb = K.expand_dims(p_priceInput, axis=-1)
        # nprEmb = K.expand_dims(n_priceInput, axis=-1)
        if self.positionMode == 1:
            p_features = Concatenate()([piEmb, paEmb, p_poisitionInput, p_priceInput])
            n_features = Concatenate()([niEmb, naEmb, n_poisitionInput, n_priceInput])
        else:
            ppEmb = positionEmbedding(p_poisitionInput)
            npEmb = positionEmbedding(n_poisitionInput)
            p_features = Concatenate()([piEmb, paEmb, ppEmb, p_priceInput])
            n_features = Concatenate()([niEmb, naEmb, npEmb, n_priceInput])


        rnn = SimpleRNN(self.dim, unroll=True)

        dense = Dense(1)

        rel_score = dense(rnn(p_features))
        irr_score = dense(rnn(n_features))

        # Subtract scores.
        diff = Subtract()([rel_score, irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        self.model = Model(
            inputs=[p_itemInput, p_actionInput, p_poisitionInput, p_priceInput, n_itemInput, n_actionInput,
                    n_poisitionInput, n_priceInput], outputs=prob)

        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.get_score = K.function([p_itemInput, p_actionInput, p_poisitionInput, p_priceInput], [rel_score])

    def get_features(self, item, impressions, prices, row):

        action = self.action_index[row['action_type']]
        position = impressions.index(item) + 1 if item in impressions else 0
        price = prices[position - 1] if position != 0 else 0
        return action, position, price

    def generate_data(self, df, mode="train"):

        sessions, itemIds, seq_items, seq_actions, seq_prices, seq_positions, labels = [], [], [], [], [], [], []
        nseq_items, nseq_actions, nseq_prices, nseq_positions = [], [], [], []

        for idx, rows in df.groupby("session_id"):
            seq_item, seq_action, seq_price, seq_position = [], [], [], []
            lastRow = rows.iloc[-1]

            if lastRow["action_type"] != "clickout item":
                continue

            if mode == "val":
                if type(lastRow['reference']) == float:
                    continue
            elif mode == "test":
                if type(lastRow['reference']) != float:
                    continue

            impressions = [self.item_index[int(i)] for i in lastRow['impressions'].split("|")]
            prices = [int(i) for i in lastRow['prices'].split("|")]

            if mode == "train":

                for _i, _r in rows.iterrows():
                    _item = self.item_index[int(_r['reference'])]
                    _action, _position, _price = self.get_features(_item, impressions, prices, _r)

                    seq_item.append(_item)
                    seq_position.append(_position)
                    seq_action.append(_action)
                    seq_price.append(_price)

                seq_items.append(seq_item)
                seq_positions.append(seq_position)
                seq_actions.append(seq_action)
                seq_prices.append(seq_price)

                # sample negative instance from impressions
                pool = impressions if len(impressions) > 1 else np.arange(len(self.item_index)).tolist()
                gtItem = self.item_index[int(lastRow['reference'])]
                sample = negative_sample(pool, gtItem)

                action, position, price = self.get_features(sample, impressions, prices, lastRow)

                # clone seq inputs and change last element to from negative instance
                nseq_item = seq_item[:]
                nseq_item[-1] = sample
                nseq_position = seq_position[:]
                nseq_position[-1] = position
                nseq_action = seq_action[:]
                nseq_price = seq_price[:]
                nseq_price[-1] = price

                nseq_items.append(nseq_item)
                nseq_positions.append(nseq_position)
                nseq_actions.append(nseq_action)
                nseq_prices.append(nseq_price)

            else:

                for _i, _r in rows[:-1].iterrows():

                    _item = self.item_index[int(_r['reference'])]
                    _action, _position, _price = self.get_features(_item, impressions, prices, _r)
                    seq_item.append(_item)
                    seq_position.append(_position)
                    seq_action.append(_action)
                    seq_price.append(_price)

                if mode == "val":
                    gtItem = self.item_index[int(lastRow['reference'])]

                for position, (item, price) in enumerate(zip(impressions, prices)):

                    _seq_item = seq_item[:]
                    _seq_item.append(item)
                    _seq_position = seq_position[:]
                    _seq_position.append(position + 1)
                    _seq_action = seq_action[:]
                    _seq_action.append(self.action_index["clickout item"])
                    _seq_price = seq_price[:]
                    _seq_price.append(price)

                    seq_items.append(_seq_item)
                    seq_positions.append(_seq_position)
                    seq_actions.append(_seq_action)
                    seq_prices.append(_seq_price)

                    if mode == "val":
                        labels.append(1 if item == gtItem else 0)

                sessions.extend([lastRow['session_id']] * len(impressions))
                itemIds.extend([i for i in lastRow['impressions'].split("|")])

        seq_items = pad_sequences(seq_items, maxlen=self.maxlen)
        seq_positions = pad_sequences(seq_positions, maxlen=self.maxlen)
        seq_actions = pad_sequences(seq_actions, maxlen=self.maxlen)
        seq_prices = pad_sequences(seq_prices, maxlen=self.maxlen)

        nseq_items = pad_sequences(nseq_items, maxlen=self.maxlen)
        nseq_positions = pad_sequences(nseq_positions, maxlen=self.maxlen)
        nseq_actions = pad_sequences(nseq_actions, maxlen=self.maxlen)
        nseq_prices = pad_sequences(nseq_prices, maxlen=self.maxlen)

        if self.positionMode == 1:
            seq_positions = np.expand_dims(seq_positions, axis=-1)
            nseq_positions = np.expand_dims(nseq_positions, axis=-1)
        seq_prices = np.expand_dims(seq_prices, axis=-1)
        nseq_prices = np.expand_dims(nseq_prices, axis=-1)

        if mode == "train":
            x = [seq_items, seq_actions, seq_positions, seq_prices, nseq_items, nseq_actions, nseq_positions,
                 nseq_prices]
            # for i in x:
            #     print(i.shape)
            y = np.ones(seq_items.shape[0])
            return x, y
        elif mode == "val":
            x = [seq_items, seq_actions, seq_positions, seq_prices]
            y = np.array(labels)
            return sessions, x, y
        else:
            x = [seq_items, seq_actions, seq_positions, seq_prices]
            return sessions, itemIds, x
