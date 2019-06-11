import numpy as np
from keras import Model

from keras.layers import Input, Embedding, Subtract, Activation, Concatenate, Dense, SimpleRNN
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from utils import negative_sample


class ContextRNN():
    def __init__(self, dim, maxlen, item_index, action_index, positionMode=1, timeMode=1):
        self.dim = dim
        self.maxlen = maxlen
        self.item_index = item_index
        self.action_index = action_index
        self.positionMode = positionMode
        self.timeMode = timeMode

        p_itemInput = Input(shape=(self.maxlen,))
        p_actionInput = Input(shape=(self.maxlen,))
        p_poisitionInput = Input(shape=(self.maxlen, 1,)) if self.positionMode == 1 else Input(shape=(self.maxlen,))
        p_priceInput = Input(shape=(self.maxlen, 1,))

        n_itemInput = Input(shape=(self.maxlen,))
        n_actionInput = Input(shape=(self.maxlen,))
        n_poisitionInput = Input(shape=(self.maxlen, 1,)) if self.positionMode == 1 else Input(shape=(self.maxlen,))
        n_priceInput = Input(shape=(self.maxlen, 1,))

        if timeMode == 2:
            p_timeInput = Input(shape=(self.maxlen, 1,))
            n_timeInput = Input(shape=(self.maxlen, 1,))
            p_stepInput = Input(shape=(self.maxlen, 1,))
            n_stepInput = Input(shape=(self.maxlen, 1,))

        itemEmbedding = Embedding(len(item_index) + 1, self.dim, mask_zero=True)
        actionEmbedding = Embedding(len(action_index) + 1, self.dim, mask_zero=True)
        positionEmbedding = Embedding(25 + 1, self.dim, mask_zero=True)

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

        if self.timeMode == 2:
            p_features = Concatenate()([p_features, p_timeInput, p_stepInput])
            n_features = Concatenate()([n_features, n_timeInput, n_stepInput])

        rnn = SimpleRNN(self.dim, unroll=True)

        dense = Dense(1)

        rel_score = dense(rnn(p_features))
        irr_score = dense(rnn(n_features))

        # Subtract scores.
        diff = Subtract()([rel_score, irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        if self.timeMode == 1:
            model_inputs = [p_itemInput, p_actionInput, p_poisitionInput, p_priceInput, n_itemInput, n_actionInput,
                            n_poisitionInput, n_priceInput]
            get_score_inputs = [p_itemInput, p_actionInput, p_poisitionInput, p_priceInput]
        elif self.timeMode == 2:
            model_inputs = [p_itemInput, p_actionInput, p_poisitionInput, p_priceInput, p_timeInput, p_stepInput,
                            n_itemInput,
                            n_actionInput, n_poisitionInput, n_priceInput, n_timeInput, n_stepInput]
            get_score_inputs = [p_itemInput, p_actionInput, p_poisitionInput, p_priceInput, p_timeInput, p_stepInput]

        self.model = Model(inputs=model_inputs, outputs=prob)
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.get_score = K.function(get_score_inputs, [rel_score])

    def get_features(self, item, impressions, prices, row):

        action = self.action_index[row['action_type']]
        position = impressions.index(item) + 1 if item in impressions else 0
        price = prices[position - 1] if position != 0 else 0
        return action, position, price

    def generate_data(self, df, mode="train"):

        sessions, itemIds, seq_items, seq_actions, seq_prices, seq_positions, labels = [], [], [], [], [], [], []
        nseq_items, nseq_actions, nseq_prices, nseq_positions = [], [], [], []
        seq_times, seq_steps = [], []
        nseq_times, nseq_steps = [], []

        for idx, rows in df.groupby("session_id"):
            seq_item, seq_action, seq_price, seq_position, seq_time, seq_step = [], [], [], [], [], []
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

                firstTime = rows.iloc[0]['timestamp']
                # print((row['timestamp'] - firstTime).total_seconds())

                for _i, _r in rows.iterrows():
                    _item = self.item_index[int(_r['reference'])]
                    _action, _position, _price = self.get_features(_item, impressions, prices, _r)

                    seq_item.append(_item)
                    seq_position.append(_position)
                    seq_action.append(_action)
                    seq_price.append(_price)
                    if self.timeMode == 2:
                        seq_time.append(int((_r['timestamp'] - firstTime) / 60))
                        seq_step.append(int(_r['step']))

                seq_items.append(seq_item)
                seq_positions.append(seq_position)
                seq_actions.append(seq_action)
                seq_prices.append(seq_price)
                seq_times.append(seq_time)
                seq_steps.append(seq_step)

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
                nseq_time = seq_time[:]
                nseq_step = seq_step[:]

                nseq_items.append(nseq_item)
                nseq_positions.append(nseq_position)
                nseq_actions.append(nseq_action)
                nseq_prices.append(nseq_price)
                nseq_times.append(nseq_time)
                nseq_steps.append(nseq_step)

            else:

                firstTime = rows.iloc[0]['timestamp']
                for _i, _r in rows.iterrows():
                    _item = self.item_index[int(_r['reference'])] if type(_r['reference']) == str else 0
                    _action, _position, _price = self.get_features(_item, impressions, prices, _r)
                    seq_item.append(_item)
                    seq_position.append(_position)
                    seq_action.append(_action)
                    seq_price.append(_price)
                    if self.timeMode == 2:
                        seq_time.append(int((_r['timestamp'] - firstTime) / 60))
                        seq_step.append(int(_r['step']))

                if mode == "val":
                    gtItem = self.item_index[int(lastRow['reference'])]

                for position, (item, price) in enumerate(zip(impressions, prices)):

                    _seq_item = seq_item[:]
                    _seq_item[-1] = item
                    _seq_position = seq_position[:]
                    _seq_position[-1] = position + 1
                    _seq_action = seq_action[:]
                    _seq_price = seq_price[:]
                    _seq_price[-1] = price

                    seq_items.append(_seq_item)
                    seq_positions.append(_seq_position)
                    seq_actions.append(_seq_action)
                    seq_prices.append(_seq_price)

                    if self.timeMode == 2:
                        seq_times.append(seq_time)
                        seq_steps.append(seq_step)

                    if mode == "val":
                        labels.append(1 if item == gtItem else 0)

                sessions.extend([lastRow['session_id']] * len(impressions))
                itemIds.extend([i for i in lastRow['impressions'].split("|")])

        seq_items = pad_sequences(seq_items, maxlen=self.maxlen)
        seq_positions = pad_sequences(seq_positions, maxlen=self.maxlen)
        seq_actions = pad_sequences(seq_actions, maxlen=self.maxlen)
        seq_prices = pad_sequences(seq_prices, maxlen=self.maxlen)
        seq_times = pad_sequences(seq_times, maxlen=self.maxlen)
        seq_steps = pad_sequences(seq_steps, maxlen=self.maxlen)

        nseq_items = pad_sequences(nseq_items, maxlen=self.maxlen)
        nseq_positions = pad_sequences(nseq_positions, maxlen=self.maxlen)
        nseq_actions = pad_sequences(nseq_actions, maxlen=self.maxlen)
        nseq_prices = pad_sequences(nseq_prices, maxlen=self.maxlen)
        nseq_times = pad_sequences(nseq_times, maxlen=self.maxlen)
        nseq_steps = pad_sequences(nseq_steps, maxlen=self.maxlen)

        if self.positionMode == 1:
            seq_positions = np.expand_dims(seq_positions, axis=-1)
            nseq_positions = np.expand_dims(nseq_positions, axis=-1)

        seq_prices = np.expand_dims(seq_prices, axis=-1)
        nseq_prices = np.expand_dims(nseq_prices, axis=-1)
        seq_times = np.expand_dims(seq_times, axis=-1)
        nseq_times = np.expand_dims(nseq_times, axis=-1)
        seq_steps = np.expand_dims(seq_steps, axis=-1)
        nseq_steps = np.expand_dims(nseq_steps, axis=-1)

        if mode == "train":
            if self.timeMode == 1:
                x = [seq_items, seq_actions, seq_positions, seq_prices, nseq_items, nseq_actions, nseq_positions,
                     nseq_prices]
            else:
                x = [seq_items, seq_actions, seq_positions, seq_prices, seq_times, seq_steps, nseq_items, nseq_actions, nseq_positions,
                     nseq_prices, nseq_times, nseq_steps]
            # for i in x:
            #     print(i.shape)
            y = np.ones(seq_items.shape[0])
            return x, y
        elif mode == "val":
            if self.timeMode == 1:
                x = [seq_items, seq_actions, seq_positions, seq_prices]
            else:
                x = [seq_items, seq_actions, seq_positions, seq_prices, seq_times, seq_steps]
            y = np.array(labels)
            return sessions, x, y
        else:
            if self.timeMode == 1:
                x = [seq_items, seq_actions, seq_positions, seq_prices]
            else:
                x = [seq_items, seq_actions, seq_positions, seq_prices, seq_times, seq_steps]
            return sessions, itemIds, x
