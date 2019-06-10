import numpy as np

from deepctr.models import DSIN
from deepctr.utils import SingleFeat
from keras.preprocessing.sequence import pad_sequences

from utils import negative_sample


class AttentionModel():
    def __init__(self, dim, maxlen, indexes):
        self.dim = dim
        self.maxlen = maxlen

        self.item_index = indexes[0]
        self.city_index = indexes[1]
        self.action_index = indexes[2]

        hash_flag = True
        iFeature = SingleFeat('item', len(self.item_index) + 1, hash_flag)
        cFeature = SingleFeat('city', len(self.city_index) + 1, hash_flag)
        pFeature = SingleFeat('position', 25 + 1, hash_flag)
        aFeature = SingleFeat('action', len(self.action_index) + 1, hash_flag)

        self.feature_dim_dict = {"sparse": [iFeature, cFeature, pFeature, aFeature],
                                 "dense": [SingleFeat('price', False)]}

        self.behavior_feature_list = ["item", "city", "position", "action"]

        self.model = DSIN(self.feature_dim_dict, self.behavior_feature_list, sess_max_count=1, sess_len_max=self.maxlen,
                          embedding_size=self.dim,
                          att_head_num=1,
                          att_embedding_size=self.dim * len(self.behavior_feature_list),
                          dnn_hidden_units=[self.dim, self.dim, self.dim ], dnn_dropout=0.5)

        self.model.compile('adam', 'binary_crossentropy',
                           metrics=['acc'])

    def get_features(self, item, impressions, prices, row):

        action = self.action_index[row['action_type']]
        city = self.city_index[row['city']]
        position = impressions.index(item) + 1 if item in impressions else 0
        # print(position)
        price = prices[position - 1] if position != 0 else 0

        return action, city, position, price

    def generate_data(self, df, mode="train"):

        sessions, itemIds, cand_items, cand_actions, cand_cities, cand_positions, cand_prices, seq_items, seq_cities, seq_actions, seq_prices, seq_positions, labels = [], [], [], [], [], [], [], [], [], [], [], [], []
        for idx, rows in df.groupby("session_id"):
            seq_item, seq_city, seq_action, seq_price, seq_position = [], [], [], [], []
            lastRow = rows.iloc[-1]

            if lastRow["action_type"] != "clickout item":
                continue

            if mode == "val":
                if type(lastRow['reference']) == float:
                    continue
            elif mode == "test":
                if type(lastRow['reference']) != float:
                    continue

            histRows = rows.iloc[:-1]

            impressions = [self.item_index[int(i)] for i in lastRow['impressions'].split("|")]
            prices = [int(i) for i in lastRow['prices'].split("|")]
            if mode == "train":
                gtItem = self.item_index[int(lastRow['reference'])]
                action, city, position, price = self.get_features(gtItem, impressions, prices, lastRow)

            if len(histRows) > 0:
                for _i, _r in histRows.iterrows():
                    _item = self.item_index[int(_r['reference'])]

                    _action, _city, _position, _price = self.get_features(_item, impressions, prices, _r)

                    seq_item.append(_item)
                    seq_position.append(_position)
                    seq_city.append(_city)
                    seq_action.append(_action)
                    seq_price.append(_price)

            if mode == "train":

                seq_items.append(seq_item)
                seq_positions.append(seq_position)
                seq_cities.append(seq_city)
                seq_actions.append(seq_action)
                seq_prices.append(seq_price)
                labels.append(1)

                # sample negative instance from impressions
                pool = impressions if len(impressions) > 1 else np.arange(len(self.item_index)).tolist()
                sample = negative_sample(pool, gtItem)
                action, city, position, price = self.get_features(sample, impressions, prices, lastRow)

                cand_items.append(sample)
                cand_actions.append(action)
                cand_cities.append(city)
                cand_positions.append(position)
                cand_prices.append(price)
                seq_items.append(seq_item)
                seq_positions.append(seq_position)
                seq_cities.append(seq_city)
                seq_actions.append(seq_action)
                seq_prices.append(seq_price)
                labels.append(0)

            else:
                _action = self.action_index[lastRow['action_type']]
                _city = self.city_index[lastRow['city']]
                if mode == "val":
                    gtItem = self.item_index[int(lastRow['reference'])]

                for _position, (_item, _price) in enumerate(zip(impressions, prices)):
                    cand_items.append(_item)
                    cand_actions.append(_action)
                    cand_cities.append(_city)
                    cand_positions.append(_position+1)
                    cand_prices.append(_price)

                    seq_items.append(seq_item)
                    seq_positions.append(seq_position)
                    seq_cities.append(seq_city)
                    seq_actions.append(seq_action)
                    seq_prices.append(seq_price)

                    if mode == "val":
                        labels.append(1 if _item == gtItem else 0)

                sessions.extend([lastRow['session_id']]*len(impressions))
                itemIds.extend([i for i in lastRow['impressions'].split("|")])


        cand_items = np.array(cand_items)
        cand_positions = np.array(cand_positions)
        cand_cities = np.array(cand_cities)
        cand_actions = np.array(cand_actions)
        cand_prices = np.array(cand_prices)
        seq_items = pad_sequences(seq_items, maxlen=self.maxlen)
        seq_positions = pad_sequences(seq_positions, maxlen=self.maxlen)
        seq_cities = pad_sequences(seq_cities, maxlen=self.maxlen)
        seq_actions = pad_sequences(seq_actions, maxlen=self.maxlen)
        seq_prices = pad_sequences(seq_prices, maxlen=self.maxlen)
        labels = np.array(labels)


        feature_dict = {'item': cand_items, 'position': cand_positions, 'city': cand_cities, 'action': cand_actions,
                        'price': cand_prices,
                        'seq_item': seq_items, 'seq_position': seq_positions, 'seq_city': seq_cities,
                        'seq_action': seq_actions, 'price': cand_prices, 'seq_price': seq_prices}

        x = [feature_dict[feat.name] for feat in self.feature_dim_dict["sparse"]] + [feature_dict[feat.name] for feat in
                                                                                     self.feature_dim_dict["dense"]] + [
                feature_dict['seq_' + feat] for feat in self.behavior_feature_list]

        x += [np.arange(len(cand_items))]

        y = labels

        # for i in [cand_items, cand_positions, cand_cities, cand_actions, cand_prices, seq_items, seq_positions, seq_cities, seq_actions, seq_prices, labels]:
        #     print(i.shape)

        if mode == "train":
            return x, y
        elif mode == "val":
            return sessions, x, y
        else:
            return sessions, itemIds, x

