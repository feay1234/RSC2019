import random


def negative_sample(pool, pos):
    neg = random.sample(pool, 1)[0]
    while neg == pos:
        neg = random.sample(pool, 1)[0]
    return neg

# for group.csv file
def indexing_items(df, df_val, df_test):

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

    # index zero is for masking
    item_index = {int(i): idx + 1 for idx, i in enumerate(allItems)}
    return item_index


# for .csv files
def indexing_item_context(df, df_val, df_test):

    allItems = set([int(i) for i in df['reference'].unique()])
    for i in df[~df['impressions'].isna()]['impressions']:
        allItems.update([int(i) for i in i.split("|")])
    for i in df_val[~df_val['impressions'].isna()]['impressions']:
        allItems.update([int(i) for i in i.split("|")])
    for i in df_test[~df_test['impressions'].isna()]['impressions']:
        allItems.update([int(i) for i in i.split("|")])

    allItems.update(set([int(i) for i in df_val[~df_val.reference.isna()]['reference'].unique()]))
    allItems.update(set([int(i) for i in df_test[~df_test.reference.isna()]['reference'].unique()]))

    # allCities = set(df['city'].unique())
    # allCities.update(set(df_test['city'].unique()))

    allActions = set(df['action_type'].unique())

    item_index = {int(i): idx + 1 for idx, i in enumerate(allItems)}
    # city_index = {i : idx + 1 for idx, i in enumerate(allCities)}
    action_index = {i : idx + 1 for idx, i in enumerate(allActions)}


    return item_index, action_index