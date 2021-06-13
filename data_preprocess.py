import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import product
import calendar

class Features():
    def __init__(self):        
        self.df = []
        self.readFile()
        
    def execute(self):
        self.remove_ex_value()
        self.remove_same_data()
        self.augumentation()
        self.create_test_col()
        self.locate_feature()
        self.encodeing()
        self.time_feature()
        self.history_saled_feature()
        self.slide_window_feature()
        self.three_month_buying_feature()
        self.history_sum_feature()
        self.another_feature()
        self.save()

    def readFile(self):
        self.test = pd.read_csv('./data/test.csv')
        self.sales = pd.read_csv('./data/sales_train.csv')
        self.shops = pd.read_csv('./data/shops.csv')
        self.items = pd.read_csv('./data/items.csv')
        self.item_cats = pd.read_csv('./data/item_categories.csv')

    def remove_ex_value(self):
        self.train = self.sales[(self.sales.item_price < 100000) & (self.sales.item_price > 0)]
        self.train = self.train[self.sales.item_cnt_day < 1001]

    def remove_same_data(self):
        self.train.loc[self.train.shop_id == 0, 'shop_id'] = 57
        self.test.loc[self.test.shop_id == 0, 'shop_id'] = 57

        self.train.loc[self.train.shop_id == 1, 'shop_id'] = 58
        self.test.loc[self.test.shop_id == 1, 'shop_id'] = 58

        self.train.loc[self.train.shop_id == 40, 'shop_id'] = 39
        self.test.loc[self.test.shop_id == 40, 'shop_id'] = 39

    def augumentation(self):
        self.index_cols = ['shop_id', 'item_id', 'date_block_num']

        for block_num in self.train['date_block_num'].unique():
            cur_shops = self.train.loc[self.sales['date_block_num'] == block_num, 'shop_id'].unique()
            cur_items = self.train.loc[self.sales['date_block_num'] == block_num, 'item_id'].unique()
            self.df.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

        self.df = pd.DataFrame(np.vstack(self.df), columns = self.index_cols,dtype=np.int32)


        self.group = self.train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
        self.group.columns = ['item_cnt_month']
        self.group.reset_index(inplace=True)

        self.df = pd.merge(self.df, self.group, on=self.index_cols, how='left')
        self.df['item_cnt_month'] = (self.df['item_cnt_month']
                                        .fillna(0)
                                        .clip(0,20)
                                        .astype(np.float16))

    def create_test_col(self):
        self.test['date_block_num'] = 34
        self.test['date_block_num'] = self.test['date_block_num'].astype(np.int8)
        self.test['shop_id'] = self.test['shop_id'].astype(np.int8)
        self.test['item_id'] = self.test['item_id'].astype(np.int16)
        df = pd.concat([self.df, self.test], ignore_index=True, sort=False, keys=self.index_cols)
        df.fillna(0, inplace=True)

    def locate_feature(self):
        self.shops['city'] = self.shops['shop_name'].apply(lambda x: x.split()[0].lower())
        self.shops.loc[self.shops.city == '!якутск', 'city'] = 'якутск'
        self.shops['city_code'] = LabelEncoder().fit_transform(self.shops['city'])

        coords = dict()
        coords['якутск'] = (62.028098, 129.732555, 4)
        coords['адыгея'] = (44.609764, 40.100516, 3)
        coords['балашиха'] = (55.8094500, 37.9580600, 1)
        coords['волжский'] = (53.4305800, 50.1190000, 3)
        coords['вологда'] = (59.2239000, 39.8839800, 2)
        coords['воронеж'] = (51.6720400, 39.1843000, 3)
        coords['выездная'] = (0, 0, 0)
        coords['жуковский'] = (55.5952800, 38.1202800, 1)
        coords['интернет-магазин'] = (0, 0, 0)
        coords['казань'] = (55.7887400, 49.1221400, 4)
        coords['калуга'] = (54.5293000, 36.2754200, 4)
        coords['коломна'] = (55.0794400, 38.7783300, 4)
        coords['красноярск'] = (56.0183900, 92.8671700, 4)
        coords['курск'] = (51.7373300, 36.1873500, 3)
        coords['москва'] = (55.7522200, 37.6155600, 1)
        coords['мытищи'] = (55.9116300, 37.7307600, 1)
        coords['н.новгород'] = (56.3286700, 44.0020500, 4)
        coords['новосибирск'] = (55.0415000, 82.9346000, 4)
        coords['омск'] = (54.9924400, 73.3685900, 4)
        coords['ростовнадону'] = (47.2313500, 39.7232800, 3)
        coords['спб'] = (59.9386300, 30.3141300, 2)
        coords['самара'] = (53.2000700, 50.1500000, 4)
        coords['сергиев'] = (56.3000000, 38.1333300, 4)
        coords['сургут'] = (61.2500000, 73.4166700, 4)
        coords['томск'] = (56.4977100, 84.9743700, 4)
        coords['тюмень'] = (57.1522200, 65.5272200, 4)
        coords['уфа'] = (54.7430600, 55.9677900, 4)
        coords['химки'] = (55.8970400, 37.4296900, 1)
        coords['цифровой'] = (0, 0, 0)
        coords['чехов'] = (55.1477000, 37.4772800, 4)
        coords['ярославль'] = (57.6298700, 39.8736800, 2) 

        self.shops['city_coord_1'] = self.shops['city'].apply(lambda x: coords[x][0])
        self.shops['city_coord_2'] = self.shops['city'].apply(lambda x: coords[x][1])
        self.shops['country_part'] = self.shops['city'].apply(lambda x: coords[x][2])

        self.shops = self.shops[['shop_id', 'city_code', 'city_coord_1', 'city_coord_2', 'country_part']]
        self.df = pd.merge(self.df, self.shops, on=['shop_id'], how='left')

    def encodeing(self):
        map_dict = {
                'Чистые носители (штучные)': 'Чистые носители',
                'Чистые носители (шпиль)' : 'Чистые носители',
                'PC ': 'Аксессуары',
                'Служебные': 'Служебные '
                }

        self.items = pd.merge(self.items, self.item_cats, on='item_category_id')

        self.items['item_category'] = self.items['item_category_name'].apply(lambda x: x.split('-')[0])
        self.items['item_category'] = self.items['item_category'].apply(lambda x: map_dict[x] if x in map_dict.keys() else x)
        self.items['item_category_common'] = LabelEncoder().fit_transform(self.items['item_category'])

        self.items['item_category_code'] = LabelEncoder().fit_transform(self.items['item_category_name'])
        self.items = self.items[['item_id', 'item_category_common', 'item_category_code']]
        self.df = pd.merge(self.df, self.items, on=['item_id'], how='left')

    def time_feature(self):
        def count_days(date_block_num):
            year = 2013 + date_block_num // 12
            month = 1 + date_block_num % 12
            weeknd_count = len([1 for i in calendar.monthcalendar(year, month) if i[6] != 0])
            days_in_month = calendar.monthrange(year, month)[1]
            return weeknd_count, days_in_month, month

        map_dict = {i: count_days(i) for i in range(35)}

        self.df['weeknd_count'] = self.df['date_block_num'].apply(lambda x: map_dict[x][0])
        self.df['days_in_month'] = self.df['date_block_num'].apply(lambda x: map_dict[x][1])

    def history_saled_feature(self):
        first_item_block = self.df.groupby(['item_id'])['date_block_num'].min().reset_index()
        first_item_block['item_first_interaction'] = 1

        first_shop_item_buy_block = self.df[self.df['date_block_num'] > 0].groupby(['shop_id', 'item_id'])['date_block_num'].min().reset_index()
        first_shop_item_buy_block['first_date_block_num'] = first_shop_item_buy_block['date_block_num']
        self.df = pd.merge(self.df, first_item_block[['item_id', 'date_block_num', 'item_first_interaction']], on=['item_id', 'date_block_num'], how='left')
        self.df = pd.merge(self.df, first_shop_item_buy_block[['item_id', 'shop_id', 'first_date_block_num']], on=['item_id', 'shop_id'], how='left')

        self.df['first_date_block_num'].fillna(100, inplace=True)
        self.df['shop_item_sold_before'] = (self.df['first_date_block_num'] < self.df['date_block_num']).astype('int8')
        self.df.drop(['first_date_block_num'], axis=1, inplace=True)

        self.df['item_first_interaction'].fillna(0, inplace=True)
        self.df['shop_item_sold_before'].fillna(0, inplace=True)
        
        self.df['item_first_interaction'] = self.df['item_first_interaction'].astype('int8')  
        self.df['shop_item_sold_before'] = self.df['shop_item_sold_before'].astype('int8') 

    def lag_feature(self, df, lags, col):
        tmp = df[['date_block_num','shop_id','item_id',col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
            shifted['date_block_num'] += i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
            df[col+'_lag_'+str(i)] = df[col+'_lag_'+str(i)].astype('float16')
        return df
        
    def slide_window_feature(self):
        
        self.df = self.lag_feature(self.df, [1, 2, 3], 'item_cnt_month')
        index_cols = ['shop_id', 'item_id', 'date_block_num']
        group = self.train.groupby(index_cols)['item_price'].mean().reset_index().rename(columns={"item_price": "avg_shop_price"}, errors="raise")
        self.df = pd.merge(self.df, group, on=index_cols, how='left')

        self.df['avg_shop_price'] = (self.df['avg_shop_price']
                                        .fillna(0)
                                        .astype(np.float16))

        index_cols = ['item_id', 'date_block_num']
        group = self.train.groupby(['date_block_num','item_id'])['item_price'].mean().reset_index().rename(columns={"item_price": "avg_item_price"}, errors="raise")


        self.df = pd.merge(self.df, group, on=index_cols, how='left')
        self.df['avg_item_price'] = (self.df['avg_item_price']
                                        .fillna(0)
                                        .astype(np.float16))

        self.df['item_shop_price_avg'] = (self.df['avg_shop_price'] - self.df['avg_item_price']) / self.df['avg_item_price']
        self.df['item_shop_price_avg'].fillna(0, inplace=True)

        self.df = self.lag_feature(self.df, [1, 2, 3], 'item_shop_price_avg')
        self.df.drop(['avg_shop_price', 'avg_item_price', 'item_shop_price_avg'], axis=1, inplace=True)

    def three_month_buying_feature(self):
        item_id_target_mean = self.df.groupby(['date_block_num','item_id'])['item_cnt_month'].mean().reset_index().rename(columns={"item_cnt_month": "item_target_enc"}, errors="raise")
        self.df = pd.merge(self.df, item_id_target_mean, on=['date_block_num','item_id'], how='left')

        self.df['item_target_enc'] = (self.df['item_target_enc']
                                        .fillna(0)
                                        .astype(np.float16))

        self.df = self.lag_feature(self.df, [1, 2, 3], 'item_target_enc')
        self.df.drop(['item_target_enc'], axis=1, inplace=True)

        #Add target encoding for item/city for last 3 months 
        item_id_target_mean = self.df.groupby(['date_block_num','item_id', 'city_code'])['item_cnt_month'].mean().reset_index().rename(columns={
            "item_cnt_month": "item_loc_target_enc"}, errors="raise")
        self.df = pd.merge(self.df, item_id_target_mean, on=['date_block_num','item_id', 'city_code'], how='left')

        self.df['item_loc_target_enc'] = (self.df['item_loc_target_enc']
                                        .fillna(0)
                                        .astype(np.float16))

        self.df = self.lag_feature(self.df, [1, 2, 3], 'item_loc_target_enc')
        self.df.drop(['item_loc_target_enc'], axis=1, inplace=True)

    def history_sum_feature(self):
        item_id_target_mean = self.df[self.df['item_first_interaction'] == 1].groupby(['date_block_num','item_category_code'])['item_cnt_month'].mean().reset_index().rename(columns={
            "item_cnt_month": "new_item_cat_avg"}, errors="raise")

        self.df = pd.merge(self.df, item_id_target_mean, on=['date_block_num','item_category_code'], how='left')

        self.df['new_item_cat_avg'] = (self.df['new_item_cat_avg']
                                        .fillna(0)
                                        .astype(np.float16))

        self.df = self.lag_feature(self.df, [1, 2, 3], 'new_item_cat_avg')
        self.df.drop(['new_item_cat_avg'], axis=1, inplace=True)

    def another_feature(self):
        def lag_feature_adv(df, lags, col):
            tmp = df[['date_block_num','shop_id','item_id',col]]
            for i in lags:
                shifted = tmp.copy()
                shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)+'_adv']
                shifted['date_block_num'] += i
                shifted['item_id'] -= 1
                df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
                df[col+'_lag_'+str(i)+'_adv'] = df[col+'_lag_'+str(i)+'_adv'].astype('float16')
            return df

        self.df = lag_feature_adv(self.df, [1, 2, 3], 'item_cnt_month')

    def save(self):
        self.df.drop(['ID'], axis=1, inplace=True, errors='ignore')
        self.df.to_pickle('df.pkl')