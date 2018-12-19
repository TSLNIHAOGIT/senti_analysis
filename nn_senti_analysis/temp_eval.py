import pandas as pd
# df=pd.read_csv('chinese_hotel_eval.csv')
# print(df['label']!=df['pred_label'])
# print(1!=df['pred_label'])
#
# df_new=df[df['label']!=df['pred_label']]
# print(df_new.head())
# a=[1,2,3]
# b=[4,5,6]
# c=[]
# c.extend(a)
# c.extend(b)
# print(c)

text_split_path='../data/data_cleaned/hotel_split.parquet.gzip'
df_text=pd.read_parquet(text_split_path)
print('df_text',df_text.head())
