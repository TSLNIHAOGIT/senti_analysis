import pandas as pd
df=pd.read_csv('chinese_hotel_eval.csv')
print(df['label']!=df['pred_label'])
print(1!=df['pred_label'])

df_new=df[df['label']!=df['pred_label']]
print(df_new.head())