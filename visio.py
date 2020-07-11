import pandas as pd
from config import Config
import os
import matplotlib
import matplotlib.pyplot as plt
# import missingno as msno
import pandas as pd
import seaborn as sns
config=Config()

# 原始数据集
train_df = pd.read_csv(config.process+'train.csv', encoding='utf-8-sig')
test_df = pd.read_csv(config.process +'test.csv', encoding='utf-8-sig')

# train_df = train_df[train_df['情感倾向'].isin(['0','1','-1'])]
# train_df['情感倾向'].value_counts().plot.bar()
# plt.title('TrainDateset sentiment(target)')
# plt.savefig('savepic_1',dpi = 100,bbox_inches='tight')

train_df['content_len'] = train_df['context'].astype(str).apply(len)
test_df['content_len'] = train_df['context'].astype(str).apply(len)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,8))
sns.distplot(train_df['content_len'] , ax=ax1, color='blue')
sns.distplot(test_df['content_len'], ax=ax2, color='green')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax1.set_title('TrainDataset')
ax2.set_title('TestDataset')
plt.show()
plt.savefig('savepic_2',dpi = 1000,bbox_inches='tight')