import sys
sys.path.append("/home/wangzhili/chilewang/CCF_ner")   # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config
import pandas as pd
from collections import Counter
import re
import os
import json

res_list=[]
file_path='/home/none404/hm/baidu_qa/MRC_result/MRC/'
file_list=os.listdir(file_path)
for file in file_list:
    path=os.path.join(file_path,file)
    with open(path,'r') as f:
        res_dict=json.load(f)
    res_list.append(res_dict)
c=0
sub_dict={}
for id in res_dict.keys():
    answer_list=[res_list[i][id] for i in range(len(res_list))]
    while '' in answer_list:
        answer_list.remove('')
    if answer_list==[]:
        c+=1
        answer=''
    else:
        answer=Counter(answer_list).most_common()[0][0]
    sub_dict[id]=answer

with open('/home/none404/hm/baidu_qa/ensemble_result/'+'sub_result_file.json','w') as re:
    R = json.dumps(sub_dict, ensure_ascii=False, indent=4)
    re.write(R)