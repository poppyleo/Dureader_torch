import numpy as np
from bert import tokenization
from tqdm import tqdm
from config import Config
import pandas as pd
import os
from bert4keras.tokenizers import Tokenizer
from bert.data_utils import split_text
vocab_file = Config().vocab_file
do_lower_case = True
re_tokenzier = Tokenizer(vocab_file, do_lower_case)
config=Config()
def load_data(data_file):
    """
    读取数据
    :param file:
    :return:
    """
    data_df = pd.read_csv(data_file)
    data_df.fillna('',inplace=True)
    lines = list(zip(list(data_df['id']), list(data_df['question']),list(data_df['context']),list(data_df['answer']),list(data_df['answer_start'])))
    return lines


def create_example(lines):
    examples = []
    for (_i, line) in enumerate(lines):
        guid = "%s" % line[0]
        question = tokenization.convert_to_unicode(str(line[1]))
        text = tokenization.convert_to_unicode(str(line[2]))
        answer =tokenization.convert_to_unicode(str(line[3]))
        answer_start = line[4]
        examples.append(InputExample(guid=guid, question=question,text=text, answer=answer,answer_start=answer_start))
    return examples


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def find_all(s,sub):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 0:
        return index_list
    else:
        return -1


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, question,text,answer,answer_start):
        self.guid = guid
        self.question = question
        self.text = text
        self.answer = answer
        self.answer_start = answer_start


class DataIterator:
    """
    数据迭代器
    """
    def __init__(self, batch_size, data_file, tokenizer, use_bert=False, seq_length=100, is_test=False,config=None):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test
        self.config=config

        if not self.is_test:
            self.shuffle()
        self.tokenizer = tokenizer
        print(self.num_records)
        STOPS = (
            '\uFF01'  # Fullwidth exclamation mark
            '\uFF1F'  # Fullwidth question mark
            '\uFF61'  # Halfwidth ideographic full stop
            '\u3002'  # Ideographic full stop
            '\u3002'
        )
        self.SPLIT_PAT = '([,，{}]”?)'.format(STOPS)
        # self.SPLIT_PAT = '([{}]”?)'.format(STOPS)
    def convert_single_example(self, example_idx):
        question= self.data[example_idx].question
        text = self.data[example_idx].text
        uid = self.data[example_idx].guid
        answer =self.data[example_idx].answer
        answer_start =self.data[example_idx].answer_start


        ntokens = []
        segment_ids = []
        input_mask = []
        """得到input的token-----start-------"""

        ntokens.append("[CLS]")

        segment_ids.append(0)
        input_mask.append(0)
        # 得到问题的token
        """question_token"""

        q_tokens = self.tokenizer.tokenize(question) #
        # 把问题的token加入至所有字的token中
        for i, token in enumerate(q_tokens):
            ntokens.append(token)
            segment_ids.append(0)
            input_mask.append(0)
        ntokens.append("[SEP]")
        input_mask.append(0)
        segment_ids.append(1)
        """question_token"""
        query_len=len(ntokens)
       #token后的start&&end
        text_token=self.tokenizer.tokenize(text)
        if answer!='' and answer_start!=-1:
            """训练集"""
            train_context=[]
            train_start=[]
            if text_token.__len__()+query_len+1>=self.seq_length:
                """太长，截断"""
                context_list,start_list=split_text(text,self.seq_length,self.SPLIT_PAT)
                neg_tag = 0
                for i in range(len(start_list)):
                    context=context_list[i]
                    start_=start_list[i]

                    if find_all(context,answer)!=-1:
                        index=find_all(context,answer) #特例如方太
                        for idx in index:
                            if idx+start_==answer_start:
                                """真正答案出处"""
                                train_context.append(context)
                                train_start.append(idx)
                    # 尝试加入负样本
                    elif self.config.addneg:
                        """每个样本中加入一个负样本"""
                        train_context.append(context)
                        train_start.append(-1)  #-1
                        neg_tag=1
            else:
                train_context=[text]
                train_start = [answer_start]
        else :
            """验证集&&测试集"""
            if text_token.__len__()+query_len+1>=self.seq_length:
                """太长，截断"""
                context_list,start_list=split_text(text,self.seq_length,self.SPLIT_PAT)
                train_context=context_list
                train_start=[-1]*len(context_list)
            else:
                train_context=[text]
                train_start = [-1]
        for idc,text in enumerate(train_context):
            """一个样本下可能被拆分成多个数据"""
            text_token = re_tokenzier._tokenize(text)
            mapping = re_tokenzier.rematch(text, text_token)
            if [] in mapping:
                print(text_token,text)
                raise EOFError
            if train_start[idc]==-1:
                start_pos,end_pos,cls= 0,0,0
            else:
                cls=1
                """训练集的实际位置"""
                answer_token=re_tokenzier._tokenize(answer)
                pre_answer_len = re_tokenzier._tokenize(text[:train_start[idc]]).__len__()
                start_pos = pre_answer_len + len(ntokens)
                end_pos = start_pos + len(answer_token) - 1
            for i, token in enumerate(text_token):
                ntokens.append(token)
                segment_ids.append(1)
                input_mask.append(1)
            if ntokens.__len__() >= self.seq_length - 1:
                ntokens = ntokens[:(self.seq_length - 1)]
                segment_ids = segment_ids[:(self.seq_length - 1)]
                input_mask = input_mask[:(self.seq_length - 1)]
            if start_pos > ntokens.__len__() - 1:
                start_pos = 0
                end_pos = 0
            elif end_pos > ntokens.__len__() - 1:
                end_pos = ntokens.__len__() - 1
            ntokens.append("[SEP]")
            input_mask.append(0)
            segment_ids.append(0)
            """得到input的token-------end--------"""
            assert end_pos >= start_pos
            """token2id---start---"""
            input_ids = re_tokenzier.tokens_to_ids(ntokens)
            if not self.config.mask_q:
                input_mask = [1] * (len(input_ids)-1) #SEP也当作padding，mask
                input_mask.append(0) #SEP也当作padding，mask
            while len(input_ids) < self.seq_length:
                # 不足时补零
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                # we don't concerned about it!
                ntokens.append("**NULL**")
            assert len(input_ids) == self.seq_length
            assert len(segment_ids) == self.seq_length
            assert len(input_mask) == self.seq_length
            """token2id ---end---"""
            return input_ids, input_mask, segment_ids, start_pos, end_pos, uid, answer, text, query_len, mapping , cls


    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        start_list = []
        end_list = []
        uid_list = []
        answer_list =[]
        text_list =[]
        querylen_list=[]
        maping_list=[]
        cls_list=[]
        num_tags = 0

        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res==None:
                continue
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, start_pos, end_pos,uid,answer,text,query_len,maping,cls= res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            start_list.append(start_pos)
            end_list.append(end_pos)
            uid_list.append(uid)
            answer_list.append(answer)
            text_list.append(text)
            querylen_list.append(query_len)
            maping_list.append(maping)
            cls_list.append(cls)
            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break

        return input_ids_list, input_mask_list, segment_ids_list, start_list,end_list,uid_list,\
               answer_list,text_list,querylen_list,maping_list,cls_list,self.seq_length


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case = True

    re_tokenzier = Tokenizer(vocab_file, do_lower_case)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


    # print(vocab_file)
    # print(print(len(tokenizer.vocab)))

    # data_iter = DataIterator(config.batch_size, data_file= config.dir_with_mission + 'train.txt', use_bert=True,
    #                         seq_length=config.sequence_length, tokenizer=tokenizer)
    #
    # dev_iter = DataIterator(config.batch_size, data_file=config.dir_with_mission + 'dev.txt', use_bert=True,
    #                          seq_length=config.sequence_length, tokenizer=tokenizer, is_test=True)
    train_iter = DataIterator(config.batch_size,
                              data_file=config.process + 'train.csv',
                              use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length,config=config)
    for input_ids_list, input_mask_list, segment_ids_list, start_list,end_list,uid_list,\
               answer_list,text_list,querylen_list,maping_list,cls_list,seq_length in tqdm(train_iter):
        # print(input_ids_list)
        # print(tokenizer.convert_ids_to_tokens(input_ids_list[0]))
        # print(start_list)
        # print(answer_list)
        print(cls_list)
        # print(query_len)
        # print(text_list[0])
        # answer_id=input_ids_list[0][start_list[0]:end_list[0]+1]
        # print(tokenizer.convert_ids_to_tokens(answer_id))
        # break

