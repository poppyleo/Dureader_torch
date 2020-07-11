import pandas as pd
import json
import torch
from tqdm import tqdm
from config import Config
from train_fine_tune import list2ts2device,softmax
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, get_cosine_schedule_with_warmup
from model_nq import BertForQuestionAnswering
from utils import DataIterator
import numpy as np
from bert import tokenization
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_test(test_iter, model_file):
    # Bert_config = BertConfig.from_pretrained(config.bert_path, num_labels=1,use_origin_bert=config.use_origin_bert,)
    # Bert_config.output_hidden_states=True  #获取每一层的输出
    # model = BertForQuestionAnswering.from_pretrained(model_file, config=Bert_config)
    model = torch.load(model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("***** Running Prediction *****")
    logger.info("  Predict Path = %s", model_file)
    model.eval()
    pred_answer_dict={}
    pred_score_dict={}
    all_answer_pred={}
    for input_ids, input_mask, segment_ids, start_list,end_list,uid_list,answer_list,text_list,querylen_list,maping_list,_,seq_length in tqdm(
            test_iter):
        input_ids = list2ts2device(input_ids)
        input_mask = list2ts2device(input_mask)
        segment_ids = list2ts2device(segment_ids)
        y_preds = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        start_preds, end_preds = (p.detach().cpu() for p in y_preds)
        start_probs = softmax(start_preds.numpy())
        end_probs = softmax(end_preds.numpy())
        max_a_len = 64
        answer_pred = []
        epsilon = 1e-3
        for i in range(len(start_probs)):
            start_ = start_probs[i][querylen_list[i]:-1]
            end_ = end_probs[i][querylen_list[i]:-1]
            start_end, score = None, -100
            for start, p_start in enumerate(start_):
                for end, p_end in enumerate(end_):
                    if end >= start and end < start + max_a_len:
                        """一定有答案"""
                        if p_start * p_end > score:
                            start_end = (start, end)
                            score = np.exp(
                                (0.5 * np.log(p_start + epsilon) + 0.5 * np.log(p_end + epsilon)) / (0.5 + 0.5))
                            score = p_start * p_end
            start, end = start_end
            start_cls = start_probs[i][0]
            end_cls = end_probs[i][0]
            pos_cls = -(start_cls + end_cls)
            if config.addneg:
                score += pos_cls
            try:
                start_pos = maping_list[i][start][0]
                end_pos = maping_list[i][end][-1]
                answer = text_list[i][start_pos:end_pos + 1]
            except:
                answer=''
            if uid_list[i] in pred_answer_dict:
                if pred_score_dict[uid_list[i]] < score:
                    pred_score_dict[uid_list[i]] = score
                    pred_answer_dict[uid_list[i]] = answer
            else:
                pred_answer_dict[uid_list[i]] = answer
                pred_score_dict[uid_list[i]] = score

            if uid_list[i] in pred_answer_dict:
                all_answer_pred[uid_list[i] + str(i)] = answer
            else:
                all_answer_pred[uid_list[i]] = answer
        # print(pred_answer_dict)
        # print(all_answer_pred)
    with open('sub_result_file.json', 'w') as re:
        R = json.dumps(pred_answer_dict, ensure_ascii=False, indent=4)
        re.write(R)
    with open('all_result_file.json', 'w') as re:
        R = json.dumps(all_answer_pred, ensure_ascii=False, indent=4)
        re.write(R)

    with open('/'.join(config.checkpoint_path.split('/')[:-1]) + '/sub_result_file.json', 'w') as re:
        R = json.dumps(pred_answer_dict, ensure_ascii=False, indent=4)
        re.write(R)
    with open('/'.join(config.checkpoint_path.split('/')[:-1]) + '/all_result_file.json', 'w') as re:
        R = json.dumps(all_answer_pred, ensure_ascii=False, indent=4)
        re.write(R)
    n_df = pd.DataFrame()
    id = list(pred_answer_dict.keys())
    answer = list(pred_answer_dict.values())
    n_df['id'] = id
    n_df['answer'] = answer
    no_df = n_df[n_df['answer'] == '']
    print('{}个没找到答案'.format(no_df.__len__()))
    no_df.to_csv('/'.join(config.checkpoint_path.split('/')[:-1]) + '/no_answer.csv', index=False,
                 encoding='utf_8_sig')
if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case =False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.test_batch_size,
                            # data_file=Config().data + 'test_mrc.csv',
                            config.process +'test.csv',
                            use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer,config=config)
    set_test(dev_iter, config.checkpoint_path)