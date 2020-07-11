import os
import time
import json
import pandas as pd
from tqdm import tqdm
import torch
from config import Config
import random
from evaluate import calc_f1_score,calc_em_score,evaluate
import logging
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, get_cosine_schedule_with_warmup
from model_nq import BertForQuestionAnswering
from prepare_data import PGD,FGM
from utils import DataIterator
import numpy as np
from bert import tokenization
from bert4keras.tokenizers import Tokenizer

gpu_id = Config().gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
result_data_dir = Config().result_file
print('GPU ID: ', str(gpu_id))
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Batch Size: ', Config().batch_size)
print('Use original bert', Config().use_origin_bert)
print('Use avg pool', Config().is_avg_pool)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed) #我的理解是：确保每次实验结果一致，不设置实验的情况下准确率这些指标会有波动，因为是随机

eval_batch_size = config.per_gpu_eval_batch_size * max(1,n_gpu)


def list2ts2device(target_list):
    """把utils的数据写入到gpu"""
    target_ts=torch.from_numpy(np.array(target_list))
    return target_ts.to(device)


"""



"""

def train(train_iter, test_iter, config):

    """"""
    # Prepare model
    Bert_config = BertConfig.from_pretrained(config.bert_path, num_labels=2)
    Bert_config.output_hidden_states=True  #获取每一层的输出
    model = BertForQuestionAnswering.from_pretrained(config.bert_file, config=Bert_config)
    # model = BertForQuestionAnswering(Bert_config,'/home/none404/hm/Model/robert_zh_l12/bert_model.ckpt')  #加载tf模型
    model.to(device)
    """对抗"""
    if config.adv=='fgm':
        fgm = FGM(model)
    elif config.adv=='pgd':
        """PGD"""
        pgd = PGD(model)
    K = 3
    """多卡训练"""
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # optimizer
    param_optimizer = [n for n in param_optimizer]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],  #bert
         'weight_decay': 0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} #下游任务
    ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-8)

    optimizer_bert = AdamW(optimizer_grouped_parameters[:1], lr=config.embed_learning_rate,)
    optimizer_tune = AdamW(optimizer_grouped_parameters[1:], lr=config.learning_rate, weight_decay=config.decay_rate)
    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", config.batch_size)
    logger.info("  Num epochs = %d", config.train_epoch)
    logger.info("  Learning rate = %f", config.learning_rate)

    best_acc = 0

    tr_loss = 0
    cum_step = 0
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(
        os.path.join(config.save_model, "runs_" + str(gpu_id), timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing to {}\n".format(out_dir))
    for i in range(config.train_epoch):
        model.train()
        for input_ids, input_mask, segment_ids,start_list,end_list,\
            uid_list,answer_list,text_list,querylen_list,maping_list,cls_list,seq_length  in tqdm(train_iter):
            # 转成张量
            start_label = list2ts2device(start_list)
            end_label = list2ts2device(end_list)
            # cls_label = list2ts2device(cls_list)
            loss = model(input_ids=list2ts2device(input_ids), token_type_ids=list2ts2device(segment_ids),
                         attention_mask=list2ts2device(input_mask), start_labels=start_label,end_labels=end_label)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            tr_loss += loss.item()
            train_loss = round(tr_loss * 1/ (cum_step + 1), 4)
            if cum_step % 10 == 0:
                format_str = 'step {}, loss {:.4f} lr {:.5f}'
                print(
                    format_str.format(
                        cum_step, train_loss, config.learning_rate)
                )
            model.train()  #正常训练
            loss.backward() # 反向传播，得到正常的grad

            # FGM对抗训练
            if config.adv=='fgm':
                fgm.attack()  # 在embedding上添加对抗扰动
                loss_adv = model(input_ids=list2ts2device(input_ids), token_type_ids=list2ts2device(segment_ids),
                             attention_mask=list2ts2device(input_mask), start_labels=start_label,end_labels=end_label)
                if n_gpu > 1:
                    loss_adv = loss_adv.mean()
                # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                loss_adv.backward()
                # 恢复embedding参数
                fgm.restore()
            elif config.adv=='pgd':
            # PGD对抗训练
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(input_ids=list2ts2device(input_ids), token_type_ids=list2ts2device(segment_ids),
                             attention_mask=list2ts2device(input_mask), start_labels=start_label,end_labels=end_label)
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数
            if (cum_step + 1) % 1 == 0:
                optimizer_bert.step()
                optimizer_bert.zero_grad()
                optimizer_tune.step()
                optimizer_tune.zero_grad()
                cum_step += 1
        print("set_test......")
        F1, EM = set_test(model, test_iter)

        print('dev set : step_{},F1_{},EM_{}'.format(cum_step, F1, EM))
        if F1 >= best_acc:
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(os.path.join(out_dir, 'model_{:.4f}_{:.4f}_{}'.format(F1, EM,str(cum_step))))
            # torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(model_to_save, output_model_file)

def set_test(model, test_iter):
    if not test_iter.is_test:
        test_iter.is_test = True
    model.eval()
    with torch.no_grad():
        pred_score_dict = {}
        pred_answer_dict = {}
        true_answer_list = []
        pred_answer_list = []
        for input_ids, input_mask, segment_ids,start_list,end_list,uid_list,answer_list,text_list,querylen_list,maping_list,_,seq_length in tqdm(
                test_iter):
            input_ids = list2ts2device(input_ids)
            input_mask = list2ts2device(input_mask)
            segment_ids =list2ts2device(segment_ids)
            y_preds = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            start_preds, end_preds = (p.detach().cpu() for p in y_preds)
            start_probs = softmax(start_preds.numpy())
            end_probs = softmax(end_preds.numpy())
            # cls_probs = softmax(cls_pred.numpy())
            max_a_len = 64
            answer_pred = []
            epsilon = 1e-3
            w1,w2=0.9,0.1
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
                                score=p_start*p_end
                                # score_ = np.exp((0.5 * np.log(p_start + epsilon) + 0.5 * np.log(p_end + epsilon)) / (0.5 + 0.5))
                start, end = start_end
                """"""
                # out_class_=cls_probs[i][1]
                start_cls = start_probs[i][0]
                end_cls = end_probs[i][0]
                pos_cls = -(start_cls + end_cls)
                if config.addneg:
                    score+=pos_cls
                # score = np.exp((w1 * np.log(out_class_ + epsilon) + w2 * np.log(score_ + epsilon)) / (w1 + w2))
                try:
                    start_pos = maping_list[i][start][0]
                    end_pos = maping_list[i][end][-1]
                    answer = text_list[i][start_pos:end_pos + 1]
                except:
                    answer = ''
                if uid_list[i] in pred_answer_dict:
                    if pred_score_dict[uid_list[i]] < score:
                        pred_score_dict[uid_list[i]] = score
                        pred_answer_dict[uid_list[i]] = answer
                else:
                    pred_answer_dict[uid_list[i]] = answer
                    pred_score_dict[uid_list[i]] = score
                answer_pred.append(answer)
            true_answer_list.extend(answer_list)
            pred_answer_list.extend(answer_pred)
        assert len(true_answer_list) == len(pred_answer_list)
        print(true_answer_list)
        print(pred_answer_list)
        c = 0
        for i in range(len(true_answer_list)):
            if true_answer_list[i] == pred_answer_list[i]:
                c += 1
        print('em分数为', c / len(true_answer_list))
        # pred_dict=dict(zip(qid_list,pred_answer_list))
        true_dict = json.load(open(config.data + 'dureader_robust-data/dev.json'))
        F1, EM, TOTAL, SKIP = evaluate(true_dict, pred_answer_dict)
        print('F1: {}, EM {}, TOTAL {}'.format(F1, EM, TOTAL))

        return F1, EM
def softmax(x, axis=1):
    """
    自写函数定义softmax
    :param x:
    :param axis:
    :return:
    """
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
    #计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def comput_p(true_list,pred_list):
    c=0
    for i in range(len(true_list)):
        if  true_list[i]==pred_list[i]:
            c+=1
    return c/(i+1)


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file  # 通用词典
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    re_tokenzier = Tokenizer(vocab_file, do_lower_case)
    train_iter = DataIterator(config.batch_size, data_file=config.process+'train.csv',
                              use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length,config=config)

    dev_iter = DataIterator(config.batch_size, data_file=config.process+'dev.csv',
                            use_bert=config.use_bert, tokenizer=tokenizer,
                            seq_length=config.sequence_length, is_test=True,config=config)
    train(train_iter, dev_iter, config)
#fold2 new_answer