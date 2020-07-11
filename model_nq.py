from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel,load_tf_weights_in_bert
from transformers import  AlbertModel, AlbertPreTrainedModel
import torch.nn as nn
import torch
from config import Config

config = Config()




class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, bert_config,checkpoint_path=None):
        super(BertForQuestionAnswering, self).__init__(bert_config)
        if config.bert=='bert':
            self.bert = BertModel(bert_config)
        elif config.bert=='albert':
            self.bert = AlbertModel(bert_config)
        if checkpoint_path:
            """加载tf模型"""
            self.bert = load_tf_weights_in_bert(self.bert, config=None, tf_checkpoint_path=checkpoint_path)

        if config.use_origin_bert=='dym':
            self.hidden_size=512
        else:
            self.hidden_size = 768

        self.qa_outputs = nn.Linear(self.hidden_size, 2)  # start/end
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        # self.cls = nn.Linear(self.hidden_size, 2) #for cls
        self.classifier = nn.Linear(768, 1)  #for dym's dense
        self.dense_final = nn.Sequential(nn.Linear(768, self.hidden_size), nn.ReLU(True)) #动态最后的维度
        # self.classifier = nn.Linear(1024, 1)  #for dym's dense
        # self.dense_final = nn.Sequential(nn.Linear(1024, self.hidden_size), nn.ReLU(True)) #动态最后的维度#large
        if config.lstm:
            num_layers=config.num_layer
            lstm_num=int(self.hidden_size/2)
            self.lstm = nn.LSTM(self.hidden_size, lstm_num,
                                num_layers, batch_first=True,#第一维度是否为batch_size
                                bidirectional=True)  #双向
        elif config.gru:
            num_layers=config.num_layer
            lstm_num=int(self.hidden_size/2)
            self.lstm = nn.GRU(self.hidden_size, lstm_num,
                                num_layers, batch_first=True,#第一维度是否为batch_size
                                bidirectional=True)  #双向

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, start_labels=None,end_labels=None,cls_label=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)
        # print('='*88)
        sequence_output = outputs[0]  #bert原生sequence_out_put [batch_size,seq_len,hidden_size]
        # pooled_output = outputs[1]    #bert原生pooled_out_put  [batch_size,hidden_size]

        if config.use_origin_bert=='dym':
            sequence_output=self.get_dym_layer(outputs)  # [batch_size,seq_len,512]
        if config.gru or config.lstm:
            sequence_output=self.lstm(sequence_output)[0]
        mrc_output = self.dropout(sequence_output)
        qa_logits = self.qa_outputs(mrc_output) #start/end二分类
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # classification
        # """有无答案的分类"""
        # pooled_output=torch.nn.functional.max_pool1d(sequence_output,config.sequence_length)
        # pooled_output=torch.reshape(pooled_output,(config.batch_size,self.hidden_size))
        # cls_output = self.dropout(pooled_output)
        # classifier_logits = self.cls(cls_output)
        """mask_padding"""
        def mask_logits(logits):
            #padding位置的mask
            if config.mask_padding:
                mask_list =attention_mask.float()
                logits =logits-(1.0 - mask_list) * 1e12 #mask位置得到一个很小的值
            return logits
        if config.mask_padding:
            start_logits=mask_logits(start_logits)
            end_logits=mask_logits(end_logits)

        
        if start_labels is not None:
            start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_logits, start_labels)#交叉熵计算损失
            end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_logits, end_labels) #交叉熵
            # class_loss = nn.CrossEntropyLoss()(classifier_logits, cls_label)
            outputs = start_loss + end_loss#  + 0.04*class_loss  #训练返回loss
        else:
            outputs = (start_logits, end_logits) #测试返回起始答案

        return outputs

    def get_dym_layer(self,outputs):
        layer_logits = []
        all_encoder_layers=outputs[2][1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))
        layer_logits = torch.cat(layer_logits, 2)
        layer_dist = torch.nn.functional.softmax(layer_logits)
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)
          ##是否真的需要激活函数
        word_embed = self.dense_final(pooled_output)
        dym_layer = word_embed
        return dym_layer