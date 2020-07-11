class Config:
    
    def __init__(self):
        
        self.embed_dense = True
        self.embed_dense_dim = 512
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        self.relation_num = 3
        self.over_sample = True

        self.decay_rate = 0.5
        self.decay_step = 5000
        self.num_checkpoints = 20 * 3

        self.train_epoch = 15
        # self.sequence_length = 384
        self.sequence_length =512#苏剑林参数
        self.query_length = 25  #苏剑林参数

        self.learning_rate = 1e-4
        # self.embed_learning_rate = 5e-5
        self.embed_learning_rate = 5e-5 #苏剑林参数
        self.batch_size =24
        self.embed_trainable = True

        self.as_encoder = True

        # large_file='/home/wangzhili/pretrained_model/roeberta_zh_L-24_H-1024_A-16/'
        # self.bert_file = large_file + 'roberta_zh_large_model.ckpt'
        # self.bert_config_file = large_file + 'bert_config_large.json'
        # self.vocab_file = large_file + 'vocab.txt'
        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.continue_training = False
        self.ensemble_source_file  = '/data/wangzhili/lei/ensemble/sourcefile/'
        self.ensemble_result_file = '/data/wangzhili/lei/ensemble/resultfile/'

        # 五折ERNIE 原生   BILSTM=256
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_4/1588590749/model_81.0862_67.6076"
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_0/1588657538/model_82.7339_70.7128"
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_1/1588664460/model_84.4784_73.5356"
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_4/1588686935/model_84.4139_72.9711"#gru
        # self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_0/1588693940/model_84.4281_73.0416_3630"#large
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_4/1588756473/model_84.2363_72.1948_3354"#lstm  #69.39075	52.8
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_3/1588756537/model_84.1749_72.9711_1118"#gru   #70.12963	53.7   512
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_6/1589024907/model_84.5161_72.6182_1210"#gru fgm  #69.9951	53.95
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_5/1589018007/model_82.6603_69.6542_1815"#gru not   67.73447
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_3/1589028153/model_84.1505_71.9831_1815"#gru token_type #69.3908
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_4/1589027861/model_84.5808_72.9711_1210"#position  #70.28586
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_3/1589089131/model_84.4115_72.6182_1815"#position  # task2 69.75615	52.1
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_7/1589077469/model_84.0526_71.9125_1815"#PGD  #69.53977	52.8
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_4/1589085318/model_84.7394_73.3239_1815"#  #  not gru lstm  69.46055	53.05
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_7/1589205635/model_84.0037_73.1122_1210"#  #  fgm3层addneg  71.34724	55.3
        self.checkpoint_path = "/home/none404/hm/DU_model_torch/runs_6/1589205655/model_84.1395_72.1242_1815"#  #  fgm2层addneg  68.57927	50.65
        self.checkpoint_path='/home/none404/hm/DU_model_torch/runs_3/1589290005/model_83.6166_72.1242_1210' #addneg ,4  70.28129	54.5
        self.checkpoint_path='/home/none404/hm/DU_model_torch/runs_4/1589289812/model_83.6668_71.8419_1815' #fgm5  68.91713	52.5
        self.checkpoint_path='/home/none404/hm/DU_model_torch/runs_6/1589289495/model_83.5629_72.3359_1210' #4层   70.53276	54.4
        self.checkpoint_path='/home/none404/hm/DU_model_torch/runs_7/1589289427/model_83.8716_71.9831_1815' #fgm3 not addneg 68.05814	50.4
        self.checkpoint_path='/home/none404/hm/DU_model_torch/runs_7/1589383301/model_83.9045_72.4771_1210' # word  gru pgd add3
        self.checkpoint_path='/home/none404/hm/DU_model_torch/runs_6/1589383174/model_83.4116_71.6302_4235' # pos  lstm pgd add3
        self.checkpoint_path='/home/none404/hm/DU_model_torch/runs_3/1589382900/model_83.9584_72.3359_1815' # gpu3 word gru add3
        self.checkpoint_path='/home/none404/hm/DU_model_torch/runs_4/1589383008/model_83.3491_71.9125_1210' # gpu3 word gru add3

        #  数据预处理的路径
        # self.data_process = '/data/wangzhili/lei/nCoV/'
        # """学校"""
        # """roberta"""
        # bert_file='/home/wangzhili/pretrained_model/roberta_zh_l12/'
        #
        # self.bert_file = bert_file+'bert_model.ckpt'
        # self.bert_config_file = bert_file+ 'bert_config.json'
        # self.vocab_file = bert_file+'vocab.txt'
        #
        # """albert"""
        # # bert_file='/home/wangzhili/pretrained_model/roberta_zh_l12/''/home/wangzhili/pretrained_model/albert_zh_xlarge'
        # # self.bert_file = bert_file+'albert_model.ckpt'
        # # self.bert_config_file = bert_file+ 'albert_config_xlarge.json'
        # # self.vocab_file = bert_file+'vocab.txt'
        # self.save_model = '/home/wangzhili/lei/DureaderQA/save_model/'
        # self.data = '/home/wangzhili/data/baidu_qa/'
        # self.process='/home/wangzhili/data/baidu_qa/processed/'

        self.save_model = '/home/none404/hm/DU_model_torch/'
        self.data = '/home/none404/hm/baidu_qa/'
        self.process='/home/none404/hm/baidu_qa/processed/'
        # """albert"""


        """RObera_pytorch"""
        self.bert = 'bert'
        self.bert_path = '/home/none404/hm/Model/Roberta_zh_l12/'
        self.bert_file = self.bert_path + 'pytorch_model.bin'
        self.bert_config_file = self.bert_path + 'config.json'
        self.vocab_file = self.bert_path + 'vocab.txt'

        # """RObera_pytorch——24"""
        # self.bert = 'bert'
        # self.bert_path = '/home/none404/hm/Model/Roberta_zh_l24/'
        # self.bert_file = self.bert_path + 'pytorch_model.bin'
        # self.bert_config_file = self.bert_path + 'config.json'
        # self.vocab_file = self.bert_path + 'vocab.txt'

        """albera_pytorch"""
        # self.bert = 'bert'
        # self.bert_path = '/home/none404/hm/Model/albert_xlarge_zh'
        # self.bert_file = self.bert_path + 'pytorch_model.bin'
        # self.bert_config_file = self.bert_path + 'config.json'
        # self.vocab_file = self.bert_path + 'vocab.txt'

        self.use_origin_bert ='dym'  # 'ori':使用原生bert, 'dym':使用动态融合bert,'hire':Hirebert
        self.is_avg_pool =True # True: 使用平均avg_pool

        self.pool='mean'  #'mean' 为平均池化，其他为最大池化
        self.fold=1
        self.compare_result=False
        self.result_file='/data/wangzhili/lei/QA/result/'
        self.gpu=2#同时可指定多个gpu并行计算
        self.keep_prob = 0.9
        #卷积参数
        self.addtype = False   #是否训练是否答案
        self.num_filters=512 #卷积核个数
        self.kernel_size = 7  # 卷积核尺寸
        self.topk = 1 #生成前n个候选文档
        #hire_bert参数
        self.gru_hidden_dim=64
        self.slide_window=400
        self.is_split = False
        self.gru=True
        self.lstm=False
        self.num_filters = 512
        self.per_gpu_eval_batch_size = 16
        self.mask_q=False
        self.mask_padding=True
        self.test_batch_size=16
        self.num_layer=1#gru,lstm层数
        self.adv='pgd'
        self.addneg=False
        self.embed_name='bert.embeddings.word_embeddings.weight'  # 词
        # self.embed_name='bert.embeddings.position_embeddings.weight'  # 位置
        # self.embed_name='bert.embeddings.token_type_embeddings.weight'  # 前后句子embedding
        #gpu4 gru dym
        #gpu3 lstm dym
        #gpu4 dym
        #gpu5 ori
        # gpu0 large_dym
        #gpu3 addneg ,4
        #4 fgm5
        #6  fgm  2层addneg   4层
        #7  pgd   fgm3层addneg   fgm3 not addneg
        #4 position
        #3 token_type  Best twotask

        #gpu3 word gru add3
        #gpu4 pos lstm add3
        #gpu6 pos  lstm pgd add3
        #gpu7 word  gru pgd add3