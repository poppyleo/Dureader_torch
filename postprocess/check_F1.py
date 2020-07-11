import os

# folder_path = '/data/wangzhili/lei/QA/Savemodel_joint/'
# folder_path = '/data/wangzhili/lei/Dureader/save_model/'
folder_path = '/home/none404/hm/DU_model_torch/'
for folder_l1 in os.listdir(folder_path):
        for folder_l2 in os.listdir(folder_path + folder_l1):
            name_list=os.listdir(os.path.join(folder_path,folder_l1, folder_l2))
            if name_list==[]:
                continue
            max_f1 = 0
            max_p = max_r = 0
            max_em=0
            max_l = ''
            try:
                for name in name_list:
                    line = name.strip().split('_')
                    f1 = float(line[-3])
                    em = float(line[-2].split('-')[0])
                    if f1 > max_f1:
                        max_f1 = f1
                        max_em = em
                        max_l = name
                print(
                    '{} : f1 {} , em {} '.format(os.path.join(folder_path ,folder_l1 , folder_l2,max_l),
                                                               max_f1,
                                                               max_em, ))
            except:
                pass