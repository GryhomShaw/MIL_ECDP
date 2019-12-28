import os
class Config:
    #MIL_train
    n_classes = 2
    root = os.environ['ECDP_ROOT']
    data_path = os.path.join(root,'data')  #数据存放位置
    data_pos_path = os.path.join(root, 'data', 'pos') #pos数据的存放位置
    data_neg_path = os.path.join(root, 'data', 'neg')
    data_append_path = os.path.join(root, 'data_append') #追加数据存放的位置
    data_split = os.path.join(root, 'dataloader/train_val_split.json')
    label_path = os.path.join(root,'dataloader/HEROHE_HER2_STATUS.xlsx')
    patch_data = os.path.join(root,'Patch')
    patch_overlap = os.path.join(data_path, 'overlap_count')

    #MIL_test
    patch_predict = os.path.join(root,'Predict')
    rnn_data_split = os.path.join(root,'dataloader/rnn_data_split.json')
    color_img_path = os.path.join(patch_predict,'color')

    weights = 0.8 # 正负比(默认0.5)
    mean = [0.5, 0.5, 0.5]
    std = [0.1, 0.1, 0.1]

cfg = Config()




