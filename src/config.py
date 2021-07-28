class Config:
    data_dir = '/data/siim-covid19-detection/'
    train_dir = data_dir + '640/train/'
    mask_dir = data_dir + '640/train_mask/'
#     study_folds_csv_path = data_dir + 'train_study_folds.csv' # drop all duplicate images 6054
    study_folds_csv_path = data_dir + 'train_study_folds_washen_v2.csv' # drop duplicate images 6227
#     study_folds_csv_path = data_dir + 'train_all_study_folds_washen.csv' # all images 6334
    test_csv_path = data_dir + 'sample_submission.csv'
    
#     ext = True  ## use external data
#     external_train_dir = data_dir + '640/external_train/'
#     external_study_folds_csv_path = data_dir + 'external_study_all_folds.csv' #images 1022
    
#     pseudo = True ## use test data
#     pseudo_train_dir = '../test/'
#     pseudo_folds_csv_path = data_dir + 'pseudo_data_folds.csv' #images 2477
    
    num_classes = 4
#     model_name = 'tf_efficientnet_b4_ns'
    model_name = 'tf_efficientnet_b5_ns'
#     model_name = 'tf_efficientnet_b7_ns'
#     model_name = 'tf_efficientnetv2_m'
#     model_name = 'tf_efficientnetv2_l'
    image_size = 512           ### input size in training

#     model_name = 'swin_base_patch4_window12_384'
#     image_size = 384            ### input size in training
    
    device = 'cuda'             ### set gpu or cpu mode
    debug = False              ### debug flag for checking your modify code
    
    gpus = 4                 ### gpu numbers
    precision = 16             ### training precision 8, 16,32, etc
    batch_size = 8       ### total batch size
    

    lr = 1e-4     ### learning rate default 1e-4,effnet, 2.5e-5,swin transformer
    min_lr = 1e-6              ### min learning rate
    weight_decay = 1e-6
#     gradient_accumulation_steps = 1
#     max_grad_norm =1000         ### 5
    num_workers = 16            ### number workers
    print_freq = 100            ### print log frequency

    seed = 42
    n_fold = 5
    trn_fold = [0,1,2,3,4] # [0,1,2,3]
    
    optimizer = 'Adam'
    scheduler = 'GradualWarmupSchedulerV2' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'GradualWarmupSchedulerV2']

    # ReduceLROnPlateau
    factor=0.2 # ReduceLROnPlateau
    patience=4 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    
    ## CosineAnnealingLR
    T_max= 10 # CosineAnnealingLR

    ## CosineAnnealingWarmRestarts
    T_0 = 10
    T_mult = 1
    
    ## GradualWarmupSchedulerV2
    warmup_epochs = 1
    cosine_epochs = 29
    epochs = warmup_epochs + cosine_epochs ### total training epochs
#     epochs = 35 ### total training epochs
    multiplier = 10
   
    freeze_epochs = 5

    ### cls loss for logits
#     criterion = 'LabelSmoothingBinaryCrossEntropy'  ### BinaryCrossEntropy, LabelSmoothingBinaryCrossEntropy
    # CrossEntropy, SCELoss, LabelSmoothingCrossEntropy, FocalCosineLoss, BiTemperedLogisticLoss, TaylorLabelSmoothingCrossEntropy
    criterion = 'LabelSmoothingBinaryCrossEntropy'  ##ClassBalancedLabelSmoothingCrossEntropy
    label_smoothing = 0.0 #best
    
    t1 = 0.8
    t2 = 1.2

    taylor_n = 2

    ### aux loss for masks
    seg_type = 'lovasz'
    seg_prob = 0.0

    save_row_num = 8
    save_dir = f'/data/output/pl-siim-covid19-study-classification/{model_name}_folds_cutout_{image_size}_{epochs}e_{optimizer}_{scheduler}_{criterion}_ls{label_smoothing}_{seg_type}{seg_prob}_v2_dropdup/'