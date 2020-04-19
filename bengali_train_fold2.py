# import necessary libraries
from bengali_dataset import *
from bengali_models import *
from bengali_trainer import *

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    
    # Define random state
    random_state=4690;set_random_state(random_state);
    
    # define batch size, number of workers for dataloader and gradient accumulation steps
    batchSize = 100
    n_workers = 4
    grad_accum_steps = 5
    
    # Load all data into a single numpy array 
    HEIGHT = 137
    WIDTH = 236
    TRAIN = ['/home/bengalidata/data/train_image_data_0.parquet',
             '/home/bengalidata/data/train_image_data_1.parquet',
             '/home/bengalidata/data/train_image_data_2.parquet',
             '/home/bengalidata/data/train_image_data_3.parquet']
    
    all_imgs = []
    for fname in TRAIN:
        print(fname)
        df = pd.read_parquet(fname)
        data = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        for idx in tqdm(range(len(df))):
            img = data[idx].astype(np.uint8)
            img = img[np.newaxis,:,:]
            all_imgs.append(img.copy())
            
    del df, data
            
    all_imgs = np.concatenate(all_imgs,axis=0)
    train_labels = np.load('/home/bengalidata/train_labels.npy')
    
    # Get folds on the basis of grapheme labels
    grapheme_labels = train_labels[:,0]
    skf = StratifiedKFold(n_splits= 5, random_state=random_state, shuffle = True)      
    for i,(train_index, val_index) in enumerate(skf.split(all_imgs,grapheme_labels)):
        if i != 1:
            continue
        
        
        X_train, X_val = all_imgs[train_index,:,:], all_imgs[val_index,:,:]
        y_train, y_val = train_labels[train_index,:], train_labels[val_index,:]    
        
        
        ds_train = BengaliAiDataset(X_train, y_train)
        dl_train = DataLoader(ds_train, shuffle=True, batch_size=batchSize, num_workers=n_workers, pin_memory=True)
        
        
        ds_val = BengaliAiDataset(X_val, y_val)
        dl_val = DataLoader(ds_val, shuffle=False, batch_size=batchSize, num_workers=n_workers, pin_memory=True)
        
        del all_imgs,X_train,y_train,X_val,y_val
            
        # first train the model initialized with pretrained weight
        model = CNN_Model(model_name='efficientnet-b5',global_dim=2048,drp=0.125)
        weight_saving_path = '/home/bengalidata/efficientnetb5_cutmix/fold2/'
        resume_checkpoint_path = '/home/bengalidata/efficientnetb5_cutmix/fold2/checkpoint_best_Average_recall.pth'    
        args = { 
                'model':model,
                'Loaders' : [dl_train,dl_val],
                'metrics':{'Loss':AverageMeter,'grapheme_root_recall':PrintMeter,'vowel_diacritic_recall':PrintMeter,
                            'consonant_diacritic_recall':PrintMeter,'Average_recall':PrintMeter},
                
                'modes' :{'Loss':'min','grapheme_root_recall':'max','vowel_diacritic_recall':'max','consonant_diacritic_recall':'max',
                          'Average_recall':'max'},
                          
                'checkpoint_saving_path' : weight_saving_path,
                'resume_train_from_checkpoint':False,
                'resume_checkpoint_path': resume_checkpoint_path,
                'lr' :0.03,
                'fold':i+1,
                'epochsTorun' : 150,
                'test_run_for_error':False,
                'do_accum':True,
                'accumSteps': grad_accum_steps,
                'accum_loss_division':True,
                'alpha': 0.80,
                'load_checkpoint': False,
                'weight_decay':0.0008,
                'problem_name': 'Bengali_ai_grapheme_detection',
                } 
        
        Trainer = ModelTrainer(**args)
        Trainer.fit()
        
        # Second, train the model initialized with the weight from the first step
        model = CNN_Model(model_name='efficientnet-b5',global_dim=2048,drp=0.125)
        weight_saving_path = '/home/bengalidata/efficientnetb5_cutmix/fold2/'
        resume_checkpoint_path = '/home/bengalidata/efficientnetb5_cutmix/fold2/checkpoint_best_Average_recall.pth'    
        args = { 
                'model':model,
                'Loaders' : [dl_train,dl_val],
                'metrics':{'Loss':AverageMeter,'grapheme_root_recall':PrintMeter,'vowel_diacritic_recall':PrintMeter,
                            'consonant_diacritic_recall':PrintMeter,'Average_recall':PrintMeter},
                
                'modes' :{'Loss':'min','grapheme_root_recall':'max','vowel_diacritic_recall':'max','consonant_diacritic_recall':'max',
                          'Average_recall':'max'},
                          
                'checkpoint_saving_path' : weight_saving_path,
                'resume_train_from_checkpoint':False,
                'resume_checkpoint_path': resume_checkpoint_path,
                'lr' :0.03,
                'fold':i+1,
                'epochsTorun' : 150,
                'test_run_for_error':False,
                'do_accum':True,
                'accumSteps': grad_accum_steps,
                'accum_loss_division':True,
                'alpha': 0.80,
                'load_checkpoint': True,
                'weight_decay':0.0008,
                'problem_name': 'Bengali_ai_grapheme_detection',
                } 
        
        Trainer = ModelTrainer(**args)
        Trainer.fit()
