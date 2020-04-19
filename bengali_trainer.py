# import libraries
from tqdm import tqdm
import os
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn as nn
import sklearn.metrics
from bengali_optimizers import *
from bengali_cutmix import *
from apex import amp

# Class for calculating average recall score during training and validation 
class RecallScore(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.solution_grapheme_root = []
        self.solution_vowel_diacritic = []
        self.solution_consonant_diacritic = []
        self.submission_grapheme_root = []
        self.submission_vowel_diacritic = []
        self.submission_consonant_diacritic = []
    def update(self, inp):
        self.solution_grapheme_root = self.solution_grapheme_root + inp[0][0].tolist()
        self.solution_vowel_diacritic = self.solution_vowel_diacritic + inp[0][1].tolist()
        self.solution_consonant_diacritic = self.solution_consonant_diacritic + inp[0][2].tolist()
        
        self.submission_grapheme_root = self.submission_grapheme_root + inp[1][0].tolist()
        self.submission_vowel_diacritic = self.submission_vowel_diacritic + inp[1][1].tolist()
        self.submission_consonant_diacritic = self.submission_consonant_diacritic + inp[1][2].tolist()
        
    def feedback(self):
        grapheme_root_recall = sklearn.metrics.recall_score(self.solution_grapheme_root, self.submission_grapheme_root, average='macro')
        vowel_diacritic_recall = sklearn.metrics.recall_score(self.solution_vowel_diacritic, self.submission_vowel_diacritic, average='macro')
        consonant_diarcitic_recall = sklearn.metrics.recall_score(self.solution_consonant_diacritic, self.submission_consonant_diacritic, average='macro')
        scores = [grapheme_root_recall,vowel_diacritic_recall,consonant_diarcitic_recall]
        final_score = np.average(scores, weights=[2,1,1])        
        return grapheme_root_recall,vowel_diacritic_recall,consonant_diarcitic_recall,final_score
    
# function for defining random state
def set_random_state(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed_value)
   
# class for calculating average values
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, inp):
        val = inp[0]
        n = inp[1]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def feedback(self):
        return self.avg

# class for storing value and return stored value when needed
class PrintMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.value = 0
    def update(self, inp):
        self.value = inp
    def feedback(self):
        return self.value

# class for storing all the training and validation parameters
class store_info(object):
    def __init__(self,metrics,modes=None):
        self.default_best_scores = {'min':9999,'max':-9999}
        self.modes = modes
        self.current_epoch = 1
        self.metrics = metrics           
        self.keys = list(metrics.keys())
        self.all_info = {'Metrics':self.keys}
        
        self.info_function = {}
        for key in metrics.keys():
            self.info_function.update({'Train'+key:metrics[key]()})
            self.info_function.update({'Val'+key:metrics[key]()})
            self.all_info.update({'BestVal'+key:self.default_best_scores[modes[key]]})
            
        self._init_new_epoch(self.current_epoch)
    def _init_new_epoch(self,epoch_no):
        train_info = {}
        val_info = {}
        for key in self.metrics.keys():
            train_info.update({'Epoch_'+key:0})
#            train_info.update({'Per_Batch_'+key:[]})
            val_info.update({'Epoch_'+key:0})
#            val_info.update({'Per_Batch_'+key:[]})
        
        self.all_info.update({'Epoch_'+str(epoch_no):{'Train':train_info,'Val':val_info}})
        self.current_epoch = epoch_no
        self.reset_info()
    def reset_info(self):
        for key in self.info_function.keys():
            self.info_function[key].reset()        
    def update_train_info(self,info_dict):
        for key in info_dict.keys():
            self.info_function['Train'+key].update(info_dict[key])           
            self.all_info['Epoch_'+str(self.current_epoch)]['Train']['Epoch_'+key] = self.info_function['Train'+key].feedback()
#            self.all_info['Epoch_'+str(self.current_epoch)]['Train']['Per_Batch_'+key].append(info_dict[key][0])
    def update_val_info(self,info_dict):
        for key in info_dict.keys():        
            self.info_function['Val'+key].update(info_dict[key])           
            self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key] = self.info_function['Val'+key].feedback()
#            self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Per_Batch_'+key].append(info_dict[key][0])
    def request_current_epoch_metric_info(self):
        info = {}
        for key in self.keys:
             info.update({'Train'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Train']['Epoch_'+key]})
             info.update({'Val'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key]})
        return info
    def request_current_epoch_train_metric_info(self):
        info = {}
        for key in self.keys:
             info.update({'Train'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Train']['Epoch_'+key]})
        return info
    def request_current_epoch_val_metric_info(self):
        info = {}
        for key in self.keys:
             info.update({'Val'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key]})
        return info
#    def request_current_epoch_allbatch_metric_info(self):
#        info = {}
#        for key in self.keys:
#             info.update({'Train'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Train']['Per_Batch_'+key]})
#             info.update({'Val'+key:self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Per_Batch_'+key]})
#        return info
    def request_allinfo(self):
        return self.all_info
    def load_info(self,all_info):
        self.all_info = all_info
    def get_info(self,epoch_no,metric,mode='Val'):
        if mode not in ['Val','Train']:
            raise print('Mode should be either Val or Train!')
        return self.all_info['Epoch_'+str(epoch_no)][mode]['Epoch_'+metric]        
    def is_best_metric(self):
        self.is_best = {}
        for key in self.metrics.keys():
            is_best = False
            if self.modes[key] == 'min':
                if self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key] < self.all_info['BestVal'+key]:
                    print('Val'+key +' is improved from {:.4f} to {:.4f}'.format(self.all_info['BestVal'+key],self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key]))
                    self.all_info['BestVal'+key] = self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key]
                    is_best = True
                else:
                    print('Val'+key +' is not improved from {:.4f}'.format(self.all_info['BestVal'+key]))
            elif self.modes[key] == 'max':
                if self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key] > self.all_info['BestVal'+key]:
                    print('Val'+key +' is improved from {:.4f} to {:.4f}'.format(self.all_info['BestVal'+key],self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key]))
                    self.all_info['BestVal'+key] = self.all_info['Epoch_'+str(self.current_epoch)]['Val']['Epoch_'+key]
                    is_best = True
                else:
                    print('Val'+key +' is not improved from {:.4f}'.format(self.all_info['BestVal'+key]))
                    
            self.is_best.update({key:[is_best,self.all_info['BestVal'+key]]})
        
        return self.is_best

# callback object for showing progress bar
class TQDM(object):
    def __init__(self):
        self.progbar_train = None
        self.progbar_val = None
        super(TQDM, self).__init__()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar_train is not None:
            self.progbar_train.close()
            
        if self.progbar_val is not None:
            self.progbar_val.close()
    
    def on_train_begin(self, logs):
        self.train_logs = logs
    def on_val_begin(self, logs):
        self.val_logs = logs
    
    def on_epoch_train_begin(self, fold, epoch):
        try:
            self.progbar_train = tqdm(total=self.train_logs['num_batches'],
                                unit=' batches')
            self.progbar_train.set_description('(Train) Fold %i Epoch %i/%i' % 
                            (fold, epoch, self.train_logs['num_epoch']))
        except:
            pass
    
    def on_epoch_train_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar_train.set_postfix(log_data)
        self.progbar_train.update()
        self.progbar_train.close()
        print('')
        
    def on_epoch_val_begin(self, fold, epoch):
        try:            
            self.progbar_val = tqdm(total=self.val_logs['num_batches'],
                                unit=' batches')
            self.progbar_val.set_description('(Valid) Fold %i Epoch %i/%i' % 
                            (fold, epoch, self.val_logs['num_epoch']))                                    
        except:
            pass
    def on_epoch_val_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}        
        self.progbar_val.set_postfix(log_data)
        self.progbar_val.update()
        self.progbar_val.close()
        print('')
        
    def on_train_batch_begin(self):
        self.progbar_train.update(1)
        
    def on_train_batch_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar_train.set_postfix(log_data)
        
    def on_val_batch_begin(self):
        self.progbar_val.update(1)
        
    def on_val_batch_end(self, logs=None):
        log_data = {key: '%.04f' % logs[key] for key in logs.keys()}
        self.progbar_val.set_postfix(log_data)

# class for early stopping the training
class EarlyStopping:    
    def __init__(self,threshold=0.01,patience=5):
        self.earlyThreshold = threshold
        self.patience = patience        
    def on_train_begin(self):
        self.loss_history = []        
    def on_epoch_end(self,loss,epoch):
        self.loss_history.append(loss)
        stop = False
        if epoch >= self.patience-1:
            best = min(self.loss_history)
            t = self.loss_history[epoch-self.patience+1:epoch+1]
            differences = [best-i for i in t]
            count = sum([1 for x in differences if (x >= -self.earlyThreshold and x<=0)])
            if count == self.patience: print('Early stopping');stop = True            
        return stop
       
#%% #################################### Model Trainer Class #################################### 
class ModelTrainer():
    def __init__(self, problem_name = None, model=None, Loaders=[None,[]], metrics=None, modes=None, fold=None, lr=0.003, epochsTorun=40,
                 save_checkpoint_based_on=None, checkpoint_saving_path=None,
                 resume_train_from_checkpoint=False, resume_checkpoint_path=None, resume_lr_checkpoint=False,
                 loss_criteria = None, observing_parameter=None,weight_decay=0.0002,
                 use_early_stopper=False, patience = 5, earlyThreshold = 0.0004, test_run_for_error=False,
                 do_accum=True,accumSteps=40,accum_loss_division=False,alpha=0.4,load_checkpoint=False):
        
        super(ModelTrainer, self).__init__()
        
        if resume_lr_checkpoint == True and resume_train_from_checkpoint==False:
            raise print('Error')
            
        self.problem_name = problem_name
        self.model = model.cuda()
        self.trainLoader = Loaders[0]
        self.valLoader = Loaders[1]        
        self.info_bbx = store_info(metrics,modes)
        self.save_checkpoint_based_on = save_checkpoint_based_on
        self.fold = fold
                
#        if self.fold != None:
#            self.checkpoint_saving_path = checkpoint_saving_path + '/fold' + str(self.fold) + '/'
#        else:
#            self.checkpoint_saving_path = checkpoint_saving_path + '/'
#            self.fold = 1
        
        self.checkpoint_saving_path = checkpoint_saving_path
#        os.makedirs(self.checkpoint_saving_path,exist_ok=True)
        
        self.lr = lr
        self.epochsTorun = epochsTorun
        self.init_epoch = -1
        self.use_early_stopper = use_early_stopper
        self.observing_parameter = observing_parameter
        self.test_run_for_error = test_run_for_error
        self.count = 1
        
        self.best_grapheme_root_recall = -9999
        self.best_vowel_diacritic_recall = -9999
        self.best_consonant_diacritic_recall = -9999
        self.best_average_recall = -9999        
        self.best_loss = 9999
        
        if self.use_early_stopper:
            self.earlyStopper = EarlyStopping(threshold= earlyThreshold, patience= patience)
        
        self.resume_checkpoint_path = resume_checkpoint_path
        self.optimizer = Over9000(params=self.model.parameters(),lr=self.lr,weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer,factor=0.5,mode='min',patience=7,verbose = True)    
        
        if load_checkpoint:
            checkpoint_dict = torch.load(resume_checkpoint_path)
            self.model.load_state_dict(checkpoint_dict['Model_state_dict'])
            print('Checkpoint Model Loaded!')
            print('Best Average recall {}'.format(checkpoint_dict['Best_average_recall']))
        
        if resume_train_from_checkpoint:
            if os.path.isfile(resume_checkpoint_path):
                print("=> Loading checkpoint from '{}'".format(resume_checkpoint_path))
                checkpoint_dict = torch.load(resume_checkpoint_path)
                self.model.load_state_dict(checkpoint_dict['Model_state_dict'])
                self.scheduler.load_state_dict(checkpoint_dict['Scheduler_state_dict'])
                self.optimizer.load_state_dict(checkpoint_dict['Optimizer_state_dict'])
                                
                self.best_loss = checkpoint_dict['Best_val_loss']
                self.best_grapheme_root_recall = checkpoint_dict['Best_grapheme_root_recall']
                self.best_vowel_diacritic_recall = checkpoint_dict['Best_vowel_diacritic_recall']
                self.best_consonant_diacritic_recall = checkpoint_dict['Best_consonant_diacritic_recall']
                self.best_average_recall = checkpoint_dict['Best_average_recall']
                
                self.info_bbx.all_info = checkpoint_dict['All_info']
                self.init_epoch = checkpoint_dict['Epoch']
                
                self.scheduler_flag = checkpoint_dict['Scheduler_flag']
            else:
                print("=> No checkpoint found at '{}' !".format(resume_checkpoint_path))
        
        
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic")
        
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.accumSteps = accumSteps
        self.do_accum = do_accum
        self.accum_loss_division = accum_loss_division
        
        self.scheduler_flag = 9999
        self.use_scheduler_flag = True
        self.use_only_non_mixup_cutmix_for_train_metric = True
        self.print_metric_value_in_end = True
        self.use_cutmix = True
        self.alpha = alpha
#%% train part starts here
    def fit(self):                
        with TQDM() as pbar:           
            pbar.on_train_begin({'num_batches':len(self.trainLoader),'num_epoch':self.epochsTorun})
            pbar.on_val_begin({'num_batches':len(self.valLoader),'num_epoch':self.epochsTorun})
            
            if self.use_early_stopper:
                self.earlyStopper.on_train_begin()
            
            self.train_recall_score = RecallScore()
            self.val_recall_score = RecallScore()
            
            for epoch in range(self.epochsTorun):
                current_epoch_no = epoch+1                               
                
                if current_epoch_no <= self.init_epoch:
                    continue                
                pbar.on_epoch_train_begin(self.fold,current_epoch_no)                
                self.info_bbx._init_new_epoch(current_epoch_no)                                
                
                self.model.train()
                torch.set_grad_enabled(True)
                self.optimizer.zero_grad()
                
                self.train_recall_score.reset()
                self.val_recall_score.reset() 
                                
                for i, data in enumerate(self.trainLoader):
                    pbar.on_train_batch_begin()                                    
                    
                    images, grapheme_root_labels,vowel_diacritic_labels,consonant_diacritic_labels = data

                    images = images.cuda()
                    grapheme_root_labels = grapheme_root_labels.cuda()
                    vowel_diacritic_labels = vowel_diacritic_labels.cuda()
                    consonant_diacritic_labels = consonant_diacritic_labels.cuda()
                    
                    if self.use_cutmix:
                        cutmix_images, cutmix_targets = cutmix(images,grapheme_root_labels,vowel_diacritic_labels,consonant_diacritic_labels,self.alpha)
                        grapheme_output,vowel_diacritic_output,consonant_diacritic_output = self.model(cutmix_images)
                        batch_loss = cutmix_criterion(grapheme_output,vowel_diacritic_output,consonant_diacritic_output,cutmix_targets,False)
                        batch_loss = batch_loss[0]+batch_loss[1]+batch_loss[2]
                    else:                                        
                        grapheme_output,vowel_diacritic_output,consonant_diacritic_output = self.model(images)                    
                        batch_loss = self.criterion(grapheme_output,grapheme_root_labels) + self.criterion(vowel_diacritic_output,vowel_diacritic_labels) +  self.criterion(consonant_diacritic_output,consonant_diacritic_labels)
                        
                    
                    if not self.do_accum:
                        self.optimizer.zero_grad()
                        with amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
#                        batch_loss.backward()
                        self.optimizer.step()
                    else:
                        if self.accum_loss_division:
                            batch_loss = batch_loss/self.accumSteps
                        with amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
#                        batch_loss.backward()
                        if (i+1)%self.accumSteps == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        elif (i+1) == len(self.trainLoader):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                                                                    
                    submission_grapheme_root_labels = torch.argmax(grapheme_output.clone(),dim=1)
                    submission_vowel_diacritic_labels = torch.argmax(vowel_diacritic_output.clone(),dim=1)
                    submission_consonant_diacritic_labels = torch.argmax(consonant_diacritic_output.clone(),dim=1)
                                        
                    self.train_recall_score.update([[grapheme_root_labels,vowel_diacritic_labels,consonant_diacritic_labels],[submission_grapheme_root_labels,submission_vowel_diacritic_labels,submission_consonant_diacritic_labels]])
                    
                    if self.print_metric_value_in_end:
                        grapheme_root_recall,vowel_diacritic_recall,consonant_diarcitic_recall,final_score = 0,0,0,0
                    else:
                        grapheme_root_recall,vowel_diacritic_recall,consonant_diarcitic_recall,final_score = self.train_recall_score.feedback()
                                                           
                    self.info_bbx.update_train_info({'Loss':[batch_loss.detach().item(),images.size(0)],'grapheme_root_recall':grapheme_root_recall,
                                                     'vowel_diacritic_recall':vowel_diacritic_recall,'consonant_diacritic_recall':consonant_diarcitic_recall,
                                                     'Average_recall':final_score})
    
                    pbar.on_train_batch_end(logs=self.info_bbx.request_current_epoch_train_metric_info())
                                       
                    torch.cuda.empty_cache()                    
                    if self.test_run_for_error:                                     
                        if i==30:
                            break
#%% validation part starts here
                grapheme_root_recall,vowel_diacritic_recall,consonant_diarcitic_recall,final_score = self.train_recall_score.feedback()                                                           
                self.info_bbx.update_train_info({'grapheme_root_recall':grapheme_root_recall,
                                                 'vowel_diacritic_recall':vowel_diacritic_recall,'consonant_diacritic_recall':consonant_diarcitic_recall,
                                                 'Average_recall':final_score})
                
                pbar.on_epoch_train_end(self.info_bbx.request_current_epoch_train_metric_info())
                pbar.on_epoch_val_begin(self.fold,current_epoch_no)
                self.model.eval()
                torch.set_grad_enabled(False)                                
                with torch.no_grad():
                    for i, data in enumerate(self.valLoader):
                        pbar.on_val_batch_begin()
                        
                        images, grapheme_root_labels,vowel_diacritic_labels,consonant_diacritic_labels = data
                        images = images.cuda()
                        grapheme_root_labels = grapheme_root_labels.cuda()
                        vowel_diacritic_labels = vowel_diacritic_labels.cuda()
                        consonant_diacritic_labels = consonant_diacritic_labels.cuda()
                        
                        grapheme_output,vowel_diacritic_output,consonant_diacritic_output = self.model(images)                                                                 
                        batch_loss = self.criterion(grapheme_output,grapheme_root_labels) + self.criterion(vowel_diacritic_output,vowel_diacritic_labels) + self.criterion(consonant_diacritic_output,consonant_diacritic_labels)
                                                
                        submission_grapheme_root_labels = torch.argmax(grapheme_output.clone(),dim=1)
                        submission_vowel_diacritic_labels = torch.argmax(vowel_diacritic_output.clone(),dim=1)
                        submission_consonant_diacritic_labels = torch.argmax(consonant_diacritic_output.clone(),dim=1)
                        
                        self.val_recall_score.update([[grapheme_root_labels,vowel_diacritic_labels,consonant_diacritic_labels],[submission_grapheme_root_labels,submission_vowel_diacritic_labels,submission_consonant_diacritic_labels]])
                        grapheme_root_recall,vowel_diacritic_recall,consonant_diarcitic_recall,final_score = self.val_recall_score.feedback()
                                                           
                        self.info_bbx.update_val_info({'Loss':[batch_loss.detach().item(),images.size(0)],'grapheme_root_recall':grapheme_root_recall,
                                                     'vowel_diacritic_recall':vowel_diacritic_recall,'consonant_diacritic_recall':consonant_diarcitic_recall,
                                                     'Average_recall':final_score})
                                               
                        pbar.on_val_batch_end(logs=self.info_bbx.request_current_epoch_val_metric_info())
                        torch.cuda.empty_cache()
                        if self.test_run_for_error:
                            if i==30:
                                break                                                                     
                    
                    pbar.on_epoch_val_end(self.info_bbx.request_current_epoch_val_metric_info())
#%% Update best parameters
                if self.best_loss >= self.info_bbx.get_info(current_epoch_no,'Loss','Val'):
                    print( ' Val Loss is improved from {:.4f} to {:.4f}! '.format(self.best_loss,self.info_bbx.get_info(current_epoch_no,'Loss','Val')) )
                    self.best_loss = self.info_bbx.get_info(current_epoch_no,'Loss','Val')
                    is_best_loss = True
                else:
                    print( ' Val Loss is not improved from {:.4f}! '.format(self.best_loss))
                    is_best_loss = False
                
                if self.best_grapheme_root_recall < self.info_bbx.get_info(current_epoch_no,'grapheme_root_recall','Val'):
                    print( ' Val grapheme_root_recall is improved from {:.4f} to {:.4f}! '.format(self.best_grapheme_root_recall,self.info_bbx.get_info(current_epoch_no,'grapheme_root_recall','Val')) )
                    self.best_grapheme_root_recall = self.info_bbx.get_info(current_epoch_no,'grapheme_root_recall','Val')
                    is_best_grapheme_root_recall = True
                else:
                    print( ' Val grapheme_root_recall is not improved from {:.4f}! '.format(self.best_grapheme_root_recall) )
                    is_best_grapheme_root_recall = False
                    
                if self.best_vowel_diacritic_recall < self.info_bbx.get_info(current_epoch_no,'vowel_diacritic_recall','Val'):
                    print( ' Val vowel_diacritic_recall is improved from {:.4f} to {:.4f}! '.format(self.best_vowel_diacritic_recall,self.info_bbx.get_info(current_epoch_no,'vowel_diacritic_recall','Val')) )
                    self.best_vowel_diacritic_recall = self.info_bbx.get_info(current_epoch_no,'vowel_diacritic_recall','Val')
                    is_best_vowel_diacritic_recall = True
                else:
                    print( ' Val vowel_diacritic_recall is not improved from {:.4f}! '.format(self.best_vowel_diacritic_recall) )
                    is_best_vowel_diacritic_recall = False
                
                if self.best_consonant_diacritic_recall < self.info_bbx.get_info(current_epoch_no,'consonant_diacritic_recall','Val'):
                    print( ' Val consonant_diacritic_recall is improved from {:.4f} to {:.4f}! '.format(self.best_consonant_diacritic_recall,self.info_bbx.get_info(current_epoch_no,'consonant_diacritic_recall','Val')) )
                    self.best_consonant_diacritic_recall = self.info_bbx.get_info(current_epoch_no,'consonant_diacritic_recall','Val')
                    is_best_consonant_diacritic_recall = True
                else:
                    print( ' Val consonant_diacritic_recall is not improved from {:.4f}! '.format(self.best_consonant_diacritic_recall) )
                    is_best_consonant_diacritic_recall = False
                    
                if self.best_average_recall < self.info_bbx.get_info(current_epoch_no,'Average_recall','Val'):
                    print( ' Val Average_recall is improved from {:.4f} to {:.4f}! '.format(self.best_average_recall,self.info_bbx.get_info(current_epoch_no,'Average_recall','Val')) )
                    self.best_average_recall = self.info_bbx.get_info(current_epoch_no,'Average_recall','Val')
                    is_best_average_recall = True
                else:
                    print( ' Val Average_recall is not improved from {:.4f}! '.format(self.best_average_recall) )
                    is_best_average_recall = False
#%%Learning Rate Schedulers
#                self.scheduler.step()
                if not self.use_scheduler_flag:
                    self.scheduler.step(self.info_bbx.get_info(current_epoch_no,'Loss','Val'))
                else:
                    if is_best_loss or is_best_average_recall:
                        self.scheduler_flag = self.scheduler_flag - 1 
                        self.scheduler.step(self.scheduler_flag)
                    else:
                        self.scheduler.step(self.scheduler_flag+1)
#%%checkpoint dict creation                                    
                checkpoint_dict = {
                    'Problem_name':self.problem_name,
                    'Epoch': current_epoch_no,
                    'Lr':self.lr,
                    'Optimizer_state_dict' : self.optimizer.state_dict(),
                    'Model_state_dict': self.model.state_dict(),
                    'Scheduler_state_dict':self.scheduler.state_dict(),
                    'All_info': self.info_bbx.request_allinfo(),
                    'Current_val_Loss': self.info_bbx.get_info(current_epoch_no,'Loss','Val'),
                    'Current_average_recall':self.info_bbx.get_info(current_epoch_no,'Average_recall','Val'),
                    'Current_grapheme_root_recall':self.info_bbx.get_info(current_epoch_no,'grapheme_root_recall','Val'),
                    'Current_vowel_diacritic_recall':self.info_bbx.get_info(current_epoch_no,'vowel_diacritic_recall','Val'),
                    'Current_consonant_diacritic_recall':self.info_bbx.get_info(current_epoch_no,'consonant_diacritic_recall','Val'),
                    'Best_val_loss' : self.best_loss,
                    'Best_average_recall':self.best_average_recall,
                    'Best_grapheme_root_recall':self.best_grapheme_root_recall,
                    'Best_vowel_diacritic_recall':self.best_vowel_diacritic_recall,
                    'Best_consonant_diacritic_recall':self.best_consonant_diacritic_recall,
                    'Scheduler_flag':self.scheduler_flag,
                    }
#%%checkpoint dict saving
                current_checkpoint_filename = self.checkpoint_saving_path+'checkpoint_{}'.format(self.count)+'.pth'
                torch.save(checkpoint_dict, current_checkpoint_filename)
                self.count += 1
                if self.count == 3:
                    self.count = 1
                
                print(' Checkpoint Saved! ' )
                
                if is_best_loss:
                    torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_loss.pth')
                    print(' Best Loss Checkpoint Saved! ' )
                                
#                if is_best_grapheme_root_recall:
                    #torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_grapheme_root_recall.pth')
                                    
#                if is_best_vowel_diacritic_recall:
                    #torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_vowel_diacritic_recall.pth')
                    
#                if is_best_consonant_diacritic_recall:
                    #torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_consonant_diacritic_recall.pth')
                                    
                if is_best_average_recall :
                    torch.save(checkpoint_dict, self.checkpoint_saving_path+'checkpoint_best_Average_recall.pth')
                    print(' Best Average Recall Checkpoint Saved! ' )
                
                del checkpoint_dict
                torch.cuda.empty_cache()
#%%Early Stopping
                if self.use_early_stopper:
                    stop = self.earlyStopper.on_epoch_end(self.info_bbx.get_info(current_epoch_no,self.observing_parameter,'Val'),epoch)
                    if stop: 
                        break
#%%Return model            
        return self.model