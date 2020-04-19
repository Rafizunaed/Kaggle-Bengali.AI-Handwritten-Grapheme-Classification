import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class CNN_Model(nn.Module):
    def __init__(self,model_name='efficientnet-b5',global_dim=2048,drp=0.15):
        
        super(CNN_Model,self).__init__()
            
        self.cnn_backbone = EfficientNet.from_pretrained(model_name)
        self.cnn_backbone.set_swish(memory_efficient=False)                                
        self.cnn_head_grapheme_root = CNN_Head(168,global_dim,1024,512,True,False,drp,drp,drp)
        self.cnn_head_vowel_diacritic = CNN_Head(11,global_dim,1024,512,True,False,drp,drp,drp)
        self.cnn_head_consonant_diacritic = CNN_Head(7,global_dim,1024,512,True,False,drp,drp,drp)                                                
                        
    def forward(self, x):
        global_feat = self.cnn_backbone.extract_features(x)                              
        global_feat = (F.adaptive_avg_pool2d(global_feat,1)+F.adaptive_max_pool2d(global_feat,1))*0.5
        global_feat = global_feat.view(global_feat.size(0), -1)
            
        grapheme_output = self.cnn_head_grapheme_root(global_feat)
        vowel_diacritic_output = self.cnn_head_vowel_diacritic(global_feat)
        consonant_diacritic_output = self.cnn_head_consonant_diacritic(global_feat)
        
        return grapheme_output,vowel_diacritic_output,consonant_diacritic_output

class CNN_Head(nn.Sequential):
    def __init__(self, classes, input_dim, linear1_output_dim, linear2_output_dim,
                 use_batch_norm=True, use_activation_function=False,
                 global_dropout=0.15, lin1_dropout=0.15, lin2_dropout=0.15): #0.10 0.10 0.10 #0.20,0.15,0.10 #0.15,0.10,0.05
        
        super(CNN_Head, self).__init__()
        
        if global_dropout > 0.0:
            self.add_module('drop_1',nn.Dropout(p=global_dropout,inplace=False))
                
        self.add_module('linear_1',nn.Linear(input_dim,linear1_output_dim))
        
        if use_batch_norm:
            self.add_module('norm_1', nn.BatchNorm1d(linear1_output_dim))
        
        if use_activation_function:
            self.add_module('activ_1',nn.ReLU(inplace=False))
            
        if lin1_dropout > 0.0:
            self.add_module('drop_2',nn.Dropout(p=lin1_dropout,inplace=False))
            
        self.add_module('linear_2',nn.Linear(linear1_output_dim,linear2_output_dim))
        
        if use_batch_norm:
            self.add_module('norm_2', nn.BatchNorm1d(linear2_output_dim))
        
        if use_activation_function:
            self.add_module('activ_2',nn.ReLU(inplace=False))
        
        if lin2_dropout > 0.0:
            self.add_module('drop_3',nn.Dropout(p=lin2_dropout,inplace=False))
            
        self.add_module('linear_final',nn.Linear(linear2_output_dim,classes))