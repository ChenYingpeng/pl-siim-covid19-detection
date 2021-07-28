import sys
#### add your timm path
sys.path.append('/home/chen/ai-competition/pytorch-image-models-master-210703')
import timm

import torch.nn.functional as F
# from timm.models.efficient import *

import torch
import torch.nn as nn

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)[:,:,0,0]
    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={str(self.eps)})"
        )


## only classification
class SIIMNet(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super(SIIMNet,self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)
#         print(self.model)
        ### effnet
        if model_name == 'tf_efficientnet_b4_ns' or model_name == 'tf_efficientnet_b5_ns':
            num_features = self.model.classifier.in_features  
            self.model.classifier = nn.Linear(num_features, num_classes)
            
        if model_name == 'efficientnetv2_rw_m' or model_name == 'tf_efficientnetv2_m_in21k':   
            num_features = self.model.classifier.in_features  
            self.model.classifier = nn.Linear(num_features, num_classes)
            
        if model_name == 'resnext50d_32x4d' or model_name == 'seresnet152d_320' or model_name == 'resnet200d_320' or model_name == 'resnest269e' or model_name == 'ecaresnet269d':
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
            
        ### vit    
        if model_name == 'vit_base_patch16_384':
            num_features = self.model.head.in_features  
            self.model.head = nn.Linear(num_features, num_classes) 
            
        if model_name == 'swin_base_patch4_window12_384':
            num_features = self.model.head.in_features  
            self.model.head = nn.Linear(num_features, num_classes) 
            
        
    def forward(self, x):
        if self.model_name == 'tf_efficientnet_b4_ns' or self.model_name == 'tf_efficientnet_b5_ns' or self.model_name == 'tf_efficientnet_b7_ns':
            x = self.model.act1(self.model.bn1(self.model.conv_stem(x)))
            x = self.model.blocks(x)
            conv5 = self.model.act2(self.model.bn2(self.model.conv_head(x)))

            feat = self.model.global_pool(conv5).view(conv5.size(0), -1)
            logits = self.model.classifier(feat)
        
        if self.model_name == 'resnet200d':
            x = self.model.act1(self.model.bn1(self.model.conv1(x)))
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            conv5 = self.model.layer4(x)

            feat = self.model.global_pool(conv5).view(conv5.size(0), -1)
            logits = self.model.fc(feat)
            
            
        if self.model_name == 'vit_base_patch16_384' or self.model_name == 'vit_large_patch16_384':
            B = x.shape[0]
            x = self.model.patch_embed(x)

            cls_tokens = self.model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.model.pos_embed
            x = self.model.pos_drop(x)

            for blk in self.model.blocks:
                x = blk(x)

            conv5 = x

            feat = self.model.norm(conv5)
            logits = self.model.head(feat[:,0])
            
        if self.model_name == 'swin_base_patch4_window12_384':
            x = self.model.patch_embed(x)
            if self.model.absolute_pos_embed is not None:
                x = x + self.model.absolute_pos_embed
            x = self.model.pos_drop(x)
            x = self.model.layers[0](x)
            x = self.model.layers[1](x)
#             print(f'layer 1 x size is {x.size()}')
            x = self.model.layers[2](x)
#             print(f'layer 2 x size is {x.size()}')
            x = self.model.layers[3](x)
#             print(f'layer 3 x size is {x.size()}')
            conv5 = x
            x = self.model.norm(x)  # B L C
            x = self.model.avgpool(x.transpose(1, 2))  # B C 1
            feat = torch.flatten(x, 1)
            logits = self.model.head(feat)
#             print(f'logits size is {x.size()}')
        
        return logits, conv5.detach()
    
    


class SIIMMaskNet(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super(SIIMMaskNet,self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        ### effnet
        if model_name == 'tf_efficientnet_b5_ns' or model_name == 'tf_efficientnet_b7_ns':
#             self.model.global_pool = GeM()
            num_features = self.model.classifier.in_features  
            self.model.classifier = nn.Linear(num_features, num_classes)
            
        if model_name == 'tf_efficientnetv2_l' or model_name == 'tf_efficientnetv2_m':   
            num_features = self.model.classifier.in_features  
            self.model.classifier = nn.Linear(num_features, num_classes)
            
        if model_name == 'swin_base_patch4_window12_384':
            num_features = self.model.head.in_features  
            self.model.head = nn.Linear(num_features, num_classes) 
            
        
        hidden_dims = 128
        
        ## for effnet 
        if model_name == 'tf_efficientnet_b5_ns' or model_name == 'tf_efficientnetv2_m':
            self.mask = nn.Sequential(
                nn.Conv2d(176, hidden_dims, kernel_size=3, padding=1), 
                nn.BatchNorm2d(hidden_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dims, 1, kernel_size=1, padding=0),
            )
                       

        if model_name == 'tf_efficientnet_b7_ns' or model_name == 'tf_efficientnetv2_l':
            self.mask = nn.Sequential(
                nn.Conv2d(224, hidden_dims, kernel_size=3, padding=1), 
                nn.BatchNorm2d(hidden_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dims, 1, kernel_size=1, padding=0),
            )    
        
        ### for transformer
        if model_name == 'swin_base_patch4_window12_384':
            self.mask = nn.Sequential(
                nn.Conv2d(512, hidden_dims, kernel_size=3, padding=1), ### swin
                nn.BatchNorm2d(hidden_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dims, 1, kernel_size=1, padding=0),
            )
        
        
    def forward(self, x):
        if self.model_name == 'tf_efficientnet_b5_ns' or self.model_name == 'tf_efficientnet_b7_ns' or self.model_name == 'tf_efficientnetv2_m' or self.model_name == 'tf_efficientnetv2_l':
            x = self.model.act1(self.model.bn1(self.model.conv_stem(x)))
            x = self.model.blocks[0](x)
            x = self.model.blocks[1](x)
            x = self.model.blocks[2](x)
            x = self.model.blocks[3](x)
            x = self.model.blocks[4](x)
 
            ####----
            mask = self.mask(x)
            ####----
            
            x = self.model.blocks[5](x)
            x = self.model.blocks[6](x)
            conv5 = self.model.act2(self.model.bn2(self.model.conv_head(x)))

            feat = self.model.global_pool(conv5).view(conv5.size(0), -1)
            logits = self.model.classifier(feat)
            
        if self.model_name == 'swin_base_patch4_window12_384':
            x = self.model.patch_embed(x)
            if self.model.absolute_pos_embed is not None:
                x = x + self.model.absolute_pos_embed
            x = self.model.pos_drop(x)
            x = self.model.layers[0](x)
            x = self.model.layers[1](x)

#             x = self.model.layers[2](x)
            for blk in self.model.layers[2].blocks:
                if not torch.jit.is_scripting() and self.model.layers[2].use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            
            ####----
            # B L C to B C L
            x_mask = x.transpose(1, 2)
            # B C L to B C H W
            x_mask = x_mask.reshape(x_mask.size(0),x_mask.size(1), 24, 24) # 24 = 384 / 16
            mask = self.mask(x_mask)
            ####----
            
            if self.model.layers[2].downsample is not None:
                x = self.model.layers[2].downsample(x)
            
            x = self.model.layers[3](x)
            conv5 = x
            x = self.model.norm(x)  # B L C
            x = self.model.avgpool(x.transpose(1, 2))  # B C 1
            feat = torch.flatten(x, 1)
            logits = self.model.head(feat)
        
        return logits, mask
        
    