import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

# from inference.models.grasp_model import LanguageGraspModel, ResidualBlock

from .grasp_model import LanguageGraspModel, ResidualBlock

import torch
from torch import nn

import torch
from torch import nn
from .UnseenObjectClustering.lib import networks

from .crossvit import *
import time
# class CrossAttention(nn.Module):
#     def __init__(self, d_model=392, nhead=8):
#         super().__init__()
#         self.d_model = d_model
#         self.nhead = nhead

#         # Linear layers for each attention head (adjusted for input shapes)
#         self.linear = nn.Linear(3136, 392)
#         self.vision_query = nn.Linear(d_model, d_model)
#         self.vision_key = nn.Linear(d_model, d_model)
#         self.vision_value = nn.Linear(d_model, d_model)
#         self.language_query = nn.Linear(d_model, d_model)
#         self.language_key = nn.Linear(d_model, d_model)
#         self.language_value = nn.Linear(d_model, d_model)

#         # Output projection (adjusted for multi-head)
#         self.output_proj = nn.Linear(d_model * nhead, d_model)

#     def forward(self, vision_features, language_features):
#         vision_features = vision_features.view(vision_features.size(0), vision_features.size(1), -1)
        

#         print(vision_features.shape)
#         language_features = language_features.reshape(language_features.size(0), language_features.size(1), -1)
#         b, n, d_model = vision_features.shape  # Batch size, num_patches, d_model

#         # Split into multiple heads
#         vision_features = vision_features.view(b, self.nhead, n, d_model // self.nhead)
#         language_features = language_features.reshape(b, self.nhead, n, d_model // self.nhead)

#         # Project each head separately
#         vision_query, vision_key, vision_value = self.project_heads(vision_features)
#         language_query, language_key, language_value = self.project_heads(language_features)

#         # Calculate cross-attention (vision attends to language)
#         attn_scores = torch.matmul(vision_query, language_key.transpose(-1, -2)) / (d_model // self.nhead)**0.5
#         attn_weights = nn.functional.softmax(attn_scores, dim=-1)
#         attended_vision = torch.matmul(attn_weights, language_value)

#         # Calculate cross-attention (language attends to vision)
#         attn_scores = torch.matmul(language_query, vision_key.transpose(-1, -2)) / (d_model // self.nhead)**0.5
#         attn_weights = nn.functional.softmax(attn_scores, dim=-1)
#         attended_language = torch.matmul(attn_weights, vision_value)

#         # Combine attended features from all heads
#         fused_features = torch.cat([attended_vision, attended_language], dim=2)  # Concatenate along head dimension
#         fused_features = fused_features.view(b, n, d_model * self.nhead)  # Reshape for output projection
#         fused_features = self.output_proj(fused_features)

#         return fused_features #.view(b, n, d_model)  # Reshape back to original shape

#     def project_heads(self, features):
#         vision_query = self.vision_query(features)
#         vision_key = self.vision_key(features)
#         vision_value = self.vision_value(features)
#         return vision_query, vision_key, vision_value

class GenerativeResnet(LanguageGraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, clip_version='ViT-B/32'):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv1_fuse = nn.Conv2d(67, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1_fuse = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        #self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,output_padding=1)
        #self.bn4 = nn.BatchNorm2d(channel_size * 2)

        #self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,#output_padding=1)
        #self.bn5 = nn.BatchNorm2d(channel_size)
        self.conv6 = nn.ConvTranspose2d(1, 1, kernel_size=9, stride=1, padding=4)
        self.conv7 = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=2, padding=(1, 1))
        self.bn7 = nn.BatchNorm2d(1)
        self.conv8 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1,output_padding=1)
        self.bn8 = nn.BatchNorm2d(1)
        #self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
        self.y_flatten = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 56),
            nn.GELU()
        )

        self.pos_output = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # Setup language modality
        self.device = torch.device("cuda")
        self.clip_version = clip_version
        self.lang_model = self._load_and_freeze_clip(self.clip_version,self.device)
        self.condition_layer_langue = condition_layer_langue(56, 128, 1000)
        self.condition_layer_amodel_mask = CrossViT(56, 128, 1000)
        pretrained_amodal ='./inference/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_early_sampling_epoch_16.checkpoint.pth'
        network_name = 'seg_resnet34_8s_embedding'
        num_classes = 2
        train_num_units = 64
        self.amodel_seg = self.load_amodel_segment(pretrained_amodal,network_name,num_classes,train_num_units,self.device)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in, prompt, query):
        st_img = time.time()
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        end_img = time.time()

        

        # Encode text
        text_st = time.time()
        device = x.device
        y_feats = self._encode_text(query, device=device)
        y_feats = self.y_flatten(y_feats)
       
        y_feats = y_feats.unsqueeze(2).expand(-1, -1, 56).unsqueeze(1).expand(-1, 128, -1, -1)
        
        text_end = time.time()
        
        # Combine textual features with the visual features
        mask_st = time.time()
        amodal_prob_masks = self.amodel_seg(x_in,None)
        mask_end = time.time()

        
        #processed_mask = []
        #for i,amodal_prob_mask in enumerate(amodal_prob_masks):
            #amodal_prob_mask = amodal_prob_mask.squeeze(0)
            # print(amodal_prob_mask.shape)
            #gray_scale = torch.sum(amodal_prob_mask,0)
            #gray_scale = gray_scale / amodal_prob_mask.shape[0]
            # print(gray_scale.shape)
            #expanded_map = gray_scale.unsqueeze(0).expand_as(x_in[i])
            #processed_mask.append(expanded_map)
        
        #processed_mask = torch.stack(processed_mask, dim=0).to(device)
        # print(processed_mask.shape)

        # feature_map = feature_map.squeeze(0)
        # gray_scale = torch.sum(feature_map,0)
        # gray_scale = gray_scale / feature_map.shape[0]
        # processed.append(gray_scale.data.cpu().numpy())
        # print(gray_scale.shape)

        #prob_map_mask_fused_image = x_in*processed_mask
        trans_st = time.time()
        prob_map_mask_fused_image = torch.cat([x_in,amodal_prob_masks],dim=1)
        x_mask = F.relu(self.bn1_fuse(self.conv1_fuse(prob_map_mask_fused_image)))
        x_mask = F.relu(self.bn2(self.conv2(x_mask)))
        x_mask = F.relu(self.bn3(self.conv3(x_mask)))
        x_mask = self.res1(x_mask)
        x_mask = self.res2(x_mask)
        x_mask = self.res3(x_mask)
        x_mask = self.res4(x_mask)
        x_mask = self.res5(x_mask)
        # print('x: ',x.shape)
        # print('x_mask',x.shape)
        # x = x+x_mask
        prob_map_mask_fused_condition = self.condition_layer_amodel_mask(x,x_mask)
        # print('prob_map_mask_fused_condition:',prob_map_mask_fused_condition.shape)
        # print('y_text:',y_feats.shape)
        prob_map_mask_fused_condition=prob_map_mask_fused_condition
        out_fusion = self.condition_layer_langue(y_feats,prob_map_mask_fused_condition)
        #out_fusion = out_fusion*prob_map_mask_fused_condition
        out_fusion = torch.unsqueeze(out_fusion, 1)

        # print('out_fusion:',out_fusion.shape)

       
        # x = torch.clone(x).detach() + y_feats
        
        out_fusion = F.relu(self.bn7(self.conv7(out_fusion)))
        out_fusion = F.relu(self.bn8(self.conv8(out_fusion)))
        out_fusion = self.conv6(out_fusion)

        # print('out:',x.shape)
        
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(out_fusion))
            cos_output = self.cos_output(self.dropout_cos(out_fusion))
            sin_output = self.sin_output(self.dropout_sin(out_fusion))
            width_output = self.width_output(self.dropout_wid(out_fusion))
        else:
            pos_output = self.pos_output(out_fusion)
            cos_output = self.cos_output(out_fusion)
            sin_output = self.sin_output(out_fusion)
            width_output = self.width_output(out_fusion)
        trans_end = time.time()
        # print('pos output',cos_output.shape)
        # print('image_end:',end_img-st_img)
        # print('mask_end:',mask_end-mask_st)
        # print('text_end:',text_end-text_st)
        # print('trans_end:',trans_end-trans_st)

        return pos_output, cos_output, sin_output, width_output #,x,x_mask

    def _load_and_freeze_clip(self, clip_version, device=None):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    def load_amodel_segment(self,pretrained_amodal,network_name,num_classes,train_num_units,device):
        if pretrained_amodal:
            network_data = torch.load(pretrained_amodal)
            if isinstance(network_data, dict) and 'model' in network_data:
                network_data = network_data['model']
            print("=> using pre-trained network '{}'".format(network_name))
        else:
            network_data = None
            print("=> creating network '{}'".format(network_name))

        network = networks.__dict__[network_name](num_classes, train_num_units, network_data).cuda()
        network = network.to(device)
        network.eval()
        for p in network.parameters():
            p.requires_grad = False

        return network
    def _encode_text(self, raw_text, device=None):
        # raw_text - list (batch_size length) of strings with input text prompts
        max_text_len = 20 # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.lang_model.encode_text(texts).float()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':

    device = torch.device("cuda")
    net = GenerativeResnet(input_channels=3,dropout=1,prob=0.1,channel_size=32)
    net.to(device)
    img = torch.rand(8,3, 224,224)
    img = img.to(device)
    querys = []
    query = ['bowl'] *8
    # for i in range(32):
    #     querys.append(query)
    # print(query)
    # querys = torch.tensor(query)
    start = time.time()
    print(start)
    out = net(img,query,query)
    end = time.time()
    print(end)

    print('inference time: ',end-start)
    params = count_parameters(net)

    print(params)
    # print(out)

