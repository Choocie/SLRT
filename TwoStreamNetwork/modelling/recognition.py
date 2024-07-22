import math
from copy import deepcopy
import random, torchvision
import numpy as np
import tensorflow as tf
import torch
from itertools import groupby

from modelling.S3D import S3D_backbone
from modelling.two_stream import S3D_two_stream_v2
from utils.misc import get_logger, neq_load_customized
from modelling.Tokenizer import GlossTokenizer_S2G
from modelling.Visualhead import VisualHead
from utils.gen_gaussian import gen_gaussian_hmap_op

def ctc_decode_func(tf_gloss_logits, input_lengths, beam_size):
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_logits, 
        sequence_length=input_lengths.cpu().detach().numpy(),
        beam_width=beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]
    tmp_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1
        )
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences


class RecognitionNetwork(torch.nn.Module):
    def __init__(self, cfg, input_type, transform_cfg, 
        input_streams=['rgb']) -> None:
        super().__init__()
        logger = get_logger()
        self.cfg = cfg
        self.input_type = input_type
        assert(input_type=='feature')
        self.gloss_tokenizer = GlossTokenizer_S2G(
            cfg['GlossTokenizer'])
        self.input_streams = input_streams
        assert(self.input_streams==['rgb'])
        self.fuse_method = cfg.get('fuse_method', 'empty')
        self.heatmap_cfg = cfg.get('heatmap_cfg',{})
        self.transform_cfg = transform_cfg
        self.preprocess_chunksize = cfg.get('preprocess_chunksize', 16)
        cfg['pyramid'] = cfg.get('pyramid',{'version':None, 'rgb':None, 'pose':None})
        if self.input_type=='video':
            if 'rgb' in input_streams and not 'keypoint' in input_streams:
                if 's3d' in cfg:
                    self.visual_backbone = S3D_backbone(in_channel=3, **cfg['s3d'], cfg_pyramid=cfg['pyramid'])
                else:
                    raise ValueError
                self.visual_backbone_keypoint, self.visual_backbone_twostream = None, None
            elif 'keypoint' in input_streams and not 'rgb' in input_streams:
                if 'keypoint_s3d' in cfg:
                    self.visual_backbone_keypoint = S3D_backbone(\
                        **cfg['keypoint_s3d'], cfg_pyramid=cfg['pyramid'])
                self.visual_backbone, self.visual_backbone_twostream = None, None
            elif 'rgb' in input_streams and 'keypoint' in input_streams: 
                self.visual_backbone_twostream = S3D_two_stream_v2(
                    use_block=cfg['s3d']['use_block'],
                    freeze_block=(cfg['s3d']['freeze_block'], cfg['keypoint_s3d']['freeze_block']),
                    pose_inchannels=cfg['keypoint_s3d']['in_channel'],
                    flag_lateral=(cfg['lateral'].get('pose2rgb',False),
                        cfg['lateral'].get('rgb2pose',False)),
                    lateral_variant=(cfg['lateral'].get('variant_pose2rgb', None),
                        cfg['lateral'].get('variant_rgb2pose', None)),
                    lateral_ksize=tuple(cfg['lateral'].get('kernel_size', (7,3,3))),
                    cfg_pyramid=cfg['pyramid'],
                    fusion_features=cfg['lateral'].get('fusion_features',['c1','c2','c3'])
                )
                self.visual_backbone, self.visual_backbone_keypoint = None, None
            else:
                raise ValueError

        if 'visual_head' in cfg:
            if 'rgb' in input_streams:
                if cfg['pyramid']['rgb'] in ['fused_head', 'fused_multi_head']:
                    cfg['visual_head']['input_size'] = 1568  #832+480+192+64
                else:
                    cfg['visual_head']['input_size'] = 832
                
                if cfg['pyramid']['rgb'] == 'shared_head':
                    cfg['visual_head']['input_size'] = None
                    self.visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['visual_head'])
                    self.fc_layers_rgb = torch.nn.ModuleList()
                    channels = [192,480,832]
                    for i in range(len(channels)):
                        self.fc_layers_rgb.append(torch.nn.Linear(channels[i], 512))
                else:
                    self.visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['visual_head'])
                
                if cfg['pyramid']['rgb'] is not None and 'multi' in cfg['pyramid']['rgb']:
                    num_remain_heads = cfg['s3d']['use_block'] if cfg['pyramid']['rgb'] == 'fused_multi_head' else cfg['s3d']['use_block']-1
                    dims = [64, 192, 480, 832]
                    if cfg['pyramid']['version'] == 'v2':
                        num_levels = cfg['pyramid'].get('num_levels', 3)
                    else:
                        num_levels = cfg['pyramid'].get('num_levels', 4)
                    num_remain_heads = num_levels - 1
                    dims = dims[-num_levels:]
                    self.visual_head_remain = torch.nn.ModuleList()
                    for i in range(num_remain_heads):
                        cfg['visual_head']['input_size'] = dims[i]
                        self.visual_head_remain.append(VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['visual_head']))
            else:
                self.visual_head = None
            
            if 'keypoint' in input_streams:
                if cfg['pyramid']['pose'] in ['fused_head', 'fused_multi_head']:
                    cfg['visual_head']['input_size'] = 1568 
                else:
                    cfg['visual_head']['input_size'] = 832

                if cfg['pyramid']['pose'] == 'shared_head':
                    cfg['visual_head']['input_size'] = None
                    self.visual_head_keypoint = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['visual_head'])
                    self.fc_layers_keypoint = torch.nn.ModuleList()
                    channels = [192,480,832]
                    for i in range(len(channels)):
                        self.fc_layers_keypoint.append(torch.nn.Linear(channels[i], 512))
                else:
                    self.visual_head_keypoint = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['visual_head'])
                
                if cfg['pyramid']['pose'] is not None and 'multi' in cfg['pyramid']['pose']:
                    num_remain_heads = cfg['s3d']['use_block'] if cfg['pyramid']['pose'] == 'fused_multi_head' else cfg['s3d']['use_block']-1
                    dims = [64, 192, 480, 832]
                    if cfg['pyramid']['version'] == 'v2':
                        num_levels = cfg['pyramid'].get('num_levels', 3)
                    else:
                        num_levels = cfg['pyramid'].get('num_levels', 4)

                    num_remain_heads = num_levels - 1
                    dims = dims[-num_levels:]
                    self.visual_head_keypoint_remain = torch.nn.ModuleList()
                    for i in range(num_remain_heads):
                        cfg['visual_head']['input_size'] = dims[i]
                        self.visual_head_keypoint_remain.append(VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['visual_head']))
            else:
                self.visual_head_keypoint = None
            if 'triplehead' in self.fuse_method:
                assert 'rgb' in input_streams and 'keypoint' in input_streams
                new_cfg = deepcopy(cfg['visual_head'])
                if 'cat' in self.fuse_method:
                    new_cfg['input_size'] = 2*832                 
                self.visual_head_fuse = VisualHead(
                    cls_num=len(self.gloss_tokenizer), **new_cfg)

        if 'pretrained_path_rgb' in cfg:
            load_dict = torch.load(cfg['pretrained_path_rgb'],map_location='cpu')['model_state']      
            backbone_dict, head_dict, fc_dict, head_remain_dict = {}, {}, {}, {}
            for k, v in load_dict.items():
                if 'visual_backbone' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone.','')] = v
                if 'visual_head' in k and 'visual_head_remain' not in k:
                    head_dict[k.replace('recognition_network.visual_head.','')] = v
                if 'fc_layers_rgb' in k:
                    fc_dict[k.replace('recognition_network.fc_layers_rgb.','')] = v
                if 'visual_head_remain' in k:
                    head_remain_dict[k.replace('recognition_network.visual_head_remain.','')] = v
            if self.visual_backbone!=None and self.visual_backbone_twostream==None:
                neq_load_customized(self.visual_backbone, backbone_dict, verbose=False)
                neq_load_customized(self.visual_head, head_dict, verbose=False)
                logger.info('Load visual_backbone and visual_head for rgb from {}'.format(cfg['pretrained_path_rgb']))
            elif self.visual_backbone==None and self.visual_backbone_twostream!=None:
                neq_load_customized(self.visual_backbone_twostream.rgb_stream, backbone_dict, verbose=False)
                neq_load_customized(self.visual_head, head_dict, verbose=False)
                if cfg['pyramid']['rgb'] == 'shared_head':
                    neq_load_customized(self.fc_layers_rgb, fc_dict, verbose=False)
                elif cfg['pyramid']['rgb'] == 'multi_head':
                    neq_load_customized(self.visual_head_remain, head_remain_dict, verbose=False)
                logger.info('Load visual_backbone_twostream.rgb_stream and visual_head for rgb from {}'.format(cfg['pretrained_path_rgb'])) 
            else:
                logger.info('No rgb stream exists in the network')

        if 'pretrained_path_keypoint' in cfg:
            load_dict = torch.load(cfg['pretrained_path_keypoint'],map_location='cpu')['model_state']
            backbone_dict, head_dict, fc_dict, head_remain_dict = {}, {}, {}, {}
            for k, v in load_dict.items():
                if 'visual_backbone_keypoint' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone_keypoint.','')] = v
                if 'visual_head_keypoint' in k and 'visual_head_keypoint_remain' not in k: #for model trained using new_code
                    head_dict[k.replace('recognition_network.visual_head_keypoint.','')] = v
                elif 'visual_head' in k and 'visual_head_keypoint_remain' not in k: #for model trained using old_code
                    head_dict[k.replace('recognition_network.visual_head.','')] = v
                elif 'visual_head_keypoint_remain' in k:
                    head_remain_dict[k.replace('recognition_network.visual_head_keypoint_remain.','')] = v
                if 'fc_layers_keypoint' in k:
                    fc_dict[k.replace('recognition_network.fc_layers_keypoint.','')] = v
            if self.visual_backbone_keypoint!=None and self.visual_backbone_twostream==None:
                neq_load_customized(self.visual_backbone_keypoint, backbone_dict, verbose=False)
                neq_load_customized(self.visual_head_keypoint, head_dict, verbose=False)
                logger.info('Load visual_backbone and visual_head for keypoint from {}'.format(cfg['pretrained_path_keypoint']), verbose=False)
            elif self.visual_backbone_keypoint==None and self.visual_backbone_twostream!=None:
                neq_load_customized(self.visual_backbone_twostream.pose_stream, backbone_dict, verbose=False)
                neq_load_customized(self.visual_head_keypoint, head_dict, verbose=False)
                if cfg['pyramid']['pose'] == 'shared_head':
                    neq_load_customized(self.fc_layers_keypoint, fc_dict, verbose=False)
                elif cfg['pyramid']['pose'] == 'multi_head':
                    neq_load_customized(self.visual_head_keypoint_remain, head_remain_dict, verbose=False)
                logger.info('Load visual_backbone_twostream.pose_stream and visual_head for pose from {}'.format(cfg['pretrained_path_keypoint']))    
            else:
                logger.info('No pose stream exists in the network')

        self.recognition_loss_func = torch.nn.CTCLoss(
            blank=self.gloss_tokenizer.silence_id, zero_infinity=True,
            reduction='sum'
        )

    def compute_recognition_loss(self, gloss_labels, gloss_lengths, gloss_probabilities_log, input_lengths):
        loss = self.recognition_loss_func(
            log_probs = gloss_probabilities_log.permute(1,0,2), #T,N,C
            targets = gloss_labels,
            input_lengths = input_lengths,
            target_lengths = gloss_lengths
        )
        loss = loss/gloss_probabilities_log.shape[0]
        return loss

    def decode(self, gloss_logits, beam_size, input_lengths):
        gloss_logits = gloss_logits.permute(1, 0, 2) #T,B,V
        gloss_logits = gloss_logits.cpu().detach().numpy()
        tf_gloss_logits = np.concatenate(
            (gloss_logits[:, :, 1:], gloss_logits[:, :, 0, None]),
            axis=-1,
        )
        decoded_gloss_sequences = ctc_decode_func(
            tf_gloss_logits=tf_gloss_logits,
            input_lengths=input_lengths,
            beam_size=beam_size
        )
        return decoded_gloss_sequences

    def generate_batch_heatmap(self, keypoints):
        B,T,N,D = keypoints.shape
        keypoints = keypoints.reshape(-1, N, D)
        n_chunk = int(math.ceil((B*T)/self.preprocess_chunksize))
        chunks = torch.split(keypoints, n_chunk, dim=0)
        heatmaps = []
        for chunk in chunks:
            hm = gen_gaussian_hmap_op(
                coords=chunk,  
                **self.heatmap_cfg) 
            _, N, H, W = hm.shape
            heatmaps.append(hm)
        heatmaps = torch.cat(heatmaps, dim=0) 
        return heatmaps.reshape(B,T,N,H,W) 

    def apply_spatial_ops(self, x, spatial_ops_func):
        B, T, C_, H, W = x.shape
        x = x.view(-1, C_, H, W)
        chunks = torch.split(x, self.preprocess_chunksize, dim=0)
        transformed_x = []
        for chunk in chunks:
            transformed_x.append(spatial_ops_func(chunk))
        _, C_, H_o, W_o = transformed_x[-1].shape
        transformed_x = torch.cat(transformed_x, dim=0)
        transformed_x = transformed_x.view(B, T, C_, H_o, W_o)
        return transformed_x    

    def augment_preprocess_inputs(self, is_train, sgn_videos=None, sgn_heatmaps=None):
        rgb_h, rgb_w = self.transform_cfg.get('img_size',224), self.transform_cfg.get('img_size',224)
        if sgn_heatmaps!=None:
            hm_h, hm_w = self.heatmap_cfg['input_size'], self.heatmap_cfg['input_size']
            if sgn_videos!=None:
                rgb_h0, rgb_w0 = sgn_videos.shape[-2],sgn_videos.shape[-1] 
                hm_h0, hm_w0 = sgn_heatmaps.shape[-2],sgn_heatmaps.shape[-1]  
                factor_h, factor_w= hm_h0/rgb_h0, hm_w0/rgb_w0 
        if is_train:
            if sgn_videos!=None:
                if  self.transform_cfg.get('color_jitter',False) and random.random()<0.3:
                    color_jitter_op = torchvision.transforms.ColorJitter(0.4,0.4,0.4,0.1)
                    sgn_videos = color_jitter_op(sgn_videos)
                i,j,h,w = torchvision.transforms.RandomResizedCrop.get_params(
                    img=sgn_videos,
                    scale=(self.transform_cfg.get('bottom_area',0.2), 1.0), 
                    ratio=(self.transform_cfg.get('aspect_ratio_min',3./4), 
                        self.transform_cfg.get('aspect_ratio_max',4./3)))
                sgn_videos = self.apply_spatial_ops(
                    sgn_videos, 
                    spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                        x, i, j, h, w, [rgb_h, rgb_w]))
            if sgn_heatmaps!=None:
                if sgn_videos!=None:
                    i2, j2, h2, w2 = int(i*factor_h), int(j*factor_w), int(h*factor_h), int(w*factor_w)
                else:
                    i2, j2, h2, w2 = torchvision.transforms.RandomResizedCrop.get_params(
                        img=sgn_heatmaps,
                        scale=(self.transform_cfg.get('bottom_area',0.2), 1.0), 
                        ratio=(self.transform_cfg.get('aspect_ratio_min',3./4), 
                            self.transform_cfg.get('aspect_ratio_max',4./3)))
                sgn_heatmaps = self.apply_spatial_ops(
                        sgn_heatmaps,
                        spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                         x, i2, j2, h2, w2, [hm_h, hm_w]))
        else:
            if sgn_videos!=None:
                spatial_ops = []
                if self.transform_cfg.get('center_crop',False)==True:
                    spatial_ops.append(torchvision.transforms.CenterCrop(
                        self.transform_cfg['center_crop_size']))
                spatial_ops.append(torchvision.transforms.Resize([rgb_h, rgb_w]))
                spatial_ops = torchvision.transforms.Compose(spatial_ops)
                sgn_videos = self.apply_spatial_ops(sgn_videos, spatial_ops)
            if sgn_heatmaps!=None:
                spatial_ops = []
                if self.transform_cfg.get('center_crop',False)==True:
                    spatial_ops.append(
                        torchvision.transforms.CenterCrop(
                            [int(self.transform_cfg['center_crop_size']*factor_h),
                            int(self.transform_cfg['center_crop_size']*factor_w)]))
                spatial_ops.append(torchvision.transforms.Resize([hm_h, hm_w]))
                spatial_ops = torchvision.transforms.Compose(spatial_ops)
                sgn_heatmaps = self.apply_spatial_ops(sgn_heatmaps, spatial_ops)                

        if sgn_videos!=None:
            sgn_videos = sgn_videos[:,:,[2,1,0],:,:] 
            sgn_videos = (sgn_videos-0.5)/0.5
            sgn_videos = sgn_videos.permute(0,2,1,3,4).float() 
        if sgn_heatmaps!=None:
            sgn_heatmaps = (sgn_heatmaps-0.5)/0.5
            sgn_heatmaps = sgn_heatmaps.permute(0,2,1,3,4).float()
        return sgn_videos, sgn_heatmaps

    def forward(self,
                is_train,
                gloss_labels,
                gls_lengths,
                sgn_features=None,
                sgn_mask=None,
                sgn_videos=None,
                sgn_lengths=None,
                sgn_keypoints=None,
                head_rgb_input=None, 
                head_keypoint_input=None):

        aux_prob_log = {'rgb':[],'keypoint':[]}
        head_outputs = self.visual_head(x=head_rgb_input, mask=sgn_mask)
        valid_len_out = sgn_lengths 


        outputs = {**head_outputs,
            'input_lengths': valid_len_out}  
        assert(self.fuse_method=='empty')
        if gloss_labels is not None:
            outputs['recognition_loss'] = self.compute_recognition_loss(
                gloss_labels=gloss_labels, gloss_lengths=gls_lengths,
                gloss_probabilities_log=head_outputs['gloss_probabilities_log'],
                input_lengths=valid_len_out
            )
        self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble','gloss_feature')
        outputs['gloss_feature'] = outputs[self.cfg['gloss_feature_ensemble']]
        for i in range(len(aux_prob_log[self.input_streams[0]])):
            if gloss_labels is not None:
                outputs['recognition_loss'] += self.cfg['pyramid']['head_weight'] * self.compute_recognition_loss(
                        gloss_labels=gloss_labels, gloss_lengths=gls_lengths,
                        gloss_probabilities_log=aux_prob_log[self.input_streams[0]][i],
                        input_lengths=head_outputs['valid_len_out_lst'][i])

        stream, feat_name = self.cfg['gloss_feature_ensemble'].split('_gloss_')
        feat_name = 'gloss_'+feat_name
        outputs['gloss_feature'] = outputs[f'{stream}_{feat_name}']
        if gloss_labels is not None:
            outputs['recognition_loss'] = outputs['recognition_loss_rgb'] + outputs['recognition_loss_keypoint'] + outputs['recognition_loss_fuse']
    
        return outputs
    

    