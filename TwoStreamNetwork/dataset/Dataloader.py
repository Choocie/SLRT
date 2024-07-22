from dataset.VideoLoader import load_batch_video
from dataset.FeatureLoader import load_batch_feature
from dataset.Dataset import build_dataset
import torch


def collate_fn_(inputs, data_cfg, task, is_train, 
    text_tokenizer=None, gloss_tokenizer=None,name2keypoint=None):
    assert(task=='S2T')
    outputs = {
        'name':[i['name'] for i in inputs],
        'gloss':[i.get('gloss','') for i in inputs],
        'text':[i.get('text','') for i in inputs],
        'num_frames':[i['num_frames'] for i in inputs]}

    tokenized_text = text_tokenizer(input_str=outputs['text'])
    outputs['translation_inputs'] = {**tokenized_text}
    if task == 'S2T':
        outputs['recognition_inputs'] = gloss_tokenizer(outputs['gloss'])
        outputs['translation_inputs']['gloss_ids'] = outputs['recognition_inputs']['gloss_labels']  
        outputs['translation_inputs']['gloss_lengths'] = outputs['recognition_inputs']['gls_lengths'] 
        for feature_name in ['sgn_features', 'head_rgb_input','head_keypoint_input']:
            if feature_name in inputs[0]:
                outputs['recognition_inputs'][feature_name], sgn_mask, sgn_lengths = \
                    load_batch_feature(features=[i[feature_name]+1.0e-8 for i in inputs])
        outputs['recognition_inputs']['sgn_mask'] = sgn_mask
        outputs['recognition_inputs']['sgn_lengths'] = sgn_lengths
   
    return outputs

def build_dataloader(cfg, split, 
    text_tokenizer=None, gloss_tokenizer=None, 
    mode='auto', val_distributed=False):
    dataset = build_dataset(cfg['data'], split)
    mode = split if mode=='auto' else mode
    if mode=='train':
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            shuffle=cfg['training']['shuffle'] and split=='train'
        )
    else:
        if val_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn=lambda x:collate_fn_(
                                                 inputs=x,
                                                 task=cfg['task'],
                                                 data_cfg=cfg['data'],
                                                 is_train=(mode=='train'),
                                                 text_tokenizer=text_tokenizer,
                                                 gloss_tokenizer=gloss_tokenizer,
                                                 name2keypoint=dataset.name2keypoints),
                                             batch_size=cfg['training']['batch_size'],
                                             num_workers=cfg['training'].get('num_workers',2),
                                             sampler=sampler,
                                             )
    return dataloader, sampler