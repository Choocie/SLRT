import torch
from modelling.recognition import RecognitionNetwork
from utils.misc import get_logger
from modelling.translation import TranslationNetwork
from modelling.translation_ensemble import TranslationNetwork_Ensemble
from modelling.vl_mapper import VLMapper

class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task, self.device = cfg['task'], cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []
        assert(self.task=='S2T')
       
        self.recognition_weight = model_cfg.get('recognition_weight',1)
        self.translation_weight = model_cfg.get('translation_weight',1)
        self.recognition_network = RecognitionNetwork(
            cfg=model_cfg['RecognitionNetwork'],
            input_type = 'feature',
            input_streams = cfg['data'].get('input_streams','rgb'),
            transform_cfg=cfg['data'].get('transform_cfg',{}))
        if model_cfg['RecognitionNetwork'].get('freeze', False)==True:
            self.logger.info('freeze recognition_network')
            self.frozen_modules.append(self.recognition_network)
            for param in self.recognition_network.parameters():
                param.requires_grad = False
            self.recognition_network.eval()

        input_type = model_cfg['TranslationNetwork'].pop('input_type','feature')
        self.translation_network = TranslationNetwork(
            input_type=input_type, 
            cfg=model_cfg['TranslationNetwork'], 
            task=self.task)
        self.text_tokenizer = self.translation_network.text_tokenizer
        self.gloss_tokenizer = self.recognition_network.gloss_tokenizer 
        if model_cfg['VLMapper'].get('type','projection') == 'projection':
            if 'in_features' in model_cfg['VLMapper']:
                in_features = model_cfg['VLMapper'].pop('in_features')
            else:
                in_features = model_cfg['RecognitionNetwork']['visual_head']['hidden_size']
        else:
            in_features = len(self.gloss_tokenizer)
        self.vl_mapper = VLMapper(
            cfg=model_cfg['VLMapper'],
            in_features = in_features,
            out_features = self.translation_network.input_dim,
            gloss_id2str=self.gloss_tokenizer.id2gloss,
            gls2embed=getattr(self.translation_network, 'gls2embed', None), 
        )


    def forward(self,
                is_train,
                translation_inputs={},
                recognition_gloss_labels=None,
                recognition_gls_lengths=None,
                recognition_inputs_sgn_features=None,
                recognition_inputs_sgn_mask=None,
                recognition_inputs_sgn_videos=None, 
                recognition_inputs_sgn_lengths=None,
                recognition_inputs_sgn_keypoints=None,
                recognition_inputs_head_rgb_input=None,
                recognition_inputs_head_keypoint_input=None,
                translation_inputs_gloss_ids=None,
                translation_inputs_gloss_length=None,
                **kwargs):
        assert(self.task=='S2T')
        recognition_outputs = self.recognition_network(
            is_train=is_train,
            gloss_labels=recognition_gloss_labels,
            gls_labels=recognition_gls_lengths,
            sgn_features=recognition_inputs_sgn_features,
            sgn_mask=recognition_inputs_sgn_mask,
            sgn_videos=recognition_inputs_sgn_videos, 
            sgn_lengths=recognition_inputs_sgn_lengths,
            sgn_keypoints=recognition_inputs_sgn_keypoints,
            head_rgb_input=recognition_inputs_head_rgb_input,
            head_keypoint_input=recognition_inputs_head_keypoint_input)
        mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)
        translation_inputs = {
            **translation_inputs,
            'input_feature':mapped_feature, 
            'input_lengths':recognition_outputs['input_lengths']} 

        translation_outputs = self.translation_network(
            gloss_ids=translation_inputs_gloss_ids,
            gloss_length=translation_inputs_gloss_length,
            input_feature=mapped_feature,
            input_lengths=recognition_outputs['input_lengths'])
        model_outputs = {**translation_outputs, **recognition_outputs}
        model_outputs['transformer_inputs'] = model_outputs['transformer_inputs'] #for latter use of decoding
        model_outputs['total_loss'] = \
            model_outputs['recognition_loss']*self.recognition_weight + \
            model_outputs['translation_loss']*self.translation_weight 
        return model_outputs

    
    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):          
        model_outputs = self.translation_network.generate(**transformer_inputs, **generate_cfg)  
        return model_outputs
    
    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths)

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()

    def set_eval(self):
        self.eval()

def build_model(cfg):
    model = SignLanguageModel(cfg)
    return model.to(cfg['device'])