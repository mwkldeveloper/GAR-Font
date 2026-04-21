from model.generator.gpt import GPTModelArgs
from model.tokenizer.tokenizer import TokenizerModelArgs
from peft import LoraConfig, TaskType

z_channel = 256
codebook_size = 2048
codebook_embed_dim = 8

img_C_in = 3 
downsample_ratio = 8 # VQ-16 => 16, VQ-8 => 8
img_size = 64


tokenizer_kwargs = dict( 
    codebook_embed_num = codebook_size,
    codebook_embed_dim = codebook_embed_dim,
    codebook_l2_norm = False,
    codebook_show_usage = True,
    commit_loss_beta = 0.35,
    entropy_loss_ratio = 0.02,
    mid_ch = 128, 
    z_channels = z_channel,
    dropout_p = 0.0,

    vit_dim = 32,
    vit_depth = 6,
    vit_mlp_dim = 32,
    vit_heads_dim = 32,
    patch_num = img_size//downsample_ratio
)

style_args = {
    'C_in': img_C_in,
    'C': 32,
    'C_out': z_channel,
    'norm': 'in',
    'activ': 'relu',
    'pad_type': 'reflect',
    'sigmoid': False,
    'scale_var': True,
    'downsample_ratio' : downsample_ratio
}
ffm_args = {
    'z_channel': z_channel,
    'n_heads': 8,
}

#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return TokenizerModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs)

def VQ_16(**kwargs):
    return TokenizerModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs)

VQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8}

gpt_kwargs = dict(
    vocab_size = codebook_size ,
    token_dropout_p = 0.1 ,
    attn_dropout_p = 0.0 ,
    resid_dropout_p = 0.1 ,
    ffn_dropout_p = 0.1 ,
    feature_dropout_prob = 0.1 ,
    img_feature_channel = z_channel*2 ,
    img_feature_code_len = (img_size//downsample_ratio)**2,
    target_token_len = (img_size//downsample_ratio)**2
)

#################################################################################
#                              AR Model Configs                                 #
#################################################################################
def GPT_314M(**kwargs):
    return GPTModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)

def GPT_141M(**kwargs):
    return GPTModelArgs(n_layer=16, n_head=16, dim=832, **kwargs)

def GPT_90M(**kwargs): 
    return GPTModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)

def GPT_30M(**kwargs): 
    return GPTModelArgs(n_layer=8, n_head=8, dim=512, **kwargs)

gpt_models = {
    'GPT-314M': GPT_314M, 'GPT-141M': GPT_141M,'GPT-90M':GPT_90M,'GPT-30M':GPT_30M
}


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["wqkv", "wo", "w1", "w2", "w3"],  
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION  
)