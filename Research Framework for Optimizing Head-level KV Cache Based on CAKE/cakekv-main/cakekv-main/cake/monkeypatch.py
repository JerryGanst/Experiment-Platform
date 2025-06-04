import transformers
from cake.model.modify_llama import llama_model_forward_cake, llama_attn_forward_cake
from cake.model.modify_mistral import mistral_model_forward_cake, mistral_attn_forward_cake
from cake.model.modify_qwen2 import qwen2_model_forward_cake, qwen2_attn_forward_cake
    

def replace_flashllama_attn_with_cakeattn():
    """替换LLaMA attention实现，兼容不同transformers版本"""
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward_cake
    
    # 检查FlashAttention2是否存在（较新版本的transformers）
    if hasattr(transformers.models.llama.modeling_llama, 'LlamaFlashAttention2'):
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_attn_forward_cake
    
    # 检查标准LlamaAttention是否存在（较旧版本或非Flash版本）
    if hasattr(transformers.models.llama.modeling_llama, 'LlamaAttention'):
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_cake

def replace_flashmistral_attn_with_cakeattn():
    """替换Mistral attention实现，兼容不同transformers版本"""
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward_cake
    
    # 检查FlashAttention2是否存在
    if hasattr(transformers.models.mistral.modeling_mistral, 'MistralFlashAttention2'):
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_attn_forward_cake
    
    # 检查标准MistralAttention是否存在
    if hasattr(transformers.models.mistral.modeling_mistral, 'MistralAttention'):
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_cake

def replace_flashqwen2_attn_with_cakeattn():
    """替换Qwen2 attention实现，兼容不同transformers版本"""
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward_cake
    
    # 检查FlashAttention2是否存在
    if hasattr(transformers.models.qwen2.modeling_qwen2, 'Qwen2FlashAttention2'):
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_attn_forward_cake
    
    # 检查标准Qwen2Attention是否存在
    if hasattr(transformers.models.qwen2.modeling_qwen2, 'Qwen2Attention'):
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attn_forward_cake

