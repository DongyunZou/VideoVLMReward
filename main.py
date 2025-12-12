from dataclasses import dataclass
from omegaconf import OmegaConf
from transformers import AutoProcessor
from qwen2_vl_reward import Qwen2VLRewardModel
import torch

@dataclass
class VideoRewardModelConfig:
    model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    model_revision: str = "main"
    output_dim: int = 1
    special_tokens: tuple[str, ...] = ("<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>")

    pretrained_path: str = "assets/reward_model.pt"

    # normalize
    VQ_mean: float = 3.6757
    VQ_std: float = 2.2476
    MQ_mean: float = 1.1646
    MQ_std: float = 1.3811
    TA_mean: float = 2.8105
    TA_std: float = 2.5121

    # data
    fps: int = 2
    max_frame_pixels: int = 200704
    prompt_template_type: str = "detailed_special"

def main():
    cfg: VideoRewardModelConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(VideoRewardModelConfig), OmegaConf.from_cli())
    )

    processor = AutoProcessor.from_pretrained(
        cfg.model_name_or_path,
        padding_side="right",
    )
    processor.tokenizer.add_special_tokens({"additional_special_tokens": cfg.special_tokens})
    special_token_ids = processor.tokenizer.convert_tokens_to_ids(cfg.special_tokens)

    model = Qwen2VLRewardModel.from_pretrained(
        cfg.model_name_or_path,
        output_dim=cfg.output_dim,
        special_token_ids=special_token_ids,
        attn_implementation="flash_attention_2",
        revision=cfg.model_revision,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(processor.tokenizer)) 
    model.load_state_dict(torch.load(cfg.pretrained_path))
    model.cuda().eval()

if __name__ == "__main__":
    main()