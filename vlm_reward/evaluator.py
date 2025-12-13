import os
import math
import torch
from torchvision import io
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from dataclasses import dataclass
from transformers import AutoProcessor
from vlm_reward.qwen2_vl_reward import Qwen2VLRewardModel
from vlm_reward.prompt_template import build_prompt

# -----------------------------------------------------------------------------
# 1. Simplest Pre-process Utils (Extracted & Simplified)
# -----------------------------------------------------------------------------

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Qwen2-VL 核心 Resize 逻辑：确保长宽是 28 的倍数，且像素总数在范围内。
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"aspect ratio must be smaller than {MAX_RATIO}")
    
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
        
    return h_bar, w_bar

def load_video_torchvision(video_path: str, fps: float = 2.0, max_pixels: int = 200704) -> torch.Tensor:
    """
    使用 torchvision 读取视频，按指定 FPS 采样，并 Resize 到符合 Qwen 要求的尺寸。
    Returns: Tensor of shape (T, C, H, W), float32, range [0, 255] (Qwen processor handles normalization)
    """
    # 1. Read Video
    if "file://" in video_path:
        video_path = video_path[7:]
    
    # output_format="TCHW" returns (T, C, H, W) in [0, 255]
    video, _, info = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
    
    total_frames = video.size(0)
    video_fps = info["video_fps"]
    
    # 2. Sample Frames (Uniformly based on FPS)
    target_nframes = int(total_frames / video_fps * fps)
    # Ensure at least minimal frames if video is short
    target_nframes = max(target_nframes, 4) 
    target_nframes = min(target_nframes, total_frames)
    
    # Round to factor of 2 (Qwen preference, though strictly defined for 'nframes' arg usually)
    target_nframes = round_by_factor(target_nframes, 2)
    
    idx = torch.linspace(0, total_frames - 1, target_nframes).round().long()
    video = video[idx]

    # 3. Smart Resize
    _, _, h, w = video.shape
    resized_height, resized_width = smart_resize(h, w, factor=IMAGE_FACTOR, max_pixels=max_pixels)
    
    video = resize(
        video, 
        [resized_height, resized_width], 
        interpolation=InterpolationMode.BICUBIC, 
        antialias=True
    ).float()
    
    return video

def process_video_tensor(video: torch.Tensor, max_pixels: int = 200704) -> torch.Tensor:
    """
    处理直接传入的 Tensor。假设用户传入的 Tensor 已经是采样的关键帧。
    Input: (T, C, H, W)
    """
    if video.dim() == 5: # Batch: B, T, C, H, W -> support single video only here for logic simplicity, assume T,C,H,W
         video = video.squeeze(0)
         
    _, _, h, w = video.shape
    resized_height, resized_width = smart_resize(h, w, factor=IMAGE_FACTOR, max_pixels=max_pixels)
    
    video = resize(
        video, 
        [resized_height, resized_width], 
        interpolation=InterpolationMode.BICUBIC, 
        antialias=True
    ).float()
    return video


# -----------------------------------------------------------------------------
# 2. Evaluator Implementation
# -----------------------------------------------------------------------------

@dataclass
class EvaluatorConfig:
    model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    model_revision: str = "main"
    output_dim: int = 1
    eval_dim: tuple[str, ...] = ("VQ", "MQ", "TA")
    special_tokens: tuple[str, ...] = ("<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>")

    pretrained_path: str = "assets/reward_model.pt"

    # normalize stats
    VQ_mean: float = 3.6757
    VQ_std: float = 2.2476
    MQ_mean: float = 1.1646
    MQ_std: float = 1.3811
    TA_mean: float = 2.8105
    TA_std: float = 2.5121

    # data
    fps: float = 2.0
    max_frame_pixels: int = 200704 # 128 * 28 * 28 approx
    prompt_template_type: str = "detailed_special"


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig, device="cuda"):
        self.cfg = cfg
        self.device = device

        print(f"Loading processor from {cfg.model_name_or_path}...")
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path,
            padding_side="left",
        )
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": cfg.special_tokens})
        special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(cfg.special_tokens)

        print(f"Loading model from {cfg.pretrained_path}...")
        self.model = Qwen2VLRewardModel.from_pretrained(
            cfg.model_name_or_path,
            output_dim=cfg.output_dim,
            special_token_ids=special_token_ids,
            attn_implementation="flash_attention_2",
            revision=cfg.model_revision,
            torch_dtype=torch.bfloat16,
        )
        self.model.resize_token_embeddings(len(self.processor.tokenizer)) 
        
        # Load weights
        if os.path.exists(cfg.pretrained_path):
            state_dict = torch.load(cfg.pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
        else:
            print(f"Warning: Checkpoint {cfg.pretrained_path} not found. Using random weights for debug.")

        self.model.to(device=self.device, dtype=torch.bfloat16).eval()

    def _normalize_reward(self, reward_dict):
        """Standardize rewards using stats from config."""
        reward_dict['VQ'] = (reward_dict['VQ'] - self.cfg.VQ_mean) / self.cfg.VQ_std
        reward_dict['MQ'] = (reward_dict['MQ'] - self.cfg.MQ_mean) / self.cfg.MQ_std
        reward_dict['TA'] = (reward_dict['TA'] - self.cfg.TA_mean) / self.cfg.TA_std
        reward_dict['Overall'] = reward_dict['VQ'] + reward_dict['MQ'] + reward_dict['TA']
        return reward_dict

    def prepare_batch(self, videos: list[str | torch.Tensor], prompts: list[str]):
        """
        Correctly uses the processor to handle padding and mRoPE alignment.
        """
        text_inputs_list = []
        video_inputs_list = []

        for video_item, prompt in zip(videos, prompts):
            if isinstance(video_item, str):
                video_tensor = load_video_torchvision(video_item, fps=self.cfg.fps, max_pixels=self.cfg.max_frame_pixels)
            elif isinstance(video_item, torch.Tensor):
                video_tensor = process_video_tensor(video_item, max_pixels=self.cfg.max_frame_pixels)
            else:
                raise ValueError("Input video must be path (str) or Tensor.")
            
            video_inputs_list.append(video_tensor)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "min_pixels": 1, "max_pixels": self.cfg.max_frame_pixels}, 
                        {"type": "text", "text": build_prompt(prompt, self.cfg.eval_dim, self.cfg.prompt_template_type)},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_inputs_list.append(text)

        batch = self.processor(
            text=text_inputs_list,
            videos=video_inputs_list,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True} 
        )

        return {k: v.to(self.device) for k, v in batch.items()}

    def reward(self, videos: list[str | torch.Tensor], prompts: list[str], use_norm=True):
        batch = self.prepare_batch(videos, prompts)
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs["logits"] # Shape: (B, 3, 1) usually or (B, 3) depends on model def
            
            # Squeeze output dim if necessary
            if logits.dim() == 3 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)

        results = []
        for i in range(len(logits)):
            # Assume order is VQ, MQ, TA based on config
            r = {
                'VQ': logits[i][0].item(), 
                'MQ': logits[i][1].item(), 
                'TA': logits[i][2].item()
            }
            if use_norm:
                r = self._normalize_reward(r)
            results.append(r)
            
        return results

# -----------------------------------------------------------------------------
# 3. Debug / Main
# -----------------------------------------------------------------------------

def debug():
    # Update paths to your local environment
    base_dir = os.path.expanduser("~/workspace/code/video-reward")
    video_paths = [
        os.path.join(base_dir, "datasets/train/videos/example_1_A.mp4"),
        os.path.join(base_dir, "datasets/train/videos/example_1_B.mp4"),
        # os.path.join(base_dir, "datasets/train/videos/example_2_A.mp4")
    ]

    prompts = [
        "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
        "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
        # "The camera follows a young explorer through an abandoned urban building at night, exploring hidden corridors and forgotten spaces, with a mix of light and shadow creating a mysterious atmosphere.",
    ]

    cfg = EvaluatorConfig(pretrained_path="assets/reward_model.pt") # Ensure path exists
    
    # Check if model path is valid (for running without errors if weight missing)
    if not os.path.exists(cfg.pretrained_path):
        print(f"Note: {cfg.pretrained_path} not found. Model initialized with random weights.")

    evaluator = Evaluator(cfg)
    
    rewards = evaluator.reward(video_paths, prompts)
    
    for i, r in enumerate(rewards):
        print(f"Video {i}: {r}")

if __name__ == "__main__":
    debug()

"""
python -m vlm_reward.evaluator
"""