from vlm_reward.evaluator import Evaluator, EvaluatorConfig
from omegaconf import OmegaConf

def main():
    cfg: EvaluatorConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(EvaluatorConfig), OmegaConf.from_cli())
    )

    evaluator = Evaluator(cfg)

    video_paths = [
        "examples/example_1_A.mp4",
        "examples/example_1_B.mp4",
    ]

    prompts = [
        "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
        "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
    ]

    rewards = evaluator.reward(video_paths, prompts)

    print(rewards)


if __name__ == "__main__":
    main()