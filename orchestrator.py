from experiments.linear_probes.probe_object_and_typo import LinearProbeExperiment
from experiments.typographic_score.typographic_score import TypographicScoreExperiment
from experiments.greedy_selection.greedy_selection import GreedySelectionExperiment
from experiments.eval.zeroshot_eval import ZeroShotEval
import time
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Orchestrator for running experiments")
    parser.add_argument(
        "--model",
        type=str,
        default="vit-l",
        choices=[
            "vit-b",
            "vit-l",
            "vit-g",
            "vit-h",
            "vit-big-g",
            "whylesionclip",
            "whyxrayclip",
        ],
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default="cls", choices=["cls", "spatial"])
    return parser.parse_args()


def main(args):
    model = args.model
    device = args.device
    mode = args.mode

    linear_probe_experiment = LinearProbeExperiment(
        model_short_name=model,
        device=device,
        batch_size=16,
        num_workers=32,
    )
    linear_probe_experiment.load_or_run_experiment()
    del linear_probe_experiment

    typographic_score_experiment = TypographicScoreExperiment(
        model_short_name=model,
        device=device,
        batch_size=16,
        num_workers=32,
        mode=mode,
    )
    typographic_score_experiment.load_or_run_experiment()
    del typographic_score_experiment

    greedy_selection_experiment = GreedySelectionExperiment(
        model_short_name=model,
        device=device,
        batch_size=16,
        num_workers=32,
        mode=mode,
    )
    greedy_selection_experiment.load_or_run_experiment()
    del greedy_selection_experiment

    non_typo_zeroshot_eval_experiment = ZeroShotEval(
        model_short_name=model,
        device=device,
        mode=mode,
        dataset_list=[
            "imagenet-100-typo",
            "food101",
            "rta100",
            "scam",
            "fgvc-aircraft",
            "disentangling",
            "isic2019",
            "melanoma",
            "chest-xray",
            "paint",
            "bcn",
            "ham10k",
        ],
    )
    non_typo_zeroshot_eval_experiment.load_or_run_experiment()
    del non_typo_zeroshot_eval_experiment


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
