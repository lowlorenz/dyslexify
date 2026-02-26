```
██████╗ ██╗   ██╗███████╗██╗     ███████╗██╗  ██╗██╗███████╗██╗   ██╗
██╔══██╗╚██╗ ██╔╝██╔════╝██║     ██╔════╝╚██╗██╔╝██║██╔════╝╚██╗ ██╔╝
██║  ██║ ╚████╔╝ ███████╗██║     █████╗   ╚███╔╝ ██║█████╗   ╚████╔╝ 
██║  ██║  ╚██╔╝  ╚════██║██║     ██╔══╝   ██╔██╗ ██║██╔══╝    ╚██╔╝  
██████╔╝   ██║   ███████║███████╗███████╗██╔╝ ██╗██║██║        ██║   
╚═════╝    ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   
```

Dyslexify investigates typographic attacks on CLIP-based vision-language models. It trains linear probes to detect typographic content, computes a typographic attention score, builds a circuit using a greedy selection strategy, and evaluates zero-shot classification robustness.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) to install the dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Usage

The main entry point is `orchestrator.py`, which runs the full pipeline:

1. **Linear Probes** -- trains probes to detect objects and typographic content
2. **Typographic Attention Score** -- computes per-head attention scores for typographic features
3. **Greedy Selection** -- builds a circuit by greedily selecting attention heads
4. **Zero-Shot Evaluation** -- evaluates zero-shot classification with and without the circuit

```bash
uv run orchestrator.py --model vit-l --device cuda:0 --mode cls
```

### Arguments

| Argument   | Default  | Options                                                                  |
|------------|----------|--------------------------------------------------------------------------|
| `--model`  | `vit-l`  | `vit-b`, `vit-l`, `vit-g`, `vit-h`, `vit-big-g`, `whylesionclip`, `whyxrayclip` |
| `--device` | `cuda:0` | any valid torch device                                                   |
| `--mode`   | `cls`    | `cls`, `spatial`                                                         |

## Datasets

See [`dyslexify/dataset/readme.md`](dyslexify/dataset/readme.md) for instructions on where to obtain each dataset.

## Issues

If you run into any problems, feel free to open a [GitHub issue](https://github.com/lowlorenz/dyslexify/issues).


```
@inproceedings{hufe2026dyslexify,
    title={Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP},
    author={Hufe, Lorenz and Venhoff, Constantin and Dreyer, Maximilian
            and Purelku, Erblina and Lapuschkin, Sebastian and Samek,
            Wojciech},
    booktitle={International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=HNGEzXwbJw}
}
```