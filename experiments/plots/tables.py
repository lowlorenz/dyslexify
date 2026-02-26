import json
from pathlib import Path


def generate_tab_dataset_comparison(results):
    tab_dataset_comparison_header = f"""\\begin{{table}}[h]
\\centering
\\caption{{Comparison of dyslexic models performance on datasets of typographic attacks across model sizes, showing accuracy changes relative to the base model, with improvements (\\textcolor{{darkgreen}}{{\\scriptsize↑}}) or declines (\\textcolor{{red}}{{\\scriptsize↓}}).}}
\\begin{{tabular}}{{l@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c}}
  \\toprule
  & \\multicolumn{{3}}{{c}}{{Real Typographic}} & \\multicolumn{{3}}{{c}}{{Synthetic Typographic}} \\\\
  Model & RTA-100 & Disentangling & Paint & IN-100-T & Food-101-T & Aircraft-T \\\\
  \\midrule
"""

    tab_dataset_comparison_footer = f"""
  \\bottomrule
  \\end{{tabular}}
  \\label{{tab:dataset_comparison}}
\\end{{table}}"""

    tab_dataset_comparison_body = ""
    for model, results in results.items():
        tab_dataset_comparison_body += f"  {model} & "
        for dataset in [
            "RTA100",
            "Disentangling",
            "Paint",
            "ImageNet100",
            "Food101",
            "FGVCAircraft",
        ]:
            dyslexified_typo_accuracy = (
                results[dataset]["dislexified_typo_accuracy"] * 100
            )
            delta = (
                results[dataset]["dislexified_typo_accuracy"]
                - results[dataset]["typo_accuracy"]
            ) * 100
            color = "darkgreen" if delta > 0 else "red"
            arrow = "↑" if delta > 0 else "↓"
            tab_dataset_comparison_body += f"{dyslexified_typo_accuracy:.2f}\\textcolor{{{color}}}{{\scriptsize{arrow}{delta:.2f}}} & "

        # Remove the last " & " and add line break
        tab_dataset_comparison_body = tab_dataset_comparison_body[:-2] + " \\\\\n"

    tab_dataset_comparison = (
        tab_dataset_comparison_header
        + tab_dataset_comparison_body
        + tab_dataset_comparison_footer
    )

    output_path = Path("results/experiments/eval/tables/dataset_comparison.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tab_dataset_comparison)

    print("Generating dataset comparison table")
    print(tab_dataset_comparison)
    print(f"Saved to {output_path}")


def generate_vision_dataset_comparison(results):
    tab_vision_comparison_header = f"""\\begin{{table}}[h]
\\centering
\\caption{{Comparison of dyslexic model performance on non-typographic datasets across model sizes, showing accuracy changes relative to the base model, with improvements (\\textcolor{{darkgreen}}{{\\scriptsize↑}}) or declines (\\textcolor{{red}}{{\\scriptsize↓}}).}}
\\begin{{tabular}}{{l@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c}}
  \\toprule
  Model & Aircraft & Food-101 & ImageNet-100 \\\\
  \\midrule
"""

    tab_vision_comparison_footer = f"""
  \\bottomrule
  \\end{{tabular}}
  \\label{{tab:vision_dataset_comparison}}
\\end{{table}}"""

    tab_vision_comparison_body = ""
    for model, model_results in results.items():
        tab_vision_comparison_body += f"  {model} & "
        for dataset_key, dataset_name in [
            ("FGVCAircraft", "Aircraft"),
            ("Food101", "Food-101"),
            ("ImageNet100", "ImageNet-100"),
        ]:
            if dataset_key in model_results:
                dislexified_accuracy = (
                    model_results[dataset_key]["dislexified_accuracy"] * 100
                )
                accuracy = model_results[dataset_key]["accuracy"] * 100
                delta = dislexified_accuracy - accuracy

                if abs(delta) < 0.005:  # Essentially zero
                    color = "gray"
                    arrow = "="
                    tab_vision_comparison_body += f"{dislexified_accuracy:.2f}\\textcolor{{{color}}}{{\\scriptsize{arrow}{delta:.2f}}}"
                else:
                    color = "darkgreen" if delta > 0 else "red"
                    arrow = "↑" if delta > 0 else "↓"
                    tab_vision_comparison_body += f"{dislexified_accuracy:.2f}\\textcolor{{{color}}}{{\\scriptsize{arrow}{abs(delta):.2f}}}"

                tab_vision_comparison_body += " & "
            else:
                tab_vision_comparison_body += "N/A & "

        # Remove the last " & " and add line break
        tab_vision_comparison_body = tab_vision_comparison_body[:-3] + " \\\\\n"

    tab_vision_comparison = (
        tab_vision_comparison_header
        + tab_vision_comparison_body
        + tab_vision_comparison_footer
    )

    output_path = Path("results/experiments/eval/tables/vision_dataset_comparison.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tab_vision_comparison)

    print("Generating vision dataset comparison table")
    print(tab_vision_comparison)
    print(f"Saved to {output_path}")

    return tab_vision_comparison


def generate_selected_heads_table():
    tab_heads_header = f"""\\begin{{table}}[h]
\\centering
\\caption{{Number of heads \\(\\mathcal{{H}}_{{i,\\ell}}\\) in \\(\\mathcal{{C}}\\) per Model}}
\\begin{{tabular}}{{lrrr}}
\\hline
Model & Selected & Total & Percentage (\\%) \\\\
\\hline
"""

    tab_heads_footer = f"""\\hline
\\end{{tabular}}
\\label{{tab:selected_heads_percentage}}
\\end{{table}}"""

    # Model configurations with total number of heads
    model_configs = {
        "B": {
            "path": "results/experiments/greedy_selection/vit-b/ablated_heads_cls.json",
            "total": 144,
        },
        "L": {
            "path": "results/experiments/greedy_selection/vit-l/ablated_heads_cls.json",
            "total": 288,
        },
        "H": {
            "path": "results/experiments/greedy_selection/vit-h/ablated_heads_cls.json",
            "total": 384,
        },
        "G": {
            "path": "results/experiments/greedy_selection/vit-g/ablated_heads_cls.json",
            "total": 480,
        },
        "Big-G": {
            "path": "results/experiments/greedy_selection/vit-big-g/ablated_heads_cls.json",
            "total": 576,
        },
    }

    tab_heads_body = ""
    for model, config in model_configs.items():
        if Path(config["path"]).exists():
            with open(config["path"]) as f:
                ablated_heads = json.load(f)
            selected = len(ablated_heads)
            total = config["total"]
            percentage = (selected / total) * 100
            tab_heads_body += (
                f"{model} & {selected} & {total} & {percentage:.1f} \\\\\n"
            )
        else:
            print(f"Warning: {config['path']} not found, skipping {model} model")

    tab_heads_table = tab_heads_header + tab_heads_body + tab_heads_footer

    output_path = Path("results/experiments/eval/tables/selected_heads_percentage.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tab_heads_table)

    print("Generating selected heads percentage table")
    print(tab_heads_table)
    print(f"Saved to {output_path}")

    return tab_heads_table


def generate_ocr_eval_table():
    ### THE PATHS ARE WRONG, NEED TO FIX THEM
    ### THEY ARE LIKE THIS BECAUSE IS REBUTTAL CRUNCHTIME 😎
    tab_ocr_header = f"""\\begin{{table}}[h]
\\centering
\\caption{{Comparison of dyslexic model performance on OCR evaluation, showing accuracy changes relative to the base model, with improvements (\\textcolor{{darkgreen}}{{\\scriptsize↑}}) or declines (\\textcolor{{red}}{{\\scriptsize↓}}).}}
\\begin{{tabular}}{{l@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c}}
\\toprule
Model & Small Lexicon & Medium Lexicon \\\\
\\midrule
"""

    tab_ocr_footer = f"""\\bottomrule
\\end{{tabular}}
\\label{{tab:ocr_eval_comparison}}
\\end{{table}}"""

    model_configs = {
        "B": "iclr_results/experiments/ocr_eval/vit-b/results_cls.json",
        "L": "iclr_results/experiments/ocr_eval/vit-l/results_cls.json",
        "H": "iclr_results/experiments/ocr_eval/vit-h/results_cls.json",
        "G": "iclr_results/experiments/ocr_eval/vit-g/results_cls.json",
        "Big-G": "iclr_results/experiments/ocr_eval/vit-big-g/results_cls.json",
    }

    tab_ocr_body = ""
    for model, filepath in model_configs.items():
        if Path(filepath).exists():
            with open(filepath) as f:
                results = json.load(f)

            tab_ocr_body += f"{model} & "

            for lexicon in ["small_lexicon", "medium_lexicon"]:
                if lexicon in results:
                    baseline_accuracy = results[lexicon]["accuracy"] * 100
                    dislexified_accuracy = (
                        results[lexicon]["dislexified_accuracy"] * 100
                    )
                    delta = dislexified_accuracy - baseline_accuracy
                    color = "darkgreen" if delta > 0 else "red"
                    arrow = "↑" if delta > 0 else "↓"
                    tab_ocr_body += f"{dislexified_accuracy:.2f}\\textcolor{{{color}}}{{\\scriptsize{arrow}{abs(delta):.2f}}} & "
                else:
                    tab_ocr_body += "N/A & "

            tab_ocr_body = tab_ocr_body[:-3] + " \\\\\n"
        else:
            print(f"Warning: {filepath} not found, skipping {model} model")

    tab_ocr_table = tab_ocr_header + tab_ocr_body + tab_ocr_footer

    output_path = Path("results/experiments/eval/tables/ocr_eval_comparison.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tab_ocr_table)

    print("Generating OCR evaluation table")
    print(tab_ocr_table)
    print(f"Saved to {output_path}")

    return tab_ocr_table


def generate_whylesionclip_table(results):
    tab_whylesionclip_header = f"""\\begin{{table}}[h]
\\centering
\\caption{{Comparison of dyslexic model performance on WhylesionCLIP dataset, showing accuracy changes relative to the base model, with improvements (\\textcolor{{darkgreen}}{{\\scriptsize↑}}) or declines (\\textcolor{{red}}{{\\scriptsize↓}}).}}
\\footnotesize
\\begin{{tabular}}{{l@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c}}
\\toprule
& \\multicolumn{{4}}{{c}}{{Not Typographic}} & \\multicolumn{{4}}{{c}}{{Synthetic Typographic}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}}
Model & ISIC2019 & Melanoma & BCN20k & HAM10k & ISIC2019-T & Melanoma-T & BCN20k-T & HAM10k-T \\\\
\\midrule
"""

    tab_whylesionclip_footer = f"""\\bottomrule
\\end{{tabular}}
\\label{{tab:whylesionclip_comparison}}
\\end{{table}}"""

    why_lesion_clip_results = results["WhylesionCLIP"]

    tab_whylesionclip_body = ""

    # WhylesionCLIP baseline row
    isic_accuracy = why_lesion_clip_results["ISIC2019Binary"]["accuracy"] * 100
    melanoma_accuracy = why_lesion_clip_results["Melanoma"]["accuracy"] * 100
    bcn20k_accuracy = why_lesion_clip_results["BCN20k"]["accuracy"] * 100
    ham10k_accuracy = why_lesion_clip_results["HAM10k"]["accuracy"] * 100
    isic_typo_accuracy = (
        why_lesion_clip_results["ISIC2019Binary"]["typo_accuracy"] * 100
    )
    melanoma_typo_accuracy = why_lesion_clip_results["Melanoma"]["typo_accuracy"] * 100
    bcn20k_typo_accuracy = why_lesion_clip_results["BCN20k"]["typo_accuracy"] * 100
    ham10k_typo_accuracy = why_lesion_clip_results["HAM10k"]["typo_accuracy"] * 100

    tab_whylesionclip_body += f"WhyLesionCLIP & {isic_accuracy:.2f} & {melanoma_accuracy:.2f} & {bcn20k_accuracy:.2f} & {ham10k_accuracy:.2f} & {isic_typo_accuracy:.2f} & {melanoma_typo_accuracy:.2f} & {bcn20k_typo_accuracy:.2f} & {ham10k_typo_accuracy:.2f} \\\\\n"

    # Our model row with improvement indicators
    tab_whylesionclip_body += f"Ours & "

    # ISIC2019Binary
    ours_isic_accuracy = (
        why_lesion_clip_results["ISIC2019Binary"]["dislexified_accuracy"] * 100
    )
    delta_isic = ours_isic_accuracy - isic_accuracy
    color_isic = "darkgreen" if delta_isic > 0 else "red"
    arrow_isic = "↑" if delta_isic > 0 else "↓"
    tab_whylesionclip_body += f"{ours_isic_accuracy:.2f} \\textcolor{{{color_isic}}}{{\\scriptsize{arrow_isic}{abs(delta_isic):.2f}}}& "

    # Melanoma
    ours_melanoma_accuracy = (
        why_lesion_clip_results["Melanoma"]["dislexified_accuracy"] * 100
    )
    delta_melanoma = ours_melanoma_accuracy - melanoma_accuracy
    color_melanoma = "darkgreen" if delta_melanoma > 0 else "red"
    arrow_melanoma = "↑" if delta_melanoma > 0 else "↓"
    tab_whylesionclip_body += f"{ours_melanoma_accuracy:.2f} \\textcolor{{{color_melanoma}}}{{\\scriptsize{arrow_melanoma}{abs(delta_melanoma):.2f}}}& "

    # BCN20k
    ours_bcn20k_accuracy = (
        why_lesion_clip_results["BCN20k"]["dislexified_accuracy"] * 100
    )
    delta_bcn20k = ours_bcn20k_accuracy - bcn20k_accuracy
    color_bcn20k = "darkgreen" if delta_bcn20k > 0 else "red"
    arrow_bcn20k = "↑" if delta_bcn20k > 0 else "↓"
    tab_whylesionclip_body += f"{ours_bcn20k_accuracy:.2f} \\textcolor{{{color_bcn20k}}}{{\\scriptsize{arrow_bcn20k}{abs(delta_bcn20k):.2f}}}& "

    # HAM10k
    ours_ham10k_accuracy = (
        why_lesion_clip_results["HAM10k"]["dislexified_accuracy"] * 100
    )
    delta_ham10k = ours_ham10k_accuracy - ham10k_accuracy
    color_ham10k = "darkgreen" if delta_ham10k > 0 else "red"
    arrow_ham10k = "↑" if delta_ham10k > 0 else "↓"
    tab_whylesionclip_body += f"{ours_ham10k_accuracy:.2f} \\textcolor{{{color_ham10k}}}{{\\scriptsize{arrow_ham10k}{abs(delta_ham10k):.2f}}}& "

    # ISIC2019Binary-Typo
    ours_isic_typo_accuracy = (
        why_lesion_clip_results["ISIC2019Binary"]["dislexified_typo_accuracy"] * 100
    )
    delta_isic_typo = ours_isic_typo_accuracy - isic_typo_accuracy
    color_isic_typo = "darkgreen" if delta_isic_typo > 0 else "red"
    arrow_isic_typo = "↑" if delta_isic_typo > 0 else "↓"
    tab_whylesionclip_body += f"{ours_isic_typo_accuracy:.2f} \\textcolor{{{color_isic_typo}}}{{\\scriptsize{arrow_isic_typo}{abs(delta_isic_typo):.2f}}}& "

    # Melanoma-Typo
    ours_melanoma_typo_accuracy = (
        why_lesion_clip_results["Melanoma"]["dislexified_typo_accuracy"] * 100
    )
    delta_melanoma_typo = ours_melanoma_typo_accuracy - melanoma_typo_accuracy
    color_melanoma_typo = "darkgreen" if delta_melanoma_typo > 0 else "red"
    arrow_melanoma_typo = "↑" if delta_melanoma_typo > 0 else "↓"
    tab_whylesionclip_body += f"{ours_melanoma_typo_accuracy:.2f} \\textcolor{{{color_melanoma_typo}}}{{\\scriptsize{arrow_melanoma_typo}{abs(delta_melanoma_typo):.2f}}}& "

    # BCN20k-Typo
    ours_bcn20k_typo_accuracy = (
        why_lesion_clip_results["BCN20k"]["dislexified_typo_accuracy"] * 100
    )
    delta_bcn20k_typo = ours_bcn20k_typo_accuracy - bcn20k_typo_accuracy
    color_bcn20k_typo = "darkgreen" if delta_bcn20k_typo > 0 else "red"
    arrow_bcn20k_typo = "↑" if delta_bcn20k_typo > 0 else "↓"
    tab_whylesionclip_body += f"{ours_bcn20k_typo_accuracy:.2f} \\textcolor{{{color_bcn20k_typo}}}{{\\scriptsize{arrow_bcn20k_typo}{abs(delta_bcn20k_typo):.2f}}}& "

    # HAM10k-Typo
    ours_ham10k_typo_accuracy = (
        why_lesion_clip_results["HAM10k"]["dislexified_typo_accuracy"] * 100
    )
    delta_ham10k_typo = ours_ham10k_typo_accuracy - ham10k_typo_accuracy
    color_ham10k_typo = "darkgreen" if delta_ham10k_typo > 0 else "red"
    arrow_ham10k_typo = "↑" if delta_ham10k_typo > 0 else "↓"
    tab_whylesionclip_body += f"{ours_ham10k_typo_accuracy:.2f} \\textcolor{{{color_ham10k_typo}}}{{\\scriptsize{arrow_ham10k_typo}{abs(delta_ham10k_typo):.2f}}}"
    # Remove the last " & " and add line break
    tab_whylesionclip_body += " \\\\\n"

    tab_whylesionclip_table = (
        tab_whylesionclip_header + tab_whylesionclip_body + tab_whylesionclip_footer
    )

    output_path = Path("results/experiments/eval/tables/whylesionclip_comparison.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tab_whylesionclip_table)

    print("Generating WhylesionCLIP table")
    print(tab_whylesionclip_table)
    print(f"Saved to {output_path}")

    return tab_whylesionclip_table


if __name__ == "__main__":

    # results = {}

    # # Load available result files
    # result_files = {
    #     "B": "results/experiments/eval/vit-b/results_cls.json",
    #     "L": "results/experiments/eval/vit-l/results_cls.json",
    #     "H": "results/experiments/eval/vit-h/results_cls.json",
    #     "G": "results/experiments/eval/vit-g/results_cls.json",
    #     "Big-G": "results/experiments/eval/vit-big-g/results_cls.json",
    # }

    # for model, filepath in result_files.items():
    #     if Path(filepath).exists():
    #         results[model] = json.load(open(filepath))
    #     else:
    #         print(f"Warning: {filepath} not found, skipping {model} model")

    # medical_results = {
    #     "WhylesionCLIP": json.load(
    #         open("results/experiments/eval/whylesionclip/results_cls.json")
    #     ),
    # }

    # generate_tab_dataset_comparison(results)
    # generate_vision_dataset_comparison(results)
    # generate_selected_heads_table()
    # generate_whylesionclip_table(medical_results)
    generate_ocr_eval_table()
