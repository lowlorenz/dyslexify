import json
from pathlib import Path


def generate_blur_nontypographic_table(results):
    tab_header = f"""\\begin{{table}}[h]
\\centering
\\caption{{Comparison of dyslexic model performance on ImageNet-100 with and without blur.}}
\\begin{{tabular}}{{l@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c}}
\\toprule
Model & Normal & Blurred & Dyslexify & Both \\\\
\\midrule
"""

    tab_footer = f"""\\bottomrule
\\end{{tabular}}
\\label{{tab:blur_nontypographic}}
\\end{{table}}"""

    # First, collect all accuracies per model
    model_accuracies = {}
    for model in ["B", "L", "H", "G", "Big-G"]:
        if model in results:
            accuracies = []
            # Normal
            if "ImageNet100" in results[model]:
                accuracies.append(
                    ("Normal", results[model]["ImageNet100"]["accuracy"] * 100)
                )
            # Blurred
            if "ImageNet100Blurred" in results[model]:
                accuracies.append(
                    ("Blurred", results[model]["ImageNet100Blurred"]["accuracy"] * 100)
                )
            # Dyslexify
            if "ImageNet100" in results[model]:
                accuracies.append(
                    (
                        "Dyslexify",
                        results[model]["ImageNet100"]["dislexified_accuracy"] * 100,
                    )
                )
            # Both
            if "ImageNet100Blurred" in results[model]:
                accuracies.append(
                    (
                        "Both",
                        results[model]["ImageNet100Blurred"]["dislexified_accuracy"]
                        * 100,
                    )
                )

            # Sort by accuracy to find best and second best
            sorted_accuracies = sorted(accuracies, key=lambda x: x[1], reverse=True)
            model_accuracies[model] = {
                "best": sorted_accuracies[0][0] if len(sorted_accuracies) > 0 else None,
                "second_best": (
                    sorted_accuracies[1][0] if len(sorted_accuracies) > 1 else None
                ),
            }

    tab_body = ""

    # Generate rows for each model
    for model in ["B", "L", "H", "G", "Big-G"]:
        tab_body += f"{model} & "

        # Normal
        if model in results and "ImageNet100" in results[model]:
            accuracy = results[model]["ImageNet100"]["accuracy"] * 100
            if (
                model in model_accuracies
                and model_accuracies[model]["best"] == "Normal"
            ):
                tab_body += f"\\textbf{{{accuracy:.2f}}} & "
            elif (
                model in model_accuracies
                and model_accuracies[model]["second_best"] == "Normal"
            ):
                tab_body += f"\\underline{{{accuracy:.2f}}} & "
            else:
                tab_body += f"{accuracy:.2f} & "
        else:
            tab_body += "N/A & "

        # Blurred
        if model in results and "ImageNet100Blurred" in results[model]:
            accuracy = results[model]["ImageNet100Blurred"]["accuracy"] * 100
            if (
                model in model_accuracies
                and model_accuracies[model]["best"] == "Blurred"
            ):
                tab_body += f"\\textbf{{{accuracy:.2f}}} & "
            elif (
                model in model_accuracies
                and model_accuracies[model]["second_best"] == "Blurred"
            ):
                tab_body += f"\\underline{{{accuracy:.2f}}} & "
            else:
                tab_body += f"{accuracy:.2f} & "
        else:
            tab_body += "N/A & "

        # Dyslexify
        if model in results and "ImageNet100" in results[model]:
            accuracy = results[model]["ImageNet100"]["dislexified_accuracy"] * 100
            if (
                model in model_accuracies
                and model_accuracies[model]["best"] == "Dyslexify"
            ):
                tab_body += f"\\textbf{{{accuracy:.2f}}} & "
            elif (
                model in model_accuracies
                and model_accuracies[model]["second_best"] == "Dyslexify"
            ):
                tab_body += f"\\underline{{{accuracy:.2f}}} & "
            else:
                tab_body += f"{accuracy:.2f} & "
        else:
            tab_body += "N/A & "

        # Both
        if model in results and "ImageNet100Blurred" in results[model]:
            accuracy = (
                results[model]["ImageNet100Blurred"]["dislexified_accuracy"] * 100
            )
            if model in model_accuracies and model_accuracies[model]["best"] == "Both":
                tab_body += f"\\textbf{{{accuracy:.2f}}} & "
            elif (
                model in model_accuracies
                and model_accuracies[model]["second_best"] == "Both"
            ):
                tab_body += f"\\underline{{{accuracy:.2f}}} & "
            else:
                tab_body += f"{accuracy:.2f} & "
        else:
            tab_body += "N/A & "

        tab_body = tab_body[:-3] + " \\\\\n"

    tab_table = tab_header + tab_body + tab_footer

    output_path = Path("results/experiments/eval/tables/blur_nontypographic.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tab_table)

    print("Generating blur non-typographic table")
    print(tab_table)
    print(f"Saved to {output_path}")

    return tab_table


def generate_blur_typographic_table(results):
    tab_header = f"""\\begin{{table}}[h]
\\centering
\\caption{{Comparison of dyslexic model performance on ImageNet-100-Typo with and without blur.}}
\\begin{{tabular}}{{l@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c@{{\\hspace{{1em}}}}c}}
\\toprule
Model & Normal & Blurred & Dyslexify & Both \\\\
\\midrule
"""

    tab_footer = f"""\\bottomrule
\\end{{tabular}}
\\label{{tab:blur_typographic}}
\\end{{table}}"""

    # First, collect all accuracies per model
    model_accuracies = {}
    for model in ["B", "L", "H", "G", "Big-G"]:
        if model in results:
            accuracies = []
            # Normal
            if "ImageNet100" in results[model]:
                accuracies.append(
                    ("Normal", results[model]["ImageNet100"]["typo_accuracy"] * 100)
                )
            # Blurred
            if "ImageNet100Blurred" in results[model]:
                accuracies.append(
                    (
                        "Blurred",
                        results[model]["ImageNet100Blurred"]["typo_accuracy"] * 100,
                    )
                )
            # Dyslexify
            if "ImageNet100" in results[model]:
                accuracies.append(
                    (
                        "Dyslexify",
                        results[model]["ImageNet100"]["dislexified_typo_accuracy"]
                        * 100,
                    )
                )
            # Both
            if "ImageNet100Blurred" in results[model]:
                accuracies.append(
                    (
                        "Both",
                        results[model]["ImageNet100Blurred"][
                            "dislexified_typo_accuracy"
                        ]
                        * 100,
                    )
                )

            # Sort by accuracy to find best and second best
            sorted_accuracies = sorted(accuracies, key=lambda x: x[1], reverse=True)
            model_accuracies[model] = {
                "best": sorted_accuracies[0][0] if len(sorted_accuracies) > 0 else None,
                "second_best": (
                    sorted_accuracies[1][0] if len(sorted_accuracies) > 1 else None
                ),
            }

    tab_body = ""

    # Generate rows for each model
    for model in ["B", "L", "H", "G", "Big-G"]:
        tab_body += f"{model} & "

        # Normal
        if model in results and "ImageNet100" in results[model]:
            accuracy = results[model]["ImageNet100"]["typo_accuracy"] * 100
            if (
                model in model_accuracies
                and model_accuracies[model]["best"] == "Normal"
            ):
                tab_body += f"\\textbf{{{accuracy:.2f}}} & "
            elif (
                model in model_accuracies
                and model_accuracies[model]["second_best"] == "Normal"
            ):
                tab_body += f"\\underline{{{accuracy:.2f}}} & "
            else:
                tab_body += f"{accuracy:.2f} & "
        else:
            tab_body += "N/A & "

        # Blurred
        if model in results and "ImageNet100Blurred" in results[model]:
            accuracy = results[model]["ImageNet100Blurred"]["typo_accuracy"] * 100
            if (
                model in model_accuracies
                and model_accuracies[model]["best"] == "Blurred"
            ):
                tab_body += f"\\textbf{{{accuracy:.2f}}} & "
            elif (
                model in model_accuracies
                and model_accuracies[model]["second_best"] == "Blurred"
            ):
                tab_body += f"\\underline{{{accuracy:.2f}}} & "
            else:
                tab_body += f"{accuracy:.2f} & "
        else:
            tab_body += "N/A & "

        # Dyslexify
        if model in results and "ImageNet100" in results[model]:
            accuracy = results[model]["ImageNet100"]["dislexified_typo_accuracy"] * 100
            if (
                model in model_accuracies
                and model_accuracies[model]["best"] == "Dyslexify"
            ):
                tab_body += f"\\textbf{{{accuracy:.2f}}} & "
            elif (
                model in model_accuracies
                and model_accuracies[model]["second_best"] == "Dyslexify"
            ):
                tab_body += f"\\underline{{{accuracy:.2f}}} & "
            else:
                tab_body += f"{accuracy:.2f} & "
        else:
            tab_body += "N/A & "

        # Both
        if model in results and "ImageNet100Blurred" in results[model]:
            accuracy = (
                results[model]["ImageNet100Blurred"]["dislexified_typo_accuracy"] * 100
            )
            if model in model_accuracies and model_accuracies[model]["best"] == "Both":
                tab_body += f"\\textbf{{{accuracy:.2f}}} & "
            elif (
                model in model_accuracies
                and model_accuracies[model]["second_best"] == "Both"
            ):
                tab_body += f"\\underline{{{accuracy:.2f}}} & "
            else:
                tab_body += f"{accuracy:.2f} & "
        else:
            tab_body += "N/A & "

        tab_body = tab_body[:-3] + " \\\\\n"

    tab_table = tab_header + tab_body + tab_footer

    output_path = Path("results/experiments/eval/tables/blur_typographic.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tab_table)

    print("Generating blur typographic table")
    print(tab_table)
    print(f"Saved to {output_path}")

    return tab_table


if __name__ == "__main__":
    results = {}

    # Load available result files
    result_files = {
        "B": "results/experiments/eval/vit-b/results_cls.json",
        "L": "results/experiments/eval/vit-l/results_cls.json",
        "H": "results/experiments/eval/vit-h/results_cls.json",
        "G": "results/experiments/eval/vit-g/results_cls.json",
        "Big-G": "results/experiments/eval/vit-big-g/results_cls.json",
    }

    for model, filepath in result_files.items():
        if Path(filepath).exists():
            results[model] = json.load(open(filepath))
        else:
            print(f"Warning: {filepath} not found, skipping {model} model")

    generate_blur_nontypographic_table(results)
    generate_blur_typographic_table(results)
