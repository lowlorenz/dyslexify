import csv
import ast
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Any


class IIIT5K(Dataset):
    def __init__(
        self,
        root: str,
        csv_file: str = "testdata.csv",
        preprocess=None,
    ):
        self.root = Path(root)
        self.preprocess = preprocess
        self.samples = []

        csv_path = self.root / csv_file
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.root / "IIIT5K-Word_V3.0" / "IIIT5K" / row["ImgName"]
                ground_truth = row["GroundTruth"]
                small_lexi = ast.literal_eval(row["smallLexi"])
                medium_lexi = ast.literal_eval(row["mediumLexi"])

                self.samples.append(
                    {
                        "image_path": img_path,
                        "ground_truth": ground_truth,
                        "small_lexi": small_lexi,
                        "medium_lexi": medium_lexi,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, str, List[str], List[str]]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.preprocess is not None:
            image = self.preprocess(image)

        return (
            image,
            sample["ground_truth"],
            sample["small_lexi"],
            sample["medium_lexi"],
        )
