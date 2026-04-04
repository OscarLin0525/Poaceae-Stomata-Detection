import json
import sys
from pathlib import Path

from ultralytics import YOLO

# Allow running directly from outputs/ without manually setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from mtkd_framework.models.mtkd_model_v2 import build_mtkd_model_v2
from mtkd_framework.utils.helpers import load_checkpoint


def main() -> None:
	exp_dir = Path("/home/oscar/Poaceae-Stomata-Detection/outputs/mtkd_smoketest")
	ckpt_path = exp_dir / "best_model.pth"
	cfg_path = exp_dir / "config.json"
	out_path = exp_dir / "student_best.pt"

	cfg = json.loads(cfg_path.read_text())
	model = build_mtkd_model_v2(cfg["model"])

	# 1) Load MTKD checkpoint (state_dict-based)
	load_checkpoint(model, str(ckpt_path), strict=False, map_location="cpu")

	# 2) Export the student branch as Ultralytics-compatible .pt
	num_classes = cfg["model"].get("num_classes", 3)
	class_names = {i: f"class_{i}" for i in range(num_classes)}
	model.student.export_ultralytics_pt(
		save_path=str(out_path),
		num_classes=num_classes,
		class_names=class_names,
	)

	# 3) Verify the exported file can be loaded by Ultralytics
	yolo_model = YOLO(str(out_path))
	print("exported:", out_path)
	print("loaded:", out_path)
	print("nc:", yolo_model.model.nc)


if __name__ == "__main__":
	main()