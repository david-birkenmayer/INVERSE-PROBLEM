"""Crop evaluation PNGs and store them under old/viz/<wdn>.

Usage:
  python old/compactify.py --wdn Anytown
  python old/compactify.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "old" / "data"
VIZ_DIR = ROOT / "old" / "viz"


def _non_white_bbox(image: Image.Image, threshold: int = 245):
	if image.mode != "RGB":
		image = image.convert("RGB")
	width, height = image.size
	pixels = image.load()

	min_x, min_y = width, height
	max_x, max_y = -1, -1

	for y in range(height):
		for x in range(width):
			r, g, b = pixels[x, y]
			if r < threshold or g < threshold or b < threshold:
				if x < min_x:
					min_x = x
				if y < min_y:
					min_y = y
				if x > max_x:
					max_x = x
				if y > max_y:
					max_y = y

	if max_x < min_x or max_y < min_y:
		return None
	return (min_x, min_y, max_x + 1, max_y + 1)


def crop_image(src_path: Path, dst_path: Path, threshold: int = 245) -> bool:
	image = Image.open(src_path)
	bbox = _non_white_bbox(image, threshold=threshold)
	if bbox is None:
		return False
	cropped = image.crop(bbox)
	dst_path.parent.mkdir(parents=True, exist_ok=True)
	cropped.save(dst_path)
	return True


def process_wdn(wdn: str, threshold: int = 245) -> int:
	src_dir = DATA_DIR / wdn / "evaluation"
	if not src_dir.exists():
		return 0

	dst_dir = VIZ_DIR / wdn
	count = 0
	for png_path in sorted(src_dir.glob("*.png")):
		dst_path = dst_dir / png_path.name
		if crop_image(png_path, dst_path, threshold=threshold):
			count += 1
	return count


def main() -> int:
	parser = argparse.ArgumentParser(description="Crop evaluation PNGs into old/viz/<wdn>.")
	parser.add_argument("--wdn", help="Network name to process")
	parser.add_argument("--all", action="store_true", help="Process all networks under old/data")
	parser.add_argument("--threshold", type=int, default=245, help="White threshold (0-255)")
	args = parser.parse_args()

	if not args.wdn and not args.all:
		parser.error("Specify --wdn or --all")

	total = 0
	if args.all:
		for wdn_dir in sorted(DATA_DIR.iterdir()):
			if not wdn_dir.is_dir():
				continue
			total += process_wdn(wdn_dir.name, threshold=args.threshold)
	else:
		total = process_wdn(args.wdn, threshold=args.threshold)

	print(f"Cropped {total} image(s).")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
