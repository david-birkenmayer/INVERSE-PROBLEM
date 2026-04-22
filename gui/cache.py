import hashlib
import json
import os
from typing import Dict


def compute_hash(payload: Dict[str, object]) -> str:
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def load_index(path: str) -> Dict[str, str]:
	if not os.path.exists(path):
		return {}
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if isinstance(data, dict):
		return {str(k): str(v) for k, v in data.items()}
	return {}


def save_index(path: str, index: Dict[str, str]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(index, f, indent=2, sort_keys=True)
