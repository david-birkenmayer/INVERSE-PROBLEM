import hashlib
import json
import os
from typing import Dict


def compute_hash(payload: Dict[str, object]) -> str:
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def _normalize_path(p: str, root: str) -> str:
	"""Return *p* as a path relative to *root*.

	If *p* is absolute:
	  - return it as-is if it still exists (same machine).
	  - otherwise try to derive a relative path by stripping the root prefix
	    (handles moves to a different machine / directory).
	If *p* is already relative, return it unchanged.
	"""
	if not os.path.isabs(p):
		return p
	if os.path.exists(p):
		# On the same machine — convert to relative so future copies work.
		try:
			return os.path.relpath(p, root)
		except ValueError:
			return p
	# Different machine or moved directory: strip any leading root-like prefix
	# and return the remaining path as relative.
	try:
		rel = os.path.relpath(p, root)
		# relpath may produce ../../… paths for unrelated roots; skip those.
		if not rel.startswith(".."):
			return rel
	except ValueError:
		pass
	return p


def load_index(path: str, root: str = "") -> Dict[str, str]:
	if not os.path.exists(path):
		return {}
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		return {}
	if not root:
		return {str(k): str(v) for k, v in data.items()}
	return {str(k): _normalize_path(str(v), root) for k, v in data.items()}


def save_index(path: str, index: Dict[str, str], root: str = "") -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	if root:
		index = {k: _normalize_path(v, root) for k, v in index.items()}
	with open(path, "w", encoding="utf-8") as f:
		json.dump(index, f, indent=2, sort_keys=True)
