import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_path = os.path.join(project_dir, "models")

path_dirs = [
        (os.path.join(project_dir, 'repositories/BLIP'), 'models/blip.py', 'BLIP', [])
    ]

paths = {}

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(project_dir, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        sys.path.append(d)
        paths[what] = d
def is_torch2_available():
    import torch.nn.functional as f
    return hasattr(f, "scaled_dot_product_attention")
