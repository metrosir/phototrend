from importlib.util import find_spec
import torch


def get_module_version(module_name):
    from importlib.metadata import version
    try:
        module_version = version(module_name)
    except Exception:
        module_version = None
    return module_version


def diffusers_enable_cpu_offload():

    if (find_spec("diffusers") is not None and compare_module_version("diffusers", "0.15.0") >= 0 and
            find_spec("accelerate") is not None and compare_module_version("accelerate", "0.17.0") >= 0 and
            torch.cuda.is_available()):
        return True
    else:
        return False


def compare_version(version1, version2):
    from packaging.version import parse
    if not isinstance(version1, str) or not isinstance(version2, str):
        return None

    if parse(version1) > parse(version2):
        return 1
    elif parse(version1) < parse(version2):
        return -1
    else:
        return 0


def compare_module_version(module_name, version_string):
    module_version = get_module_version(module_name)

    result = compare_version(module_version, version_string)
    return result if result is not None else -2


def torch_mps_is_available():
    if compare_module_version("torch", "2.0.1") < 0:
        if not getattr(torch, "has_mps", False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False
    else:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()