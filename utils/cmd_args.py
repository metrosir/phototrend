import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--sd_host', type=str, default='http://127.0.0.1:7860', help='server host')
parser.add_argument('--ip', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=7777)
parser.add_argument("--xformers", action='store_true', help="enable xformers for cross attention layers")
parser.add_argument('--setup_mode', action='store_true', help='setup mode')
# xformers_available

if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    opts = parser.parse_args()
else:
    opts, _ = parser.parse_known_args()