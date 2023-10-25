import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--sd_host', type=str, default='http://127.0.0.1:7860', help='server host')
parser.add_argument('--ip', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=7777)


if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    opts = parser.parse_args()
else:
    opts, _ = parser.parse_known_args()