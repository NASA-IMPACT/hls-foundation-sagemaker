# copied over from https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/multi_model_bring_your_own/container/dockerd-entrypoint.py
import os
import shlex
import subprocess
import sys
from subprocess import CalledProcessError

from retrying import retry
from sagemaker_inference import model_server


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


@retry(stop_max_delay=1000 * 50, retry_on_exception=_retry_if_error)
def _start_service():
    # by default the number of workers per model is 1, but we can configure it through the
    # environment variable below if desired.
    # os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = '2'
    model_server.start_model_server(handler_service="/home/model-server/infer.py:handle")


def main():
    if sys.argv[1] == "serve":
        print("!!!!", sys.argv)
        _start_service()
    else:
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

    # prevent docker exit
    subprocess.call(["tail", "-f", "/dev/null"])

main()
