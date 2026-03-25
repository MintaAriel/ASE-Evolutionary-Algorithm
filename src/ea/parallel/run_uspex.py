from pygulp.relaxation.relax import Gulp_relaxation_noadd
import subprocess
import time
import os

NPROCMAX = 14
CODE = "gulp64"

class ParallelGULP:
    def __init__(self):
        self.optimizer = Gulp_relaxation_noadd()
        print(self.optimizer)




def count_running_jobs():
    result = subprocess.run(
        ["pgrep", "-f", CODE],
        capture_output=True,
        text=True
    )
    if result.stdout.strip() == "":
        return 0
    return len(result.stdout.strip().split("\n"))


def run_uspex():
    subprocess.run(["USPEX", "-r"])


def is_done():
    return os.path.exists("USPEX_IS_DONE")


while not is_done():

    run_uspex()

    time.sleep(5)

    while count_running_jobs() > NPROCMAX:
        time.sleep(5)