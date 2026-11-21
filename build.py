import subprocess
import sys

subprocess.run(['cmake', '-S', '.', '-B', 'build'], check=True)
subprocess.run(['cmake', '--build', 'build', '--parallel'], check=True)
