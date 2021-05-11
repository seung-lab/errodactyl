"""Worker script for errodactyl - pulls tasks from the supplied task queue URL"""
import argparse
from errodactyl import worker


parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("queueurl")
parser.add_argument("--lease_seconds", type=int, default=300)

worker.run(**vars(parser.parse_args()))
