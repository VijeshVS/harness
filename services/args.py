import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=False)
    parser.add_argument("--agent", type=str, required=False)
    return parser.parse_args()
