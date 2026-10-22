import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=False)
    parser.add_argument("--agent", type=str, required=False)
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use the plain terminal loop instead of the TUI.",
    )
    return parser.parse_args()
