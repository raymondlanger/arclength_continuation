import sys
from arclength_continuation.args import parse_args
from arclength_continuation.parameters import ContinuationParameters, DEFAULTS
from arclength_continuation.continuation import run


def main():
  # Parse command-line arguments
  args = parse_args()

  # Create your parameter dataclass using the new from_args method.
  params = ContinuationParameters.from_args(args)

  # Run the continuation process
  run(params, x=args.x, y=args.y)


if __name__ == "__main__":
  main()
