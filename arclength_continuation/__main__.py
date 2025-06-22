import sys
from .args import parse_args
from .parameters import ContinuationParameters
from .continuation import run

def main():
    # Parse command-line arguments
    args = parse_args()

    # Create your parameter dataclass
    params = ContinuationParameters(
        delta_s=args.delta_s,
        n_continuation=args.n_continuation,
        tol=args.tol,
        max_iter=args.max_iter,
        predictor=args.predictor
    )

    # Run the continuation process
    run(params, x=args.x, y=args.y)

if __name__ == "__main__":
    main()