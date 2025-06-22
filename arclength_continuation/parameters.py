# MIT License
#
# Copyright (c) 2025 Raymond Langer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass

DEFAULTS = {
    "delta_s": 0.2,
    "n_continuation": 30,
    "tol": 1.0e-08,
    "max_iter": 20,
    "y": 0.6,
    "x": 0.6,
    "predictor": True
}


@dataclass
class ContinuationParameters:
  """
  Data class that encapsulates all the relevant parameters
  for pseudo-arclength continuation.
  
  Attributes:
      delta_s: Step size (positive -> clockwise, negative -> counterclockwise)
      n_continuation: Number of arc-length steps
      tol: Newton method convergence tolerance
      max_iter: Maximum iterations for Newton's method
      predictor: Whether to use a predictor step in continuation
  """
  delta_s: float = DEFAULTS["delta_s"]
  n_continuation: int = DEFAULTS["n_continuation"]
  tol: float = DEFAULTS["tol"]
  max_iter: int = DEFAULTS["max_iter"]
  predictor: bool = DEFAULTS["predictor"]

  @classmethod
  def from_args(cls, args):
    return cls(delta_s=args.delta_s
               if args.delta_s is not None else DEFAULTS['delta_s'],
               n_continuation=args.n_continuation if args.n_continuation
               is not None else DEFAULTS['n_continuation'],
               tol=args.tol if args.tol is not None else DEFAULTS['tol'],
               max_iter=args.max_iter
               if args.max_iter is not None else DEFAULTS['max_iter'],
               predictor=args.predictor
               if args.predictor is not None else DEFAULTS['predictor'])

  def __str__(self) -> str:
    return ("ContinuationParameters:\n"
            f"  delta_s        = {self.delta_s}\n"
            f"  n_continuation = {self.n_continuation}\n"
            f"  tol            = {self.tol}\n"
            f"  max_iter       = {self.max_iter}\n"
            f"  predictor      = {self.predictor}")
