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

import argparse
from arclength_continuation.parameters import DEFAULTS


def parse_args():
  """
  Parse command-line arguments for the pseudo-arclength continuation script.

  :param argv: Optional list of arguments (default: None uses sys.argv[1:])
  :return: argparse.Namespace containing parsed flags
  """
  parser = argparse.ArgumentParser(
      description="Pseudo-arclength solution of a unit circle.")

  # Use defaults from parameters.DEFAULTS
  parser.add_argument(
      "--delta_s",
      "-d",
      type=float,
      default=DEFAULTS["delta_s"],
      help=
      "Step size of the continuation (negative -> counterclockwise, positive -> clockwise)."
  )

  parser.add_argument("--n_continuation",
                      "-n",
                      type=int,
                      default=DEFAULTS["n_continuation"],
                      help="Number of continuation steps.")

  parser.add_argument("--tol",
                      "-t",
                      type=float,
                      default=DEFAULTS["tol"],
                      help="Newton convergence tolerance.")

  parser.add_argument(
      "--max_iter",
      "-m",
      type=int,
      default=DEFAULTS["max_iter"],
      help="Maximum Newton iterations before divergence is declared.")

  parser.add_argument("--y",
                      "-y",
                      type=float,
                      default=DEFAULTS["y"],
                      help="Initial guess for y (y-coordinate).")

  parser.add_argument("--x",
                      "-x",
                      type=float,
                      default=DEFAULTS["x"],
                      help="Initial guess for x (x-coordinate).")

  # Here we add predictor with a default of True, but allow disabling it
  group = parser.add_mutually_exclusive_group()

  group.add_argument(
      "--predictor",
      "-p",
      dest="predictor",
      action="store_true",
      default=DEFAULTS["predictor"],
      help=
      "Use a predictor step that improves the initial guess of every continuation step with only one extra solve step (usually a back substitution, but here the 2x2 matrix is inverted analytically)."
  )

  group.add_argument("--no-predictor",
                     dest="predictor",
                     action="store_false",
                     help="Disable the predictor step.")

  return parser.parse_args()
