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
  delta_s: float
  n_continuation: int
  tol: float
  max_iter: int
  predictor: bool

  def __str__(self) -> str:
    return ("ContinuationParameters:\n"
            f"  delta_s        = {self.delta_s}\n"
            f"  n_continuation = {self.n_continuation}\n"
            f"  tol            = {self.tol}\n"
            f"  max_iter       = {self.max_iter}\n"
            f"  predictor      = {self.predictor}")
