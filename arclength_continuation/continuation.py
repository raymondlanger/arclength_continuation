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

# This module implements the routines to numerically computes points on the
# unit circle using Newton's method embedded within a pseudo-arclength
# continuation [1]. The continuation proceeds through points where the
# derivative of the solved function is singular. The unit circle,
#
#    G(y,x) = y^2 + x^2 - 1 = 0,
#
# has such points at y = 0, where x is considered a parameter. In contrast to the
# pseudo-arclength continuation, Newton's method would fail at these points in a
# natural parameter continuation. No solution exists for |x| > 1.0, and attempts
# to find a solution for such values of x would also result in converge failures
# in natural continuations.

# [1] Keller, H.B., Numerical solution of bifurcation and nonlinear eigenvalue
#     problems) in Applications of Bifurcation Theory (P. Rabinowitz, ed.)
#     Academic Press, New York (1977) 359-384.

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arclength_continuation.args import parse_args
from arclength_continuation.parameters import ContinuationParameters

COLORS = {
    "reference": "gray",
    # light_blue
    "converged": "#72a0c1",
    "diverged": "red"
}


# See def A() for an explanation
def Ainv(y: float, x: float, yp: float, xp: float) -> NDArray[np.float64]:
  det = 2.0 * (y * yp + x * xp)
  return 1.0 / det * np.array([[yp, -2.0 * x], [xp, 2.0 * y]])


# A is the Jacobian of the system
# H = (G, N)^T = 0 (^T means transposed)
#
# A = d H / d(x,y)
#
# x and y are both considered to be
# functions of the curve parameter.
#
# A is not used; This is only for documentation and testing purposes
def A(y: float, x: float, yp: float, xp: float) -> NDArray[np.float64]:
  return np.array([[2.0 * y, 2.0 * x], [-xp, yp]])


# The nonlinear equation G = 0 to be solved.
def G(y: float, x: float) -> float:
  """
  Return the value of y^2 + x^2 - 1, for the current guess (y, x).
  Must be zero for points on the unit circle.
  """
  return y * y + x * x - 1.0


def tangent(solution: NDArray[np.float64]) -> NDArray[np.float64]:
  """
  Return the 2D tangent vector as a NumPy array.

  """
  # index 0 is the y-direction
  # index 1 is the x-direction
  y = solution[0]
  x = solution[1]
  return np.array([y, -x], dtype=np.float64)


def N(y: float, x: float, yp: float, xp: float, ds: float) -> float:
  """
  TDLR:
  Pseudo-arclength normalization condition:
  N = t^T * (s - sp) - delta_s = 0
  (where t is the tangent vector).

  :param y, x: Current solution guess
  :param yp, xp: Previous solution guess
  :param ds: Step size
  :return: Value of the normalization condition

  Full explanation:
  The solution is advanced from sp to s along the solution curve in the unit
  tangent t=t(s0) direction. The normalization condition N = 0 constrains the
  arclength between sp and s. In other words, the normalization condition must
  be satisfied to achieve the desired step size. However, it is not required
  to constrain the exact arclength. Instead, it suffices to constrain an
  approximation of the arclength, hence the term pseudo-arclength. Here, we
  project the vector (s-sp) onto the tangent t and constrain the projection
  vector's length to delta_s:
  N = t^T * (s-sp) - delta_s = 0 (^T means transposed, so it is a dot product)

  """
  tan = tangent(np.array([yp, xp]))
  tan_in_A_direction = tan[0]
  tan_in_T_direction = tan[1]
  return tan_in_A_direction * (x - xp) + tan_in_T_direction * (y - yp) - ds


##
## Tests begin
##

# G must be close to zero
assert (abs(G(0.9995498987, 0.03)) < 1.0e-06)

# N = 0 must be satisfied by the selected points. The first point is above
# the x-axis at y = 0.03. The second point is below the axis at y = -0.03.
x_t = 0.9995498987
y_t = 0.03
assert (abs(N(-y_t, x_t, y_t, x_t, 0.059972993922)) < 1.0e-06)

##
## Tests end
##


def compute_first_solution(
    x: float, y: float,
    params: ContinuationParameters) -> Tuple[float, float, float]:

  y_init = y
  x_init = x
  n_iter = 0
  #
  # Execute Newton's method to find the first solution.
  #
  while (np.linalg.norm(G(y, x)) > params.tol and params.max_iter > n_iter):
    n_iter += 1
    if (abs(y) < 1.0e-10):
      raise RuntimeError(
          f"Invalid initial guess y={y_init}, x={x_init}, too close to y=0.0.")
    dGdy = 2.0 * y
    y -= G(y, x) / dGdy

  # Check whether the initial point is sufficiently close to the circle.
  res = np.linalg.norm(G(y, x))
  return x, y, res


def plot_first_solution(x: float, y: float,
                        delta_s: float) -> Tuple[Axes, Figure]:
  """
  Plot the first converged solution on a square axes figure.

  :param x: x-coordinate of the solution
  :param y: y-coordinate of the solution
  :param delta_s: step size (used to draw a tangent arrow)
  :return: (ax, fig) where ax is the Matplotlib Axes and fig is the Figure
  """
  # square plot
  fig, ax = plt.subplots(1)
  ax.set_aspect(1)
  # Show the initial guess as a dot.
  ax.scatter(x,
             y,
             marker='o',
             color=COLORS["converged"],
             label='Initial guess')

  # Highlight the first converged solution with a bold cross
  ax.scatter(x,
             y,
             marker='X',
             color=COLORS["converged"],
             label='First solution')

  # Draw the tangent arrow that indicates the direction
  # of the continuation.
  ax.annotate("",
              xy=(x + 0.5 * delta_s / abs(delta_s) * y,
                  y - 0.5 * delta_s / abs(delta_s) * x),
              xytext=(x, y),
              arrowprops=dict(arrowstyle="->", color=COLORS["converged"]))
  return ax, fig


def pseudo_arclength_continuation(
    x: float, y: float, ax: Axes,
    params: ContinuationParameters) -> List[Tuple[float, float]]:
  # p means (p)revious
  solutions = []

  # s means (s)olution
  s = np.array([y, x])

  #
  # Start the pseudo-arclength continuation
  #
  sp = np.copy(s)
  for c in range(params.n_continuation):
    # initialize the Newton iteration
    n_iter = 0
    # defining Ai here is only required to extent the scope of
    # the variable Ai to the predictor step below
    Ai = [[0.0, 0.0], [0.0, 0.0]]
    # test_stop > tol such that at least one iteration is executed
    test_stop = 10.0 * params.tol
    # Newton's method
    while (np.linalg.norm(test_stop) > params.tol
           and n_iter < params.max_iter):
      n_iter += 1
      Ai = Ainv(s[0], s[1], sp[0], sp[1])
      rhs = np.array(
          [G(s[0], s[1]),
           N(s[0], s[1], sp[0], sp[1], params.delta_s)])
      step = np.matmul(-Ai, rhs)
      s += step
      test_stop = np.linalg.norm(step)

    # summarize what happened
    if (np.linalg.norm(test_stop) < params.tol):
      print("step " + "{:4d}".format(c + 1) + ": It =" +
            "{:2d}".format(n_iter) + "     |step| = " +
            "{:.2e}".format(np.linalg.norm(step)) + "     |G| = " +
            "{:.2e}".format(np.linalg.norm(G(s[0], s[1]))) + " (successful)")
      solutions.append((s[0], s[1]))
      sp = np.copy(s)
      if (params.predictor):
        predictor_step = np.matmul(Ai, np.array([0.0, 1.0]))
        predictor_step = predictor_step / np.linalg.norm(predictor_step)
        s = np.add(s, predictor_step * params.delta_s)
      if (params.n_continuation == c + 1):
        print("All " + str(params.n_continuation) + " steps were successful.")
    else:
      print("step " + "{:4d}".format(c + 1) +
            " FAILED to find a converged solution: It =" +
            "{:2d}".format(n_iter) + "     |step| = " +
            "{:.2e}".format(np.linalg.norm(step)) + "     |G| = " +
            "{:.2e}".format(np.linalg.norm(G(s[0], s[1]))))
      ax.scatter(s[1],
                 s[0],
                 marker="x",
                 color=COLORS["diverged"],
                 s=50,
                 label="diverged solution")
      # Not expected.
      # Stop here because the solution is crap now.
      # It is unclear why Newton's method failed.
      break

  return solutions


def plot_solutions(solutions: List[Tuple[float, float]]):
  """
  Plot all solutions from the continuation steps, if any.
  """
  if not solutions:
    # no solutions found
    return
  y_vals, x_vals = zip(*solutions)
  plt.plot(x_vals,
           y_vals,
           marker='+',
           markersize=8,
           linestyle='',
           color=COLORS["converged"],
           label=str(len(x_vals)) + ' solutions from the continuation')
  plt.legend(loc='center')


def run(params: ContinuationParameters, x: float, y: float):

  # Validate arguments if needed (like your original asserts)
  if not (abs(params.delta_s) < 1.0 and abs(params.delta_s) != 0.0):
    raise ValueError("delta_s must be nonzero and within (-1, 1).")
  if params.n_continuation <= 0:
    raise ValueError("n_continuation must be positive.")
  if not (0.0 < params.tol < 0.5):
    raise ValueError("tol must lie between 0.0 and 0.5.")
  if not (0 < params.max_iter < 1000):
    raise ValueError("max_iter must lie between 1 and 999.")
  if not (abs(y) <= 1.1):
    raise ValueError("|y| must be <= 1.1.")
  if not (abs(x) <= 1.0):
    raise ValueError("|x| must be <= 1.0.")

  print(params)
  print(f"Initial guess y = {y}")
  print(f"Initial guess x = {x}\n")

  x, y, res = compute_first_solution(x, y, params)
  if (res < params.tol):
    print(
        "First converged solution: |G(y = " + "{:.2f}".format(y) + ", x = " +
        "{:.2f}".format(x) + ")| =", "{:.2e}".format(res))
  else:
    raise RuntimeError("Initial guess (y_init = " + "{:.2f}".format(args.y) +
                       ", x = " + "{:.2f}".format(args.x) +
                       ") diverged: |G(y = " + "{:.2f}".format(y) + ", x = " +
                       "{:.2f}".format(x) + ")| = " + "{:.4f}".format(res) +
                       ". This is unexpected.")

  ax, fig = plot_first_solution(x, y, params.delta_s)

  solutions = pseudo_arclength_continuation(x, y, ax, params)

  title = "$\\Delta s = $" + "{:.2e}".format(params.delta_s)
  if (params.delta_s > 0.0):
    title += " (clockwise)"
  else:
    title += " (counterclockwise)"

  plt.title(title)
  plt.xlabel("$x$")
  plt.ylabel("$y$")

  #
  # Plot the reference solution
  #
  ax.add_patch(
      plt.Circle((0.0, 0.0), 1.0, color=COLORS["reference"], fill=False))

  plt.grid()

  plot_solutions(solutions)

  plt.show()
