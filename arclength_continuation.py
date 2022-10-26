#!/usr/bin/env python

# This program numerically computes points on the unit circle using Newton's
# method embedded within a pseudo-arclength continuation [1]. The continuation
# proceeds through points where the derivative of the solved function is
# singular. The unit circle,
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

# MIT License
#
# Copyright (c) 2022 Raymond Langer
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

import numpy as np
import matplotlib.pyplot as plt

##
## Input begin
##

# The step size of the continuation steps:
# - delta_s > 0.0 means clockwise
# - delta_s < 0.0 means counterclockwise
delta_s = 0.2
assert (abs(delta_s) < 1.0)
assert (abs(delta_s) != 0.0)

# The number of steps to be taken along the arc.
n_continuation = 30
assert (n_continuation > 0)

# If the 2-norm of the newton step vector is smaller than
# the tolerance "tol" the solution is considered converged.
tol = 1.0e-08
assert (tol > 0.0)
assert (tol < 0.5)

# The maximum number of iterations before Newton's method is considered to have
# diverged.
max_iter = 20
assert (max_iter > 0)
assert (max_iter < 1000)

# The starting values:
#T = -0.99996666
#a = -0.00816578

T = 0.6
#T = 1.0 / np.sqrt(2.0)
assert (abs(T) <= 1.1)
#a = np.sqrt(1.0 - T * T)
a = 0.6
assert (abs(a) <= 1.0)

ref_solution_color = "gray"
light_blue = "#72a0c1"
converged_color = light_blue
diverged_color = "red"

# Use a predictor step that improves the initial guess of every
# continuation step with only one extra solve step (usually a back
# substitution, but here the 2x2 matrix is inverted analytically)
predictor = True

##
## Input end
##

# square plot
figure, axes = plt.subplots(1)
axes.set_aspect(1)
# Show the initial guess as a dot.
axes.scatter(a, T, marker='o', color=converged_color, label='Initial guess')


# p means (p)revious

# See def A() for an explanation
def Ainv(T, a, Tp, ap):
  det = 2.0 * (T * Tp + a * ap)
  return 1.0 / det * np.array([[Tp, -2.0 * a], [ap, 2.0 * T]])

# A is the Jacobian of the system 
# H = (G, N)^T = 0 (^T means transposed)
#
# A = d H / d(x,y)
# 
# x and y (a and T in the source code) are both considered to be  
# functions of the curve parameter.
#
# A is not used; This is only for documentation and testing purposes
def A(T, a, Tp, ap):
  return np.array([[2.0 * T, 2.0 * a], [-ap, Tp]])


# The nonlinear equation G = 0 to be solved.
def G(T, a):
  return T * T + a * a - 1.0


def Tangent(solution):
  # index 0 is the a-direction
  # index 1 is the T-direction
  T = solution[0]
  a = solution[1]
  return np.array([T, -a])


# The solution is advanced from sp to s along the solution curve in the unit 
# tangent t=t(s0) direction. The normalization condition N = 0 constrains the 
# arclength between sp and s. In other words, the normalization condition must 
# be satisfied to achieve the desired step size. However, it is not required 
# to constrain the exact arclength. Instead, it suffices to constrain an 
# approximation of the arclength, hence the term pseudo-arclength. Here, we 
# project the vector (s-sp) onto the tangent t and constrain the projection 
# vector's length to delta_s: 
# N = t^T * (s-sp) - delta_s = 0 (^T means transposed, so it is a dot product)
def N(T, a, Tp, ap, ds):
  tan = Tangent(np.array([Tp, ap]))
  tan_in_A_direction = tan[0]
  tan_in_T_direction = tan[1]
  return tan_in_A_direction * (a - ap) + tan_in_T_direction * (T - Tp) - ds


##
## Tests begin
##

# G must be close to zero
assert (abs(G(0.9995498987, 0.03)) < 1.0e-06)

# N = 0 must be satisfied by the selected points. The first point is above
# the x-axis at y = 0.03. The second point is below the axis at y = -0.03.
a_t = 0.9995498987
T_t = 0.03
assert (abs(N(-T_t, a_t, T_t, a_t, 0.059972993922)) < 1.0e-06)

##
## Tests end
##

T_init = T
a_init = a
n_iter = 0

#
# Execute Newton's method to find the first solution.
#
while (np.linalg.norm(G(T, a)) > tol and max_iter > n_iter):
  n_iter += 1
  if (abs(T) < 1.0e-10):
    print("Invalid inital guess: T_init =", T_init, " a = ", a_init,
          " converges to a solution that is too close to T = 0.0.")
    quit()
  dGdT = 2.0 * T
  T -= G(T, a) / dGdT

# Check whether the initial point is sufficiently close to the circle.
res = np.linalg.norm(G(T, a))
if (res < tol):
  print(
      "First converged solution: |G(T = " + "{:.2f}".format(T) + ", a = " +
      "{:.2f}".format(a) + ")| =", "{:.2e}".format(res))
else:
  print("Initial guess (T_init = " + "{:.2f}".format(T_init) + ", a = " +
        "{:.2f}".format(a_init) + ") diverged: |G(T = " + "{:.2f}".format(T) +
        ", a = " + "{:.2f}".format(a) + ")| = " + "{:.4f}".format(res) +
        ". This is unexpected.")
  quit()

# Highlight the first converged solution with a bold cross
axes.scatter(a, T, marker='X', color=converged_color, label='First solution')

# Draw the tangent arrow that indicates the direction
# of the continuation.
axes.annotate("",
              xy=(a + 0.5 * delta_s / abs(delta_s) * T,
                  T - 0.5 * delta_s / abs(delta_s) * a),
              xytext=(a, T),
              arrowprops=dict(arrowstyle="->", color=converged_color))
a_history = []
T_history = []

# s means (s)olution
s = np.array([T, a])

title = "$\\Delta s = $" + "{:.2e}".format(delta_s)
if (delta_s > 0.0):
  title += " (clockwise)"
else:
  title += " (counterclockwise)"

plt.title(title)
plt.xlabel("$x$")
plt.ylabel("$y$")

#
# Plot the reference solution
#
axes.add_patch(
    plt.Circle((0.0, 0.0), 1.0, color=ref_solution_color, fill=False))

#
# Start the pseudo-arclength continuation
#
sp = np.copy(s)
for c in range(n_continuation):
  # initialize the Newton iteration
  n_iter = 0
  # defining Ai here is only required to extent the scope of
  # the variable Ai to the predictor step below
  Ai = [[0.0, 0.0], [0.0, 0.0]]
  # test_stop > tol such that at least one iteration is executed
  test_stop = 10.0 * tol
  # Newton's method
  while (np.linalg.norm(test_stop) > tol and n_iter < max_iter):
    n_iter += 1
    Ai = Ainv(s[0], s[1], sp[0], sp[1])
    rhs = np.array([G(s[0], s[1]), N(s[0], s[1], sp[0], sp[1], delta_s)])
    step = np.matmul(-Ai, rhs)
    s += step
    test_stop = np.linalg.norm(step)

  # summarize what happened
  if (np.linalg.norm(test_stop) < tol):
    print("step " + "{:4d}".format(c + 1) + ": It =" + "{:2d}".format(n_iter) +
          "     |step| = " + "{:.2e}".format(np.linalg.norm(step)) +
          "     |G| = " + "{:.2e}".format(np.linalg.norm(G(s[0], s[1]))) +
          " (successful)")
    T_history.append(s[0])
    a_history.append(s[1])
    sp = np.copy(s)
    if (predictor):
      predictor_step = np.matmul(Ai, np.array([0.0, 1.0]))
      predictor_step = predictor_step / np.linalg.norm(predictor_step)
      s = np.add(s, predictor_step * delta_s)
    if (n_continuation == c + 1):
      print("All " + str(n_continuation) + " steps were successful.")
  else:
    print("step " + "{:4d}".format(c + 1) +
          " FAILED to find a converged solution: It =" +
          "{:2d}".format(n_iter) + "     |step| = " +
          "{:.2e}".format(np.linalg.norm(step)) + "     |G| = " +
          "{:.2e}".format(np.linalg.norm(G(s[0], s[1]))))
    axes.scatter(s[1],
                 s[0],
                 marker="x",
                 linestyle='',
                 color=diverged_color,
                 s=50,
                 label="diverged solution")
    # Not expected.
    # Stop here because the solution is crap now.
    # It is unclear why Newton's method failed.
    break

plt.grid()
plt.plot(a_history,
         T_history,
         marker='+',
         markersize=8,
         linestyle='',
         color=converged_color,
         label=str(len(a_history)) + ' solutions from the continuation')
plt.legend(loc='center')
plt.show()
