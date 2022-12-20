def sNorm(a: float, b:float):
  return max(0, a+b-1)

def tNorm(a: float, b: float):
  return min(1, a+b)

def triangle(a, t, b):
  def f(x: float):
    if x < a:
      return 0
    elif x < t:
      return 1/(t-a)*(x-a)
    elif x < b:
      return -1/(b-t)*(x-b)
    else:
      return 0

  return f

def trapezeL(a, t):
  def f(x: float):
    if x < a:
      return 0
    elif x < t:
      return 1/(t-a)*(x-a)
    else:
      return 1

  return f

def trapezeR(t, b):
  def f(x: float):
    if x < t:
      return 1
    elif x < b:
      return -1/(b-t)*(x-b)
    else:
      return 0

  return f
