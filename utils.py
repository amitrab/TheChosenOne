import logging
from functools import wraps
from time import monotonic

def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = monotonic()
    result = f(*args, **kwds)
    elapsed = monotonic() - start
    # TODO timedelta nice str
    logging.debug(f"{f.__name__} took {elapsed/1000:.3f} sec to finish")
    return result
  return wrapper

def timed_ctx(name):
    start = monotonic()
    yield
    elapsed = monotonic() - start
    # TODO timedelta nice str
    logging.debug(f"{name} took {elapsed/1000:.3f} sec to finish")

