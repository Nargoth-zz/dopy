import time

def timing(f):
  def wrap(*args):
    time1 = time.time()
    ret = f(*args)
    time2 = time.time()
    print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
    return ret
  return wrap


from contextlib import contextmanager

@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] took {} ms'.format(name, int(elapsedTime * 1000)))
