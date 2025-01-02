from . import generic
from . import selectors
from . import limiters


class Uniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0):
    if samples_per_insert:
      limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      limiter = limiters.MinSize(min_size)
    assert not capacity or min_size <= capacity
    super().__init__(
        length=length, # 64
        capacity=capacity, # 1e6/2e6
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiter, # MinSize
        directory=directory,
        online=online, # False
        chunks=chunks, # 1024
    )
