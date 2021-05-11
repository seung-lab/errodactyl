"""General utility functions"""
import time
import functools
import torch
import cloudvolume


def bboxatpoint(point, boxwidth):
    """Creates a cloudvolume Bbox centered at the given point"""
    pt = cloudvolume.lib.Vec(*point)

    return cloudvolume.lib.Bbox(pt - boxwidth, pt + boxwidth)


def timed(func):
    """Decorator for timing functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{func.__name__}")
        start = time.time()
        returnvals = func(*args, **kwargs)
        print(f"{func.__name__} finished in {time.time() - start:.3f}s")

        return returnvals

    return wrapper


@timed
def inference(model, *inputdata):
    """A small wrapper around inference for timing"""
    with torch.no_grad():
        return model(*inputdata)
