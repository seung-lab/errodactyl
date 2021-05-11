"""Task Generators"""
import inspect
import functools

from .. import tasks


def ptgenerator(func):
    """Creates a generator that iterates over points to create tasks"""

    def generator(pts, *args, **kwargs):
        sig = inspect.signature(func)

        for pt in pts:
            checkargs(sig, *args, pt=pt, **kwargs)
            curried = functools.partial(func, *args, pt=pt, **kwargs)
            yield(curried)

    return generator


def checkargs(signature, *args, **kwargs):
    """Checks whether arguments satisfy a function's signature"""
    try:
        signature.bind(*args, **kwargs)

    except TypeError:
        raise Exception(f"invalid args: {args} {kwargs}")


gensimpleyacn = ptgenerator(tasks.simpleyacn)
