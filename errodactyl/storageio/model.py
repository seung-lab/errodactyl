"""Cached IO for PyTorch models

The cache stores each model along with a timestamp indicating the
last time it was requested.
"""
import io
import torch
import datetime
from . import storageio
from ..utils import timed


MODELCACHE = dict()
CACHELIMIT = datetime.timedelta(hours=1)


@timed
def readmodel(modelfilename, chkptfilename):
    """Reading a pytorch model from storage"""
    model = checkcache(modelfilename, chkptfilename)
    if model is not None:
        return model

    model = storageio.readpymodule(modelfilename, "model").InferenceModel

    chkptbytes = readchkpt(chkptfilename)
    model.load_state_dict(torch.load(chkptbytes))

    cache(model, modelfilename, chkptfilename)

    return model


def readchkpt(filename):
    """
    Reads a checkpoint from a file and returns the contents
    as a binary stream
    """
    rawbytes = storageio._readfile(filename, raw=False, decompress=False)
    buff = io.BytesIO(rawbytes)

    return buff


def cache(model, modelfilename, chkptfilename):
    """Adds a model to the cache"""
    key = (modelfilename, chkptfilename)
    MODELCACHE[key] = (model, datetime.datetime.now())


def checkcache(modelfilename, chkptfilename):
    """Checks whether a model is in the cache, specified by filenames"""
    key = (modelfilename, chkptfilename)

    removestalemodels()

    if key in MODELCACHE:
        model, cachetime = MODELCACHE[key]
        # refresh cache timestamp
        cache(model, modelfilename, chkptfilename)
        return model
    else:
        return None


def removestalemodels():
    now = datetime.datetime.now()
    for (k, (model, storagetime)) in MODELCACHE.items():
        if now - storagetime >= CACHELIMIT:
            MODELCACHE.pop(k)


def clearcache():
    """Clears the cache and allows models to be garbage collected"""
    MODELCACHE.clear()
