"""Basic IO functions"""
import os
import sys
import json
import importlib
import numpy as np
import cloudfiles
import cloudvolume
from . import utils
from .model import readmodel
from ..utils import timed


NGLHEADERLINE = ("Coordinate 1,Coordinate 2,Ellipsoid Dimensions,"
                 "Tags,Description,Segment IDs,Parent ID,Type,ID\n")

# CONVENTION: functions that are not timed have an underscore.
# timed functions generally call untimed functions


@timed
def readbbox(cvpath, bbox, agglomerate=False,
             parallel=1, progress=False, volumes=0,
             resolution=(8, 8, 40), timestamp=None):
    """Main function for reading from cloudvolume"""
    if agglomerate:
        cv = cloudvolume.CloudVolume(cvpath, mip=resolution, agglomerate=True,
                                     parallel=parallel, progress=progress)
    else:
        cv = cloudvolume.CloudVolume(cvpath, mip=resolution,
                                     parallel=parallel, progress=progress)

    cv.fill_missing = True
    cv.bounded = False

    data = cv.download(bbox, timestamp=timestamp)

    return data[..., volumes].transpose((2, 1, 0))


def readmask(cvpath, segid, bbox,
             resolution=(8, 8, 40), timestamp=None):
    """Reading a segment mask from a CloudVolume"""
    seg = readbbox(cvpath, bbox, agglomerate=True,
                   resolution=resolution,
                   timestamp=timestamp)

    return seg == segid


@timed
def readfile(path, raw=False, decompress=False):
    """Pulls the contents of a file from storage"""
    return _readfile(path, raw=raw, decompress=decompress)


def _readfile(path, raw=False, decompress=False):
    """Reading a data file from object storage bypassing timing"""
    bucket, key = utils.splitpath(path)

    cf = cloudfiles.CloudFiles(bucket)
    binary = cf.get(key)

    if binary is None:
        raise ValueError(f"File not found: {path}")

    if decompress:
        return cloudfiles.compression.gunzip(binary)
    else:
        return binary


@timed
def readjson(path, raw=False, decompress=False):
    """Reads a json file from storage"""
    return _readjson(path, raw=raw, decompress=decompress)


def _readjson(path, raw=False, decompress=False):
    content = _readfile(path, raw=raw, decompress=decompress)

    return json.loads(content.decode("utf-8"))


def readpymodule(path, modulename="module"):
    """Reading a python module from source code"""
    code = _readfile(path, raw=False, decompress=False).decode("utf-8")

    modulespec = importlib.util.spec_from_loader(modulename, loader=None)
    module = importlib.util.module_from_spec(modulespec)

    exec(code, module.__dict__)
    sys.modules[modulename] = module

    return module


@timed
def readjsons(paths):
    return [_readjson(path) for path in paths]


def writevol(volume, path, compresslevel=9):
    data = volume.tobytes('F')

    if compresslevel is not None:
        data = cloudfiles.compression.compress(
                   data, method="gzip", compress_level=compresslevel)

    bucket, key = utils.splitpath(path)
    cf = cloudfiles.Cloudfiles(bucket)

    cf.put(key, data)


@timed
def writefile(content, path, compress=False):
    _writefile(content, path, compress=compress)


def _writefile(content, path, compress=False):
    """Writing a file to object storage bypassing timing"""
    bucket, key = utils.splitpath(path)

    cf = cloudfiles.CloudFiles(bucket)

    binary = content.encode("utf-8")
    if compress:
        binary = cloudfiles.compression.gzip(binary)

    cf.put(key, binary)


@timed
def writejson(content, path):
    bucket, key = utils.splitpath(path)

    cf = cloudfiles.CloudFiles(bucket)
    cf.put_json(key, content)


@timed
def writeinferencevols(towrite, storagestr, point):
    tag = utils.pointtag(point)
    for (name, vol) in towrite.items():
        outputfilename = os.path.join(storagestr, f"{tag}_{name}")

        utils.writevol(vol, outputfilename)


@timed
def writeannotations(annotations, storagestr, point):
    tag = utils.pointtag(point)
    for (i, ann) in enumerate(annotations):
        outputfilename = os.path.join(storagestr,
                                      f"{tag}_annotation{i}.json")
        writejson(ann.tojson(), outputfilename)


@timed
def write_ngl_annotations(annotations, path):
    content = NGLHEADERLINE
    for ann in annotations:
        content += ann.tongl() + "\n"

    writefile(content, path)


@timed
def readpts(filename):
    """Reads a point list file.

    Points are assumed to be specified line-by-line in a way that
    can be eval'd
    """
    with open(filename) as f:
        return list(map(eval, f))
