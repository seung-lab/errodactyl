"""Error detection result annotations"""
import json
from dataclasses import dataclass
from typing import List

from . import storageio


BASERESOLUTION = [4, 4, 40]


@dataclass
class Annotation:
    pt: List[int]
    minpt: List[int]
    maxpt: List[int]
    confidence: float
    tags: List[str]
    description: str

    def tojson(self):
        rawstr = str(self.__dict__)
        jsonstr = rawstr.replace("'", '"').replace('(', '[').replace(')', ']')
        return json.loads(jsonstr)

    @classmethod
    def fromjson(self, json):
        return Annotation(json["pt"], json["minpt"], json["maxpt"],
                          json["confidence"], json["tags"],
                          json["description"])

    def tongl(self):
        tagstr = ','.join(self.tags)
        return (f'"{tuple(self.minpt)}","{tuple(self.maxpt)}",'
                f',"{tagstr}","{self.description}",,,AABB,')


def scalept(pt, resolution):
    factors = [r/br for (br, r) in zip(BASERESOLUTION, resolution)]
    assert all(int(f) == f for f in factors), "not int multiple resolution"
    assert all(f >= 1 for f in factors), "scale factor < 1"

    return tuple(int(coord*f) for (coord, f) in zip(pt, factors))


def compileannotations(filenames, errorfilename, nonerrorfilename):
    annotations = [Annotation.fromjson(j) for j in
                   storageio.readjsons(filenames)]

    errors = list()
    nonerrors = list()
    for ann in annotations:
        if "error" in ann.tags:
            errors.append(ann)
        else:
            nonerrors.append(ann)

    errors = sorted(errors, key=lambda ann: 1-ann.confidence)
    nonerrors = sorted(nonerrors, key=lambda ann: 1-ann.confidence)

    storageio.write_ngl_annotations(errors, errorfilename)
    storageio.write_ngl_annotations(nonerrors, nonerrorfilename)


def classification(value, threshold, pt, bbox, resolution, tags):
    """Create a classification annotation from an output value"""
    if value >= threshold:
        tags = ["error"] + tags
        confidence = (value - threshold) / (1 - threshold)
    else:
        tags = ["correct"] + tags
        confidence = 1 - (value / threshold)

    minpt = scalept(bbox.minpt, resolution)
    maxpt = scalept(bbox.maxpt, resolution)
    description = f"Confidence = {confidence:.3f}"

    return [Annotation(pt, minpt, maxpt, confidence, tags, description)]
