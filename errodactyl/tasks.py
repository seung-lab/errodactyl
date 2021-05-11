import numpy as np
import torch
from taskqueue import queueable

from . import storageio
from . import annotations
from . import utils


@queueable
@utils.timed
def simpleyacn(imgcvpath, segcvpath, rootid, pt, bboxwidth,
               modelfilename, chkptfilename, annotationstoragestr,
               resolution=(8, 8, 40), timestamp=None, threshold=0.5,
               inferencestoragestr=None, annotationtags=[],
               outputindex=1):

    # Read
    bbox = utils.bboxatpoint(pt, bboxwidth)
    img = storageio.readbbox(imgcvpath, bbox,
                             resolution=resolution, timestamp=timestamp)
    msk = storageio.readmask(segcvpath, rootid, bbox,
                             resolution=resolution, timestamp=timestamp)
    img = torch.from_numpy(
              img.reshape(1, 1, *img.shape).astype(np.float32) / 255.
              ).cuda()
    msk = torch.from_numpy(
              msk.reshape(1, 1, *msk.shape).astype(np.float32)).cuda()

    model = storageio.readmodel(modelfilename, chkptfilename).cuda()

    # Inference
    outputs = [torch.sigmoid(o) for o in utils.inference(model, img, msk)]
    anns = annotations.classification(
               outputs[outputindex].max().item(),
               threshold, pt, bbox, resolution, annotationtags)

    # Write
    if inferencestoragestr is not None:
        towrite = dict(image=img, objmask=msk,
                       o5=outputs[3], o4=outputs[2],
                       o3=outputs[1], recon=outputs[0])
        storageio.writeinferencevols(towrite, inferencestoragestr, pt)

    if annotationstoragestr is not None:
        storageio.writeannotations(anns, annotationstoragestr, pt)
