"""Task generation leveraging pcg_skel

See https://github.com/AllenInstitute/pcg_skel
"""
import time

from annotationframeworkclient import FrameworkClient
import pcg_skel


def l2skelgenerator(ptgenerator, rootid, datastackname, rootpt, *genargs,
                    voxelres=[8, 8, 40], collapse_soma=True, n_parallel=8,
                    pointtypes="bpep",
                    **genkwargs):
    """L2-skeleton task generation

    Extracts points from a PCG L2 skeleton and feeds them to a point
    generator function
    """
    client = FrameworkClient(datastackname)

    refine = "bpep" if pointtypes in ["bpep", "epbp"] else None  # crude
    print("Extracting PCG L2 skeleton")
    start = time.time()
    l2skel = pcg_skel.pcg_skeleton(
                 rootid, client=client, refine=refine,
                 root_point=rootpt, root_point_resolution=voxelres,
                 collapse_soma=True, n_parallel=n_parallel)
    elapsed = time.time() - start
    print(f"Extracting PCG L2 skeleton completed in {elapsed:.3f}s")

    points = list()
    if "bp" in pointtypes:
        bps = l2skel.vertices[l2skel.branch_points] // voxelres
        points.extend(map(tuple, bps))

    if "ep" in pointtypes:
        eps = l2skel.vertices[l2skel.end_points] // voxelres
        points.extend(map(tuple, eps))

    return ptgenerator(points, *genargs, rootid=rootid, **genkwargs)
