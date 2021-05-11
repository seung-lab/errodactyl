"""Basic IO utilities"""
import os


def splitpath(path):
    """Parsing a (potentially remote) file path for CloudFiles

    Splits the path into a (protocol + bucket, key) tuple, 
    where the protocol+bucket could be a cloud protocol or 'file://'.
    The key specfies a file within the bucket or directory.
    """
    # Does the string contain '://'? If so, split it.
    fields = path.split("://")
    if len(fields) == 2:
        protocol = f"{fields[0]}://"
        bucket = fields[1].split('/')[0]
        key = '/'.join(fields[1].split('/')[1:])

    elif len(fields) == 1:  # assume filesystem
        protocol = "file://"
        # including all but the last directory here to limit potential damage
        bucket = '/'.join(fields[0].split('/')[:-1])
        key = fields[0].split('/')[-1]

    else:
        raise ValueError(f"unrecognized file pattern: {path}")

    # the join call ensures an ending slash
    return os.path.join(protocol + bucket, ""), key


def pointtag(point):
    return f"{point[0]}_{point[1]}_{point[2]}"
