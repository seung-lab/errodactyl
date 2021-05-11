from taskqueue import TaskQueue
from . import tasks


def run(queueurl, lease_seconds=300):

    with TaskQueue(qurl=queueurl) as tq:
        tq.poll(lease_seconds=int(lease_seconds),
                verbose=True)
