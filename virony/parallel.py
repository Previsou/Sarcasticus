""" Adapted from Brian Refsdal parallel map module:

Parallel map using multiprocessing
This module (courtesy Brian Refsdal at SAO) implements a parallelized 
version of the native Python map function that utilizes the Python 
multiprocessing module to divide and conquer an iterable.

Last modified: 25/01/2013

See: http://www.astropython.org/snippet/2010/3/Parallel-map-using-multiprocessing
"""


import itertools
import Queue
import multiprocessing
import time

def chunker(sequence, nb_parts):

    chunks = [ [] for i in range(nb_parts) ]
    for taskID, item in enumerate(sequence):
        chunks[taskID%nb_parts].append((taskID,item))
    return chunks
        
def worker(f, chunk, out_q, err_q, lock):
    """
    A worker function that maps an input function over a
    slice of the input iterable.

    :param f  : callable function that accepts argument from iterable
    :param chunk: slice of input iterable
    :param out_q: thread-safe output queue
    :param err_q: thread-safe queue to populate on exception
    :param lock : thread-safe lock to protect a resource
            ( useful in extending parallel_map() )
    """

    # iterate over slice 
    for taskID, val in chunk:
        try:
            result = f(val)
        except Exception, e:
            err_q.put(e)
            result = None
        # output the result and task ID to output queue
        out_q.put( (taskID,result) )

def task_generator(procs, err_q, out_q, timeout=0.1):
    """
    A generator that executes populated processes and processes
    the resultant array. Checks error queue for any exceptions.

    :param procs: list of Process objects
    :param out_q: thread-safe output queue
    :param err_q: thread-safe queue to populate on exception
    """
    # function to terminate processes that are still running.
    die = (lambda vals : [val.terminate() for val in vals
             if val.exitcode is None])

    try:
        map(lambda proc: proc.start(), procs)
        time.sleep(timeout)
        
        while True:
            yield out_q.get(block=True, timeout=timeout)
            
            if not err_q.empty():
                # kill all on any exception from any one slave
                raise err_q.get()
    except Queue.Empty:
        pass
    finally:
        # kill all slave processes on ctrl-C
        die(procs)
        

def parallel_map(function, *sequences, **kwargs):
    """
    A parallelized version of the native Python map function that
    utilizes the Python multiprocessing module to divide and 
    conquer sequence.

    :param function: callable function (arity should match number of sequences)
    :param sequence: iterable sequences
    :param n_jobs: number of cores to use
    """
    
    final_size = min(map(len, sequences))
    answers = [None] * final_size
    
    # Order results by index
    for index, result in parallel_imap(function, *sequences, **kwargs):
        answers[index] = result
            
    return answers

def initPool(function, sequence, n_jobs):
    """
    Initialize the pool (start processes and manager).
    
    :param function: callable function that accepts argument from iterable
    :param sequence: iterable sequence 
    :param n_jobs: number of cores to use
    """
    if not callable(function):
        raise TypeError("input function '%s' is not callable" %
              repr(function))

    try:
        iter(sequence)
    except:
        raise TypeError("input '%s' is not iterable" %
              repr(sequence))


    # Returns a started SyncManager object which can be used for sharing 
    # objects between processes. The returned manager object corresponds
    # to a spawned child process and has methods which will create shared
    # objects and return corresponding proxies.
    manager = multiprocessing.Manager()

    # Create FIFO queue and lock shared objects and return proxies to them.
    # The managers handles a server process that manages shared objects that
    # each slave process has access to. Bottom line -- thread-safe.
    out_q = manager.Queue()
    err_q = manager.Queue()
    lock = manager.Lock()

    # if sequence is less than n_jobs, only use len sequence number of 
    # processes
    size = len(sequence)
    if size < n_jobs:
        n_jobs = size 

    # group sequence into n_jobs-worth of chunks
    chunks = chunker(sequence, n_jobs)

    procs = [multiprocessing.Process(target=worker, 
                                    args=(function, chunk, out_q, err_q, lock))
            for chunk in chunks]
         
    return procs, err_q, out_q

def parallel_imap(function, *sequences, **kwargs):
    """
    A parallelized version of the native Python itertools.imap function that
    utilizes the Python multiprocessing module to divide and 
    conquer sequence.
    
    :param function: callable function (arity should match number of sequences)
    :param sequence: iterable sequences
    :param n_jobs: number of cores to use
    """
    # Handle function application over multiple sequences
    sequence = zip(*sequences)
    tupled_function = lambda Xs: function(*Xs)
    
    n_jobs = kwargs.get("n_jobs", -1)
    timeout = kwargs.get("timeout", 0.1)
    if n_jobs==1:
        return enumerate(itertools.imap(tupled_function, sequence))
    elif n_jobs < 0 or n_jobs > multiprocessing.cpu_count():
        n_jobs = multiprocessing.cpu_count()

    pool = initPool(tupled_function, sequence, n_jobs)
    return task_generator(*pool, timeout=timeout)

