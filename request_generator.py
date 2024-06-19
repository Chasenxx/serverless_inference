import numpy as np
import time
from multiprocessing import Process, Queue
log = print

def PoissonProcess(rate=5):
    # rate=5 --> 5 requests per second
    interval = np.random.exponential(1 / rate)
    return interval


def NormalProcess(mean=5, std_dev=1):
    """
    Generates a time interval following a normal distribution.

    Parameters:
    mean (float): The mean of the normal distribution.
    std_dev (float): The standard deviation of the normal distribution.

    Returns:
    float: A time interval drawn from the normal distribution.
    """
    interval = np.random.normal(loc=mean, scale=std_dev)
    return interval


def request_generator(mean=1, std_dev=0.1, request_queue=None, max_requests=10):
    count = 0
    while count < max_requests:
        interval = NormalProcess(mean, std_dev)
        time.sleep(interval)  # Sleep for the interval time
        if request_queue:
            request_queue.put(f"Request generated at interval: {interval:.2f} seconds")
        else:
            log(f"Request generated at interval: {interval:.2f} seconds")
        count += 1
    return request_queue


if __name__ == "__main__":
    # Using multiprocessing queue to store requests
    request_queue = Queue()

    # Create and start a process for the request generator
    process = Process(target=request_generator, args=(1, 0.1, request_queue, 5))
    process.start()

    # Collect and print generated requests
    while not request_queue.empty() or process.is_alive():
        while not request_queue.empty():
            log(request_queue.get())

    # Wait for the process to complete
    process.join()


    
