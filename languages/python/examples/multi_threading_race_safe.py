import threading
import time

lock = threading.Lock()

shared_data = 0

def increment():
    global shared_data
    for _ in range(1000):
        with lock:  # Acquire lock before modifying shared_data
            current = shared_data
            time.sleep(0.001)  # Artificial delay to force context switching
            shared_data = current + 1

threads = [threading.Thread(target=increment) for _ in range(4)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print(f"Final shared_data: {shared_data}")
