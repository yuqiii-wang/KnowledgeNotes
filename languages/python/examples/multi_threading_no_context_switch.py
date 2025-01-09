import threading

shared_data = 0

def increment():
    global shared_data
    for _ in range(1000):
        current = shared_data
        shared_data = current + 1

threads = [threading.Thread(target=increment) for _ in range(4)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print(f"Final shared_data: {shared_data}")
