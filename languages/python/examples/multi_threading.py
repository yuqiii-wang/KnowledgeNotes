import threading

# Shared variable
shared_variable = 0

# Function to be run by each thread
# For GIL, there will be no conflicts to updating shared_variable
def increment():
    global shared_variable
    for _ in range(100000000):  
        shared_variable += 1

# Create four threads
threads = []
for i in range(4):
    thread = threading.Thread(target=increment)
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print(f"Final value of shared_variable: {shared_variable}")
