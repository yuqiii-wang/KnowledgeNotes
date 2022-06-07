class MyCircularQueue:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        """
        self.len = k
        self.crntLen = 0
        self.front = 0
        self.rear = -1
        self.queue = [None] * k

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        """
        if self.isFull(): return False

        self.queue[(self.rear + 1) % self.len] = value
        self.crntLen += 1
        self.rear += 1
        return True

    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        """
        if self.isEmpty(): return False

        self.queue[self.front % self.len] = None
        self.crntLen -= 1
        self.front += 1

        return True

    def Front(self) -> int:
        """
        Get the front item from the queue.
        """
        if self.queue[self.front % self.len] is not None:
            return self.queue[self.front % self.len]
        else:
            return -1

    def Rear(self) -> int:
        """
        Get the last item from the queue.
        """
        if self.queue[self.rear % self.len] is not None:
            return self.queue[self.rear % self.len]
        else:
            return -1
        

    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return self.crntLen <= 0
        

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return self.crntLen == self.len


# Your MyCircularQueue object will be instantiated and called as such:
obj = MyCircularQueue(3)
param_1 = obj.enQueue(1)
param_1 = obj.enQueue(2)
param_1 = obj.enQueue(3)
param_1 = obj.enQueue(4)
param_2 = obj.deQueue()
param_1 = obj.enQueue(5)
param_2 = obj.deQueue()
param_2 = obj.deQueue()
param_2 = obj.deQueue()
param_1 = obj.enQueue(0)
param_3 = obj.Front()
param_4 = obj.Rear()
param_5 = obj.isEmpty()
param_6 = obj.isFull()
print(param_1,
    param_2,
    param_3,
    param_4,
    param_5,
    param_6)