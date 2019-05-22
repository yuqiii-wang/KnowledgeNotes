class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        
    def push(self, x: int) -> None:
        self.stack.append(x)

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        # to get min value of a stack
        return min(self.stack)

# Your MinStack object will be instantiated and called as such:
obj = MinStack()
obj.push(-2)
obj.push(0)
obj.push(-1)
param_1 = obj.getMin()
param_2 = obj.top()
obj.pop()
param_3 = obj.getMin()
print(param_1, param_2, param_3)