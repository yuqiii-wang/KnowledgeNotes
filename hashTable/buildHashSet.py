"""
Design a HashSet without using any built-in hash table libraries.

To be specific, your design should include these functions:

add(value): Insert a value into the HashSet. 
contains(value) : Return whether the value exists in the HashSet or not.
remove(value): Remove a value in the HashSet. If the value does not exist in the HashSet, do nothing.
"""

class buildHashSet:

	def __init__(self, num_elem=1000):
		self.arr = [False] * num_elem

	def add(self, key):
		self.arr[key] = True

	def remove(self, key):
		self.arr[key] = False

	def contains(self, key):
		self.__print(self.arr[key])
		return self.arr[key]

	def __print(self, val):
		print(str(val))

if __name__=="__main__":
	hashSet = buildHashSet()
	hashSet.add(1)         
	hashSet.add(2)     
	hashSet.contains(1)    
	hashSet.contains(3)
	hashSet.add(2)      
	hashSet.contains(2)
	hashSet.remove(2)         
	hashSet.contains(2)
	hashSet.add(100)  
	hashSet.contains(100)