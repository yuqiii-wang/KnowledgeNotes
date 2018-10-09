"""
Design a HashMap without using any built-in hash table libraries.

To be specific, your design should include these functions:

put(key, value) : Insert a (key, value) pair into the HashMap. If the value already exists in the HashMap, update the value.
get(key): Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
remove(key) : Remove the mapping for the value key if this map contains the mapping for the key.

All keys and values will be in the range of [0, 1000000].
The number of operations will be in the range of [1, 10000].
"""

class designHashMap:

	def __init__(self, capacity=10000, val_range=1000000):
		self.capacity = capacity
		self.val_range = val_range
		self.buckets = [[]] * self.capacity

	def put(self, key, val):
		bucket_idx_record, i_record = self.__findBucket(key)

		if i_record is -1:
			self.buckets[bucket_idx_record].append((key, val))
		else:
			self.buckets[bucket_idx_record][i_record] = (key, val)

	def get(self, key):
		bucket_idx_record, i_record = self.__findBucket(key)

		if i_record is -1:
			return None
		return self.buckets[bucket_idx_record][i_record]

	def remove(self, key):
		bucket_idx_record, i_record = self.__findBucket(key)

		if i_record is -1:
			return None
		self.buckets[bucket_idx_record].pop(i_record)

	def __findBucket(self, key):
		# every bucket can hold items up to val_range/capacity amount.
		bucket_idx = key % self.capacity 
		# this is the for loop operator that should have limited range [1, 10000].
		# i: index; (k, v) for (key, value) stored as each item in each bucket that all gather to form a big hash map
		bucket_idx_record = None
		i_record = None
		for i, (k, v) in enumerate(self.buckets[bucket_idx]):
			if k == key:
				bucket_idx_record = bucket_idx
				i_record = i
				break
		if bucket_idx_record is None and i_record is None:
			bucket_idx_record = bucket_idx
			i_record = -1

		return bucket_idx_record, i_record

if __name__=="__main__":
	designHashMap_obj = designHashMap()
	designHashMap_obj.put(123, 1234)
	print(designHashMap_obj.get(123))
	designHashMap_obj.remove(123)
	print(designHashMap_obj.get(123))

