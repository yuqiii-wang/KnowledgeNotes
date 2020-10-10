"""
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
"""

import numpy as np
from binaryTreeConstrct import binaryTree
from travsTreeByLevel import travsTreeByLevel

class lowestCommonAncestor:
	# assume all element values in the tree are unique

	def __init__(self, treeRoot, p, q):
		self.commonAncestor = None
		self.__lowestCommonAncestor(treeRoot, p, q)

	# Return val of this function is boolean that indicates whether the currently processing node is p or q
	def __lowestCommonAncestor(self, Node, p, q):
		if Node.root is None:
			return None

		if Node.root==p:
			return p
		elif Node.root==q:
			return q

		tempLeftAncestor = None
		tempRightAncestor = None

		if self.__lowestCommonAncestor(Node.left, p, q) is not None:
			tempLeftAncestor = Node.root

		if self.__lowestCommonAncestor(Node.right, p, q) is not None:
			tempRightAncestor = Node.root

		if tempLeftAncestor==tempRightAncestor and (tempLeftAncestor is not None or tempRightAncestor is not None):
			self.commonAncestor = Node.root
		elif tempLeftAncestor is not None:
			return tempLeftAncestor
		elif tempRightAncestor is not None:
			return tempRightAncestor

		return None

	def __str__(self):
		return str(self.commonAncestor)

if __name__=='__main__':
	# set parameteres of a binary tree
	np.random.seed(111)
	val_range = 10
	size_range = 10
	array = list(np.random.randint(val_range, size=size_range))
	array = list(range(size_range))
	tree = binaryTree(element=array, _Root_Start=True)
	print(travsTreeByLevel(tree))
	print(lowestCommonAncestor(tree, 1, 9))