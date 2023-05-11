# Binary Tree

* A binary tree is made of nodes, where each node contains a left pointer (called the left subtree), a right pointer(called the right subtree), and a data element(data to be stored).
* The root pointer(node->root) points to the topmost node in the tree. A null pointer(`nullptr`) represents a binary tree with no elements — >the empty tree.
* Order (insert/search rule) would be: compare the inserting element with the root, if less than the root, then recursively call the left subtree, else recursively call the right subtree.

```bash
                  10
                /    \
	           6      14
			  / \    /  \
		     5   8  11  18
```

## Balance Tree

## Red-Black Tree