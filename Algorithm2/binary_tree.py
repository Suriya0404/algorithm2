"""
Module contains binary tree implementation using classes and tree traversal using below methods
pre order - order of traversal - root node, left nodes, right nodes
post order - order of traversal - left nodes, root node, right nodes
in order - order of traversal - left nodes, right nodes, root node.
"""

def preOrder(tree):
    """
    Recursive function - to traverse the nodes of a tree.
    :param tree: Instance of BinaryTree
    :return: None
    """
    if tree:
        print(tree.getRootValue())
        preOrder(tree.getLeftChild())
        preOrder(tree.getRightChild())

def postOrder(tree):
    """
    Recursive function - to traverse the nodes of a tree.
    :param tree: Instance of BinaryTree
    :return: None
    """
    if tree:
        postOrder(tree.getLeftChild())
        postOrder(tree.getRightChild())
        print(tree.getRootValue())

def inOrder(tree):
    """
    Recursive function - to traverse the nodes of a tree.
    :param tree: Instance of BinaryTree
    :return: None
    """
    if tree:
        inOrder(tree.getLeftChild())
        print(tree.getRootValue())
        inOrder(tree.getRightChild())


class BinaryTree(object):

    def __init__(self, rootObj):
        """
        Constructor - Initialize the variables
        :param rootObj: Key value of the node.
        """
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        """
        Insert a new node left of current node.
        If current node already has a left node then existing left node becomes left node
        of new inserted node.
        :param newNode:
        :return: None
        """
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)        # Assign new node to left child of current node.
        else:
            nnode = BinaryTree(newNode)                 # create new Node
            nnode.leftChild = self.leftChild            # Assign left node of current node to new node's left child
            self.leftChild = nnode                      # Assign new node to left child of current node.
        return self.leftChild                           # Added by SM

    def insertRight(self, newNode):
        """
        Insert a new node right of the current node
        If current node already has a right node the insert existing right node to right of new node.
        :param newNode: Key value of new node
        :return: None
        """
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)       # Assign new node to right child of current node.
        else:
            nnode = BinaryTree(newNode)                 # Create a new node
            nnode.rightChild = self.rightChild          # Assign right node of current node to new node's right child
            self.rightChild = nnode                     # Assign new node to right Child of current node
        return self.rightChild                          # Added by SM

    def getRightChild(self):
        """
        Get the right child of current node.
        :return: Instance of BinaryTree. Right child of current node.
        """
        return self.rightChild

    def getLeftChild(self):
        """
        Get the left child of current node.
        :return: Instance of BinaryTree. Left child of current node.
        """
        return self.leftChild

    def setRootValue(self, obj):
        """
        Set the key value of current node
        :param obj: key value
        :return: None
        """
        self.key = obj

    def getRootValue(self):
        """
        Returns the key value of current node.
        :return: Return the key value of current node.
        """
        return self.key

    def preorder(self):
        """
        Recursive method - to traverse all nodes in tree.
        :return:
        """
        print(self.key)

        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()


if __name__ == '__main__':
    r = BinaryTree('a')

    print(r.getRootValue())

    b = r.insertLeft('b')
    b.insertRight('e')
    b.insertLeft('d')

    c = r.insertRight('c')
    c.insertLeft('f')
    c.insertRight('g')

    print(r.getLeftChild().getRootValue())
    print(r.getRightChild().getRootValue())

    print('Pre-order')
    preOrder(r)

    print('Post-order')
    postOrder(r)

    print('Inorder')
    inOrder(r)

