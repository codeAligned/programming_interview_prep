# Stores all custom - implemented data structures
# Alex Chao



class node:
    # constructor
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next = next_node

    def __str__(self):
        return str(self.data)



class linked_list:
    def __init__(self, head=None):
        self.head = head

    def append(self, data):
        # if its empty
        current = self.head
        if current is None:
            self.head = node(data)
        else:
            # traverse the list
            while current.next is not None:
                current = current.next
            current.next = node(data)

    def remove(self, data):
        current = self.head
        # if we are removing the head
        if current.data == data:
            self.head = current.next
        else:
            while current.next.data != data:
                current = current.next
            # stop if next is the one to delete
            current.next = current.next.next

    def __str__(self):
        # traverse the list
        if self.head == None:
            return "Empty List"
        else:
            total_string = ''
            current = self.head
            while current is not None:
                value_string = str(current.data)
                current = current.next
                total_string += value_string + ' -> '
        return total_string

