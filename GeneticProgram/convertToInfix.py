from ExpressionGenerator import GenMember


class ToInfixParser:
    """
    Class to convert the prefix expression to infix notation.
    """

    def __init__(self):
        self.stack = []

    @staticmethod
    def deconstruct_tree(list_nodes):
        """

        :param list_nodes: list of nodes belonging to tree
        :return: the values of the tree in prefix notation.
        """
        pref = list()
        for i in list_nodes:
            pref.append(str(i.value))
        return pref

    def get_infix_notation(self, prefix_expr):
        """
        Function to convert the prefix expression back to infix notation.
        :param prefix_expr: prefix expression to be converted.
        :return: infix expression.
        """
        for e in prefix_expr[::-1]:
            if e not in GenMember.operations:
                self.stack.append(e)

            else:
                try:
                    operand1 = self.stack.pop(-1)
                    operand2 = self.stack.pop(-1)
                    op = e
                    self.stack.append("({}{}{})".format(operand1, op, operand2))
                except IndexError:
                    new = GenMember()
                    new_mem = new.generate_expression()
                    return new_mem

        return self.stack.pop()[1:-1]


if __name__ == "__main__":
    t = ToInfixParser()
    x = t.get_infix_notation(['-', '*', '3', '4', '/', 'X5'])
    print(x)
