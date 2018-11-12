"""---------------------------------------------
Script initially found at :
https://gist.github.com/StuartGordonReid/eb59113cb29e529b8105/revisions
but heavily modified to handle systems and to take advantage of spase matrices.
A few bugs have also been corrected.

This is a optimized version of the gaussian elimination that tries to keep the matrice sparse.
---------------------------------------------"""
import copy
import time

class BinarySystem:
    def __init__(self, matrix, right_hand_side, rows, cols):
        """
        :param matrix: the matrix we want to compute the rank for
        :param right_hand_side: the right and side of the system
        :param rows: the number of rows
        :param cols: the number of columns
        :return: a BinaryMatrix object
        """
        self.M = rows
        self.Q = cols
        self.A = matrix
        self.B = right_hand_side
        self.m = min(rows, cols)

    def compute_rank(self):
        """
        While this function originally computed the rank of the matrice by echelonizing it, it has been corrected,
        optimized and modified to handle a right_hand_side.
        Return the number of interted variables.
        The final state is of the form:
            -self.M becomes a diagonalized matrice (if possible, else an echelonized form)
            -self.B is modified accordingly to self.M and so contains the solution of the system (if self.M is diagonalized)
        """

        #premier passage pour faire une matrice triangulaire superieure
        for i in range(self.m):
            if (self.A[i][i] != 1):
                self.find_unit_element_swap(i, True)
            if (self.A[i][i] == 1):
                self.perform_row_operations(i, True)
                #self.perform_row_operations(i, False) #redondant avec le second passage
            else : #no matching row found but this row could be used later -> add it to the next
                for k in range(self.Q):
                    self.A[i+1][k] = (self.A[i+1][k] ^ self.A[i][k])
                self.B[i+1] = self.B[i+1] ^ self.B[i]
                
        #second passage pour echelonner
        for i in range(self.m-1, 0, -1):
            if (self.A[i][i] == 1):
                self.perform_row_operations(i, False)
                
        return self.determine_rank() #Instead of the rank, return the number of inverted variables

    def perform_row_operations(self, i, forward_elimination):
        """
        This method performs the elementary row operations. This involves xor'ing up to two rows together depending on
        whether or not certain elements in the matrix contain 1's if the "current" element does not.
        :param i: the current index we are are looking at
        :param forward_elimination: True or False.
        
        Amelioration : tient compte du fait que ligne A[i] ne contient que des zeros jusqu'a A[i][i]
        """
        if forward_elimination:
            j = i + 1
            while j < self.M:
                if self.A[j][i] == 1:
                    for k in range(i,self.Q):
                        self.A[j][k] = (self.A[j][k] ^ self.A[i][k])
                    self.B[j] = self.B[j] ^ self.B[i]
                j += 1
        else:
            j = i - 1
            while j >= 0:
                if self.A[j][i] == 1:
                    for k in range(i,self.Q):
                        self.A[j][k] = (self.A[j][k] ^ self.A[i][k])
                    self.B[j] = self.B[j] ^ self.B[i]
                j -= 1

    def find_unit_element_swap(self, i, forward_elimination):
        """
        This given an index which does not contain a 1 this searches through the rows below the index to see which rows
        contain 1's, if they do then they swapped. This is done on the forward and backward elimination
        :param i: the current index we are looking at
        :param forward_elimination: True or False.
        """
        row_op = 0
        if forward_elimination:
            index = i + 1
            while index < self.M and self.A[index][i] == 0:
                index += 1
            if index < self.M:
                row_op = self.swap_rows(i, index)
        else:
            index = i - 1
            while index >= 0 and (self.A[index][i] == 0 or self.A[index][index] == 1):  #DO NOT SWAP if the row found was well placed
                index -= 1
            if index >= 0:
                row_op = self.swap_rows(i, index)
        return row_op

    def swap_rows(self, i, ix):
        """
        This method just swaps two rows in a matrix. Had to use the copy package to ensure no memory leakage
        :param i: the first row we want to swap and
        :param ix: the row we want to swap it with
        :return: 1
        """
        temp_A = copy.copy(self.A[i])
        self.A[i] = self.A[ix]
        self.A[ix] = temp_A
        temp_B = copy.copy(self.B[i])
        self.B[i] = self.B[ix]
        self.B[ix] = temp_B
        return 1

    def determine_rank(self):
        """
        This method determines the rank of the transformed matrix
        :return: the rank of the transformed matrix
        """
        rank = self.M
        i = 0
        while i < self.M:
            all_zeros = 1
            for j in range(self.Q):
                if self.A[i][j] == 1:
                    all_zeros = 0
            if all_zeros == 1:
                rank -= 1
            i += 1
        return rank
