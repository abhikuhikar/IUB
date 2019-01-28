#!/usr/bin/env python2
# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, 2016
# Updated by Zehua Zhang, 2017
#
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

import sys

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] ) 

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] ) 

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ " ".join([ "R" if col else "_" for col in row ]) for row in board])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

#Gives the index of the first empty row from start
def getFirstEmptyRowIndex(board):
    for i in range(0,N):
        if sum(board[i]) == 0:
            return i
    return 0

#check the position of the Rook in the row
def checkRookPositionInRow(board, row):
    pos = -1
    for i in range(0,N):
        if board[row][i] == 1:
            pos = i
    return pos

# Get list of successors of given board state
def successors(board):
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]

# Question 3 : Resolved two problems - Get list of successors of given board state
def successors2(board):
    if count_pieces(board) < N:
        return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) if board[r][c] == 0]
    return []

#Get list of successors of given board : optimized code : Adding pieces only from the leftmost empty columns
def successors3(board):
    if(count_pieces(board) >= N):
        return []
    emptyRowNum = getFirstEmptyRowIndex(board)
    if emptyRowNum == 0:
        return [ add_piece(board, emptyRowNum, N - col - 1) for col in range(0,N)]
    sol = []
    for a in range(0, N):
        if count_on_col(board, N - a - 1) < 1 and board[emptyRowNum][N - a - 1] == 0:
            sol = sol + [add_piece(board, emptyRowNum, N - a - 1)]
    return sol

def successors4(board):
    if count_pieces(board) < N:
        return [ add_piece(board, r, r) for r in range(0, count_pieces(board) + 1) if board[r][r] == 0 ]

# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] )

# check if board is a goal state
def is_goal_faster(board):
    return count_pieces(board) == N and \
        ( sum(board[r][r] for r in range(0,N)) == N or
             sum(board[r][r] for r in range(0,N)) == N )

#modified is_goal
def is_goal_NRooks(board):
    if(count_pieces(board) != N):
        return False
    for i in range(0,N):
        if(sum(board[i]) > 1):
            return False
        col = checkRookPositionInRow(board, i)
        if(count_on_col(board, col) > 1):
           return False
    return True

# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors4( fringe.pop() ):
            if is_goal_faster(s):
                return(s)
            fringe.append(s)
    return False

# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[1])
# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.

initial_board = [[0]*N]*N
solution = solve(initial_board)
print (printable_board(solution) if solution else "Sorry, no solution found. :(")

