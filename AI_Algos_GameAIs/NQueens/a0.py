#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 13:59:21 2018

@author: Abhilash
"""
#Assignment 0 file to calculate N Rooks and N Queens

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

# Return a string with the board rendered in a human-friendly format for n queens
def printable_board(board):
    piece = ""
    if Problem_Type == "nqueen":
        piece = "Q "
    if Problem_Type == "nrook":
        piece = "R "
    if Problem_Type == "nknight":
        piece = "K "
    printable_board = ""
    for r in range(0, len(board)):
        for c in range(0, len(board)):
            if board[r][c] == 1:
                printable_board+=piece
            if board[r][c] == 0:
                printable_board+="_ "
            if board[r][c] == -1:
                printable_board+="X "
        printable_board+= "\n"
        
    return printable_board

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

#Check if the coordinates are not restricted
def isNotRestricted(row, col):
    for i in range(0,len(RestrictedCoordinates)):
        if row == RestrictedCoordinates[i][0] and col == RestrictedCoordinates[i][1]:
            return False
    return True
    

#Get list of successors of given board : optimized code : Adding
    # pieces only from the topmost empty rows and adding only to the empty column
def successorsNRooks(board):
    if(count_pieces(board) >= N):
        return []
    emptyRowNum = getFirstEmptyRowIndex(board)
    if emptyRowNum == 0:
        return [ add_piece(board, emptyRowNum, N - col - 1) for col in range(0,N) if isNotRestricted(emptyRowNum, N - col - 1)]
    sol = []
    for a in range(0, N):
        if count_on_col(board, N - a - 1) < 1 and board[emptyRowNum][N - a - 1] == 0 and \
        isNotRestricted(emptyRowNum,N - a - 1):
            sol = sol + [add_piece(board, emptyRowNum, N - a - 1)]
    return sol


#Gives the index of the first empty row from start
def getFirstEmptyRowIndex(board):
    for i in range(0,N):
        if sum(board[i]) == 0:
            return i
    return 0
        
#Returns the count on diagonals : Optimized code in linear time
def count_diag(board, row, col):
    if N == 1 and board[row][col] == 1:
        return 1
    n = N - abs(row - col)
    if(row <= col):
        sum_diag1 = sum([board[i][col - row + i] for i in range(0,n)])
        
    if(row>col):
        sum_diag1 = sum([board[row - col + i][i] for i in range(0,n)])

    i = 0
    j = 0
    sum_diag2 = 0
    while(i < N - row and j <= col):
        sum_diag2 = sum_diag2 + board[row+i][col-j]
        i = i + 1
        j = j + 1

    i = 1
    j = 1
    while(i <= row and j < N - col):
        sum_diag2 = sum_diag2 + board[row-i][col+j]
        i = i + 1
        j = j + 1
    
    if board[row][col] == 1:
        return sum_diag1 + sum_diag2 - 1
    return sum_diag1 + sum_diag2    

#Returns the position of the piece in the row
def checkPiecePositionInRow(board, row):
    pos = -1
    for i in range(0,N):
        if board[row][i] == 1:
            pos = i
    return pos

#successor function for nQueens problem
def successorsNQueens(board):
    if(count_pieces(board) >= N):
        return []
    emptyRowNum = getFirstEmptyRowIndex(board)
    if emptyRowNum == 0:
        return [ add_piece(board, emptyRowNum, col) for col in range(0,N) if isNotRestricted(emptyRowNum, col)]
    prevRowQueen = checkPiecePositionInRow(board, emptyRowNum-1)
    sol = []
    for a in range(0, prevRowQueen - 1):
        if count_on_col(board, a) < 1 and count_diag(board,emptyRowNum,a) < 1 and \
            board[emptyRowNum][a] == 0 and isNotRestricted(emptyRowNum,a):
            sol = [add_piece(board, emptyRowNum, a)]
    for b in range(prevRowQueen+2, N):
        if count_on_col(board, b) < 1 and count_diag(board,emptyRowNum,b) < 1 and \
            board[emptyRowNum][b] == 0 and isNotRestricted(emptyRowNum,b):
            sol = sol + [add_piece(board, emptyRowNum, b)]
    return sol

#Get list of successors of given board : optimized code : Adding
    # pieces only from the topmost empty rows and adding only to the empty column
def successorsNKnights(board):
    if(count_pieces(board) >= N):
        return []
    sol = []
    for r in range(0,N):
        for c in range(0,N):
            if(count_KillerKnights(board, r, c) < 1) and board[r][c] == 0 \
                and isNotRestricted(r,c):
                sol = sol + [add_piece(board, r, c)]
    return sol

def isValidandEmpty(board, r, c):
    return r<N and r>=0 and c<N and c>=0 and board[r][c]
#Returns the count of the knights that can take it
def count_KillerKnights(board, r, c):
    count = 0
    count = isValidandEmpty(board,r-2,c-1) + isValidandEmpty(board,r-1,c-2) \
    + isValidandEmpty(board,r+1,c-2) + isValidandEmpty(board,r+2,c-1) + \
    isValidandEmpty(board,r+2,c+1) + isValidandEmpty(board,r+1,c+2) + \
    isValidandEmpty(board,r-1,c+2) + isValidandEmpty(board,r-2,c+1)
    
    return count

def is_goal_NKnights(board):
    if(count_pieces(board) == N):
        return True
    return False

#Check if board is a goal state for n Queens
def is_goal_NQueens(board):
    if(count_pieces(board) != N):
        return False
    for i in range(0,N):
        if(sum(board[i]) > 1):
            return False
        col = checkPiecePositionInRow(board, i)
        if(count_on_col(board, col) > 1):
            return False
        if(count_diag(board, i, col) > 1):
            return False
    return True

#Check if board is a goal state for n Rooks
def is_goal_NRooks(board):
    if(count_pieces(board) != N):
        return False
    for i in range(0,N):
        if(sum(board[i]) > 1):
            return False
        col = checkPiecePositionInRow(board, i)
        if(count_on_col(board, col) > 1):
            return False
    return True

# Solve n-rooks!
def solveRooks(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successorsNRooks( fringe.pop() ):
            if is_goal_NRooks(s):
                return(s)
            fringe.append(s)
    return False

# Solve n-queens!
def solveQueens(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successorsNQueens( fringe.pop() ):
            if is_goal_NQueens(s):
                return(s)
            fringe.append(s)
    return False

# Solve n-rooks!
def solveKnights(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successorsNKnights( fringe.pop() ):
            if is_goal_NKnights(s):
                return(s)
            fringe.append(s)
    return False

#mark restricted positions with -1 in a solution board
def markRestrictedPositions(board):
    for r in range(0, len(RestrictedCoordinates)):
        board[RestrictedCoordinates[r][0]][RestrictedCoordinates[r][1]] = -1
    return board

#Problem type is defined by nrook or nqueen
Problem_Type = str(sys.argv[1])

# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[2])

#N_Restricted is the number of restricted positions
N_Restricted = int(sys.argv[3])

#All the arguments after 2nd are the restricted positions
RestrictedCoordinates = [[int(sys.argv.pop(4)) - 1, int(sys.argv.pop(4)) - 1] for r in range(0,N_Restricted)]

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0 for x in range(N)] for y in range(N)] 

solution = []
if Problem_Type == "nqueen":
    solution = solveQueens(initial_board)
if Problem_Type == "nrook":
    solution = solveRooks(initial_board)
if Problem_Type == "nknight":
    solution = solveKnights(initial_board)
if solution:
    solution = markRestrictedPositions(solution)
print (printable_board(solution) if solution else "Sorry, no solution found. :(")

