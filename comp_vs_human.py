import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras import models
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import math
from collections import deque
import os
import time
from datetime import datetime
import h5py
import copy

# Hyperparameters
model_path = "GoodUltimate2019-03-03 21_06_38+MCTS600+cpuct4.h5"
mcts_search = 400
MCTS = True
cpuct = 2


def get_empty_board():
    board = []
    
    for i in range(9):
        board.append([[" "," "," "],
        [" "," "," "],
        [" "," "," "]])
    return board

def print_board(totalBoard):
    firstRow = ""
    secondRow = ""
    thirdRow = ""

    # Takes each board, saves the rows in a variable, then prints the variables
    for boardIndex in range(len(totalBoard)):
        firstRow = firstRow + "|" + " ".join(totalBoard[boardIndex][0]) + "|"
        secondRow = secondRow + "|" + " ".join(totalBoard[boardIndex][1]) + "|"
        thirdRow = thirdRow + "|" + " ".join(totalBoard[boardIndex][2]) + "|"

        # if 3 boards have been collected, then it prints the boards out and resets the variables (firstRow, secondRow, etc.)
        if boardIndex > 1 and (boardIndex + 1) % 3 == 0:
            print (firstRow)
            print (secondRow)
            print (thirdRow)
            print ("---------------------")
            firstRow = ""
            secondRow = ""
            thirdRow = ""

def possiblePos(board, subBoard):

    if subBoard == 9:
        return range(81)
    
    possible = []


    # otherwise, finds all available spaces in the subBoard
    if board[subBoard][1][1] != 'x' and board[subBoard][1][1] != 'o':
        for row in range(3):
            for coloumn in range(3):
                if board[subBoard][row][coloumn] == " ":
                    possible.append((subBoard * 9) + (row * 3) + coloumn)
        if len(possible) > 0:
            return possible

    # if the subboard has already been won, it finds all available spaces on the entire board
    for mini in range(9):
        if board[mini][1][1] == "x" or board[mini][1][1] == "o":
            continue
        for row in range(3):
            for coloumn in range(3):
                if board[mini][row][coloumn] == " ":
                    possible.append((mini * 9) + (row * 3) + coloumn)

    return possible


def move(board,action, player):

    if player == 1:
        turn = 'X'
    if player == -1:
        turn = "O"
    
    bestPosition = []

    bestPosition.append(int (action / 9))
    remainder = action % 9
    bestPosition.append(int (remainder/3))
    bestPosition.append(remainder%3)

    # place piece at position on board
    board[bestPosition[0]][bestPosition[1]][bestPosition[2]] = turn

    emptyMiniBoard = [[" "," "," "], [" "," "," "], [" "," "," "]]

    wonBoard = False
    win = False
    mini = board[bestPosition[0]]
    subBoard = bestPosition[0]
    x = bestPosition[1]
    y = bestPosition[2]

    #check for win on verticle
    if mini[0][y] == mini[1][y] == mini [2][y]:
        board[subBoard] = emptyMiniBoard
        board[subBoard][1][1] = turn.lower()
        wonBoard = True

    #check for win on horozontal
    if mini[x][0] == mini[x][1] == mini [x][2]:
        board[subBoard] = emptyMiniBoard
        board[subBoard][1][1] = turn.lower()
        wonBoard = True

    #check for win on negative diagonal
    if x == y and mini[0][0] == mini[1][1] == mini [2][2]:
        board[subBoard] = emptyMiniBoard
        board[subBoard][1][1] = turn.lower()
        wonBoard = True

    #check for win on positive diagonal
    if x + y == 2 and mini[0][2] == mini[1][1] == mini [2][0]:
        board[subBoard] = emptyMiniBoard
        board[subBoard][1][1] = turn.lower()
        wonBoard = True

    #set new subBoard
    newsubBoard = (bestPosition[1] * 3) + bestPosition[2]

    # if the subBoard was won, checking whether the entire board is won as well
    if wonBoard == True:
        win = checkWinner(board, subBoard, turn)
    
    #if win:
    #    print ("won game!")
    #    print_board(board)

    return board, newsubBoard, win


def checkWinner(board,winningSubBoard, turn):

    # getting coordinates of winning subBoard
    for i in range(3):
        if (winningSubBoard - i) % 3 == 0:
            row = int((winningSubBoard - i) /3)
            winningSubBoardCoordinate = [row,i]
            break

    # making winning subBoard using just centre pieces
    winningBoard = [
    [board[0][1][1], board[1][1][1], board[2][1][1]],
    [board[3][1][1], board[4][1][1], board[5][1][1]],
    [board[6][1][1], board[7][1][1], board[8][1][1]]
    ]

    # horozontal wins
    if turn.lower() == winningBoard[winningSubBoardCoordinate[0]][0] == winningBoard[winningSubBoardCoordinate[0]][1] == winningBoard[winningSubBoardCoordinate[0]][2]:
        return True
    # vertical wins
    elif turn.lower() == winningBoard[0][winningSubBoardCoordinate[1]] == winningBoard[1][winningSubBoardCoordinate[1]] == winningBoard[2][winningSubBoardCoordinate[1]]:
        return True
    # top left to bottom right diagonal
    elif turn.lower() == winningBoard[0][0] == winningBoard[1][1] == winningBoard[2][2]:
        return True
    # bottom left to top right diagonal
    elif turn.lower() == winningBoard[2][0] == winningBoard[1][1] == winningBoard[0][2]:
        return True
    else:
        return False

def human_turn(board,subBoard,turn):
    possible = possiblePos(board, subBoard)

    print_board(board)
    print ("It is " + turn + "'s turn")

    #check if the subBoard has already been won, and takes new subBoard as input
    if subBoard == 9 or board[subBoard][1][1] == "x" or board[subBoard][1][1] == "o" or len(possible) > 9:
        while True: 
            try: 
                newsubBoard = int(input("Wow, which sub-board would you like to play on")) -1
            except ValueError:
                print ("That was not a valid integer, please try again")
                continue
            if newsubBoard not in range(9):
                print("Please enter a valid input between 1 and 9")
                continue
            if board[newsubBoard][1][1] == "x" or board[newsubBoard][1][1] == "o":
                print("That board has been taken, please enter a valid board")
                continue
            else:
                subBoard = newsubBoard
                break

    #takes placement of piece as input
    print ("You can only play on board number", subBoard + 1)
    while True:
        try:
            y = int(input("Please enter y coordinate")) -1
            x = int(input("Please enter x coordinate")) -1
        except ValueError:
            print ("One of those inputs were not valid integers, please try again")
            continue
        if y not in range(3) or x not in range(3):
            print ("Integers must be between 1 and 3, please try again")
            continue
        if board[subBoard][y][x] != " ":
            print ("That space has already been taken, please try again")
            continue
        else:
            return subBoard * 9 + y * 3 + x

# ---------------------------------
# Functions for neural network
# --------------------------------


# initializing search tree
Q = {}  # state-action values
Nsa = {}  # number of times certain state-action pair has been visited
Ns = {}   # number of times state has been visited
W = {}  # number of total points collected after taking state action pair
P = {}  # initial predicted probabilities of taking certain actions in state


def fill_winning_boards(board):

    # takes in a board in its normal state, and converts all suboards that have been won to be filled with the winning player's piece

    new_board = []
    for suboard in board:
        if suboard[1][1] =='x':
            new_board.append([["X","X","X"],["X","X","X"],["X","X","X"]])
        elif suboard[1][1] =='o':
            new_board.append([["O","O","O"],["O","O","O"],["O","O","O"]])
        else:
            new_board.append(suboard)
    return new_board


def letter_to_int(letter, player):
    # based on the letter in a box in the board, replaces 'X' with 1 and 'O' with -1
    if letter == 'v':
        return 0.1
    elif letter == " ":
        return 0
    elif letter == "X":
        return 1 * player
    elif letter =="O":
        return -1 * player

def board_to_array(boardreal, mini_board, player):
    
    # makes copy of board, so that the original board does not get changed
    board = copy.deepcopy(boardreal)

    # takes a board in its normal state, and returns a 9x9 numpy array, changing 'X' = 1 and 'O' = -1
    # also places a 0.1 in all valid board positions

    board = fill_winning_boards(board)
    tie = True

    # if it is the first turn, then all of the cells are valid moves
    if mini_board == 9:
        return np.full((9,9), 0.1)

    # replacing all valid positions with 'v'
    # checking whether all empty values on the board are valid
    if board[mini_board][1][1] != 'x' or board[mini_board][1][1] != 'o':
        for line in range(3):
            for item in range(3):
                if board[mini_board][line][item] == " ":
                    board[mini_board][line][item] = 'v'
                    tie = False
    # if not, then replacing empty cells in mini board with 'v'
    else:
        for suboard in range (9):
            for line in range(3):
                for item in range(3):
                    if board[suboard][line][item] == " ":
                        board[suboard][line][item] = 'v'

    # if the miniboard ends up being a tie
    if tie:
        for suboard in range (9):
            for line in range(3):
                for item in range(3):
                    if board[suboard][line][item] == " ":
                        board[suboard][line][item] = 'v'


    array = []
    firstline = []
    secondline = []
    thirdline = []
    
    for suboardnum in range(len(board)):
            
        for item in board[suboardnum][0]:
            firstline.append(letter_to_int(item, player))
        
        for item in board[suboardnum][1]:
            secondline.append(letter_to_int(item, player))
        
        for item in board[suboardnum][2]:
            thirdline.append(letter_to_int(item, player))
        
        if (suboardnum + 1) % 3 == 0:
            array.append(firstline)
            array.append(secondline)
            array.append(thirdline)
            firstline = []
            secondline = []
            thirdline = []

    nparray = np.array(array)
    
    return nparray



def mcts(s, current_player, mini_board):

    if mini_board == 9:
        possibleA = range(81)
    else:
        possibleA = possiblePos(s, mini_board)

    sArray = board_to_array(s, mini_board, current_player)
    sTuple = tuple(map(tuple, sArray))

    if len(possibleA) > 0:
        if sTuple not in P.keys():

            policy, v = nn.predict(sArray.reshape(1,9,9))
            v = v[0][0]
            valids = np.zeros(81)
            np.put(valids,possibleA,1)
            policy = policy.reshape(81) * valids
            policy = policy / np.sum(policy)
            P[sTuple] = policy

            Ns[sTuple] = 1

            for a in possibleA:
                Q[(sTuple,a)] = 0
                Nsa[(sTuple,a)] = 0
                W[(sTuple,a)] = 0
            return -v

        best_uct = -100
        for a in possibleA:

            uct_a = Q[(sTuple,a)] + cpuct * P[sTuple][a] * (math.sqrt(Ns[sTuple]) / (1 + Nsa[(sTuple,a)]))

            if uct_a > best_uct:
                best_uct = uct_a
                best_a = a
        
        next_state, mini_board, wonBoard = move(s, best_a, current_player)

        if wonBoard:
            v = 1
        else:
            current_player *= -1
            v = mcts(next_state, current_player, mini_board)
    else:
        return 0

    W[(sTuple,best_a)] += v
    Ns[sTuple] += 1
    Nsa[(sTuple,best_a)] += 1
    Q[(sTuple,best_a)] = W[(sTuple,best_a)] / Nsa[(sTuple,best_a)]
    return -v



def get_action_probs(init_board, current_player, mini_board):

    for _ in range(mcts_search):
        s = copy.deepcopy(init_board)
        value = mcts(s, current_player, mini_board)
    
    print ("done one iteration of MCTS")

    actions_dict = {}

    sArray = board_to_array(init_board, mini_board, current_player)
    sTuple = tuple(map(tuple, sArray))

    for a in possiblePos(init_board, mini_board):
        actions_dict[a] = Nsa[(sTuple,a)] / Ns[sTuple]
    print ("actions dict-", actions_dict)
    action_probs = np.zeros(81)
    
    for a in actions_dict:
        np.put(action_probs, a, actions_dict[a], mode='raise')
    
    return action_probs


nn = load_model(model_path)
def playgame():

    board = get_empty_board()
    mini_board = 9
    global nn

    while True:
        action = human_turn(board, mini_board, 'X')
        next_board, mini_board, wonBoard = move(board, action, 1)

        if wonBoard:
            print ("Wow you're really good. You just beat a computer")
            break
        else:
            board = next_board
        
        if MCTS:
            policy = get_action_probs(board, -1, mini_board)
            policy = policy / np.sum(policy)
        else:
            policy, value = nn.predict(board_to_array(board, mini_board, -1).reshape(1,9,9))
            possibleA = possiblePos(board,mini_board)
            valids = np.zeros(81)
            np.put(valids,possibleA,1)
            policy = policy.reshape(81) * valids
            policy = policy / np.sum(policy)

        action = np.argmax(policy)
        print ("action", action)
        print ("policy")
        print (policy)
        
        next_board, mini_board, wonBoard = move(board, action, -1)

        if wonBoard:
            print ("Awww you lost. Better luck next time")
            break
        else:
            board = next_board

playgame()