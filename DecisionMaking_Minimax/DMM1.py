from tic_tac_toe import *
from copy import deepcopy

my_board = [
	["1", "2", "X"],
	["4", "5", "6"],
	["7", "8", "9"]
]

new_board = deepcopy(my_board)
select_space(new_board, 5, "O")
print_board(new_board)
