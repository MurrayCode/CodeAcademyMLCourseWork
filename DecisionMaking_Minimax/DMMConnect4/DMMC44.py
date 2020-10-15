from connect_four import *
import random
random.seed(108)

def minimax(input_board, is_maximizing, depth, alpha, beta):
  # Base case - the game is over, so we return the value of the board
  if game_is_over(input_board) or depth == 0:
    return [evaluate_board(input_board), "", alpha, beta]
  best_move = ""
  if is_maximizing == True:
    best_value = -float("Inf")
    symbol = "X"
  else:
    best_value = float("Inf")
    symbol = "O"
  for move in available_moves(input_board):
    new_board = deepcopy(input_board)
    select_space(new_board, move, symbol)
    hypothetical_value = minimax(new_board, not is_maximizing, depth - 1, alpha, beta)[0]
    if is_maximizing == True and hypothetical_value > best_value:
      best_value = hypothetical_value
      best_move = move
      alpha = max(alpha, best_value)
    if is_maximizing == False and hypothetical_value < best_value:
      best_value = hypothetical_value
      best_move = move
      beta = min(beta, best_value)
    if alpha > beta:
      break
  return [best_value, best_move, alpha, beta]
  
print_board(board)  
print(minimax(board, True, 6, -float("Inf"), float("Inf")))
