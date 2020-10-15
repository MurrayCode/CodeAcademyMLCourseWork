from connect_four import *

def two_ai_game():
    my_board = make_board()
    while not game_is_over(my_board):
      # Fill in the third parameter for the first player's "intelligence"
      result = minimax(my_board, True, 6, -float("Inf"), float("Inf"))
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        #Fill in the third parameter for the second player's "intelligence"
        result = minimax(my_board, False, 6, -float("Inf"), float("Inf"))
        print( "O Turn\nO selected ", result[1])
        print(result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")

two_ai_game()
