import numpy as np
import torch
class Game:
    def __init__(self):
       self.board = np.zeros((3,3))
       self.current_player = 1

    def make_move(self,x,y):
       self.board[x][y] = self.current_player
       self.current_player*=-1
    
    def unmake_move(self,x,y):
       self.board[x][y] = 0
       self.current_player*=-1
   
    def legal_moves(self):
       flat = self.board.flatten()
       return np.where(flat == 0)[0]
    
    def game_state(self):
       board_string = str((self.current_player*self.board.astype(np.int64)).flatten()) ## canonical board states
       return board_string
    
    def game_state_tensor(self):
       return torch.tensor(self.current_player*self.board.astype(np.float32).flatten())
    
    def game_status(self):
       column_sum = np.sum(self.board,axis = 0) 
       row_sum = np.sum(self.board,axis = 1)
       p_diagonal = np.trace(self.board)
       nonp_diagonal = np.trace(np.fliplr(self.board))
      
       if (-3 in column_sum) or (3 in column_sum):
          return -self.current_player
       if (-3 in row_sum) or (3 in row_sum):
          return -self.current_player
       if (-3 == p_diagonal) or (3 == p_diagonal):
          return -self.current_player
       if (-3 == nonp_diagonal) or (3 == nonp_diagonal):
          return -self.current_player
       if np.all(self.board != 0):
          return 0
       
       return None # game still going on
    
    def display(self):
       print(f"Player {self.current_player} will make move")
       print(f"Current game status = {self.game_status()}")
       print(self.board)
       print("------------------")
       print()