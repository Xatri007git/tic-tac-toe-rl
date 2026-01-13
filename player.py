import numpy as np
import torch.nn as nn
import torch
from itertools import product
from game import Game
import random
from math import inf
class RLPlayer:
    def __init__(self):
        self.p_values = [0,1,-1]
        p_states = np.array(list(product(self.p_values,repeat=9))).reshape(-1,9)
        p_board_strings = [str(row) for row in p_states]
        self.qtable = dict(zip(p_board_strings,[np.random.rand(9) for _ in range(pow(3,9))]))
        self.ntable = dict(zip(p_board_strings,[np.zeros(9,dtype=int) for _ in range(pow(3,9))]))
    
    def evaluate(self,state):
        return self.qtable[state]

class NNRLPlayer(nn.Module):
   def __init__(self):
    super(NNRLPlayer,self).__init__()
    self.fc1 = nn.Linear(9,32)
    self.fc2 = nn.Linear(32,16)
    self.fc3 = nn.Linear(16,9)

   def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    return x

class MMPlayer:
    def minimax(self,game:Game,depth,player,max_player):
       if player == max_player:
           best = [-1,-1,-inf]
       else:
           best = [-1,-1,+inf]
    
       if depth == 0 or game.game_status() is not None: # the game is over
           if game.game_status() == max_player:
               score = 1
           elif game.game_status() == -max_player:
               score = -1
           else:
               score = 0 ## draw or game not over
           return [-1,-1,score]
       
       legal_moves = game.legal_moves()
       for move in legal_moves:
           game.make_move(move//3,move%3)
           score = self.minimax(game,depth-1,-player,max_player)
           game.unmake_move(move//3,move%3)
           score[0] = move // 3
           score[1] = move % 3 
           
           if player == max_player:
               if score[2] > best[2]:
                   best = score
           else:
               if score[2] < best[2]:
                   best = score
               
       return best

class manual_player:
    def manual_move(self,game:Game):
       a, b = map(int, input("Enter two numbers separated by a comma: ").split(','))
       game.make_move(a,b)

class random_player:
   def random_move(self,game:Game):
      move = random.choice(game.legal_moves())
      game.make_move(move//3,move%3)

class match_maker:
    def create_perfect_match(self,max_player:int,rlPlayer:RLPlayer): # max_player is the perfect player
        game = Game()
        perfectPlayer = MMPlayer()
        # first move is manual player
        while game.game_status() == None:
         #game.display()
         legal_moves = game.legal_moves()

         if game.current_player != max_player: 
           optimal_move = legal_moves[0]
           for move in legal_moves:
               if rlPlayer.qtable[game.game_state()][move] > rlPlayer.qtable[game.game_state()][optimal_move]:
                   optimal_move = move
           game.make_move(optimal_move//3,optimal_move%3)

         else:
           best_move = perfectPlayer.minimax(game,len(legal_moves),game.current_player,max_player)
           game.make_move(best_move[0],best_move[1])
        
        #game.display()
        return game.game_status()
    
    def create_random_match(self,rand_player:int,rlPlayer:RLPlayer):
        game = Game()
        randomPlayer = random_player()
        # first move is manual player
        while game.game_status() == None:
         #game.display()
         legal_moves = game.legal_moves()

         if game.current_player != rand_player: 
           optimal_move = legal_moves[0]
           for move in legal_moves:
               if rlPlayer.qtable[game.game_state()][move] > rlPlayer.qtable[game.game_state()][optimal_move]:
                   optimal_move = move
           game.make_move(optimal_move//3,optimal_move%3)

         else:
           randomPlayer.random_move(game)
        
        #game.display()
        return game.game_status()
    
    def create_nn_match(self,rand_player:int, rlplayer: NNRLPlayer):
        game = Game()
        randomPlayer = random_player()
        # first move is manual player
        while game.game_status() == None:
         #game.display()
         legal_moves = game.legal_moves()

         if game.current_player != rand_player: 
           optimal_move = legal_moves[0]
           for move in legal_moves:
               if rlplayer(game.game_state_tensor())[move] > rlplayer(game.game_state_tensor())[optimal_move]:
                   optimal_move = move
           game.make_move(optimal_move//3,optimal_move%3)

         else:
           randomPlayer.random_move(game)
        
        #game.display()
        return game.game_status()
           