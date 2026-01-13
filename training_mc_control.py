import numpy as np
import pickle
import random
import os
import ast
from math import inf 
from game import Game
from player import RLPlayer,match_maker
from collections import deque
from test import logger

def read_tuple_file(file_name):
  tuples = []
  with open(file_name, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            tuples.append(ast.literal_eval(line))
  return tuples

q_table_mcc_old = "q_table_mcc_v4.pkl"
q_table_mcc_new = "q_table_mcc_v4.pkl"
wins_rl_mcc = "wins_rl_mcc_v4.txt"
wins_pl_mcc = "wins_pl_mcc_v4.txt"
draws_mcc = "draws_mcc_v4.txt"
plot_mcc = "plot_mcc_v4.png"

rlplayer = RLPlayer()
if os.path.exists(q_table_mcc_old):
 with open(q_table_mcc_old,"rb") as f:
  rlplayer.qtable = pickle.load(f)

theGame = None
Logger = logger()
epochs = 200000
test_interval = 100
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9999885
min_epsilon = 0.1
n_test_games = 100
alpha = 0.2 ## learning rate
cap = 10000
batch_size = 32
checkpoint_interval = 1000
replay_buffer = deque() ## will only store the latest completed episode 

wins_rl = [] #(rl wins as first player, rl wins as second player)
wins_mm = [] #(minimax wins as first player, minimax wins as second player)
draws = [] #(draws with rl as first player, draws with rl as second player)

for epoch in range(1,epochs+1):
 theGame = Game()
 replay_buffer.clear()

 while theGame.game_status() == None:
    ## epsilon greedy approach
    legal_moves = theGame.legal_moves()
    player = theGame.current_player ## the player who takes the action
    state = theGame.game_state() 
    next_legal_moves = None
    reward = 0
    next_state = None
    action = None
    done = False
    reward = None
    
    if np.random.rand() < epsilon :
        # random move 
        action = random.choice(legal_moves)
        theGame.make_move(action//3,action%3)
        next_state = theGame.game_state()
        done = theGame.game_status() != None
        next_legal_moves = theGame.legal_moves()
    else:
        all_actions = rlplayer.qtable[state]
        action = legal_moves[0]
        for move in legal_moves:
            if all_actions[move] > all_actions[action]:
                action = move
        
        theGame.make_move(action//3,action%3)
        next_state = theGame.game_state()
        done = theGame.game_status() != None
        next_legal_moves = theGame.legal_moves()
     
    if done :
      if theGame.game_status() == 0:
          reward = 0
      elif theGame.game_status() == player:
          reward = 1
      else:
          reward = -1
    else:
      reward = 0
    
    replay_buffer.append((state,action,reward,next_state,next_legal_moves,done))
 
 t = len(replay_buffer)-1
 G = 0
 while(t >= 0):
   G *= gamma
   data = replay_buffer[t]
   s,a,r,ns,nls,d = data
   G += r
   rlplayer.ntable[s][a] += 1
   rlplayer.qtable[s][a] = rlplayer.qtable[s][a] + ( (G - rlplayer.qtable[s][a])/rlplayer.ntable[s][a] )
   t -= 1;

 if epoch % test_interval == 0:
    print(f"This is {epoch//test_interval}th testing")
    match = match_maker()

    rl_player_wins_asfp = 0 ## rl player as first player
    rl_player_wins_assp = 0 ## rl player as second player
    draws_with_rl_asfp = 0 ## no. of draws with rl player as first
    draws_with_rl_assp = 0 ## no. of draws with rl player as second
    p_player_wins_asfp = 0
    p_player_wins_assp = 0
       
    count = 0
    for i in range(n_test_games):
      result_rl_asfp = match.create_random_match(-1,rlplayer)
        
      if i % 5 == 0:
        print((100*count)//(2*n_test_games),end='\r')
        
      count += 1

      if result_rl_asfp == 1:
        rl_player_wins_asfp += 1
      elif result_rl_asfp == -1:
        p_player_wins_assp += 1
      else:
        draws_with_rl_asfp += 1
       
    for i in range(n_test_games):
      result_rl_assp = match.create_random_match(1,rlplayer)

      if i % 5 == 0:
        print((100*count)//(2*n_test_games),end='\r')
        
      count += 1

      if result_rl_assp == 1:
        p_player_wins_asfp += 1
      elif result_rl_assp == -1:
        rl_player_wins_assp += 1
      else:
        draws_with_rl_assp += 1
        
    wins_rl.append((rl_player_wins_asfp*100/n_test_games,rl_player_wins_assp*100/n_test_games))
    wins_mm.append((p_player_wins_asfp*100/n_test_games,p_player_wins_assp*100/n_test_games))
    draws.append((draws_with_rl_asfp*100/n_test_games,draws_with_rl_assp*100/n_test_games))
       
    print()
 
 if epoch % checkpoint_interval == 0:
    with open(q_table_mcc_new,"wb") as f: ## writing the qtable
     pickle.dump(rlplayer.qtable,f)
    
    with open(wins_rl_mcc,"w") as file:
      for v in wins_rl:
       file.write(f"{v}\n")
    
    with open(wins_pl_mcc,"w") as file:
      for v in wins_mm:
       file.write(f"{v}\n")
    
    with open(draws_mcc,"w") as file:
      for v in draws:
       file.write(f"{v}\n")
    
    Logger.plot(wins_rl_mcc,plot_mcc)
    
 epsilon = max(epsilon*epsilon_decay,min_epsilon)
 

    # experience replay
    