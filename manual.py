from player import manual_player, random_player, MMPlayer
from game import Game
from player import RLPlayer
import pickle

Man = MMPlayer()
theGame = Game()
player = RLPlayer()
q_table = "q_table_sarsa_ofpol_v5.pkl"

with open(q_table,'rb') as file:
  player.qtable = pickle.load(file)

print(len(player.qtable))
x = 1
total_wins = 0
for i in range(1):
  print(f'{i+1}')
  theGame = Game()
  while theGame.game_status() == None:
     
     if(x == -1):
        best = Man.minimax(theGame,len(theGame.legal_moves()),x,x)
        theGame.make_move(best[0],best[1])
      #   Man.manual_move(theGame)
     else:
        print(player.qtable[theGame.game_state()])
        legal_moves = theGame.legal_moves()
        all_actions = player.qtable[theGame.game_state()]
        action = legal_moves[0]
        
        for move in legal_moves:
            if all_actions[move] > all_actions[action]:
                action = move

        theGame.make_move(action // 3,action % 3)
      
     x *= -1
     if(theGame.game_status() == 0):
      total_wins += 1
     
     theGame.display()

print(total_wins)