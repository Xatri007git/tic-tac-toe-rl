import ast
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class logger:
 def plot(self,file_name,plot_name):
  tuples = []
  with open(file_name, "r") as f:
     for line in f:
        line = line.strip()
        if line:
            tuples.append(ast.literal_eval(line))

  wins_rl_asfp = []
  wins_rl_assp = []
  for x,y in tuples:
   wins_rl_asfp.append(x)
   wins_rl_assp.append(y)

  x_axis = range(len(wins_rl_asfp))
  plt.figure(figsize=(16,9))
  plt.plot(x_axis,wins_rl_asfp)
  plt.plot(x_axis,wins_rl_assp)
  plt.savefig(plot_name)

 def avg_plot(self,moving_avg_1,moving_avg_2,window,plot_name):
    plt.figure(figsize=(16,9))
    plt.plot(moving_avg_1, label=f'{window}-Game Average', color='blue', linewidth=1)
    plt.plot(moving_avg_2, label=f'{window}-Game Average', color='orange', linewidth=1)
    plt.title('DQN Learning Progress (Tic-Tac-Toe)')
    plt.xlabel('Eval')
    plt.ylabel(f'Avg win rate over last {window} tests')
    plt.legend()
    plt.savefig(plot_name)

