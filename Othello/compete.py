from Coach import Coach
from OthelloGame import OthelloGame as Game
from model.NNet import NNetWrapper as nn
from model.NNet import NNetWrapper_origin as nno
from utils import dotdict
from Arena import Arena
from MCTS import MCTS, MCTS_origin
import numpy as np

args = dotdict({
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.给定一个棋面，MCTS共进行的模拟次数
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'load_folder_file': ('/temp','best.pth.tar'),
    # 'load_folder_file': ('/checkpoint','best.pth.tar'),
    'load_folder_file_test': ('/temp_noise_uct','best_only_addnoise.pth.tar'),

})
g = Game(8)

# 当运行原始模型的结果，即'load_folder_file': ('/checkpoint','best.pth.tar')
# pnet = nno(g)

# 当运行改进网络的模型结果，即'load_folder_file': ('/temp','best.pth.tar')
pnet = nn(g)

# 当运行改进网络及MCT的新模型结果，即'load_folder_file_test': ('/temp_noise_uct','best_only_addnoise.pth.tar')
nnet = nn(g)

pnet.load_checkpoint(folder = args.load_folder_file[0], filename = args.load_folder_file[1])
nnet.load_checkpoint(folder = args.load_folder_file_test[0], filename = args.load_folder_file_test[1])

# 选择原始的MCTS算法
pmcts = MCTS_origin(g, pnet, args)

# 选择新的MCTS算法
nmcts = MCTS(g, nnet, args)
print('开始与之前网络对抗。')
arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
              lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), g)
pwins, nwins, draws = arena.playGames(args.arenaCompare)
print('对抗中新网络的胜场数目：{:2d}'.format(nwins))
print('对抗中新网络的胜率：{:4f}'.format(nwins/args.arenaCompare))
print('--------------------------------')

with open('compete.txt', 'a') as file:
                file.write('对抗中新网络的胜场数目：{:2d},对抗中新网络的胜率：{:4f}\n'.format( nwins, nwins/args.arenaCompare))