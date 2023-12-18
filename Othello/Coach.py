from collections import deque
from Arena import Arena
from MCTS import MCTS, MCTS_origin
import numpy as np
import os, sys
from pickle import Pickler, Unpickler
from random import shuffle

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False    # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        # 实现自我对弈，每一步棋保存iterationTrainExamples中，用于输入神经网络中
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        board = self.game.getInitBoard()
        cur_player = 1
        step = 0
        temp = 1
        train_examples=[]
        while True:
            if step > self.args.tempThreshold: #self.args.tempThreshold 设置为15，temp如果小于15，则按照概率选择下一步棋；temp如果大于15，
                # 则只下最大概率的棋。为什么这样呢？就如同围棋，刚开始下棋的时候，大局观重要，下错了，关系不大，按照概率下棋比较好；
                # 后面到了贴身搏杀的时候，下错一步，大龙被杀，所以只能下确定性的棋（最大概率）。
                temp = 0
                # self.args.is_dis = False
            step += 1
            #由于mcts返回的所有pi都是针对cur_play=1的情况
            #所以要把board转换以得到正确的pi
            #随后仍是用现在的board进行self-play
            #但是放入train_examples的board是转换后的
            c_board = self.game.getCanonicalForm(board,cur_player)
            # 2.内部执行指定次数的MCTS模拟，得到联合策略pi。# 调用了蒙特卡罗树返回下一步落子的概率分布pi
            pi = self.mcts.getActionProb(c_board,temp)
            # 根据棋盘对称性，生成8盘棋，增加训练样本
            sym = self.game.getSymmetries(c_board,pi)

            # 3.将这时的board信息和对应策略存储，用于后续神经网络训练
            for (b,p) in sym:
                train_examples.append([b,p,cur_player])
            # 4.从策略中随机选择一个作为行动
            action = np.random.choice(self.game.getActionSize(),size=1,p=pi)
            # 5.根据前面的board，player和对应的action获取下一时刻状态
            board,cur_player = self.game.getNextState(board,cur_player,action)

            result = self.game.getGameEnded(board,cur_player) 
            if result!=0:
                #self-play 结束：此时完成了一局棋的自对弈模拟
                # 如果当前玩家 (cur_player) 是胜利者（e[2] 表示当前训练样本中的玩家），则 e[2] != cur_player 为 False，整个表达式等于 result * (-1) ** False，即奖励值 result 不变。
                # 如果当前玩家不是胜利者，那么 e[2] != cur_player 为 True，整个表达式等于 result * (-1) ** True，即奖励值 result 取反（乘以 -1）。
                return [ (e[0],e[1],result*((-1)**(e[2]!=cur_player))) for e in train_examples ]

    def learn(self):
        """
        执行指定numIters次数的迭代，每次迭代包含numEps episodes of self-play。每次迭代后，都会将trainExamples输入网络进行训练，
        每一局棋，重新初始化mcts（蒙特卡罗树）
        调用executeEpisode()函数下一局棋
        将100局棋谱保存到iterationTrainExamples链表中；

        将iterationTrainExamples棋谱链表增加到trainExamplesHistory中
        trainExamplesHistory为一个长度为20的双向队列，元素为20万棋谱链表；
        如果trainExamplesHistory的长度超过20，则删除第0个元素

        利用棋谱，训练nnet网络，得到新的模型

        创建arena对象，使新模型和老模型相互对弈
        如果新模型胜率低于60%，则拒绝新模型；若相反，则接受
        """
        for iter_idx in range(self.args.iteration+1,self.args.iteration+1+self.args.numIters):
            print('--------------------------------')
            print('Iter {:3d}:'.format(iter_idx))
            iteratorExam = deque([], maxlen=self.args.maxlenOfQueue)
            for ep in range(self.args.numEps): # numEps为100，即
                # 训练时，在自对弈前30步，添加狄利克雷噪声
                self.args.is_dis = True
                self.mcts=MCTS(self.game,self.nnet,self.args)
                examples = self.executeEpisode()
                iteratorExam.extend(examples)
            # 2.将一轮迭代的结果放入训练样例 trainExamplesHistory
            self.trainExamplesHistory.append(iteratorExam)
            #如果经验池满了，就丢弃掉最早的经验:
            print('经验池长度: {:3d}'.format(len(self.trainExamplesHistory)))
            if len(self.trainExamplesHistory) >= self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)
            self.saveTrainExamples(iter_idx)#保存训练样本
            trainExamples = []
            for exam in self.trainExamplesHistory:
                trainExamples.extend(exam)
            shuffle(trainExamples)#为了训练，重排样本
            print('训练样本长度:{}'.format(len(trainExamples)))
            #保存当前的网络
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            #读取最近保存的网络
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            #对抗时不添加狄利克雷噪声
            pmcts = MCTS(self.game,self.pnet,self.args)
            #训练
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game,self.nnet,self.args)
            #与之前的网络对抗
            print('开始与之前网络对抗。')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            if pwins+nwins == 0 or nwins/(nwins+pwins)< self.args.updateThreshold:
                #新网络胜场不足，不够厉害，不保留，回退到上一次的checkpoint
                print('新网络胜场不足，不保存，回退到上一次参数')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('保存新网络')
                #变强了，保存下来
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='{:2d}.pth.tar'.format(iter_idx))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best_only_addnoise.pth.tar')
            print('对抗中新网络的胜场数目：{:2d}'.format(nwins))
            print('对抗中新网络的胜率：{:4f}'.format(nwins/self.args.arenaCompare))
            print('--------------------------------')


            with open('rate_noise.txt', 'a') as file:
                file.write('numIters {}: nwins {:2d}, rate {:.4f}, train {}\n'.format(iter_idx, nwins, nwins/self.args.arenaCompare, len(trainExamples)))
            
              

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0],
            'checkpoint_' + str(self.args.iteration) + '.pth.tar')
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
