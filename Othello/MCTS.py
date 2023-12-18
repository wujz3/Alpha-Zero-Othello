import numpy as np
EPS = 1e-8

# 添加了迪利克雷噪声和修改了UCT的MCTS
class MCTS():

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # 在状态s下采取动作a的预期奖励；
        self.Nsa = {}       # (s,a)被访问的次数
        self.Ns = {}        # 棋盘状态s被访问的次数；
        self.Ps = {}        # 对于s,由网络模拟的q-function得到的policy
        self.Es = {}        # 棋盘状态s对应的游戏状态，获胜、失败、打平或棋局没有结束；
        self.Vs = {}        # stores game.getValidMoves for board s
        self.rng = np.random.default_rng()

    def getActionProb(self, canonicalBoard, temp=1):
        """
        函数在 stimulate MCTS 与 canonicalBoard 对应节点 numMCTSSims次之后,
        返回当前canonicalBoard 对应的policy
        格式为：
        [ probability for a in all action ]
        """
        for i in range(self.args.numMCTSSims):
            self.stimulate(canonicalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        pi = [ self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 \
                for a in range(self.game.getActionSize())]
        if temp == 0:
            #返回one-hot形式的结果
            return [ 1 if a == np.argmax(pi) else 0 for a in range(len(pi)) ] # temp==0的情况下，设定最大概率的aciton为1，其他为0，返回
        else:
            #返回直接归一化的结果，temp!=0的情况下，把各步棋概率归一化，然后返回
            sum_pi = float(np.sum(pi))
            return [ p/sum_pi for p in pi ]

    def stimulate(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Es:
            #是不是端节点
            self.Es[s] = self.game.getGameEnded(canonicalBoard,1)
        if self.Es[s] != 0: # 如果棋局结束了
            return -self.Es[s]
            #由于canonicalBoard是固定当前player为1的，
            #所以回传上一个节点的v值时要取反
        if s not in self.Ps:
            #是不是未扩展过的新节点，最开始的根节点就是这个状态，由于算法之前并未见过这个棋面，因此输入的棋面实际上是一个“叶子”状态节点，需要先“扩展”这个节点，扩展后它就不再是叶子节点了
            self.Ps[s],v = self.nnet.predict(canonicalBoard) # 网络计算Policy和V值
            valids = self.game.getValidMoves(canonicalBoard,1)
            # self.Ps[s] = self.Ps[s]*valids # 将非法动作的概率置为0
            # 添加迪利克雷噪声
            self.applyDirNoise(self.Ps[s], valids)
            ps_sum = np.sum(self.Ps[s])
            if ps_sum > 0:
                self.Ps[s] = self.Ps[s]/ps_sum # 归一化概率分布
            else:
                #如果Ps[s]的总和为0，那么证明所有的valid action都是0
                #如果这种情况多次出现，网络的训练很可能出现了问题
                print("All the valid actions are masked!!!")
                self.Ps[s]+=valids
                self.Ps[s]=self.Ps[s]/np.sum(self.Ps[s])
            self.Vs[s]=valids
            self.Ns[s]=0
            #这里并没有访问这个节点，只是创建了它
            return -v
        # 如果为非叶子节点，而是中间节点，合法落子可以从以前保存的信息中取出来。
        valids = self.Vs[s] # 选择阶段
        best_u = -float('Inf')
        best_a = -1
        self.Ns[s] += 1
        for a in range(len(valids)):
            #当前action合法
            if valids[a] == 1:
                if (s,a) in self.Qsa:
                    # 根据论文，修改UCT的计算公式，将PI放入根号
                    u = self.Qsa[(s,a)]+ \
                        self.args.cpuct*np.sqrt(self.Ps[s][a]*np.log(self.Ns[s]+EPS)/(1+self.Nsa[(s,a)]))
                else:
                    u = self.args.cpuct*np.sqrt(self.Ps[s][a]*np.log(self.Ns[s]+EPS))
                if u > best_u:
                    best_u = u
                    best_a = a
        a = best_a
        b,next_player = self.game.getNextState(canonicalBoard,1,a)
        next_canonicalBoard = self.game.getCanonicalForm(b,next_player)
        v = self.stimulate(next_canonicalBoard) # 展开与评估阶段
        if (s,a) not in self.Qsa:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
        else: # 回传阶段
            self.Qsa[(s,a)] = (self.Qsa[(s,a)]*self.Nsa[(s,a)]+v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        return -v
    
    def applyDirNoise(self, Ps, Vs):
        dir_values = self.rng.dirichlet([0.3] * np.count_nonzero(Vs))
        dir_idx = 0
        for idx in range(len(Ps)):
            if Vs[idx]:
                Ps[idx] = (0.75 * Ps[idx]) + (0.25 * dir_values[dir_idx])
                dir_idx += 1
    
    def softmax(Ps, softmax_temp):
        if softmax_temp == 1.:
            return Ps
        result = Ps ** (1. / softmax_temp)
        normalise(result)
        return result.astype(np.float32)

    def normalise(vector):
        sum_vector = np.sum(vector)
        vector /= sum_vector
            

# 原始的MCTS        
class MCTS_origin():

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # 在状态s下采取动作a的预期奖励；
        self.Nsa = {}       # (s,a)被访问的次数
        self.Ns = {}        # 棋盘状态s被访问的次数；
        self.Ps = {}        # 对于s,由网络模拟的q-function得到的policy
        self.Es = {}        # 棋盘状态s对应的游戏状态，获胜、失败、打平或棋局没有结束；
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        函数在 stimulate MCTS 与 canonicalBoard 对应节点 numMCTSSims次之后,
        返回当前canonicalBoard 对应的policy
        格式为：
        [ probability for a in all action ]
        """
        for i in range(self.args.numMCTSSims):
            self.stimulate(canonicalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        pi = [ self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 \
                for a in range(self.game.getActionSize())]
        if temp == 0:
            #返回one-hot形式的结果
            return [ 1 if a == np.argmax(pi) else 0 for a in range(len(pi)) ] # temp==0的情况下，设定最大概率的aciton为1，其他为0，返回
        else:
            #返回直接归一化的结果，temp!=0的情况下，把各步棋概率归一化，然后返回
            sum_pi = float(np.sum(pi))
            return [ p/sum_pi for p in pi ]

    def stimulate(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Es:
            #是不是端节点
            self.Es[s] = self.game.getGameEnded(canonicalBoard,1)
        if self.Es[s] != 0: # 如果棋局结束了
            return -self.Es[s]
            #由于canonicalBoard是固定当前player为1的，
            #所以回传上一个节点的v值时要取反
        if s not in self.Ps:
            #是不是未扩展过的新节点，最开始的根节点就是这个状态，由于算法之前并未见过这个棋面，因此输入的棋面实际上是一个“叶子”状态节点，需要先“扩展”这个节点，扩展后它就不再是叶子节点了
            self.Ps[s],v = self.nnet.predict(canonicalBoard) # 网络计算Policy和V值
            valids = self.game.getValidMoves(canonicalBoard,1)
            self.Ps[s] = self.Ps[s]*valids # 将非法动作的概率置为0
            ps_sum = np.sum(self.Ps[s])
            if ps_sum > 0:
                self.Ps[s] = self.Ps[s]/ps_sum # 归一化概率分布
            else:
                #如果Ps[s]的总和为0，那么证明所有的valid action都是0
                #如果这种情况多次出现，网络的训练很可能出现了问题
                print("All the valid actions are masked!!!")
                self.Ps[s]+=valids
                self.Ps[s]=self.Ps[s]/np.sum(self.Ps[s])
            self.Vs[s]=valids
            self.Ns[s]=0
            #这里并没有访问这个节点，只是创建了它
            return -v
        # 如果为非叶子节点，而是中间节点，合法落子可以从以前保存的信息中取出来。
        valids = self.Vs[s] # 选择阶段
        best_u = -float('Inf')
        best_a = -1
        self.Ns[s] += 1
        for a in range(len(valids)):
            #当前action合法
            if valids[a] == 1:
                if (s,a) in self.Qsa:
                    #这里的uct公式不是最初的uct公式，在exploration的模块额外乘了p，以便在最初
                    #没有访问经验时能够做出选择。
                    u = self.Qsa[(s,a)]+ \
                        self.args.cpuct*self.Ps[s][a]*np.sqrt(np.log(self.Ns[s]+EPS)/(1+self.Nsa[(s,a)]))
                else:
                    u = self.args.cpuct*self.Ps[s][a]*np.sqrt(np.log(self.Ns[s]+EPS))
                if u > best_u:
                    best_u = u
                    best_a = a
        a = best_a
        b,next_player = self.game.getNextState(canonicalBoard,1,a)
        next_canonicalBoard = self.game.getCanonicalForm(b,next_player)
        v = self.stimulate(next_canonicalBoard) # 展开与评估阶段
        if (s,a) not in self.Qsa:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
        else: # 回传阶段
            self.Qsa[(s,a)] = (self.Qsa[(s,a)]*self.Nsa[(s,a)]+v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        return -v