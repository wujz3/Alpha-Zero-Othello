import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
#from utils import *
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class OthelloNNet1(nn.Module):
    def __init__(self, game, args):
        self.board_size = game.getBoardSize() # 一个二元组(width,height)
        self.action_size= game.getActionSize()# action的size：width*height+1
        self.args = args
        super(OthelloNNet1, self).__init__()
        #初始化卷积层
        self.conv1 = nn.Conv2d(1,self.args.num_channels,3,padding=1)
        self.conv2 = nn.Conv2d(self.args.num_channels,self.args.num_channels,3,padding=1)
        self.conv3 = nn.Conv2d(self.args.num_channels,self.args.num_channels,3)
        self.conv4 = nn.Conv2d(self.args.num_channels,self.args.num_channels,3)
        self.bn1 = nn.BatchNorm2d(self.args.num_channels)
        self.bn2 = nn.BatchNorm2d(self.args.num_channels)
        self.bn3 = nn.BatchNorm2d(self.args.num_channels)
        self.bn4 = nn.BatchNorm2d(self.args.num_channels)
        #初始化连接层
        self.fc1 = nn.Linear(self.args.num_channels*(self.board_size[0]-4)*(self.board_size[1]-4),1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,self.action_size)
        self.fc4 = nn.Linear(self.action_size,1)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_bn2 = nn.BatchNorm1d(512)
    def forward(self, s):
        #s是当前的state，也就是棋盘的输入
        s = s.view(-1,1,self.board_size[0],self.board_size[1])#改变输入形状
        s = F.relu((self.bn1(self.conv1(s))))
        s = F.relu((self.bn2(self.conv2(s))))
        s = F.relu((self.bn3(self.conv3(s))))
        s = F.relu((self.bn4(self.conv4(s))))
        #卷积，由于有两层没有pad1，所以宽和高分别减少了2*2=4
        s = s.view(-1,self.args.num_channels*(self.board_size[0]-4)*(self.board_size[1]-4))
        s = F.dropout(self.fc_bn1(self.fc1(s)),p=self.args.dropout,training=self.training)
        s = F.dropout(self.fc_bn2(self.fc2(s)),p=self.args.dropout,training=self.training)
        pi = self.fc3(s)
        v = self.fc4(pi)
        return F.log_softmax(pi,dim=1),F.tanh(v)


'''
    残差块结构
    深度学习残差网络的基本结构
'''
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

'''
    深度神经网络模型类：网络对象
    只包含网络结构，网络训练、预测在Net.py中
'''
class OthelloNNet2(nn.Module):
    
    '''
        初始化网络
        网络参数：
        board_x、board_y:棋盘大小
        action_num:动作最多数量
        args：参数选择，通过训练网络传入
    '''
    def __init__(self, game, args):
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # 网络卷积层、残差块设置
        super(OthelloNNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.resblock1 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock2 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock3 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock4 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock5 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock6 = ResidualBlock(args.num_channels, args.num_channels, stride=1)
        self.resblock7 = ResidualBlock(args.num_channels, args.num_channels, stride=1)

        # 网络BN层设置
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)

        # 网络全连接层及其BN层设置
        self.fc1 = nn.Linear(args.num_channels * (self.board_size[0]-4)*(self.board_size[1]-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    
    '''前向传播过程，代码后的注释为输入的大小（输出后的大小）'''
    def forward(self, s):
                                                                    # batch_size * board_x * board_y
        s = s.view(-1, 1, self.board_size[0], self.board_size[1])               # batch_size * 1 * board_x * board_y
        s = F.relu(self.bn1(self.conv1(s)))                         # batch_size * num_channels * board_x * board_y 
        s = self.resblock1(s)
        s = self.resblock2(s)
        s = self.resblock3(s) 
        s = self.resblock4(s) 
        s = self.resblock5(s)
        s = self.resblock6(s)
        s = self.resblock7(s)
        s = F.relu(self.bn2(self.conv2(s)))                         # batch_size * num_channels * (board_x-2) * (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))                         # batch_size * num_channels * (board_x-4) * (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_size[0]-4) * (self.board_size[1]-4))

        # 使用dropout层来增强网络效果
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p= self.args.dropout, training= self.training)  # batch_size * 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p= self.args.dropout, training= self.training)  # batch_size * 512

        # 损失,分别对应action概率向量(策略)和分数(值)
        p = self.fc3(s)                                                                          # batch_size * action_size
        v = self.fc4(s)                                                                          # batch_size * 1

        return F.log_softmax(p, dim=1), torch.tanh(v)




# 改进后的网络
class NNetWrapper():
    def __init__(self, game):

        self.nnet = OthelloNNet2(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.loss_p=nn.CrossEntropyLoss()#对policy的交叉熵
        self.loss_v=nn.MSELoss()#对v的最小均方误差
        if args.cuda:
            self.nnet.cuda(1)
        #提前设置cuda

    def train(self, examples):
        """
        输入一个元素为example的列表，列表长度为经验池的长度
        每个example组成为
        board: 8*8 ndarray
        pi : 65 ndarray
        r : 1
        """
        optimizer = optim.Adam(self.nnet.parameters(),lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_idx = 0
            idx_list = np.arange(len(examples))
            np.random.shuffle(idx_list)
            #重排numpy的序号列表，然后直接按间隔取出要训练的sample对应的序号
            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = idx_list[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).cuda(1)
                target_pis = torch.FloatTensor(np.array(pis)).cuda(1)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).cuda(1)

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.cuda(1), target_pis.cuda(1), target_vs.cuda(1)
                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v.view(-1))
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                #loss分开记录
                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                batch_idx += 1
            print('pi_loss: {:.4f}'.format(pi_losses.avg))
            print('v_loss : {:.4f}'.format(v_losses.avg))
            with open('losses_noise.txt', 'a') as file:
                file.write('Epoch {}: pi_loss {:.4f}, v_loss {:.4f}\n'.format(epoch + 1, pi_losses.avg, v_losses.avg))
        # with open('losses.txt', 'r') as file:
        #     lines = file.readlines()

        # epochs = []
        # pi_losses = []
        # v_losses = []

        # # Parse the data
        # for line in lines:
        #     parts = line.split(': ')
        #     epoch_part, losses_part = parts[0], parts[1]
        #     epoch = int(epoch_part.split()[1])
        #     pi_loss, v_loss = map(float, losses_part.split(','))
        #     epochs.append(epoch)
        #     pi_losses.append(pi_loss)
        #     v_losses.append(v_loss)

        # # Plot the losses
        # plt.plot(epochs, pi_losses, label='pi_loss')
        # plt.plot(epochs, v_losses, label='v_loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig('loss_plot.png')
        # plt.show()
            
    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        #board = torch.FloatTensor(board.astype(np.float64)).cuda()
        if torch.cuda.is_available():
            board = torch.FloatTensor(board.astype(np.float64)).cuda(1)
        else:
            board = torch.FloatTensor(board.astype(np.float64))
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        '''
        求两个输入的交叉熵损失。
        由于onnet返回pi时，用了torch.log_softmax函数。
        所以这里按照交叉熵公式直接做点积就可。
        '''
        return -torch.sum(targets*outputs)/targets.size()[0]


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

# 原始的网络
class NNetWrapper_origin():
    def __init__(self, game):
        
        self.nnet = OthelloNNet1(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.loss_p=nn.CrossEntropyLoss()#对policy的交叉熵
        self.loss_v=nn.MSELoss()#对v的最小均方误差
        if args.cuda:
            self.nnet.cuda(0)
        #提前设置cuda

    def train(self, examples):
        """
        输入一个元素为example的列表，列表长度为经验池的长度
        每个example组成为
        board: 8*8 ndarray
        pi : 65 ndarray
        r : 1
        """
        optimizer = optim.Adam(self.nnet.parameters(),lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_idx = 0
            idx_list = np.arange(len(examples))
            np.random.shuffle(idx_list)
            #重排numpy的序号列表，然后直接按间隔取出要训练的sample对应的序号
            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = idx_list[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).cuda(0)
                target_pis = torch.FloatTensor(np.array(pis)).cuda(0)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).cuda(0)

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.cuda(0), target_pis.cuda(0), target_vs.cuda(0)
                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v.view(-1))
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                #loss分开记录
                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                batch_idx += 1
            print('pi_loss: {:.4f}'.format(pi_losses.avg))
            print('v_loss : {:.4f}'.format(v_losses.avg))
            with open('losses_noise.txt', 'a') as file:
                file.write('Epoch {}: pi_loss {:.4f}, v_loss {:.4f}\n'.format(epoch + 1, pi_losses.avg, v_losses.avg))
        # with open('losses.txt', 'r') as file:
        #     lines = file.readlines()

        # epochs = []
        # pi_losses = []
        # v_losses = []

        # # Parse the data
        # for line in lines:
        #     parts = line.split(': ')
        #     epoch_part, losses_part = parts[0], parts[1]
        #     epoch = int(epoch_part.split()[1])
        #     pi_loss, v_loss = map(float, losses_part.split(','))
        #     epochs.append(epoch)
        #     pi_losses.append(pi_loss)
        #     v_losses.append(v_loss)

        # # Plot the losses
        # plt.plot(epochs, pi_losses, label='pi_loss')
        # plt.plot(epochs, v_losses, label='v_loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig('loss_plot.png')
        # plt.show()
            
    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        #board = torch.FloatTensor(board.astype(np.float64)).cuda()
        if torch.cuda.is_available():
            board = torch.FloatTensor(board.astype(np.float64)).cuda(0)
        else:
            board = torch.FloatTensor(board.astype(np.float64))
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        '''
        求两个输入的交叉熵损失。
        由于onnet返回pi时，用了torch.log_softmax函数。
        所以这里按照交叉熵公式直接做点积就可。
        '''
        return -torch.sum(targets*outputs)/targets.size()[0]


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
if __name__ == "__main__":
    met = NNetWrapper()
    import pdb; pdb.set_trace()