from MCTS import MCTS
from OthelloGame import OthelloGame
from OthelloLogic import Board
from model.NNet import NNetWrapper as NNet
import numpy as np
from utils import *
import pygame

N = 8
background = pygame.image.load('./back.jpg')
WIDTH = 720
BOX = 720/(8+2) # 棋盘每行有8个棋子，两侧各有两个棋子空间作为边界，理解不了就运行一下看画出来的实际效果
screen = pygame.display.set_mode((WIDTH,WIDTH))
pygame.display.set_caption("黑白棋")

def show_board(board,surf = screen,valid = [],last=None):
    # 使用blit给对应窗口绘制图像，参数:图像数据，图像位置
    surf.blit(background,(0,0))
    
    BOUNDS = [
        ((BOX,BOX),(BOX,WIDTH-BOX)), # 定义一个包含绘制棋盘边界线的坐标的列表，注意这里是边界线，不包含棋盘内部的棋盘格子线 a
        ((WIDTH-BOX,BOX),(BOX,BOX)), # b
        ((WIDTH-BOX,WIDTH-BOX),(WIDTH-BOX,BOX)), # c
        ((WIDTH-BOX,WIDTH-BOX),(BOX,WIDTH-BOX))# d
    ]

    for line in BOUNDS:
        pygame.draw.line(surf,(0,0,0),line[0],line[1],1)

    for i in range(N-1): # 0-7
        pygame.draw.line(surf, (0,0,0), # 画出棋盘格的竖线
                         (BOX * (2 + i), BOX),
                         (BOX * (2 + i), WIDTH - BOX))
        pygame.draw.line(surf, (0,0,0), # 画出棋盘格的横线
                         (BOX, BOX * (2 + i)),
                         (WIDTH - BOX, BOX * (2 + i)))
    for i in range(board.shape[0]):
        for j in range(board.shape[0]):
            if board[i][j] == 1: # board值为1绘制白棋
                t = (int((j+1.5)*BOX),int((i+1.5)*BOX))
                pygame.draw.circle(surf,(255,255,255),t,int(BOX/2))
            if board[i][j] == -1: # board值为-1绘制黑棋
                t = (int((j+1.5)*BOX),int((i+1.5)*BOX))
                pygame.draw.circle(surf,(0,0,0),t,int(BOX/2)) # t是绘制的圆的中心点的坐标，int(BOX/2)表示绘制的圆的半径
    
    for i in range(len(valid)): # 遍历有效移动列表，如果某个位置可行，则在相应位置绘制一个绿色小圆，索引指标0---len(valid)-1
        if valid[i] == 1: 
            x = int(i/8)
            y = i%8
            t = (int((y+1.5)*BOX),int((x+1.5)*BOX))
            pygame.draw.circle(surf,(0,255,0),t,int(BOX/4))

    if last != None: # 如果存在上一步的位置信息 (last)，则在该位置绘制一个粉色小圆。
        x = last[0]
        y = last[1]
        t = (int((y+1.5)*BOX),int((x+1.5)*BOX))
        pygame.draw.circle(surf,(255,195,203),t,int(BOX/4))
    pygame.display.flip()



g = OthelloGame(8)
n1 = NNet(g)
n1.load_checkpoint('E:\\强化学习\\code\\checkpoint','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


if __name__ == "__main__":
    curplayer = -1
    step = 1    #先后手 step = 1表示人类先手

    if step == 1 and curplayer == -1:print("本局您执黑棋，先手")
    elif step == 1 and curplayer == 1: print("本局您执白棋，先手")
    elif step == 0 and curplayer == -1: print("本局您执黑棋，后手")
    else: print("本局您执白棋，后手")
    
    board = g.getInitBoard()
    show_board(board)
    last = None
    running = True
    while running:
        # if g.getGameEnded(board, 1)!=0 and g.getGameEnded(board,-1) != 0: break
        if g.getEmpty(board) == False: break;
        valid = g.getValidMoves(g.getCanonicalForm(board, curplayer), 1) # 获取最初的可以移动的位置，初始化棋面有四个子之后就会有可以下的位置（绿点提示）g.getCanonicalForm(board, curplayer)完成棋面矩阵board的每个元素和curplayer(1)的相乘操作
        if valid[-1] == 1: #当前应该下棋的人没有符合规则的可以下的位置，所以只能让对手下，也即跳步
            print("当前没有符合规则的落子位置,请跳步")
            curplayer = -curplayer
            step = 0
        else:
            show_board(board,valid=valid,last=last)

        b = Board(8)
        b.pieces = np.copy(board)
        print("棋面白子数量:",b.countDiff(1), "棋面黑子数量:",b.countDiff(-1)) # 这里不是绝对数量，而是相对数量，countDiff(1)即白子比黑子多几个，countDiff(-1)黑子比白子多几个

        while step%2:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # 检查您是否按下了pygame事件窗口上的交叉按钮
                    step = 0 # 用于退出while step%2:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN: # 鼠标按下
                    grid = (int(event.pos[1] / (BOX + .0))-1,int(event.pos[0] / (BOX + .0))-1) # 计算出落子位置，横纵坐标范围[0,7]
                    if grid[0] >= 0 and grid[0] < 8 and grid[1] >= 0 and grid[1] < 8:
                        a = g.n*grid[0]+grid[1] # valid的索引是1维的，所以才会有47-49行的处理
                        if valid[a]: # 表示是有效的落子位置
                            print("有效落子：", grid[0], grid[1])
                            board, curplayer = g.getNextState(board, curplayer, a) # 更新board矩阵状态，将刚刚落子位置的值从0改为curplayer的值（1或者-1），返回的curplayer是现在的curplayer的相反值，因为你刚刚已经落子了，所以轮换棋手下棋
                            step = 0 #while step%2只会支持1次选手的落子，落完就结束了while循环，而不是支持一个回合（人和电脑都）落子
                        else:
                            print("落子不符合黑白棋规则，请重新落子！")
                    else:
                        print("落子超出棋盘边界，请重新落子！")
        if running == False: break
        # if g.getGameEnded(board, 1) != 0 and g.getGameEnded(board,-1) != 0: break
        if g.getEmpty(board) == False: break;
        show_board(board)# 每次人落子后都实时显示出来

        b = Board(8)
        b.pieces = np.copy(board)
        print("棋面白子数量:",b.countDiff(1), "棋面黑子数量:",b.countDiff(-1))

        action = n1p(g.getCanonicalForm(board, curplayer)) # AI落子
        valid = g.getValidMoves(g.getCanonicalForm(board,curplayer),1)
        last = (int(action/8),int(action%8))
        print("AI落子:", last[0], last[1])

        board, curplayer = g.getNextState(board, curplayer, action)
        step = 1

    b = Board(8)
    b.pieces = np.copy(board)

    if g.getEmpty(board) == False and b.countDiff(1) > 0:
        print("白棋胜利！胜",b.countDiff(1),"子")
    elif g.getEmpty(board) == False and b.countDiff(-1) > 0:
        print("黑棋胜利！胜",b.countDiff(-1),"子")
    elif g.getEmpty(board) == False and b.countDiff(1) == 0:
        print("游戏平局！")
    else:
        print("游戏尚可继续，但您强制结束了游戏！")

    running = True # 140-146保证棋面结束后仍然展示给人看，知道手动关闭pygame窗口
    show_board(board)

    while(running):
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False