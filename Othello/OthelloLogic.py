#x col y row

directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)] # 表示8个方向

class Board():
    def __init__(self,n):
        self.n = n
        self.pieces = [[0]*n for i in range(n)]
        t = int(n/2)
        self.pieces[t-1][t] = 1 #白
        self.pieces[t][t-1] = 1
        self.pieces[t-1][t-1] = -1
        self.pieces[t][t] = -1
    
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color): # 返回棋面矩阵board中指定颜色的棋子比另一个颜色的棋子多几个
        cnt = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.pieces[i][j] == color:
                    cnt+=1
                if self.pieces[i][j] == -color:
                    cnt-=1
        return cnt

    def get_legal_moves(self, color):
        moves = set()

        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color: 
                    move = self.get_moves_for_square((x,y))
                    if move != None:
                        moves.update(move)
        return list(moves)

    def has_legal_moves(self, color):
        for x in range(self.n):
            for y in range(self.n):
                if self[x][y] == color: 
                    move = self.get_moves_for_square((x,y))
                    if len(move) > 0: return True

        return False

    def get_moves_for_square(self,square):
        x,y = square
        moves = []
        for direction in directions:
            move = self.find_move((x,y),direction, self[x][y])# 元组，方向，颜色
            if move != None:
                moves.append(move)

        return moves

    def find_move(self,square,direction,color):
        x,y = square# 检索起点
        flip = False # 用于表示沿途是否遇到了对手的棋子，遇到了（True）才能下在当前（x，y）的那个位置（黑白棋规则）
        moves = [] # 创建一个空列表 moves 用于存储沿着指定方向的所有可能移动。
        i = 1
        while True:
            point1 = x+i*direction[0]
            point2 = y+i*direction[1]
            if point1 >= 0 and point1 < self.n and point2 >= 0 and point2 < self.n:
                moves.append((point1,point2))
                i+=1
            else:
                break
        #print(square,direction,moves)
        for p1,p2 in moves:
            if self[p1][p2] == 0:
                if flip == True:
                    return (p1,p2)
                else:
                    return None # 不是合法的落子位置
            if self[p1][p2] == -color:
                flip = True
                continue
            if self[p1][p2] == color:
                return None
        
    def execute_move(self, move, color):
        flips = []
        for direction in directions:
            t = self._get_flips(move,direction,color)
            if t:
                #print(t)
                flips.extend(t)
        #print(flips)
        assert len(list(flips))>0
        for x, y in flips:
            #print(self[x][y],color)
            self[x][y] = color
    

    def _get_flips(self,square,direction,color):
        #print('zzf')
        x,y = square
        flips = [(x,y)]
        moves = []
        i = 1
        while True:
            point1 = x+i*direction[0]
            point2 = y+i*direction[1]
            if point1 >= 0 and point1 < self.n and point2 >= 0 and point2 < self.n:
                moves.append((point1,point2))
                i+=1
            else:
                break
        #print(moves)
        #print((x,y),direction,moves)
        for p1, p2 in moves:
            #print(x,y)
            if self[p1][p2] == 0:
                return []
            if self[p1][p2] == -color:
                flips.append((p1, p2))
            elif self[p1][p2] == color and len(flips) > 0:
                return flips

        return []
        

