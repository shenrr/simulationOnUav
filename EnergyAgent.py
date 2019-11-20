import math
import numpy as np

class EnergyAgent:
    def __init__(self, state_shape, action_bound, action_dim,
                 ):
        self.state_shape = state_shape
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.discount_on_redirect = 0.1
        self.power_scale=0
        self.power_change_scale =0.005
        self.countgo=100
        self.dangermemories=0
        self.danger=False
        self.dangerch=32
        self.smoothcache=[0]*64

    def observe(self, state, action, reward, post_state, terminal):
        return

    def act(self, state, noise=True):
        self.countgo+=1
        action = [0]*self.action_dim
        if abs(state[1][2])<0.01 and abs(state[1][3])<0.01 or self.countgo>50:
            self.countgo=0
            return self.redirect(state)

        #压缩截远近
        matrix=self.weightProcess(state)

        #平滑
        smooth=self.smoothing(matrix)

        # 计算量
        ax = state[1][0]
        ay = state[1][1]
        astandard = math.sqrt(ax * ax + ay * ay)
        bx = state[1][2]
        by = state[1][3]
        bstandard = math.sqrt(bx * bx + by * by)
        ax /= astandard
        ay /= astandard
        bx /= bstandard
        by /= bstandard

        smooth=self.addWeightDependTarget(smooth, ax, ay, bx, by)
        #smooth2=smooth
        #smooth=self.smoothcache
        #self.smoothcache=smooth2
        print(smooth)

        #取最低
        min=0
        minvalue=100
        leftcount=0
        rightcount=0
        for i in range(0, 32):
            leftcount+=smooth[i]
        for i in range(32, 64):
            rightcount += smooth[i]
        print("左边{},右边{}".format( leftcount,rightcount))
        if abs(leftcount-rightcount)>100 and leftcount>1500 and rightcount>1500 and self.dangermemories<=0:

            self.dangermemories=3
            if leftcount<=rightcount:
                self.dangerch=1
            else:
                self.dangerch=63
        if leftcount<=rightcount:
            print("优先尝试找左边")
            for i in range(24, 32):
                if smooth[i] < minvalue:
                    minvalue = smooth[i]
                    min = i
            for i in range(0, 32):
                if smooth[i] < minvalue:
                    minvalue = smooth[i]
                    min = i
        else:
            print("优先尝试找右边")
            for i in range(32, 40):
                if smooth[i] < minvalue:
                    minvalue = smooth[i]
                    min = i
            for i in range(32, 64):
                if smooth[i] < minvalue:
                    minvalue = smooth[i]
                    min = i
        for i in range(0, 64):
            if smooth[i] < minvalue:
                minvalue = smooth[i]
                min = i

        #计算角偏移
        if  self.dangermemories>0:
            min=self.dangerch
            self.dangermemories-=1
            print("恐惧回想中")

        print("最小{},最小值{}".format(min,minvalue))

        if min>28 and min<36:
            print("前进")
            xnew = bx
            ynew = by
        elif min < 32:
            anglepoint=32-min
            standard=math.sqrt(anglepoint*anglepoint+32*32*3)
            thetacos=32*math.sqrt(3)/standard
            thetasin=anglepoint/standard
            #xnew=bx*thetacos-by*thetasin
            #ynew=bx*thetasin+by*thetacos
            xnew = bx * thetacos + by * thetasin
            ynew = by * thetacos - bx * thetasin
            print("左转")
        else:
            anglepoint =  min -32
            standard = math.sqrt(anglepoint * anglepoint + 32 * 32 * 3)
            thetacos = 32 * math.sqrt(3) / standard
            thetasin = anglepoint / standard
            #xnew = bx * thetacos + by * thetasin
            #ynew = by * thetacos - bx * thetasin
            xnew = bx * thetacos - by * thetasin
            ynew = bx * thetasin + by * thetacos
            print("右转")
        action[0]=xnew
        action[1]=ynew
        return action

    def redirect(self,state):
        print("Drone Redirect")
        tarX = state[1][0]
        tarY = state[1][1]
        action=[0]*self.action_dim
        if tarX!=0:
            action[0] = tarX / math.sqrt(tarX * tarX + tarY * tarY)
        if tarY != 0:
            action[1] = tarY / math.sqrt(tarX * tarX + tarY * tarY)
        return action

    def weightProcess(self,state):
        # 压缩
        imageCompact = []
        for i in range(26, 36):
            imageCompact.append(state[0][i][:])
        imageCompact = np.array(imageCompact)

        # 截远近
        matrix = imageCompact.copy()
        showcontent = imageCompact.copy()
        # print(showcontent)
        for i in range(10):
            for j in range(64):
                matrix[i][j] = 1 - matrix[i][j]
                if matrix[i][j] < 0.7:
                    matrix[i][j] = 0
                elif matrix[i][j] < 0.85:
                    matrix[i][j] *= 0
                elif matrix[i][j] < 0.9:
                    pass
                elif matrix[i][j] < 0.95:
                    matrix[i][j] *= 15
                else:
                    matrix[i][j] *= 200
        # print(matrix)
        return matrix

    def smoothing(self,matrix):
        #转成线加平滑
        power=np.mean(matrix,axis=0)
        smooth=power.copy()
        smooth[0]=(power[0]+power[1])/2
        smooth[63] = (power[62] + power[63]) / 2
        for i in range(2,62):
            smooth[i]=(power[i-2]+power[i-1]+power[i]+power[i+1]+power[i+2])/5
        return smooth

    def addWeightDependTarget(self,smooth,ax,ay,bx,by):
        alphadirect = ax * by - ay * bx
        alphacos = ax * bx + ay * by
        power_scale = 0  # 具体价值
        if alphacos < math.sqrt(3) / 2:
            if alphadirect < 0:
                for i in range(0, 32):
                    smooth[i] -= power_scale
                    power_scale -= self.power_change_scale
            else:
                for i in range(32, 64):
                    smooth[i] -= power_scale
                    power_scale -= self.power_change_scale
        else:
            alphasin = math.sqrt(1 - alphacos * alphacos)
            pixs = math.floor(alphasin / (32 * math.sqrt(3) / alphacos))
            if alphadirect < 0:
                pixs = 32 - pixs
                temp = power_scale
                for i in range(pixs, -1, -1):
                    smooth[i] -= temp
                    temp -= self.power_change_scale
                temp = power_scale
                for i in range(pixs, 64):
                    smooth[i] -= temp
                    temp -= self.power_change_scale
            else:
                pixs = 32 + pixs
                temp = power_scale
                for i in range(pixs, -1, -1):
                    smooth[i] -= temp
                    temp -= self.power_change_scale
                temp = power_scale
                for i in range(pixs, 64):
                    smooth[i] -= temp
                    temp -= self.power_change_scale
        return smooth

    #def buildmap(self):



