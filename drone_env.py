import copy
import time

import cv2
import numpy as np
from PIL import Image

import AirSimClient

goal_threshold = 12
np.set_printoptions(precision=3, suppress=True)
IMAGE_VIEW = True

class drone_env:#无人机控制
	def __init__(self,start = [0,0,-5],aim = [32,38,-4]):
		self.start = np.array(start)
		self.aim = np.array(aim)
		self.client = AirSimClient.MultirotorClient()#获取airsimclient  remove call 连接
		self.client.confirmConnection()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.threshold = goal_threshold

	def reset(self):
		self.client.reset()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.client.moveToPosition(self.start.tolist()[0],self.start.tolist()[1],self.start.tolist()[2],5,max_wait_seconds = 10)
		time.sleep(1)


	def isDone(self):
		pos = self.client.getPosition()
		if distance(self.aim,pos) < self.threshold:
			return True
		return False

	def moveByDist(self,diff, forward = False):
		temp = AirSimClient.YawMode()
		temp.is_rate = not forward
		sta=self.client.getVelocity()
		self.client.moveByVelocity(diff[0], diff[1], diff[2]*0.8, 1 ,drivetrain = AirSimClient.DrivetrainType.ForwardOnly, yaw_mode = temp)
		time.sleep(0.25)
		return 0

	def render(self,extra1 = "",extra2 = ""):
		pos = v2t(self.client.getPosition())
		goal = distance(self.aim,pos)
		print (extra1,"distance:",int(goal),"position:",pos.astype("int"),extra2)

	def help(self):
		print ("drone simulation environment")
		
		

		
#-------------------------------------------------------
# height control
# continuous control
		
class drone_env_heightcontrol(drone_env):
	def __init__(self,start = [-23,0,-8],aim = [-23,125,-8],scaling_factor = 1,img_size = [64,64]):
		drone_env.__init__(self,start,aim)
		self.scaling_factor = scaling_factor
		self.aim = np.array(aim)
		self.height_limit = -30
		self.count = 0
		self.initDistance=1000
		self.dx=0
		self.dy=0
		self.loseCome=0
		self.rand = False
		self.totalsuccess = 0
		self.imagecache1=np.zeros((64,64))
		self.imagecache2 = np.zeros((64, 64))


		if aim == None:
			self.rand = True
			self.start = np.array([0,0,-4])
		else:
			self.aim_height = self.aim[2]

	def reset_aim(self):
		self.aim = (np.random.rand(3)*200).astype("int")-100
		self.aim[2] = -3#-np.random.randint(2) - 2
		print ("Our aim is: {}".format(self.aim).ljust(80," "),end = '\r')
		self.aim_height = self.aim[2]

	def reset(self):
		if self.rand:
			self.reset_aim()
		drone_env.reset(self)
		self.count = 0
		self.imagecache1=np.zeros((64,64))
		self.imagecache2 = np.zeros((64, 64))
		self.state = self.getState()
		self.loseCome = 0
		img = copy.deepcopy(self.state[0])
		relativeState=[img, np.array([self.aim[0]-self.state[1][0],self.aim[1]-self.state[1][1],0,0])]
		self.initDistance=np.sqrt(abs(relativeState[1][0]) ** 2 + abs(relativeState[1][1]) ** 2 )
		self.cacheDistance=self.initDistance
		self.dx = 0
		self.dy = 0

		return relativeState

	def getState(self):
		pos = v2t(self.client.getPosition())
		vel = v2t(self.client.getVelocity())
		img = self.getImg()
		state = [img, np.array([pos[0],pos[1],pos[2] - self.aim_height])]
		return state

	def step(self,action):#一步行为
		pos = v2t(self.client.getPosition())
		dpos = self.aim - pos


		if abs(action[0]) > 1 :
			print ("action value error")
			action[0] = action[0] / abs(action[0])
		if abs(action[1]) > 1:
			print ("action value error")
			action[1] = action[1] / abs(action[1])



		dx = action[0] * self.scaling_factor
		dy = action[1] * self.scaling_factor


		#angleloss=abs(initDiretion-self.cacheAngle)
		pos = self.state[1][2]
		dz =self.aim[2]-pos
		drone_env.moveByDist(self,[dx,dy,dz],forward = True)
		state_ = self.getState()
		#sta = self.client.getVelocity()
		#print(sta)
		vel = v2t(self.client.getVelocity())
		#print("one",vel[0],vel[1],vel[2])

		info = None
		done = False
		reward2 = self.rewardf(self.state,state_,dx,dy)
		#sta = self.client.getVelocity()
		#print(sta)
		self.dx=dx
		self.dy=dy
		#cache=reward
		CollisionOrNot=False
		countCollision=0
		realCollision=0
		for pixels in state_[0]:
			for pixel in pixels:
				if pixel<=0.001:
					realCollision+=1
				if pixel<=0.004:
					countCollision+=1
		if countCollision>16*64 and realCollision>10:
			CollisionOrNot = True

		if self.isDone():
			if self.rand:

				reward2 = 500
				info = "success"
				done = True
				self.reset_aim()
				self.totalsuccess+=1
				self.count=0
			else:
				done = True
				reward2 = 50
				info = "success"
				self.totalsuccess += 1
		if abs(self.dx)==1 and abs(self.dy)==1:
			self.loseCome+=1
			if self.loseCome>40:
				done=True
				reward=-50
				info="gridient disappear"
				self.loseCome =0

		if  CollisionOrNot:
			reward = -50
			done = True
			info = "judgeing collision"

		if self.client.getCollisionInfo().has_collided :
			reward = -50
			done = True
			info = "collision"
		dis_ = distance(state_[1][0:2], self.aim[0:2])
		dis = distance(self.state[1][0:2], self.aim[0:2])
		if dis_>140 and dis<dis_:
			reward = -50
			done = True
			info = "too far"

		if (pos ) < self.height_limit-8:
			done = True
			info = "too high"
			reward = -50

		if (self.count) >= 500:
			done = True
			info = "too slow"
			reward = -50

		self.state = state_
		vel=v2t(self.client.getVelocity())
		#print("two", vel[0], vel[1], vel[2])

		img = copy.deepcopy(state_[0])
		relativeState = [np.array(img), np.array([self.aim[0] - self.state[1][0], self.aim[1] - self.state[1][1],vel[0], vel[1]])]
		self.imagecache2=self.imagecache1
		self.imagecache1=img
		#print(relativeState[0])

		#reward /= 50
		#print(reward2)
		print("与目标距离: {} ".format(dis_))
		return relativeState,reward2,done,info

	def isDone(self):
		pos = v2t(self.client.getPosition())
		pos[2] = self.aim[2]
		if distance(self.aim,pos) < self.threshold:
			return True
		return False


	def rewardf(self,state,state_,dx,dy):
		pos = state[1][2]
		pos_ = state_[1][2]
		#reward = - abs(pos_) + 5
		dis = distance(state[1][0:2], self.aim[0:2])
		dis_ = distance(state_[1][0:2], self.aim[0:2])
		'''
		if dis<dis_:
		   reward2 =dis-dis_
		   reward2 = reward2

		else:
		   reward2=dis-dis_
		   reward2 = reward2 * 2
		'''
		vectorMulti=dx*(self.aim[0]-state[1][0])+dy*(self.aim[1]-state[1][1])
		mod1 = np.sqrt(abs(dx) ** 2 + abs(dy) ** 2)
		mod2 = np.sqrt(abs(state[1][0]-self.aim[0]) ** 2 + abs(state[1][1]-self.aim[1]) ** 2)
		reward2=vectorMulti/(mod1*mod2)
		reward2=reward2*mod1

		#reward2+=(0.5-abs(self.cacheAngle))*0.7*np.pi
		#reward2-=angleloss
		#if abs(self.cacheAngle * np.pi) > 3:
		#	reward2 *= 12

		print("distance: {}".format(dis_).ljust(20," "),"relative position: {}".format(state_[1]-self.aim).ljust(20," "),end = "\r")
		return reward2


	def getImg(self):
		
		responses = self.client.simGetImages([AirSimClient.ImageRequest(0, AirSimClient.AirSimImageType.DepthPerspective, True, False)])
		img1d = np.array(responses[0].image_data_float, dtype=np.float)

		img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

		image = Image.fromarray(img2d)
		im_final = np.array(image.resize((64, 64)).convert('L'), dtype=np.float)/255
		im_final.resize((64,64))

		#responses2 = self.client.simGetImages([AirSimClient.ImageRequest(0, AirSimClient.AirSimImageType.Segmentation, True, False)])
		#img1d2 = np.array(responses2[0].image_data_float, dtype=np.float)
		#img2d2 = np.reshape(img1d2, (responses[0].height, responses[0].width))
		#image2= Image.fromarray(img2d2)
		#im_final2 = np.array(image2.resize((64, 64)).convert('P'), dtype=np.float) / 255

		if IMAGE_VIEW:
			cv2.imshow("view",im_final)
			key = cv2.waitKey(1) & 0xFF;
		return im_final


def v2t(vect):
	if isinstance(vect,AirSimClient.Vector3r):
		res = np.array([vect.x_val, vect.y_val, vect.z_val])
	else:
		res = np.array(vect)
	return res


def distance(pos1,pos2):
	pos1 = v2t(pos1)
	pos2 = v2t(pos2)
	#dist = np.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2 + abs(pos1[2]-pos2[2]) **2)
	dist = np.linalg.norm(pos1-pos2)
	return dist