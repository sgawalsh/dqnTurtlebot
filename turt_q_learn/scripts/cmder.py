#!/usr/bin/env python
from geometry_msgs.msg import  Pose,Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelStates
from collections import deque
import rospy, random, math, os, tensorflow as tf, numpy as np, pdb, matplotlib.pyplot as plt, datetime

actionSpace = [2.84,1.42,0,-1.42,-2.84]

EPISODES = 350
EPISODE_LENGTH = 500
GOAL_MIN_DISTANCE = .3
SCAN_MIN_DISTANCE = .2
LASER_SCAN_RATIO = 15
CRASH_PENALTY = -200
GOAL_REWARD = 200
TURTLEBOT_NAME = "turtlebot3_burger"
REWARD_DIRECTION = True

class modelClass():

	def __init__(
		self, epsilonInit = 1.0,
		epsilonMin = .05,
		epsilonDecay = .99,
		gamma = .99,
		memoryLength = 1000000,
		optimizer = tf.keras.optimizers.RMSprop,
		loss = tf.compat.v1.losses.huber_loss,
		batchSize = 100,
		resetTargetEpisodes = 20,
		lr = 0.00025,
		saveName = "myModel - " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	):
		self.model = genModel(optimizer, loss, lr)
		self.targetModel = genModel(optimizer, loss, lr)
		self.targetModel.set_weights(self.model.get_weights())
		self.epsilon = epsilonInit
		self.epsilonMin = epsilonMin
		self.epsilonDecay = epsilonDecay
		self.gamma = gamma
		self.memory = deque(maxlen=memoryLength)
		self.batchSize = batchSize
		self.stateSpace = 360 // LASER_SCAN_RATIO + 2
		self.resetTargetEpisodes = resetTargetEpisodes
		self.turt_q_learn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
		self.saveName = saveName
		self.paramsString = "Episodes: {}\nEpisode Length: {}\nCrash Penalty: {}\nEpisodes: {}\nGoal Reward: {}\nEpsilon Initial: {}\nEpsilon Decay: {}\nEpsilon Min: {}\nUpdate Target Cycles: {}\nGamma: {}\nState space: {}\Learning Rate: {}\nOptimzer: {}\nLoss: {}\nBatch Size: {}\n".format(EPISODES, EPISODE_LENGTH, CRASH_PENALTY, GOAL_REWARD, REWARD_DIRECTION, self.epsilon, self.epsilonDecay, self.epsilonMin, self.resetTargetEpisodes, self.gamma, self.stateSpace, lr, optimizer, loss, batchSize)
		# self.paramsDict = {
			# "Episodes": EPISODES,
			# "Episode Length": EPISODE_LENGTH,
			# "Crash Penalty": CRASH_PENALTY,
			# "Goal Reward": GOAL_REWARD,
			# "Reward Direction": REWARD_DIRECTION,
			# "Epsilon Initial": self.epsilon,
			# "Epsilon Decay": self.epsilonDecay,
			# "Epsilon Min": self.epsilonMin,
			# "Reset Target": self.resetTargetEpisodes,
			# "Gamma": self.gamma,
			# "State space": self.stateSpace,
			# "Learning Rate": lr,
			# "Optimzer": optimizer,
			# "Loss": loss,
			# "Batch Size": batchSize
		# }
		 
	def getAction(self, state):
		if random.random() > self.epsilon:
			action = np.argmax(self.model.predict(state))
		else:
			action = random.randint(0, 4)
		return action
	
	def playEpisodes(self):
		myEnv = envWrapper()
		scores = []
		for epNum in range(EPISODES):
			print("Episode {0}".format(epNum))
			score = 0
			state = myEnv.resetState(epNum > 0)
			
			if not epNum % self.resetTargetEpisodes:
				self.targetModel.set_weights(self.model.get_weights()) # update target model to current model
			
			for stepNum in range(EPISODE_LENGTH):
				print(stepNum)
				#preds = self.model.predict(state)[0]
				#newState, reward, done = myEnv.step(np.argmax(preds))
				action = self.getAction(state)
				newState, reward, done = myEnv.step(action)
				self.memory.append([state, action, newState, reward, done])
				state = newState
				score += reward
				if len(self.memory) > self.batchSize:
					self.dqn(epNum)
				if done:
					break
			
			if self.epsilon > self.epsilonMin:
				self.epsilon *= self.epsilonDecay
			
			scores.append(score)
		
		if not os.path.exists(self.turt_q_learn_path + "/dqnmodels/" + self.saveName):
			os.makedirs(self.turt_q_learn_path + "/dqnmodels/" + self.saveName)
		self.model.save(self.turt_q_learn_path + "/dqnmodels/" + self.saveName + "/model")
		plt.plot(scores)
		plt.ylabel("Scores")
		plt.xlabel("Episodes")
		plt.savefig(self.turt_q_learn_path + '/dqnmodels/' + self.saveName + '/plot.png', bbox_inches='tight')
		plt.show()
		
		with open(self.turt_q_learn_path + '/dqnmodels/' + self.saveName + '/params.txt', "w") as f:
			f.write(self.paramsString)
			
		#with open(self.turt_q_learn_path + '/dqnmodels/params/' + self.saveName + '.pickle', 'wb') as p:
			#pickle.dump(self.paramsDict, p, protocol=pickle.HIGHEST_PROTOCOL)
					
	def dqn(self, epNum):
		
		mini_batch = random.sample(self.memory, self.batchSize)
		X_batch = np.empty((0, self.stateSpace), dtype=np.float64)
		Y_batch = np.empty((0, len(actionSpace)), dtype=np.float64)
		
		for mem in mini_batch:# get original predictions, get q value of next state, and update original predictions, orig state = x, updated preds = y
			q = self.model.predict(mem[0]) # get prediction from state
			if mem[4]: # check if next state is terminal
				qn = mem[3] # if so, return reward
			else:
				qn = mem[3] + self.gamma * np.amax(self.targetModel.predict(mem[2])) # return reward plus max q-value of next state
			
			
			q[0][mem[1]] = qn # replace predicted q values with calculated value for action taken
			
			X_batch = np.append(X_batch, mem[0]) # append state to X values
			Y_batch = np.append(Y_batch, q) # append updated predictions to Y values
		
		self.model.fit(X_batch.reshape(self.batchSize, self.stateSpace), Y_batch.reshape(self.batchSize, len(actionSpace)))
						
class envWrapper():
	
	def __init__(self):
		self.actionPublisher = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		self.goalX, self.goalY = getGoalCoord()
		spawnModel("goal", self.goalX, self.goalY)
		self.goalDistanceOld = None
		
	def getState(self, ranges):
		ranges = [range2State(r) for i, r in enumerate(ranges) if not i % LASER_SCAN_RATIO ]
		goalInfo = self.getGoalStateInfo()
		return np.asarray(ranges + goalInfo).reshape(1,26)
		
	def getGoalStateInfo(self):
		odomData = None
		while odomData is None:
			try:
				odomData = rospy.wait_for_message('odom', Odometry, timeout=5)
			except Exception as e:
				pass
		
		modelX = odomData.pose.pose.position.x
		modelY = odomData.pose.pose.position.y
		
		goalDistance = math.hypot(self.goalX - modelX, self.goalY - modelY)
		goalAngle = math.atan2(self.goalY - modelY, self.goalX - modelX)
		modelAngle = odomData.pose.pose.orientation
				
		yaw = math.atan2(+2.0 * (modelAngle.w * modelAngle.z + modelAngle.x * modelAngle.y), 1.0 - 2.0 * (modelAngle.y * modelAngle.y + modelAngle.z * modelAngle.z))
		
		heading = goalAngle - yaw
		if heading > math.pi:
			heading -= 2 * math.pi
		elif heading < -math.pi:
			heading += 2 * math.pi
			
		return [goalDistance, heading]
		
	def step(self, action):
		self.actionPublisher.publish(genTwist(action))
		
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('scan', LaserScan, timeout=5).ranges
			except:
				pass
		
		newState = self.getState(data)
		reward, done = self.getReward(newState[0])
		
		return newState, reward, done
		
	def resetState(self, respawnModels):
		if respawnModels:
			self.goalX, self.goalY = getGoalCoord()
			#self.teleportModel("goal", self.goalX, self.goalY)
			deleteModel("goal")
			spawnModel("goal", self.goalX, self.goalY)
			print("New goal at {0}, {1}!".format(self.goalX, self.goalY))
			self.teleportModel(TURTLEBOT_NAME, -1, 0)
			
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('scan', LaserScan, timeout=5).ranges
			except:
				pass
				
		return self.getState(data)
		
	def respawnModel(self):
		while True:
			if not self.check_model:
				rospy.wait_for_service('gazebo/spawn_sdf_model')
				spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
				spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goalPosition, "world")
				rospy.loginfo("Goal position : %.1f, %.1f", self.goalPosition.position.x,
							  self.goalPosition.position.y)
				break
			else:
				pass

	def getReward(self, state): # check if crash occured, if goal was reached, if distance to target increased or decreased, return reward + doneState accordingly
		for range in state[:len(state) - 2]:
			if range < SCAN_MIN_DISTANCE:
				print("Crashed!")
				return CRASH_PENALTY, True
				
		if state[len(state) - 2] < GOAL_MIN_DISTANCE:
			print("Reached Goal!")
			self.goalX, self.goalY = getGoalCoord(self.goalX, self.goalY)
			#self.teleportModel("goal", self.goalX, self.goalY)
			deleteModel("goal")
			spawnModel("goal", self.goalX, self.goalY)
			print("New goal at {0}, {1}!".format(self.goalX, self.goalY))
			return GOAL_REWARD, False
			
		if REWARD_DIRECTION: # reward based on heading
			return math.cos(state[len(state) - 1] ), False
		else: # reward based on current distance vs prev distance
			if state[len(state) - 2] < self.goalDistanceOld:
				reward = 1
			else:
				reward = -1
			self.goalDistanceOld = state[len(state) - 2] 
			return reward, False

	def teleportModel(self, modelName, x, y):
		rospy.wait_for_service('gazebo/set_model_state')
		apparate = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
		smsReq = SetModelStateRequest()
		smsReq.model_state.model_name = modelName
		smsReq.model_state.pose.position.x = x
		smsReq.model_state.pose.position.y = y
		if modelName != "goal":
			self.actionPublisher.publish(Twist()) # stop current twist command
		apparate(smsReq)
		
def deleteModel(modelName): # using delete on turtlebot3 crashes gazebo
		rospy.wait_for_service('gazebo/delete_model')
		del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
		del_model_prox(modelName)
				
def spawnModel(modelName, x, y):
		rospy.wait_for_service('gazebo/spawn_sdf_model')
		spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
		modelPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_"
		if modelName == TURTLEBOT_NAME:
			#modelPath += "waffle"
			modelPath = TURTLEBOT_NAME.split("_")[1]
		elif modelName == "goal":
			modelPath += "square/goal_box"
		else:
			raise ValueError("Required model name not available")
		with open (modelPath + '/model.sdf', 'r') as xml_file:
			model_xml = xml_file.read().replace('\n', '')
		spawnPose = Pose()
		spawnPose.position.x = x
		spawnPose.position.y = y
		spawn_model_prox(modelName, model_xml, '', spawnPose, "world")		
		
def genTwist(index):
	retTwist = Twist()
	retTwist.linear.x = .22
	retTwist.angular.z = actionSpace[index]
	return retTwist

def talker():
	pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
	rospy.init_node('talker', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	while not rospy.is_shutdown():
		twist = genTwist()
		pub.publish(twist)
		rate.sleep()

def range2State(r):
	if str(r) == 'inf':
		return 3.5
	else:
		return r

def getGoalCoord(removeX = None, removeY = None):
	while True:
		x = random.choice([-1.5,-1,-.5,0,.5,1,1.5])
		y = random.choice(([-1.5,-1,-.5,0,.5,1,1.5] if bool(x % 1) else [-1.5, -.5, .5, 1.5]))
		if not (x == removeX and y == removeY):
			break
	return x,  y

def genModel(optimizer, loss, lr):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(32, input_shape = (360 // LASER_SCAN_RATIO + 2,), activation = tf.nn.relu))
	model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))
	model.add(tf.keras.layers.Dense(5, activation = tf.compat.v1.keras.activations.linear))
	model.compile(optimizer=optimizer(lr = lr), loss=loss)
	return model


if __name__ == '__main__':
	try:
		rospy.init_node('qLearner')
		model = modelClass()
		model.playEpisodes()
	except rospy.ROSInterruptException:
		pass
		
''''
Algorithm:
	For each episode
		For each step
			If random val > epsilon, Predict best action with model, else choose random action
			Step Environment forward
			
			Create memory of old state, action, reward, and new state
			check random memories, calculate new state q vals, fit model to generated values
'''