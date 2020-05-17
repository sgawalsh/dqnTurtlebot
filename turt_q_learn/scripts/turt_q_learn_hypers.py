#!/usr/bin/env python
from geometry_msgs.msg import  Pose,Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState, SetModelStateRequest, GetWorldProperties
from collections import deque
from itertools import product
import rospy, datetime, random, math, os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt

crashDistances = { # wall collision distance per model
	"turtlebot3_burger": .12,
	"turtlebot3_burger_front": .20,
	"turtlebot3_waffle": .22,
	"turtlebot3_waffle_front": .22,
}

ACTION_SPACE = [2.5,1.25,0,-1.25,-2.5] # angular velocities for bot
TURTLEBOT_NAME = "turtlebot3_burger"
SCAN_MIN_DISTANCE = crashDistances[TURTLEBOT_NAME] # distance to detect wall collision
SCAN_MIN_DISTANCE_FRONT = crashDistances[TURTLEBOT_NAME + "_front"]
GOAL_MIN_DISTANCE = .35 # distance to detect goal arrival
MODEL_SPEED = .12 # linear speed of model
FINAL_SLICE = 50 # number of scores considered for final average
ROLLING_AVERAGE_SAMPLES = 25 # number of samples used for rolling average graph

class modelClass():

	def __init__(
		self,
		hyperParams,
		env,
		saveName = "myModel - "
	): # build models, set required parameters
		self.env = env
		self.turt_q_learn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
		if hyperParams["Load Model"]:
			self.model = tf.compat.v1.keras.models.load_model(self.turt_q_learn_path + "/load_model/model" )
		else:
			self.model = genModel(hyperParams["Optimizer"], hyperParams["Loss"], hyperParams["Learning Rate"], hyperParams["First Activation"], hyperParams["Hidden Activations"], hyperParams["Last Activation"], hyperParams["State Space"], hyperParams["Initializer"])
		self.targetModel = tf.compat.v1.keras.models.clone_model(self.model)
		self.targetModel.set_weights(self.model.get_weights())
		self.episodes = hyperParams["Episodes"]
		self.episodeLength = hyperParams["Episode Length"]
		self.epsilon = hyperParams["Epsilon Initial"]
		self.epsilonMin = hyperParams["Epsilon Min"]
		self.epsilonDecay = hyperParams["Epsilon Decay"]
		self.gamma = hyperParams["Gamma"]
		self.memory = deque(maxlen=hyperParams["Memory Length"])
		self.batchSize = hyperParams["Batch Size"]
		self.stateSpace = hyperParams["State Space"]
		self.resetTargetMemories = hyperParams["Reset Target"]
		self.turt_q_learn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
		self.saveName = saveName + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		self.paramsString = dictToStr(hyperParams)
		self.memoryCount = 0
		 
	def getAction(self, state): # generate random value, if greater than epsilon, choose action according to model, else, choose action randomly
		return (np.argmax(self.model.predict(state)) if random.random() > self.epsilon else random.randint(0, 4))
	
	def playEpisodes(self): # complete set amount of episodes, return numpy array of scores from each
		scores = []
		for epNum in range(self.episodes):
			print("Episode {0}:".format(epNum))
			score = 0
			state = self.env.resetState()
			
			for stepNum in range(self.episodeLength): # generate action, get new state, create memory, perform dqn if memory is large enough
				action = self.getAction(state)
				newState, reward, done = self.env.step(action)
				self.memory.append([state, action, newState, reward, done])
				state = newState
				score += reward
				if len(self.memory) > self.batchSize:
					self.dqn()
				if done:
					break
			
			if self.epsilon > self.epsilonMin:
				self.epsilon *= self.epsilonDecay
			
			scores.append(score)
			print("	Score: {}".format(score))
				
		folderPath = "/dqnmodels/{}".format(SESSION_NAME) + "/"
		if not os.path.exists(self.turt_q_learn_path + folderPath + self.saveName):
			os.makedirs(self.turt_q_learn_path + folderPath + self.saveName)
		self.model.save(self.turt_q_learn_path + folderPath + self.saveName + "/model")
		plt.figure(2)
		averages = movingAverage(scores)
		plt.plot(scores, label = "Scores")
		plt.plot(np.arange(ROLLING_AVERAGE_SAMPLES - 1, self.episodes), averages, label = "Average")
		plt.ylabel("Scores")
		plt.xlabel("Episodes")
		plt.legend()
		plt.savefig(self.turt_q_learn_path + folderPath + self.saveName + '/plot.png', bbox_inches='tight')
		plt.close()
		#plt.show()
		
		with open(self.turt_q_learn_path + folderPath + self.saveName + '/params.txt', "w") as f:
			f.write(self.paramsString)
			
		return np.array(scores), averages, self.saveName
					
	def dqn(self):
		mini_batch = random.sample(self.memory, self.batchSize)
		X_batch = np.empty((0, self.stateSpace), dtype=np.float64)
		Y_batch = np.empty((0, len(ACTION_SPACE)), dtype=np.float64)
		
		for mem in mini_batch:# get original predictions, get q value of next state, and update original predictions, orig state = x, updated preds = y
			q = self.model.predict(mem[0]) # get prediction from state
			if mem[4]: # check if next state is terminal
				qn = mem[3] # if so, return reward
			else:
				qn = mem[3] + self.gamma * np.amax(self.targetModel.predict(mem[2])) # return reward plus max q-value of next state
			
			
			q[0][mem[1]] = qn # replace predicted q values with calculated value for action taken
			
			X_batch = np.append(X_batch, mem[0]) # append state to X values
			Y_batch = np.append(Y_batch, q) # append updated predictions to Y values
		
		self.model.fit(X_batch.reshape(self.batchSize, self.stateSpace), Y_batch.reshape(self.batchSize, len(ACTION_SPACE)), verbose = 0)
		
		self.memoryCount += 1
		if self.memoryCount >= self.resetTargetMemories:
			self.targetModel.set_weights(self.model.get_weights()) # update target model to current model
			self.memoryCount = 0
			print("reset weights")
						
class envWrapper():
	
	def __init__(self, params): # Set required parameters
		self.actionPublisher = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		self.goalX, self.goalY = 0, 0
		self.goalDistanceOld = None
		self.rewardDirection = params["Reward Direction"]
		self.scanRatio = params["Scan Ratio"]
		self.crashPenalty = params["Crash Penalty"]
		self.goalReward = params["Goal Reward"]
		self.directionScalar = params["Direction Scalar"]
		self.stateSpace = params["State Space"]
		self.maxScanRange = params["Max Scan Range"]
		self.scanRewardScalar = params["Scan Reward Scaler"]
		self.modelX, self.modelY = None, None
		
	def getState(self, ranges): # read laser ranges, get goal info
		#ranges = [range2State(r) for i, r in enumerate(ranges) if not i % self.scanRatio]
		ranges = [(self.maxScanRange if str(r) == 'inf' else min(r, self.maxScanRange)) for i, r in enumerate(ranges) if not i % self.scanRatio]
		goalInfo = self.getGoalOdomStateInfo()
		return np.asarray(ranges + goalInfo).reshape(1, self.stateSpace)
		
	def getGoalOdomStateInfo(self): # get distance and angle to goal
		odomData = None
		while odomData is None:
			try:
				odomData = rospy.wait_for_message('odom', Odometry, timeout=5)
			except Exception as e:
				pass
		
		# get goal data
		self.modelX = odomData.pose.pose.position.x
		self.modelY = odomData.pose.pose.position.y
		
		goalDistance = math.hypot(self.goalX - self.modelX, self.goalY - self.modelY)
		goalAngle = math.atan2(self.goalY - self.modelY, self.goalX - self.modelX)
		modelAngle = odomData.pose.pose.orientation
				
		yaw = math.atan2(+2.0 * (modelAngle.w * modelAngle.z + modelAngle.x * modelAngle.y), 1.0 - 2.0 * (modelAngle.y * modelAngle.y + modelAngle.z * modelAngle.z))
		
		heading = goalAngle - yaw
		if heading > math.pi:
			heading -= 2 * math.pi
		elif heading < -math.pi:
			heading += 2 * math.pi
			
		# get odom data
		#modelZ = odomData.twist.twist.angular.z
			
		return [odomData.twist.twist.angular.z, goalDistance, heading]
		
	def step(self, action): # publish action and read new state, reward, and done variables
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
		
	def resetState(self): # respawn turtle and goal models, get initial state
		deleteModel("goal")
		goalSpawned = False
		self.goalX, self.goalY = getGoalCoord(-1, 0)
		self.teleportModel(TURTLEBOT_NAME, -1.5, 0)
		while not goalSpawned:
			spawnModel("goal", self.goalX, self.goalY)
			goalSpawned = checkModelPresent("goal")
		print("	New goal at {0}, {1}!".format(self.goalX, self.goalY))
			
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('scan', LaserScan, timeout=5).ranges
			except:
				pass
				
		return self.getState(data)

	def getReward(self, state): # check if crash occured, if goal was reached, if distance to target increased / angle towards target, return reward + doneState accordingly
		
		#pdb.set_trace()
		if min(state[1:len(state) - 3] ) < SCAN_MIN_DISTANCE or state[0] < SCAN_MIN_DISTANCE_FRONT:
			print("	Crashed!")
			return self.crashPenalty, True
							
		if state[len(state) - 2] < GOAL_MIN_DISTANCE:
			print("	Reached Goal!")
			self.goalX, self.goalY = getGoalCoord(self.goalX, self.goalY)
			#self.teleportModel("goal", self.goalX, self.goalY)
			deleteModel("goal")
			spawnModel("goal", self.goalX, self.goalY)
			print("	New goal at {0}, {1}!".format(self.goalX, self.goalY))
			return self.goalReward, False
			
		if self.rewardDirection: # reward based on heading
			return self.directionScalar * math.cos(state[len(state) - 1] ), False
		else: # reward based on current distance vs prev distance
			return self.directionScalar * (1 if state[len(state) - 2] < self.goalDistanceOld else -1), False
			
	def scanReward(self, state):
		return (1 - min(state[1:len(state) - 3] ) / self.maxScanRange) * self.scanRewardScalar

	def teleportModel(self, modelName, x, y): # used as workaround to reset turtlebot model
		rospy.wait_for_service('gazebo/set_model_state')
		apparate = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
		smsReq = SetModelStateRequest()
		smsReq.model_state.model_name = modelName
		smsReq.model_state.pose.position.x = x
		smsReq.model_state.pose.position.y = y
		if modelName != "goal":
			self.actionPublisher.publish(Twist()) # stop current twist command
		apparate(smsReq)
		
def deleteModel(modelName): # using delete on turtlebot3 crashes gazebo, check if goal exists, delete if present
	while True:
		rospy.wait_for_service('gazebo/get_world_properties')
		get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
		resp = get_world_properties()
		if "goal" in resp.model_names:
			break
		else:
			return
	
	rospy.wait_for_service('gazebo/delete_model')
	del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
	del_model_prox(modelName)
				
def spawnModel(modelName, x, y): #  use service to spawn model
	rospy.wait_for_service('gazebo/spawn_sdf_model')
	spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
	modelPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_"
	if modelName == TURTLEBOT_NAME:
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

def checkModelPresent(modelName):
	rospy.wait_for_service('gazebo/get_world_properties')
	get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
	resp = get_world_properties()
	return modelName in resp.model_names
	
def genTwist(index): # build twist to be sent to gazebo
	retTwist = Twist()
	retTwist.linear.x = MODEL_SPEED
	retTwist.angular.z = ACTION_SPACE[index]
	return retTwist

def range2State(r): # convert 'inf' to float value
	return (3.5 if str(r) == 'inf' else r)

def getGoalCoord(removeX = None, removeY = None): # generate new coordinates for goal
	while True:
		x = random.choice([-1.5,-.5,.5,1.5])
		y = random.choice([-1.5,-.5,.5,1.5])
		if not (x == removeX and y == removeY):
			break
	return x,  y

def genModel(optimizer, loss, lr, first, hidden, last, stateSpace, initializer): # build neural net to be used for DDQN
	model = tf.compat.v1.keras.models.Sequential()
	model.add(tf.compat.v1.keras.layers.Dense(32, input_shape = (stateSpace,), activation = first, kernel_initializer = initializer))
	model.add(tf.compat.v1.keras.layers.Dense(32, activation = hidden, kernel_initializer = initializer))
	model.add(tf.compat.v1.keras.layers.Dense(16, activation = hidden, kernel_initializer = initializer))
	model.add(tf.compat.v1.keras.layers.Dense(len(ACTION_SPACE), activation = last))
	model.compile(optimizer=optimizer(lr = lr), loss=loss)
	return model

def saveParams(path, dict):
	with open(path, "w") as f:
		f.write(dictToStr(dict))
	
def dictToStr(dict):
	str = ""
	for key, value in sorted(dict.iteritems()):
		str += "{0}: {1}\n".format(key, value)
	return str

def movingAverage(a) :
    ret = np.cumsum(a, dtype = float)
    ret[ROLLING_AVERAGE_SAMPLES:] = ret[ROLLING_AVERAGE_SAMPLES:] - ret[:-ROLLING_AVERAGE_SAMPLES]
    return ret[ROLLING_AVERAGE_SAMPLES - 1:] / ROLLING_AVERAGE_SAMPLES

def cartProd(paramDict, fn, **kwargs):
	keys, values = zip(*paramDict.items())
	for v in product(*values):
			experiment = dict(zip(keys, v)) # generate hyperparameter combination
			print("Using hyperparams: " + str(experiment))
			fn(experiment)

if __name__ == '__main__':
	rospy.init_node('qLearner')
	try:
		SESSION_NAME = "mySession - " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		hyperParameterList = { # set hyperparameter combinations
			"Episodes": [500],
			"Episode Length": [350],
			"Crash Penalty": [-2000],
			"Goal Reward": [200],
			"Reward Direction": [True], # if true, reward is given in proportion to angle towards goal, if false, reward is given if bot 'stepped' closer
			"Epsilon Initial": [1],
			"Epsilon Decay": [.992],
			"Epsilon Min": [.05],
			"Reset Target": [2000], #  memories examined count to sync target net to model net
			"Gamma": [.99], # Q-learning gamma variable, rate or reward depreciation between states
			"Scan Ratio": [12], # how wany of the 360 scans are read, larger number -> less scan count
			"Max Scan Range": [1], # how far each scan ray sees, max 3.5
			"Scan Reward Scaler": [1],
			"Learning Rate": [0.0002],
			"Optimizer": [tf.compat.v1.keras.optimizers.RMSprop],
			"Loss": [tf.compat.v1.losses.huber_loss],
			"Batch Size": [100], # memories examined per step
			"Memory Length": [1000000],
			"Direction Scalar": [1], # changes value of reward for non-terminal steps
			"First Activation": [tf.compat.v1.keras.activations.relu],
			"Hidden Activations": [tf.compat.v1.keras.activations.relu],
			"Last Activation": [tf.compat.v1.keras.activations.linear],
			"Initializer": [tf.variance_scaling_initializer(scale=2)],
			"Load Model" : [False] # load model contained in turt_q_learn/load_model/model
		}
		
		FINAL_SLICE = (hyperParameterList["Episodes"] // 2 if FINAL_SLICE >= hyperParameterList["Episodes"] else FINAL_SLICE) # set and save session vars
		hyperParameterList["State Space"] = [360 // r + 3 for r in hyperParameterList["Scan Ratio"]]
		folderPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dqnmodels/{}".format(SESSION_NAME)
		
		if not os.path.exists(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dqnmodels/" + SESSION_NAME ):
			os.makedirs(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dqnmodels/" + SESSION_NAME)
		
		saveParams(folderPath + '/allParams.txt', hyperParameterList)
		saveParams(folderPath + '/envVars.txt', {
			"Action Space": ACTION_SPACE,
			"Turtlebot Name": TURTLEBOT_NAME,
			"Goal Min Distance": GOAL_MIN_DISTANCE,
			"Scan Min Distance": SCAN_MIN_DISTANCE,
			"Model Speed": MODEL_SPEED,
			"Final Slice": FINAL_SLICE,
			"Rolling Average Samples": ROLLING_AVERAGE_SAMPLES
		})
		
		keys, values = zip(*hyperParameterList.items())
		maxAvg, maxFinalAvg = -1000000, -1000000
		bestParams, bestFinalParams = {}, {}
		
		for v in product(*values):
			experiment = dict(zip(keys, v)) # generate hyperparameter combination
			print("Using hyperparams: " + str(experiment))
			
			env = envWrapper(experiment)
			model = modelClass(experiment, env)
			hyperSetScores, averages, setName = model.playEpisodes()
			
			if np.mean(hyperSetScores) > maxAvg: # get most succesful combination
				maxAvg = np.mean(hyperSetScores)
				bestParams = experiment
			
			if np.mean(hyperSetScores[len(hyperSetScores) - FINAL_SLICE::]) > maxFinalAvg: # over last x episodes
				maxFinalAvg = np.mean(hyperSetScores[len(hyperSetScores) - FINAL_SLICE::])
				bestFinalParams = experiment
			
			plt.figure(1)
			
			plt.subplot(211) # throws incorrect warning, see https://github.com/matplotlib/matplotlib/issues/12513
			plt.plot(hyperSetScores, label = setName)
			plt.subplot(212)
			plt.plot(np.arange(ROLLING_AVERAGE_SAMPLES - 1, experiment["Episodes"]), averages, label = setName)
		
		plt.subplot(211)
		plt.ylabel("Scores")
		plt.xlabel("Episodes")
		plt.subplot(212)
		plt.ylabel("Average")
		plt.xlabel("Episodes")
		plt.legend()
		plt.savefig(folderPath + '/plot.png', bbox_inches='tight')
		plt.close()
		
		saveParams(folderPath + '/bestParams.txt', bestParams)
		saveParams(folderPath + '/bestFinalParams.txt', bestFinalParams)
	except rospy.ROSInterruptException:
		pass
