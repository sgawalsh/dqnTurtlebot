#!/usr/bin/env python
from pdb import set_trace
from matplotlib import pyplot
from geometry_msgs.msg import Twist
from operator import itemgetter
from nav_msgs.msg import Odometry
import os, rospy, numpy as np, math, bisect


mapPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/map/map.pgm"

class cell():
	
	def __init__(self, f = 0, g = 0, h = 0, walkable = False, y = 0, x = 0):
		self.G, self.F, self.H = g, f, h
		self.y, self.x = y, x
		self.parent = None
		self.walkable = walkable
		
	def __lt__(self, other):
         return self.F < other.F
		
	def __repr__(self):
		return 'F({})'.format(self.F)

class pathFinder():
	
	def __init__(self):
		self.collRadius = .3
		with open(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/map/map.pgm", 'rb') as f:
			self.map = readPgm(f)
			
		self.originOffset = (-10, -10)
		self.resolution = .05
		self.path = []
		self.pathColour = 128
		self.actionPublisher = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
		
	def genPathMap(self): # takes slam, map, marks unexplored areas, walls, and areas close to walls as -1, all other walkable cells are marked as 1
		pixRadius = int(round(self.collRadius / self.resolution))
		self.pathMap = np.array(self.map.tolist(), dtype = np.int16)
		wallList = []
		for y, row in enumerate(self.map):
			for x, cell in enumerate(row):
				if cell == 205:
					self.pathMap[y][x] = -1
				elif cell == 0:
					for y2 in range(y - pixRadius, y + pixRadius):
						for x2 in range(x - pixRadius, x + pixRadius):
							if x2 >= 0 and y2 >= 0:
								if self.map[y2][x2] != 0 and math.sqrt((x2 - x) ** 2 + (y2 - y) ** 2) < pixRadius:
									self.pathMap[y2][x2] = -1
					wallList.append([y, x])
				elif cell == 254:
					self.pathMap[y][x] = 1
					
		for coord in wallList:
			self.pathMap[coord[0]][coord[1]] = -1
	
	def genAStarMap(self, target): # generate A star F and H values according to distance from target
		self.aMap = []
		for y, row in enumerate(self.map):
			newRow = []
			for x in range(len(row)):
				h = (min(abs(x - target[1]), abs(y - target[0])) * 14) + (abs(abs(x - target[1]) - abs(y - target[0])) * 10) # value of 14 for each diagonal step, 10 for each adjacent step
				newRow.append(cell(h, 0, h, self.map[y][x] == 254, y, x))
			self.aMap.append(newRow)
	
	def aStar(self, goal, start): # a star algorithm, check most promising target, add neighbours to list and mark as explored, repeat until target is found
		self.genAStarMap(goal)
		closedList, openList = [], [self.aMap[start[0]][start[1]]]
		
		while openList:
			#openList.sort(key = lambda el: el.F, reverse = True)
			current = openList.pop(0)
			closedList.append(current)
			print("checking {}, {}".format(current.y, current.x))
			if current.y == goal[0] and current.x == goal[1]:
				self.path = (getPath(current))
				print(self.path)
				self.pathToWaypoints()
				print(self.path)
				return
			for y in range(-1, 2):
				for x in range(-1,2):
					if (y + current.y) >= 0 and (x + current.x) >=0:
						checkCell = self.aMap[y + current.y][x + current.x]
						if not checkCell.walkable or checkCell in closedList:
							continue
						else:
							newG = current.G + (10 if abs(y) - abs(x) else 14)
							if not checkCell in openList:
								checkCell.parent = current
								checkCell.G = newG
								checkCell.F = checkCell.G + checkCell.H
								bisect.insort_left(openList, checkCell)
							elif newG < checkCell.G:
								checkCell.G = newG
								checkCell.F = checkCell.G + checkCell.H
								checkCell.parent = current
								openList.sort(key = lambda el: el.F)
						
		print("NO PATH")
	
	def gradDesc(self, goal, start): # populates grid with values increasing with disctance from goal, then returns path from target by taking the minimum neighbour until target is reached
		self.genPathMap()
		if (self.pathMap[goal[0]][goal[1]] != 1 or self.pathMap[start[0]][start[1]] != 1):
			print "Error!!!!!!!!"
			return		
			
		neighbours = [[-1, True], [1, True], [-1, False], [1, False]]
		self.pathMap[goal[0], goal[1]] = 0
		counter = 2
		coordList = [[goal[0], goal[1]]]
		while len(coordList):
			newList = []
			for coord in coordList:
				for neighbour in neighbours:
					nCoord = [coord[0] + neighbour[0] * neighbour[1], coord[1] + neighbour[0] * (not(neighbour[1]))]
					if self.pathMap[nCoord[0], nCoord[1]] == 1:
						self.pathMap[nCoord[0], nCoord[1]] = counter
						newList.append([nCoord[0], nCoord[1]])
			counter += 1
			coordList = newList
			
		# pyplot.imshow(self.pathMap, pyplot.cm.gray)
		# pyplot.show()

		self.path = [[start[0], start[1]]]
		currVal = self.pathMap[start[0]][start[1]]
		coord = [start[0], start[1]]
				
		while self.pathMap[coord[0], coord[1]] > 1:
			min = self.pathMap[coord[0], coord[1]]
			for neighbour in neighbours:
				nCoord = [coord[0] + neighbour[0] * neighbour[1], coord[1] + neighbour[0] * (not(neighbour[1]))]
				if self.pathMap[nCoord[0], nCoord[1]] < min and self.pathMap[nCoord[0], nCoord[1]]  >= 0:
					min = self.pathMap[nCoord[0], nCoord[1]]
					newCoord = [nCoord[0], nCoord[1]]
			self.path.append(newCoord)
			coord = newCoord
			
		print(self.path)
		self.drawPath()
		self.pathToWaypoints()
		self.toGazeboCoord()

	def pathToWaypoints(self): # detects turns in path, and generates list of waypoints
		slope = None
		currPoint = self.path[0]
		wayPoints = []
		
		for coord in self.path[1::]:
			if slope != [coord[1] - currPoint[1], coord[0] - currPoint[0]]:
				wayPoints.append(currPoint)
				slope = [coord[1] - currPoint[1], coord[0] - currPoint[0]]
			currPoint = coord
			
		if wayPoints[len(wayPoints) - 1] != self.path[len(self.path) - 1] :
			wayPoints.append(self.path[len(self.path) - 1])
			
		self.path = wayPoints

	def toGazeboCoord(self): # translate grid coordinates to gazebo coordinates
		self.path = [[(coord[0] - self.originOffset[0] - (len(self.map) / 2)) * -self.resolution, (coord[1] - (len(self.map[0]) / 2) + self.originOffset[1]) * self.resolution] for coord in self.path]
		print(self.path)
		
	def drawPath(self): # draw path on image
		for coord in self.path:
			self.pathMap[coord[0], coord[1]] = self.pathColour

	def drivePath(self): # driving function for bot
		# turn towards waypoint, when facing, drive, checking bearing, repeat for each waypoint
		for wayPoint in self.path:
			self.turnToPoint(wayPoint)
			self.driveToPoint(wayPoint)
			
	def turnToPoint(self, point):
		while True:
			heading = getHeading(point, False)
			print(heading)
			
			self.actionPublisher.publish(genTwist(0, heading))
			if abs(heading) < .1:
				return

	def driveToPoint(self, point):
		while True:
			heading, distance = getHeading(point, True)
			print(heading)
			
			self.actionPublisher.publish(genTwist(.15, heading))
			if distance < .1:
				self.actionPublisher.publish(genTwist(0,0))
				return
	
def getPath(node):
	nodeList = []
	while node.parent:
		nodeList.append([node.y, node.x])
		node = node.parent
	nodeList.append([node.y, node.x])
	return list(reversed(nodeList))
	
def readPgm(pgmf):
	assert pgmf.readline() == 'P5\n'
	test = pgmf.readline()
	(width, height) = [int(i) for i in pgmf.readline().split()]
	depth = int(pgmf.readline())
	assert depth <= 255
		
	return np.fromfile(pgmf, dtype=np.uint8).reshape((height, width))

def genTwist(speed, z): # build twist to be sent to gazebo
	retTwist = Twist()
	retTwist.linear.x = speed
	retTwist.angular.z = z
	return retTwist

def getHeading(point, isDriving):
	odomData = None
	while odomData is None:
		try:
			odomData = rospy.wait_for_message('odom', Odometry, timeout=5)
		except Exception as e:
			pass
	
	# get goal data
	modelX = odomData.pose.pose.position.x
	modelY = odomData.pose.pose.position.y
	
	goalAngle = math.atan2(point[0] - modelY, point[1] - modelX)
	modelAngle = odomData.pose.pose.orientation
			
	yaw = math.atan2(+2.0 * (modelAngle.w * modelAngle.z + modelAngle.x * modelAngle.y), 1.0 - 2.0 * (modelAngle.y * modelAngle.y + modelAngle.z * modelAngle.z))
	
	if isDriving:
		return goalAngle - yaw, math.hypot(point[1] - modelX, point[0] - modelY)
	else:
		return goalAngle - yaw

if __name__ == '__main__':
	rospy.init_node('pathFinder')
	try:
		pf = pathFinder()
		#pf.gradDesc([168, 168], [200, 210])
		pf.aStar([168, 168], [200, 210])
		#pyplot.imshow(pf.pathMap, pyplot.cm.gray)
		#pyplot.show()
		pf.drivePath()
	except rospy.ROSInterruptException:
		pass