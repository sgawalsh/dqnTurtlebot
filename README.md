# dqnTurtlebot

The original goal of this project was to apply deep Q-learning in order to train a neural network drive a simulated bot from the turtle_bot3 package within an Gazebo 3-D phyics simulation environment while avoiding obstacles. The code for this module is contained in the `turt_q_learn.py` file in the scripts folder. In addition to the deep Q-learning algorithm, the script is structured so that a user can easily compare and contrast different types of models, or alter specific environment variables.

The environment variables are listed at the top of the script. The model parameters are contained in a `hyperParameterList` variable which is used to generate a dictionary for every possible combination of hyperparameters selected. The models and graphs detailing the results of each parameter combination are automatically saved in the `dqnmodels` folder. A user can also use a previously built model by adding a `model` file to the `load_model` folder and setting the `Load Model` value in the `hyperparmeterList` to `True`.

A video including a demonstration of the learning process in action, as well as a walkthrough of the code is located [here](https://www.youtube.com/watch?v=3VI_wHK4FtI).

The project was then extended to include a pathfinding module, using a map generated from the `turtlebot3_slam` ROS package as input, a user can set a 'start' and 'goal' coordinate. The script will generate a path using either the wavefront or a-star pathfinding algorithms, selected by calling the pathFinder.gradDesc or pathFinder.aStar functions respectively. The path is then converted to a set of waypoints which is translated into gazebo environment coordinates which are fed to a simple pathwalking algorithm in order to drive the bot through the set of waypoints.
