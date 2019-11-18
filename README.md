# dqnTurtlebot

This project implements Deep Q-Learning for a turtlebot in a gazebo environment using ROS. Results of the learning process including the final model as well as result plots are saved in the dqnmodels folder. Hyperparameters can be changed and compared by editing the `hyperParameterList` variable in the `turt_q_learn_hypers.py` file. Each hyperparameter combination is saved and plotted separately, with a combined plot being at the top level. The project also supports loading a previously constructed model by placing a Tensorflow Keras model with the filename `model` in the load_model folder, and setting the `hyperParameterList["Load Model"]` value to `True`.

1. Launch Gazebo environment by running `roslaunch turt_q_learn brick_world.launch`
2. Run DQN script with `rosrun turt_q_learn turt_q_learn_hypers.py`
