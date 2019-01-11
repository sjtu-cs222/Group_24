# Qlearning for robot navigation in dynamic environments--source code introduction

## qlearning_motion_planning.py
The main program.
The map class, robot class are defined here.
To add static obstacles, just add points to the sta_obst list in the main function part at the end of the code. 
To set dynamic obstacles number, just set parameters "dyn_obstacle" to create a map class object.
To create a new q-table, use the method init_Q() for Robot class objects.
To train, use train() method.
Use draw_train_track() to get a img for the train scenarios.
Use save_q_table() to save your trained q-table.
Use run() to use the trained q-table to instruct a robot.

## dynamic_obstacles.py
The dynamic obstacle class is defined here.
The dynamic obstacles will generate randomly in the map and do random walk.

## animation_class_test.py
The test program for the qlearning algorithm result.
Using matplotlib animation package to simulate a robot navigating in a dynamic environment using the Qlearning algorithm result.
To add static obstacles, just add points to the sta_obst list in the main function part at the end of the code. 
To set dynamic obstacles number, just set parameters "dyn_obstacle" to create a map class object.
First, you should create a map object, a robot object.
Then, create an Animator object.
Last, use plt.show() to see the process of simulation.