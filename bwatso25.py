### AUTHOR: Bryce Watson ###
### Student ID: 220199390 ####
### UNE Account name: bwatso25 ###

### COSC350 Programming Assignment 1 2022 ###

import numpy as np

### Intitialize Globals ###

#width and height dimensions of the maze
maze_rows = 8
maze_columns = 8
#the percentage of time we should take the most promising action
epsilon = 0.9
#learning rate
alpha = 0.9
#discount factor
gamma = 0.5
#number of training episodes
episodes = 500

#define actions 0 = up, 1 = right, 2 = down, 3 = left
actions = ['up', 'right', 'down', 'left']

### Environment for our agent to explore- an 8x8 maze
# g = Goal. When the agent gets here the training episode will be over. A huge reward will be given for this area
# 1 = Wall. Same with goal, when the agent gets here the training epsiode will be over. However unlike the goal the wall punishes the agent greatly.
# 0 = Empty space. Minor punishment to encourage the agent to find the goal in as short a path as possible
# s = starting location.
maze_map = [['s','1','1','1','1','1','1','1'],
            ['0','0','0','1','0','0','0','1'],
            ['0','1','0','1','0','1','0','0'],
            ['0','0','1','0','0','1','0','1'],
            ['1','0','1','0','1','1','0','0'],
            ['0','0','1','0','0','0','1','0'],
            ['0','1','0','0','1','0','1','0'],
            ['0','0','0','1','0','0','1','g']]

#3d array that stores the q-value for each state and action, full of 0s for now
q_values = np.zeros((maze_rows, maze_columns, 4))

#2d array to store the reward for each state, full of 0s for now
rewards = np.zeros((maze_rows, maze_columns))


### End Globals ###

### Start Helper Functions ###

### Function to define rewards for each location in the environment.
def initialize_rewards():
    for i in range(len(maze_map)):
        for j in range(len(maze_map)):
            # if the locations is the goal location
            if maze_map[i][j] == 'g':
                rewards[i][j] = 100
            #If the location is a wall
            elif maze_map[i][j] == '1':
                rewards[i][j] = -100
            #Otherwise the location is a free space/starting point
            else:
                rewards[i][j] = -1

#determine if the location is an end_state (If the current location is a wall or goal)
def is_end_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -1:
        return False
    else:
        return True

#simple function to read the maze map for the starting symbol "s" and set it as the initial location of the agent
def get_start_location():
    for i in range(len(maze_map)):
        for j in range(len(maze_map)):
            if maze_map[i][j] == 's':
                return i, j

#Epsilon greedy algorithm that will choose the next action. With an epsilon value of 0.9, 90% of the time, the agent will chose the most promising action. The rest of the time It will choose a random action
#This encourages exploration.
def get_next_action(current_row_index, current_column_index, epsilon):
    #if a random value is less than epsilon
    if np.random.random() < epsilon:
        #choose optimal move from q_values
        return np.argmax(q_values[current_row_index, current_column_index])
    # Otherwise choose a random move
    else:
        #choose a random move
        return np.random.randint(4)

# Function that gets the next location based the action the agent chose
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < maze_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < maze_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

#Function that finds and returns the shortest path from the given starting location to the goal state.
def get_path(start_row_index, start_column_index):
    current_row_index, current_column_index = start_row_index, start_column_index
    #list in which to store the shortest path through the maze
    path = []
    #Add the starting location to the path
    path.append([current_row_index, current_column_index])
    #Continue moving through the maze until we reach the goal. The agent will not run into a wall because it will have been trained before calling this function, and the epsilon value for the epsilon greedy algorithim we are using to determing the agent's next
    #moves is at epsilon = 1. This means that the agent will choose the most promising action 100% of the time.
    while not is_end_state(current_row_index, current_column_index):
        action_index = get_next_action(current_row_index, current_column_index, 1.)
        #move to the next location and add the new location to the path.
        current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
        path.append([current_row_index, current_column_index])

    return path

#simple function that prints program information to console
def get_info():
    print("\nCOSC350 PROGRAMMING ASSIGNMENT 1- PYTHON MAZE AI\n")

    print("AUTHOR: Bryce Watson\n")
    print("STUDENT NUMBER: 220199390\n")
    print("UNE Account name: bwatso25\n")

    print("ABOUT THE PROGRAM:\n This program implements an AI Reinforcement Q-Learning Agent to solve a hard-coded maze in Python.\n It utilises temporal-difference learning (tdl) to learn about its environment and uses the epsilon-greedy algorithm to determine its next move at each state.\n")
    print("An Learning agent utilising temporal-difference learning learns by taking an action, and observing the reward for that action. It then updates its internal model by updating the reward for that state. In this way the AI can learn the optimal move at each state.")
    print("The Epsilon-greedy algorithm involves choosing the most promising action based on the value of epsilon. At Epsilon = 0.9, The agent will choose the action with the highest reward at each state 90%* of the time. the rest of the time it will choose a random action.\n")
    print("Here is some example psuedocode:")
    print("If random.value < epsilon: return max(q_values[row,column')")
    print("else: return random_action()\n")


    print("FINDING THE SHORTEST PATH THROUGH THE MAZE: Once the agent has been trained, it should have enough information about each state and action to find a path through the maze. When generating the optimal path, the Ai will use epislon greedy at epsilon = 100")
    print("which means it will choose the best action to take at every state. After every action, it moves to the new state and adds that location to the path. Once the agent has reached the goal, it will print the optimal path to console.\n")

### End Helper Functions ###

### Function that will train the agent through a specified amount of training episodes. The agent will start at the 's' location in the maze
# The training episode will end if it reachs an "end location" (Goal or wall).
# The goal of the agent is to accumulate as much rewards as possible.

def train():
    #Run through training episodes
    print("Agent is now doing", episodes, "Training episodes in the maze.\n")
    for episode in range(episodes):
        #Determine start location
        row_index, column_index = get_start_location()

        #Run until the agent runs into a wall or reaches the goal
        while not is_end_state(row_index, column_index):
            #decide what action to take using epsilon greedy algorithm
            action_index = get_next_action(row_index, column_index, epsilon)

            #take action, and move to the next location
            old_row_index, old_column_index = row_index, column_index
            row_index, column_index = get_next_location(row_index, column_index, action_index)

            #get reward and calculate temporal differnece
            reward = rewards[row_index, column_index]
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (gamma * np.max(q_values[row_index, column_index])) - old_q_value

            #update the q-value for the previous state and action
            new_q_value = old_q_value + (alpha * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value

    print('Training Complete!\n')


### Main function
def main():
    #Display program information to user
    get_info()
    #set rewards
    initialize_rewards()
    #print rewards table
    print("This is the rewards table for the maze. -1 is an empty space, -100 means a wall while 100 is the end goal: \n")
    for row in rewards:
        print(row)
    print("\n")
    #train the agent
    train()
    #set start location at 's' in the maze
    x, y = get_start_location()
    #get the shortest path through the maze
    print("The shortest path through the maze: \n", get_path(x, y))


if __name__ == "__main__":
    main()

