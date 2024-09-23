# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:34:29 2024

@author: eghorbanioskalaei
"""
import numpy as np
import random
import math
from collections import deque
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

class CapacitatedDispersion:
    def __init__(self):
        self.n = 0  # Number of nodes
        self.b = 0  # Required capacity
        self.capacity = []  # Capacities of the nodes
        self.distance = None  # Distance matrix
        self.sortedDistances = []  # List to store sorted distances

    def readInstance(self, s):
        with open(s) as instance:
            i = 1
            fila = 0
            for line in instance:
                if line == "\n":
                    continue
                if i == 1:
                    self.n = int(line)
                    self.distance = np.zeros((self.n, self.n))
                elif i == 2:
                    self.b = int(line)
                elif i == 3:
                    l = line.rstrip('\t\n ')
                    self.capacity = [float(x) for x in l.split('\t')]
                else:
                    l = line.rstrip('\t\n ')
                    d = [float(x) for x in l.split('\t')]
                    for z in range(self.n):
                        if d[z] != 0:
                            self.distance[fila, z] = d[z]
                            self.sortedDistances.append((fila, z, d[z]))
                    fila += 1
                i += 1
        self.sortedDistances.sort(key=lambda x: x[2], reverse=True)

    def state_from_index(self, index):
        return [int(x) for x in format(index, f'0{self.n}b')]

    def index_from_state(self, state):  #[1,0,0,1]>>['1','0','0','1']
        return int(''.join(map(str, state)), 2)

    def action_space(self, state): #List of action is a list consisting the opossite state of all the nodes
        actions = []
        for i in range(self.n):
            new_state = state.copy()
            new_state[i] = 1 - state[i]
            actions.append((i, new_state))
        return actions

    def reward(self, state):
        selected_nodes = [i for i, x in enumerate(state) if x == 1]
        if sum(self.capacity[i] for i in selected_nodes) < self.b:
            return -1  # Penalize infeasible states
        if len(selected_nodes) < 2:
            return 0
        min_distance = float('inf')
        for i in range(len(selected_nodes)):
            for j in range(i + 1, len(selected_nodes)):
                min_distance = min(min_distance, self.distance[selected_nodes[i], selected_nodes[j]])
        return min_distance  #the output is just one distance between two selcted nodes which are more closer!!!!
    
    
    # def reward(self, state):
    #     selected_nodes = [i for i, x in enumerate(state) if x == 1]
    #     unselected_nodes = [i for i, x in enumerate(state) if x == 0]

    #     # Penalize infeasible states: if the capacity of selected nodes is less than required
    #     if sum(self.capacity[i] for i in selected_nodes) < self.b:
    #         return -1  # Infeasible solution
        
    #     # If fewer than two nodes are selected, return 0 as there is no meaningful dispersion
    #     if len(selected_nodes) < 2:
    #         return 0  # No selected nodes
        
    #     # Find the closest selected node to each unselected node
    #     min_distance = float('inf')  # Start with a large number
        
    #     for v in unselected_nodes:
    #         minDist = float('inf')  # Minimum distance for this unselected node
    #         vMin = None
            
    #         # Iterate through the selected vertices in the solution
    #         for s in selected_nodes:
    #             # Calculate the distance from the selected vertex (s) to the target vertex (v)
    #             d = self.distance[s][v]
                
    #             # If the calculated distance is smaller than the current minimum distance
    #             if d < minDist:
    #                 # Update the minimum distance and the closest vertex
    #                 minDist = d
            
    #         # After looping over all selected nodes, update the overall minimum distance
    #         if minDist < min_distance:
    #             min_distance = minDist

    #     # The reward is the minimum distance found between an unselected node and its closest selected node
    #     return min_distance
    
    
    
    
def biased_random_choice(action_values, beta):
    sorted_actions = sorted(enumerate(action_values), key=lambda x: -x[1])  # Sort actions by value descending
    c_list = [action for action, _ in sorted_actions]
    chosen_index = int(math.log(random.random()) / math.log(1 - beta))
    chosen_index = min(chosen_index, len(c_list) - 1)  # Ensure index is in range
    return c_list[chosen_index]

def tabu_search(instance, initial_state, max_iterations=50, tabu_size=10):
    """Tabu search to refine the solution."""
    current_state = initial_state.copy()
    tabu_list = deque(maxlen=tabu_size)
    best_state = current_state.copy()
    best_value = instance.reward(current_state)

    for _ in range(max_iterations):
        neighborhood = instance.action_space(current_state)
        candidate_states = [(state, instance.reward(state)) for _, state in neighborhood if state not in tabu_list]
        if not candidate_states:
            break

        next_state, next_value = max(candidate_states, key=lambda x: x[1])

        if next_value > best_value:
            best_state = next_state
            best_value = next_value

        tabu_list.append(current_state)
        current_state = next_state

    return best_state, best_value




def plot_nodes_with_mds(instance, optimal_state, title="Node Locations with Selected Nodes"):
    # Use MDS to reduce the distance matrix to 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(instance.distance)  # The positions are derived from the distance matrix

    # Create a plot
    plt.figure(figsize=(8, 8))
    
    # Plot all nodes
    for i in range(instance.n):
        if optimal_state[i] == 1:  # If node is selected
            plt.scatter(positions[i][0], positions[i][1], color='red', s=100, label='Selected Nodes' if i == 0 else "")  # Plot selected nodes in red
        else:
            plt.scatter(positions[i][0], positions[i][1], color='blue', s=50, label='Other Nodes' if i == 0 else "")  # Plot unselected nodes in blue
    
    # Plot distances between selected nodes
    selected_nodes = [i for i, x in enumerate(optimal_state) if x == 1]
    for i in range(len(selected_nodes)):
        for j in range(i + 1, len(selected_nodes)):
            x_values = [positions[selected_nodes[i]][0], positions[selected_nodes[j]][0]]
            y_values = [positions[selected_nodes[i]][1], positions[selected_nodes[j]][1]]
            plt.plot(x_values, y_values, 'gray', linestyle='--')  # Connect selected nodes
    
    plt.title(title)
    plt.xlabel('X-coordinate (MDS)')
    plt.ylabel('Y-coordinate (MDS)')
    plt.legend()
    plt.grid(True)
    plt.show()




def run_q_learning_with_tabu_search(instance_file_path, num_episodes=20, alpha=0.6, gamma=0.9, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.01, beta=0.4, tabu_iterations=10, tabu_size=10):
    np.random.seed(42)  # Set seed for numpy random number generation
    random.seed(42) 
    instance = CapacitatedDispersion()
    instance.readInstance(instance_file_path)

    Q = {}  # Q-values dictionary

    for e in range(num_episodes):
        state = [random.randint(0, 1) for _ in range(instance.n)]  # State is a list that shows the selected nodes and not selected ones 
        done = False
        while not done:
            state_index = instance.index_from_state(state)
            if state_index not in Q:
                Q[state_index] = np.zeros(instance.n)

            if np.random.rand() <= epsilon:  # exploration step
                action, next_state = random.choice(instance.action_space(state))
            else:  # exploitation step
                action_values = Q[state_index]
                action = biased_random_choice(action_values, beta)
                next_state = [state[i] if i != action else 1 - state[i] for i in range(instance.n)]
                #action_values = Q[state_index]
                #best_action = np.argmax(action_values)  #choose the best available action
                #action = best_action
                #next_state = [state[i] if i != best_action else 1 - state[i] for i in range(instance.n)]
                
                

            reward = instance.reward(next_state)
            next_state_index = instance.index_from_state(next_state)

            if next_state_index not in Q:  #this part added the new state to the Q dic, and get 0 vlaue to that
                Q[next_state_index] = np.zeros(instance.n)

            done = (reward == -1)  # determines whether the current episode of the Q-learning algorithm should terminate
            if not done:
                #best_next_action = biased_random_choice(Q[next_state_index], beta)
                best_next_action = np.argmax(Q[next_state_index])
                Q[state_index][action] = Q[state_index][action] + alpha * (reward + gamma * Q[next_state_index][best_next_action] - Q[state_index][action])

            state = next_state

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # After Q-learning, apply Tabu Search to refine the solution
        refined_state, refined_value = tabu_search(instance, state, max_iterations=tabu_iterations, tabu_size=tabu_size)
        
    optimal_policy = {s: np.argmax(Q[s]) if s in Q else None for s in Q}
    optimal_state_index = max(Q, key=lambda s: np.max(Q[s]))
    optimal_state = instance.state_from_index(optimal_state_index)
    optimal_value = instance.reward(optimal_state)
    # Example usage
    plot_nodes_with_mds(instance, optimal_state, title="Optimal Selected Nodes with MDS-based Visualization")
    return optimal_policy, optimal_state, optimal_value, refined_state, refined_value



# Example usage
instance_file_path = 'CDP/GKD-b_11_n50_b02_m5.txt'  # Adjust this path as needed

# Run Q-learning with biased randomization and Tabu Search
optimal_policy, optimal_state, optimal_value, refined_state, refined_value = run_q_learning_with_tabu_search(instance_file_path)

#print("Optimal Policy (Q-learning):")
#print(optimal_policy)
print("Optimal State (Q-learning):")
print(optimal_state)
print("Optimal Value (Q-learning):")
print(optimal_value)
print("Refined State (Tabu Search):")
print(refined_state)
print("Refined Value (Tabu Search):")
print(refined_value)





