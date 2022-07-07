from ast import Str
from map import Location, Agent, Corridor
from pbs import make_default_corridor_direction, state_plus_time_offset, make_default_agent_history, make_default_agent_goal, update_agent_corridor_direction, update_agent_goal_dict, states_to_dict
import random
import yaml
import os

class Environment:
    def __init__(self, name: Str, dimension: [int,int], mapf_solver, agents: [Agent], corridors: [Corridor]):
        if(dimension[0] < mapf_solver.map.dimension[0] or dimension[1] < mapf_solver.map.dimension[1]):
            print("Environment's dimension is not big enough for mapf_solver's map.")

        # map init
        self.name = name
        self.dimension = dimension
        self.mapf_solver = mapf_solver
        self.map = mapf_solver.map
        self.agents = agents
        self.corridors = corridors
        self.corridor_direction_defaults = [corridor.reverse for corridor in self.corridors]
        self.highway_type = "none"
        self.set_highway_type("none")
        
        # log init
        self.spaces = []
        self.agent_history = []
        self.agent_goal = []
        self.corridor_direction = []
        
        self.total_finished_tasks = 0
        self.total_forward_distance = 0
        
    def reset(self, seed: int):
        self.total_finished_tasks = 0
        self.total_forward_distance = 0
        self.spaces = self.map.get_spaces()
        random.seed(seed)
        random.shuffle(self.spaces)

        for idx, agent in enumerate(self.agents.values()):
            location = self.spaces[idx]
            agent.location = Location(location[0], location[1])

        random.shuffle(self.spaces)
        for agent in self.agents.values():
            goal = self.spaces.pop(0)
            agent.goal = Location(goal[0], goal[1])

        self.reset_corridors()
            
        self.agent_history = make_default_agent_history(self.agents)
        self.agent_goal = make_default_agent_goal(self.agents)
        self.corridor_direction = make_default_corridor_direction(self.corridors)

        if self.highway_type == "strict":
            self.mapf_solver.map.fit_corridors(self.corridors)
        else:
            self.map.reset()

    def reset_corridors(self):
        for corridor, direction in zip(self.corridors, self.corridor_direction_defaults):
            corridor.reverse = direction
    
    def find_location_in_corridors(self, x, y):
        """
        Find the index of the corridor from self.corridors, which contains the location.
        """
        for idx, corridor in enumerate(self.corridors):
            if ((x == corridor.start.x and corridor.start.x == corridor.end.x and corridor.start.y <= y and corridor.end.y >= y) or
                (y == corridor.start.y and corridor.start.y == corridor.end.y and corridor.start.x <= x and corridor.end.x >= x)):
                return idx
        return None
    
    def set_highway_type(self, highway_type="strict"):
        assert(highway_type=="strict" or highway_type=="soft" or highway_type=="none")
        self.highway_type = highway_type
        if self.highway_type == "strict":
            self.reset_corridors()
            self.mapf_solver.map.fit_corridors(self.corridors)
        else:
            self.map.reset()
    
    def step(self, time_step=0):
        update_agent_goal_dict(self.agents, self.agent_goal, time_step)

        update_agent_corridor_direction(self.corridors, self.corridor_direction, time_step)
        solution, reach_nodes, expand_nodes = self.mapf_solver.search(self.agents, return_info=True) 

        for agent_name, history in solution.items():
            state_plus_time_offset(history, time_step, sub_state_at_time_zero=True)
            self.agent_history[agent_name] += history

        finished_tasks = 0
        forward_distance = 0

        for agent_name, agent in self.agents.items():
            last_history = self.agent_history[agent_name][-1]
            forward_distance += self.mapf_solver.get_agent_forward_distance(last_history, agent)
            agent.location = last_history.location
            if(agent.location == agent.goal):
                finished_tasks += 1
                self.spaces.append((agent.goal.x, agent.goal.y))
                goal = self.spaces.pop(0)
                agent.goal = Location(goal[0], goal[1])
                
        self.total_finished_tasks += finished_tasks
        self.total_forward_distance += forward_distance

        no_solution_in_time = False if solution else True
        return finished_tasks, forward_distance, no_solution_in_time, reach_nodes, expand_nodes
     
    def output_yaml_history(self, directory: Str, output_filename: Str):
        # Output History
        agent_history = self.agent_history
        agent_goal = self.agent_goal
        
        for agent_name, history in agent_history.items():
            agent_history[agent_name] = states_to_dict(history)

        for agent_name, goal in agent_goal.items():
            agent_goal[agent_name] = states_to_dict(goal)

        output = {} 
        output["tasks"] = self.total_finished_tasks
        output["schedule"] = agent_history
        output["goal"] = agent_goal
        if not self.highway_type == "none":
            output["direction"] = self.corridor_direction

        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + self.name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + output_filename, 'w+') as output_yaml:
            yaml.safe_dump(output, output_yaml)