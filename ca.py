import argparse
from os import stat
from re import S
import yaml
from map import Location, Map, Agent, Corridor, PrioritySet
from math import fabs
from copy import deepcopy
from itertools import combinations
from random import shuffle, seed
from numpy.random import shuffle as shffle_v2
from numpy.random import seed as seed_v2
import time
        
class State:
    def __init__(self, time, location):
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location.x) + str(self.location.y))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location.x, self.location.y))

def state_plus_time_offset(states: [State], time_offset: int, sub_state_at_time_zero: bool):
    if(sub_state_at_time_zero):
        for state in states:
            if(state.time == 0):
                states.remove(state)
                break
    for state in states:
        state.time += time_offset

def states_to_dict(states: [State]):
    return [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in states]

class Conflict:
    VERTEX = 1
    EDGE = 2
    def __init__(self, time, conflict_type, agent_1, agent_2, location_1, location_2):
        self.time = time
        self.type = conflict_type

        self.agent_1 = agent_1
        self.agent_2 = agent_2

        self.location_1 = location_1
        self.location_2 = location_2

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint:
    def __init__(self, time, location, agent_name):
        self.time = time
        self.location = location
        self.agent_name = agent_name

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash((self.time, self.location.x, self.location.y))
    def __str__(self):
        return '(' + str(self.agent_name) + ': ' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint:
    def __init__(self, time, location_1, location_2, agent_name):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
        self.agent_name = agent_name
        
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash((self.time, self.location_1.x, self.location_1.y, self.location_2.x, self.location_2.y))
    def __str__(self):
        return '(' + str(self.agent_name) + ': ' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints:
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def constraint_filter(self, filter_list):
        new_constraints = Constraints()
        for v_c in self.vertex_constraints:
            if v_c.agent_name in filter_list:
                new_constraints.vertex_constraints |= {v_c}
        for e_c in self.edge_constraints:
            if e_c.agent_name in filter_list:
                new_constraints.edge_constraints |= {e_c}
        return new_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

# class Stack:
#     def __init__(self):
#         self.list = []

#     def pop(self, index=-1):
#         return self.list.pop(index)

#     def push(self, node):
#         self.list.append(node)

#     def push_by_priority(self, node):
#         if len(self.list):
#             idx = -1 
#             while(len(node.priorities.priority_list) == len(self.list[idx].priorities.priority_list)):   
#                 if node.cost == self.list[idx].cost:
#                     if node.priorities.priority_list < self.list[idx].priorities.priority_list:
#                         break
#                 elif node.cost < self.list[idx].cost:
#                     break
#                 idx -= 1
#                 if -idx > len(self.list):
#                     break
#             self.list.insert(idx, node)
#         else:
#             self.list.append(node)

class CA:
    def __init__(self, map, window_size = 20, buffer_size = 10, use_manhat = True, heuristic_distance_map=None, abstract_distance_map=None):
        self.map = map
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.plan_full_paths = True
        self.use_manhat = use_manhat
        self.heuristic_distance_map = heuristic_distance_map # Valid if "use_manhat is False"
        self.abstract_distance_map = abstract_distance_map   # Valid if "use_manhat is False"
        self.shffle = shffle_v2
        seed_v2(0)
            # for i in self.distance_map[:][::-1][0][0]: print(i)
            # import pdb; pdb.set_trace()
    
    def search(self, agents, time_limit=60, return_info=False):
        reach_nodes = 0
        expand_nodes = 0

        time_start = time.time()

        priorities = [agent for agent in agents]

        while (time.time()-time_start) < time_limit:
            reach_nodes += 1
            expand_nodes += 1
            solution = self.compute_solution(agents, priorities)

            if solution:
                if return_info:
                    return self.clip_solution(solution), reach_nodes, expand_nodes
                else: 
                    return self.clip_solution(solution)
            
            shffle_v2(priorities)

        print("No solution found.")
        if return_info:
            return {}, 0, 0 
        else: 
            return {}

    def clip_solution(self, solution):
        for agent_name, path in solution.items():
            end = min(len(path), self.buffer_size+1)
            solution[agent_name] = path[:end]
        return solution
    
    def print_solution(self, solution):
        for agent_name, path in solution.items():
            print(agent_name + ":")
            for step in path:
                print(step)
        return solution

    def get_neighbors(self, state, agent, constraints):
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_up and self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_down and self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_left and self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_right and self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
        return neighbors
    
    def state_wait(self, state):
        return State(state.time + 1, state.location)
    
    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    
    def location_in_vertex_constraint_after_time(self, constraints, state):
        for v in constraints:
            if v.location == state.location and v.time >= state.time:
                return True
        return False

    def state_valid(self, state, constraints, is_goal):
        if is_goal:
            return state.location.x >= 0 and state.location.x < self.map.dimension[0] \
                and state.location.y >= 0 and state.location.y < self.map.dimension[1] \
                and self.location_in_vertex_constraint_after_time(constraints, state) \
                and not self.map.grid[state.location.y][state.location.x].grid_property.is_obstacle
        else:
            return state.location.x >= 0 and state.location.x < self.map.dimension[0] \
                and state.location.y >= 0 and state.location.y < self.map.dimension[1] \
                and VertexConstraint(state.time, state.location, "") not in constraints \
                and not self.map.grid[state.location.y][state.location.x].grid_property.is_obstacle

    def transition_valid(self, state_1, state_2, edge_constraints):
        return EdgeConstraint(state_1.time, state_1.location, state_2.location, "") not in edge_constraints

    def admissible_heuristic(self, state, agent):
        if(self.use_manhat):
            return fabs(state.location.x - agent.goal.x) + fabs(state.location.y - agent.goal.y)
        else:
            return self.heuristic_distance_map[state.location.y][state.location.x][agent.goal.y][agent.goal.x]
    
    def get_agent_forward_distance(self, new_location, agent):
        if(self.use_manhat):
            return fabs(agent.goal.x-agent.location.x) + fabs(agent.goal.y-agent.location.y) \
                - fabs(agent.goal.x-new_location.x) - fabs(agent.goal.y-new_location.y)
        else:
            return self.heuristic_distance_map[agent.location.y][agent.location.x][agent.goal.y][agent.goal.x] \
                - self.heuristic_distance_map[new_location.y][new_location.x][agent.goal.y][agent.goal.x]
    
    def is_at_goal(self, state, agent):
        return state.location == agent.goal
    
    def compute_solution(self, agents, priorities):
        solution = {}

        constraints = Constraints()
        for agent_name in priorities:
            local_solution, local_cost = self.astar(agents[agent_name], constraints)
            if not local_solution:
                return None
            solution[agent_name] = local_solution
            self.add_constraint_from_solution(agent_name, local_solution, constraints)
        # print(local_cost)
        return solution
    
    def compute_solution_cost(self, costs):
        return sum(list(costs.values()))

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        end = min(self.window_size+1, len(path))
        return path[:end]
    
    def add_constraint_from_solution(self, agent_name, solution, constraints):
        last_state = None
        end = min(self.window_size+1, len(solution))
        for state in solution[:end]:
            v_constraint = VertexConstraint(state.time, state.location, agent_name)
            constraints.vertex_constraints |= {v_constraint}
            if last_state and not last_state.is_equal_except_time(state):
                e_constraint1 = EdgeConstraint(last_state.time, last_state.location, state.location, agent_name)
                e_constraint2 = EdgeConstraint(last_state.time, state.location, last_state.location, agent_name)
                constraints.edge_constraints |= {e_constraint1}
                constraints.edge_constraints |= {e_constraint2}
            last_state = state
        for i in range(len(solution), self.window_size+1):
            v_constraint = VertexConstraint(i, solution[-1].location, agent_name)
            constraints.vertex_constraints |= {v_constraint}

    def astar(self, agent, constraints):
        # print(constraints)
        """
        low level search 
        """
        initial_state = State(0, agent.location)
        step_cost = 1
        
        closed_set = set()
        open_set = PrioritySet()

        came_from = {}

        g_score = {} 
        g_score[initial_state] = 0

        f_score = {} 

        f_score[initial_state] = self.admissible_heuristic(initial_state, agent)

        open_set.add(initial_state, (f_score[initial_state], g_score[initial_state], initial_state.location.x, initial_state.location.y))

        while open_set.set:
            current = open_set.pop()

            if self.is_at_goal(current, agent) or (not self.plan_full_paths and g_score.setdefault(current, float(-1)) >= self.window_size):

                # if (agent.name == "agent5") and agent.goal.x == 8 and agent.goal.y == 6:
                #     with open("./log1.txt", "a+") as f:
                #         f.write("  agent5"+str(agent.location)+"\n")
                #         print("  agent5"+str(agent.location)+"\n")
                #         for i in self.reconstruct_path_full(came_from, current):
                #             f.write(str(i)+'\n')
                #         f.write(str(constraints)+'\n')
                #         print(constraints)
                # if (agent.name == "agent5" and agent.location == Location(4,3)):
                #     print(agent.goal, agent.location)
                #     print(f_score[current], g_score[current])
                #     self.reconstruct_path_print(came_from, current)
                return self.reconstruct_path(came_from, current), f_score[current]

            closed_set |= {current}

            neighbor_list = self.get_neighbors(current, agent, constraints)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                if neighbor not in open_set.set:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent)
                    open_set.add(neighbor, (f_score[neighbor], g_score[neighbor], neighbor.location.x, neighbor.location.y))
                    
                elif tentative_g_score < g_score.setdefault(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent)
                    open_set.update(neighbor, (f_score[neighbor], g_score[neighbor], neighbor.location.x, neighbor.location.y))

        return None, None


def read_input_file(input):
    # Read from input file
    with open(input, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    map = Map(param["map"]["dimensions"][0], param["map"]["dimensions"][1])
    map.fit_obstacles(param["map"]["obstacles"])

    agents = {}
    for agent in param['agents']:
        agents[agent['name']] = Agent(agent['name'], Location(agent['start'][0], agent['start'][1]), Location(agent['goal'][0], agent['goal'][1]))

    corridors = []
    for corridor in param["map"]["corridor"]:
        corridors.append(Corridor(Location(corridor['start'][0], corridor['start'][1]),Location(corridor['end'][0], corridor['end'][1]), corridor['reverse']))
    
    return map, agents, corridors

def make_default_agent_history(agents: dict):
    # make default history of agents with origin places
    agent_history = {}
    for agent in agents.values():
        agent_history[agent.name] = [State(0, agent.location)]
    return agent_history

def make_default_agent_goal(agents: dict):
    # make default dictionary of goals of agents
    agent_goal = {}
    for agent in agents.values():
        agent_goal[agent.name] = []
    return agent_goal

def make_default_agent_idlesteps(agents: dict):
    # make default dictionary of the counter of idle steps of agents
    agent_idlesteps= {}
    for agent in agents.values():
        agent_idlesteps[agent.name] = 0
    return agent_idlesteps

def make_default_agent_movesteps(agents: dict):
    # make default dictionary of the counter of moving steps of agents
    agent_movesteps= {}
    for agent in agents.values():
        agent_movesteps[agent.name] = 0
    return agent_movesteps

def make_default_agent_expectedsteps(agents: dict):
    # make default dictionary of the expected value of moving steps to the goals of agents
    agent_expectedsteps= {}
    for agent in agents.values():
        agent_expectedsteps[agent.name] = 0
    return agent_expectedsteps

def make_default_corridor_direction(corridors):
    # make default dictionary of direction of corridors
    corridor_direction = {}
    for i in range(len(corridors)):
        corridor_direction[i] = []
    return corridor_direction

def update_agent_goal_dict(agents: dict, agent_goals: dict, time: int):
    # make dictionary of goals of agents at the time
    for agent in agents.values():
        agent_goals[agent.name] += [State(time, agent.goal)]

def update_agent_idle_and_movesteps(agent_paths: dict, agent_idlesteps: dict, agent_movesteps: dict, buffer_size: int):
    # update the counter of moving steps of agents
    for agent, paths in agent_paths.items():
        if len(paths) > 1:
            for step in range(1, len(paths)):
                if not (paths[step-1].location == paths[step].location):
                    agent_movesteps[agent] += 1
                else:
                    agent_idlesteps[agent] += 1
        agent_idlesteps[agent] += buffer_size + 1 - len(paths)


def update_agent_corridor_direction(corridors: [Corridor], corridor_directions: dict, time: int):
    # make dictionary of direction of corridors at the time
    for idx, corridor in enumerate(corridors):
        corridor_directions[idx] += [{'t':time, 'd':corridor.reverse}]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input file containing map and obstacles")
    parser.add_argument("output", type=str, help="output file with the schedule")
    parser.add_argument("--use_manhat", type=str2bool, default=False, help="use Manhattan Distance for distance estimation or calculate distance for every pair of points at start")
    parser.add_argument("--use_highway_heuristic", type=str2bool, default=True, help="use Highway Setup for distance estimation or calculate distance for every pair of points at start")
    parser.add_argument("--highway_heuristic_setup", type=int, default=None, help="use Strict Highway Limitation or Soft Highway Heuristic Function")
    args = parser.parse_args()

    map, agents, corridors = read_input_file(args.input)
    agent_history = make_default_agent_history(agents)

    # get spaces for assigning tasks
    spaces = map.get_spaces()
    seed(1)
    shuffle(spaces)
    shuffle(spaces)

    if(len(spaces) < len(agents)):
        print("Spaces are not enough for agents.")
        return

    for agent in agents.values():
        goal = spaces.pop()
        agent.goal = Location(goal[0], goal[1])


    # 1. Setup the planning algorithm.
    WINDOW_SIZE = 5
    BUFFER_SIZE = 5
    ITERATIONS = 50

    if args.use_highway_heuristic:
        map.fit_corridors(corridors)
        heuristic_distance_map = map.get_distance_map(args.highway_heuristic_setup)
        abstract_distance_map = heuristic_distance_map  # grid_map.get_distance_map() if highway_heuristic_setup else heuristic_distance_map
        map.reset()
        # for row in heuristic_distance_map[0][0]: print(row)
        # if not highway_heuristic_setup:
        #     import pdb; pdb.set_trace()
    else:
        heuristic_distance_map = map.get_distance_map()
        abstract_distance_map = heuristic_distance_map

    pbs = CA(map, WINDOW_SIZE, BUFFER_SIZE, args.use_manhat, heuristic_distance_map, abstract_distance_map)

    # map.fit_corridors(corridors)

    time_total = 0.0
    finished_tasks = 0

    map.show()

    for iteration in range(ITERATIONS):

        # 2. Search the paths of agents.
        time_start = time.time()
        solution = pbs.search(agents)
        time_end = time.time()
        print("computing time: ", time_end - time_start)
        time_total += time_end - time_start
        
        # cost.append(pbs.compute_solution_cost(solution))

        # 3. Save the paths of agents.
        for agent_name, history in solution.items():
            state_plus_time_offset(history, iteration*BUFFER_SIZE, sub_state_at_time_zero=True)
            agent_history[agent_name] += history

        # 4. Assign tasks to the agents that have finished their tasks.
        finished_agents = []
        for agent_name, agent in agents.items():
            # print(agent)
            last_history = agent_history[agent_name][-1]
            agent.location = last_history.location
            if(agent.location == agent.goal):
                finished_tasks += 1
                spaces.append((agent.goal.x, agent.goal.y))
                finished_agents.append(agent)

        shuffle(spaces)
        for agent in finished_agents:
            goal = spaces.pop()
            agent.goal = Location(goal[0], goal[1])
            
    print("total computing time: ", time_total, ", finished tasks", finished_tasks)

    for agent_name, history in agent_history.items():
        agent_history[agent_name] = states_to_dict(history)

    output = {} 
    output["schedule"] = agent_history
    # output["cost"] = cost
    with open(args.output, 'w+') as output_yaml:
        yaml.safe_dump(output, output_yaml)

if __name__ == "__main__":
    main()