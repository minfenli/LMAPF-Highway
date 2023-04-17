import argparse
import yaml
from map import Location, Map, Agent, Corridor, PrioritySet
from math import fabs
from copy import deepcopy
from itertools import combinations
from random import shuffle, seed
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

class HighLevelNode:
    def __init__(self): 
        self.solution = {}
        self.constraint_dict = {}
        self.costs = {}
        self.cost = 0
        self.anti_direction_counts = {}

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash(self.cost)
    
    def __lt__(self, other):
        return self.cost < other.cost

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
        return '(' + str(self.time) + ', ' + self.agent_1.name + ', ' + self.agent_2.name + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint:
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash((self.time, self.location.x, self.location.y))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint:
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash((self.time, self.location_1.x, self.location_1.y, self.location_2.x, self.location_2.y))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints:
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class CBS:
    def __init__(self, map, window_size = 20, buffer_size = 10, use_manhat = True, heuristic_distance_map=None, abstract_distance_map=None, inflate_g_value=False):
        self.map = map
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.plan_full_paths = True
        self.use_manhat = use_manhat
        self.heuristic_distance_map = heuristic_distance_map # Valid if "use_manhat is False"
        self.abstract_distance_map = abstract_distance_map   # Valid if "use_manhat is False"
        self.inflate_g_value = inflate_g_value
        
    def search(self, agents, time_limit=60, return_info=False):
        reach_nodes = 0
        expand_nodes = 0

        time_start = time.time()

        open_set = set()
        closed_set = set()
        start = HighLevelNode()
        start.constraint_dict = {}
        for agent in agents.keys():
            start.constraint_dict[agent] = Constraints()
        start.solution, start.costs, start.anti_direction_counts = self.compute_solution(agents, start.constraint_dict)
        
        if not start.solution:
            if return_info:
                return {}, 0, 0, {} 
            else: 
                return {}
        start.cost = self.compute_solution_cost(start.costs)

        open_set |= {start}
        expand_nodes += 1
        while open_set:
            if (time.time()-time_start) >= time_limit:
                print("Time out.")
                break
            reach_nodes += 1
            P = min(open_set)
            open_set -= {P}
            closed_set |= {P}

            self.constraint_dict = P.constraint_dict
            conflict = self.get_first_conflict(P.solution)
            if not conflict:    
                if return_info:
                    return self.clip_solution(P.solution), reach_nodes, expand_nodes, P.anti_direction_counts
                else: 
                    return self.clip_solution(P.solution)
                    
            constraint_dict = self.create_constraints_from_conflict(conflict)

            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])
                conflict_agents = {conflict.agent_1: agents[conflict.agent_1], conflict.agent_2: agents[conflict.agent_2]}
                new_node.solution, new_node.costs, new_node.anti_direction_counts = self.compute_solution(conflict_agents, new_node.constraint_dict, new_node.solution, new_node.costs, new_node.anti_direction_counts)
                if not new_node.solution:
                    continue
                new_node.cost = self.compute_solution_cost(new_node.costs)

                if new_node not in closed_set:
                    open_set |= {new_node}
                    expand_nodes += 1
        print("No solution found.")
        if return_info:
            return {}, 0, 0, {} 
        else: 
            return {}

    def clip_solution(self, solution):
        for agent_name, path in solution.items():
            end = min(len(path), self.buffer_size+1)
            solution[agent_name] = path[:end]
        return solution

    def get_neighbors(self, state, agent, constraints):
        neighbors = []
        step_costs = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n, constraints.vertex_constraints) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_wait)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_up and self.state_valid(n, constraints.vertex_constraints) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_up)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_down and self.state_valid(n, constraints.vertex_constraints) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_down)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_left and self.state_valid(n, constraints.vertex_constraints) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_left)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_right and self.state_valid(n, constraints.vertex_constraints) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_right)
        return neighbors, step_costs

    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        check_region = min(max_t, self.window_size+1)
        for t in range(check_region):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    return Conflict(t, Conflict.VERTEX, agent_1, agent_2, state_1.location, Location())

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    return Conflict(t, Conflict.EDGE, agent_1, agent_2, state_1a.location, state_1b.location)
        return False

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict
    
    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state, constraints):
        return state.location.x >= 0 and state.location.x < self.map.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.map.dimension[1] \
            and VertexConstraint(state.time, state.location) not in constraints \
            and not self.map.grid[state.location.y][state.location.x].grid_property.is_obstacle

    def transition_valid(self, state_1, state_2, edge_constraints):
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in edge_constraints

    def admissible_heuristic(self, state, agent):
        if(self.use_manhat):
            return fabs(state.location.x - agent.goal.x) + fabs(state.location.y - agent.goal.y)
        else:
            return self.heuristic_distance_map[state.location.y][state.location.x][agent.goal.y][agent.goal.x]
    
    def get_agent_forward_distance(self, new_location, agent):
        if(self.use_manhat):
            return fabs(agent.goal.x-agent.location.x) + fabs(agent.goal.y-agent.location.y) \
                - fabs(agent.goal.x-new_location.location.x) - fabs(agent.goal.y-new_location.location.y)
        else:
            return self.heuristic_distance_map[agent.location.y][agent.location.x][agent.goal.y][agent.goal.x] \
                - self.heuristic_distance_map[new_location.y][new_location.x][agent.goal.y][agent.goal.x]
    
    def is_at_goal(self, state, agent):
        return state.location == agent.goal
    
    def compute_solution(self, agents, constraints, solution = {}, cost = {}, anti_direction_count = {}):
        for agent_name, agent in agents.items():
            local_solution, local_cost, local_anti_direction_count = self.astar(agent, constraints[agent_name])
            if not local_solution:
                return None, None, None
            solution[agent.name] = local_solution
            cost[agent.name] = local_cost
            anti_direction_count[agent.name] = local_anti_direction_count
        return solution, cost, anti_direction_count
    
    def compute_solution_cost(self, costs):
        return sum(list(costs.values()))

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        end = min(self.window_size+1, len(path))
        return path[:end]

    def astar(self, agent, constraints):
        """
        low level search 
        """
        initial_state = State(0, agent.location)
        
        closed_set = set()
        open_set = PrioritySet()

        came_from = {}

        g_score = {} 
        g_score[initial_state] = 0

        f_score = {} 
        f_score[initial_state] = self.admissible_heuristic(initial_state, agent)

        anti_direction_count = {}
        anti_direction_count[initial_state] = 0

        open_set.add(initial_state, (f_score[initial_state], g_score[initial_state], initial_state.location.x, initial_state.location.y))

        while open_set.set:
            current = open_set.pop()

            if self.is_at_goal(current, agent) or (not self.plan_full_paths and current.time >= self.window_size):
                return self.reconstruct_path(came_from, current), f_score[current], anti_direction_count[current]

            closed_set |= {current}

            neighbor_list, step_cost_list = self.get_neighbors(current, agent, constraints)

            for neighbor, step_cost in zip(neighbor_list, step_cost_list):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + (step_cost if step_cost and self.inflate_g_value else 1)

                if neighbor not in open_set.set:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent)
                    open_set.add(neighbor, (f_score[neighbor], -neighbor.time, neighbor.location.x, neighbor.location.y))
                    anti_direction_count[neighbor] = (anti_direction_count[current] + 1) if step_cost and neighbor.time<=self.window_size else anti_direction_count[current]
                    
                elif tentative_g_score < g_score.setdefault(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent)
                    open_set.update(neighbor, (f_score[neighbor], -neighbor.time, neighbor.location.x, neighbor.location.y))
                    anti_direction_count[neighbor] = (anti_direction_count[current] + 1) if step_cost and neighbor.time<=self.window_size else anti_direction_count[current]

        return None, None, None

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
        abstract_distance_map = map.get_distance_map()
        map.fit_corridors(corridors)
        heuristic_distance_map = map.get_distance_map(args.highway_heuristic_setup)
        map.reset_direction_limitation()
    else:
        abstract_distance_map = map.get_distance_map()
        heuristic_distance_map = abstract_distance_map

    cbs = CBS(map, WINDOW_SIZE, BUFFER_SIZE, args.use_manhat, heuristic_distance_map, abstract_distance_map)

    # map.fit_corridors(corridors)

    time_total = 0.0
    finished_tasks = 0

    map.show()

    for iteration in range(ITERATIONS):

        # 2. Search the paths of agents.
        time_start = time.time()
        solution = cbs.search(agents)
        time_end = time.time()
        print("computing time: ", time_end - time_start)
        time_total += time_end - time_start

        # 3. Save the paths of agents.
        for agent_name, history in solution.items():
            state_plus_time_offset(history, iteration*BUFFER_SIZE, sub_state_at_time_zero=True)
            agent_history[agent_name] += history

        # 4. Assign tasks to the agents that have finished their tasks.
        finished_agents = []
        for agent_name, agent in agents.items():
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
    with open(args.output, 'w+') as output_yaml:
        yaml.safe_dump(output, output_yaml)

if __name__ == "__main__":
    main()