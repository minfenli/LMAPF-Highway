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
        return hash(str((self.time, self.location.x, self.location.y)))
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
        self.priorities = Priorities()
        self.costs = {}
        self.cost = 0
        self.anti_direction_counts = {}

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.priorities.priorities == other.priorities.priorities

    def __hash__(self):
        s = ""
        for i in self.priorities.priority_list:
            s += i
        return hash(s)
    
    def __lt__(self, other):
        return len(self.priorities.priority_list) > len(other.priorities.priority_list) if not len(self.priorities.priority_list) == len(other.priorities.priority_list) else (self.priorities.priority_list < other.priorities.priority_list if self.cost == other.cost else self.cost < other.cost)

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

class Priorities:
    def __init__(self):
        self.priorities = dict()
        self.priorities_reverse = dict()
        self.priority_list = []

    def add_priority(self, agent_1, agent_2):
        if agent_1 not in self.priorities: self.priorities[agent_1] = []
        if agent_2 not in self.priorities[agent_1]: self.priorities[agent_1].append(agent_2)

        if agent_2 not in self.priorities_reverse: self.priorities_reverse[agent_2] = []
        if agent_1 not in self.priorities_reverse[agent_2]: self.priorities_reverse[agent_2].append(agent_1)

        self.priority_list = self._topological(self.priorities)
        return False if self.priority_list == None else True

    def _topological(self, graph):
        order, enter, state = [], list(graph), {}
        def dfs(node):
            state[node] = 0
            if node in graph:
                for k in graph[node]:
                    sk = state.get(k, None)
                    if sk == 0: 
                        return False # loop check
                    if sk == 1: continue
                    if k in enter: enter.remove(k)
                    if not dfs(k): return False
            order.insert(0, node)
            state[node] = 1
            return True
        while enter: 
            if not dfs(enter.pop(-1)):
                return None
        return order

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class PBS:
    def __init__(self, map, window_size = 20, buffer_size = 10, use_manhat = True, heuristic_distance_map=None, abstract_distance_map=None, inflate_g_value=False):
        self.map = map
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.plan_full_paths = True
        self.search = self.search_standard_stack
        self.use_manhat = use_manhat
        self.heuristic_distance_map = heuristic_distance_map # Valid if "use_manhat is False"
        self.abstract_distance_map = abstract_distance_map   # Valid if "use_manhat is False"
        self.inflate_g_value = inflate_g_value  # Inflate step cost when moving across grids during low-level planning based on grid properties. Otherwise, step cost = 1.
        
    def search_standard_stack(self, agents, time_limit=60, return_info=False):
        reach_nodes = 0
        expand_nodes = 0

        time_start = time.time()

        open_set = []
        closed_set = set()
        start = HighLevelNode()
        start.solution, start.costs, start.anti_direction_counts = self.compute_solution(agents, start.priorities, {}, {}, {})
        
        if not start.solution:
            if return_info:
                return {}, 0, 0, {} 
            else: 
                return {}
        start.cost = self.compute_solution_cost(start.costs)

        open_set += [start]
        expand_nodes += 1
        while open_set:
            if (time.time()-time_start) >= time_limit:
                print("Time out.")
                break
            reach_nodes += 1

            # even if use stack, still pick the smallest-cost node from the nodes with same level(depth-first-search)
            P = open_set.pop()
            
            closed_set |= {P}

            conflict = self.get_first_conflict(P.solution)

            if not conflict:
                if return_info:
                    return self.clip_solution(P.solution), reach_nodes, expand_nodes, P.anti_direction_counts
                else: 
                    return self.clip_solution(P.solution)

            new_node = deepcopy(P)
            if new_node.priorities.add_priority(conflict.agent_1, conflict.agent_2):
                if new_node not in closed_set:
                    new_node.solution, new_node.costs, new_node.anti_direction_counts = self.compute_solution(agents, new_node.priorities, new_node.solution, new_node.costs, new_node.anti_direction_counts)
                    if new_node.solution:
                        new_node.cost = self.compute_solution_cost(new_node.costs)
                        open_set += [new_node]
                        expand_nodes += 1

            new_node = deepcopy(P)

            if new_node.priorities.add_priority(conflict.agent_2, conflict.agent_1):
                if new_node not in closed_set:
                    new_node.solution, new_node.costs, new_node.anti_direction_counts = self.compute_solution(agents, new_node.priorities, new_node.solution, new_node.costs, new_node.anti_direction_counts)
                    if new_node.solution:
                        new_node.cost = self.compute_solution_cost(new_node.costs)
                        open_set += [new_node]
                        expand_nodes += 1

        print("No solution found.")
        if return_info:
            return {}, 0, 0, {}
        else: 
            return {}

    def search_modified_stack(self, agents, time_limit=60, return_info=False):
        reach_nodes = 0
        expand_nodes = 0

        time_start = time.time()

        open_set = []
        closed_set = set()
        start = HighLevelNode()
        start.solution, start.costs, start.anti_direction_counts = self.compute_solution(agents, start.priorities, {}, {}, {})
        
        if not start.solution:
            if return_info:
                return {}, 0, 0, {} 
            else: 
                return {}
        start.cost = self.compute_solution_cost(start.costs)

        open_set += [start]
        expand_nodes += 1
        while open_set:
            if (time.time()-time_start) >= time_limit:
                print("Time out.")
                break
            reach_nodes += 1

            # even if use stack, still pick the smallest-cost node from the nodes with same level(depth-first-search)
            if len(open_set) >= 2:
                idx = -2
                smallest = -1
                while(len(open_set[smallest].priorities.priority_list) == len(open_set[idx].priorities.priority_list)):       
                    if open_set[smallest].cost == open_set[idx].cost:
                        if open_set[smallest].priorities.priority_list > open_set[idx].priorities.priority_list:
                            smallest = idx
                    elif open_set[smallest].cost > open_set[idx].cost:
                        smallest = idx
                    idx -= 1
                    if -idx > len(open_set):
                        break
                P = open_set.pop(smallest)
            else:
                P = open_set.pop()
            
            closed_set |= {P}

            conflict = self.get_first_conflict(P.solution)

            if not conflict:
                if return_info:
                    return self.clip_solution(P.solution), reach_nodes, expand_nodes, P.anti_direction_counts
                else: 
                    return self.clip_solution(P.solution)

            new_node = deepcopy(P)
            if new_node.priorities.add_priority(conflict.agent_1, conflict.agent_2):
                if new_node not in closed_set:
                    new_node.solution, new_node.costs, new_node.anti_direction_counts = self.compute_solution(agents, new_node.priorities, new_node.solution, new_node.costs, new_node.anti_direction_counts)
                    if new_node.solution:
                        new_node.cost = self.compute_solution_cost(new_node.costs)
                        open_set += [new_node]
                        expand_nodes += 1

            new_node = deepcopy(P)

            if new_node.priorities.add_priority(conflict.agent_2, conflict.agent_1):
                if new_node not in closed_set:
                    new_node.solution, new_node.costs, new_node.anti_direction_counts = self.compute_solution(agents, new_node.priorities, new_node.solution, new_node.costs, new_node.anti_direction_counts)
                    if new_node.solution:
                        new_node.cost = self.compute_solution_cost(new_node.costs)
                        open_set += [new_node]
                        expand_nodes += 1

        print("No solution found.")
        if return_info:
            return {}, 0, 0, {} 
        else: 
            return {}
    
    def search_set(self, agents, time_limit=60, return_info=False):
        reach_nodes = 0
        expand_nodes = 0

        time_start = time.time()

        open_set = set()
        closed_set = set()
        start = HighLevelNode()
        start.solution, start.costs, start.anti_direction_counts = self.compute_solution(agents, start.priorities, {}, {}, {})
        
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

            conflict = self.get_first_conflict(P.solution)

            if not conflict:
                if return_info:
                    return self.clip_solution(P.solution), reach_nodes, expand_nodes, P.anti_direction_counts
                else: 
                    return self.clip_solution(P.solution)

            new_node = deepcopy(P)
            if new_node.priorities.add_priority(conflict.agent_1, conflict.agent_2):
                if new_node not in closed_set:
                    new_node.solution, new_node.costs, new_node.anti_direction_counts = self.compute_solution(agents, new_node.priorities, new_node.solution, new_node.costs, new_node.anti_direction_counts)
                    if new_node.solution:
                        
                        open_set |= {new_node}
                        expand_nodes += 1

            new_node = deepcopy(P)

            if new_node.priorities.add_priority(conflict.agent_2, conflict.agent_1):
                if new_node not in closed_set:
                    new_node.solution, new_node.costs, new_node.anti_direction_counts = self.compute_solution(agents, new_node.priorities, new_node.solution, new_node.costs, new_node.anti_direction_counts)
                    if new_node.solution:
                        new_node.cost = self.compute_solution_cost(new_node.costs)
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
        if self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_wait)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_up and self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_up)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_down and self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_down)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_left and self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_left)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.map.grid[state.location.y][state.location.x].grid_property.is_right and self.state_valid(n, constraints.vertex_constraints, self.is_at_goal(state, agent)) and self.transition_valid(state, n, constraints.edge_constraints):
            neighbors.append(n)
            step_costs.append(self.map.grid[state.location.y][state.location.x].grid_property.step_cost_right)
        return neighbors, step_costs
    
    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        check_region = min(max_t, self.window_size+1)
        agent_names = sorted(list(solution.keys()))
        for t in range(check_region):
            for agent_1, agent_2 in combinations(agent_names, 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    return Conflict(t, Conflict.VERTEX, agent_1, agent_2, state_1.location, Location())

            for agent_1, agent_2 in combinations(agent_names, 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    return Conflict(t, Conflict.EDGE, agent_1, agent_2, state_1a.location, state_1b.location)
        return False

    def create_priorities_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        return constraint_dict
    
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
    
    def compute_solution(self, agents, priorities, solution = {}, cost = {}, anti_direction_count = {}):
        if not solution:
            agents_without_priorities = list(agents.keys())
            for agent_name in agents_without_priorities:
                local_solution, local_cost, local_anti_direction_count = self.astar(agents[agent_name], Constraints())
                if not local_solution:
                    return None, None, None
                solution[agent_name] = local_solution
                cost[agent_name] = local_cost
                anti_direction_count[agent_name] = local_anti_direction_count

        constraints_dict = {}
        for agent_name in priorities.priority_list:
            constraints_dict[agent_name] = Constraints()
        for agent_name in priorities.priority_list:
            constraints = Constraints()
            for i in priorities.priorities_reverse.get(agent_name, []):
                constraints.add_constraint(constraints_dict[i])
            local_solution, local_cost, local_anti_direction_count = self.astar(agents[agent_name], constraints)
            if not local_solution:
                return None, None, None
            solution[agent_name] = local_solution
            cost[agent_name] = local_cost
            anti_direction_count[agent_name] = local_anti_direction_count
            self.add_constraint_from_solution(agent_name, local_solution, constraints_dict[agent_name])
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

                # step cost != None if moving against highway direction
                # So, step cost == 1 if the step cost is None. (not an anti-highway-movement)
                tentative_g_score = g_score[current] + (step_cost if step_cost and self.inflate_g_value else 1)

                if neighbor not in open_set.set:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent)
                    open_set.add(neighbor, (f_score[neighbor], -neighbor.time, neighbor.location.x, neighbor.location.y))
                    anti_direction_count[neighbor] = (anti_direction_count[current] + 1) if step_cost and neighbor.time<=self.buffer_size else anti_direction_count[current]
                elif tentative_g_score < g_score.setdefault(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent)
                    open_set.update(neighbor, (f_score[neighbor], -neighbor.time, neighbor.location.x, neighbor.location.y))
                    anti_direction_count[neighbor] = (anti_direction_count[current] + 1) if step_cost and neighbor.time<=self.buffer_size else anti_direction_count[current]

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

    pbs = PBS(map, WINDOW_SIZE, BUFFER_SIZE, args.use_manhat, heuristic_distance_map, abstract_distance_map)

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
    # output["cost"] = cost
    with open(args.output, 'w+') as output_yaml:
        yaml.safe_dump(output, output_yaml)

if __name__ == "__main__":
    main()