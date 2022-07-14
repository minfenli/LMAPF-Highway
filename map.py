import argparse
import yaml
import random 
import heapq

class Location:
    def __init__(self, x: int = -1, y: int = -1):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __str__(self):
        return str((self.x, self.y))

class GridProperty:
    def __init__(self, is_obstacle: bool = False, is_up: bool = True,\
         is_down: bool = True, is_left: bool = True, is_right: bool = True):
        self.is_obstacle = is_obstacle
        self.is_up = is_up
        self.is_down = is_down
        self.is_left = is_left
        self.is_right = is_right
    def __eq__(self, other):
        return self.is_obstacle == other.is_obstacle and self.is_up == other.is_up and self.is_down == other.is_down and self.is_left == other.is_left and self.is_right == other.is_right
    def __str__(self):
        return str((self.is_obstacle, self.is_up, self.is_down, self.is_left, self.is_right))
    def reset_all_direction(self):
        """
        Reset all the direction attributes as true except "is_obstacle"
        """
        self.is_up = True
        self.is_down = True
        self.is_left = True
        self.is_right = True

class Grid:
    def __init__(self, location: Location, grid_property: GridProperty):
        self.location = location
        self.grid_property = grid_property
    def __eq__(self, other):
        return self.location == other.location
    def __str__(self):
        return str(self.location) + str(self.grid_property)

class Corridor:
    def __init__(self, start: Location, end: Location, reverse: bool):
        self.start = start
        self.end = end
        self.reverse = reverse

class Map:
    def __init__(self, x: int = 0, y: int = 0, only_allow_main_direction: bool = False):
        self.grid = []
        self.dimension = [x, y]
        for y_idx in range(y):
            row = []
            for x_idx in range(x):
                location = Location(x_idx, y_idx)
                grid_property = GridProperty()
                row.append(Grid(location, grid_property))
            self.grid.append(row)
        self.only_allow_main_direction = only_allow_main_direction
    def __str__(self):
        string = ""
        for row in self.grid:
            for grid in row:
                string += str(grid) + "\n"
        return string
    def get_obstacles(self):
        """
        Return a list of tuples (x, y) that represents the locations of the obastacles.
        """
        obstacles= []
        for y_idx, row in enumerate(self.grid):
            for x_idx, grid in enumerate(row):
                if grid.grid_property.is_obstacle:
                    obstacles.append((x_idx, y_idx))
        return obstacles
    def get_spaces(self):
        """
        Return a list of tuples (x, y) that represents the locations of spaces (not obstacle).
        """
        spaces= []
        for y_idx, row in enumerate(self.grid):
            for x_idx, grid in enumerate(row):
                if not grid.grid_property.is_obstacle:
                    spaces.append((x_idx, y_idx))
        return spaces

    def fit_obstacles(self, obstacles: [tuple]):
        """
        Set locations in "obstacles" as obstacles.
        """
        for obstacle in obstacles:
            self.grid[obstacle[1]][obstacle[0]].grid_property.is_obstacle = True

    # If only_allow_main_direction is True, turning left or right are not available. Otherwise, block only the reversed direction of highway.
    def fit_corridors(self, corridors: [Corridor]):
        """
        Set the direction limitation based on "corridors".
        """
        self.reset()
        for corridor in corridors:
            # vertical 
            if(self.find_corridor_direction(corridor)):
                x = corridor.start.x
                if(corridor.reverse):
                    for y in range(corridor.start.y-1, corridor.end.y+1):
                        self.grid[y][x].grid_property.is_up = False
                        if self.only_allow_main_direction and not y == corridor.start.y-1:
                            self.grid[y][x].grid_property.is_left = False
                            self.grid[y][x].grid_property.is_right = False
                else:
                    for y in range(corridor.start.y, corridor.end.y+2):
                        self.grid[y][x].grid_property.is_down = False
                        if self.only_allow_main_direction and not y == corridor.end.y+1:
                            self.grid[y][x].grid_property.is_left = False
                            self.grid[y][x].grid_property.is_right = False
            # horizontal
            else:
                y = corridor.start.y
                if(corridor.reverse):
                    for x in range(corridor.start.x-1, corridor.end.x+1):
                        self.grid[y][x].grid_property.is_right = False
                        if self.only_allow_main_direction and not x == corridor.start.x-1:
                            self.grid[y][x].grid_property.is_up = False
                            self.grid[y][x].grid_property.is_down = False
                else:
                    for x in range(corridor.start.x, corridor.end.x+2):
                        self.grid[y][x].grid_property.is_left = False
                        if self.only_allow_main_direction and not x == corridor.end.x+1:
                            self.grid[y][x].grid_property.is_up = False
                            self.grid[y][x].grid_property.is_down = False
    
    def find_corridor_direction(self, corridor: Corridor):
        """
        Find the direction of the corridor, assuming that no obstacles in front of or in back of the corridor.
        Return True if it's vertical, False if it's horizontal.
        """
        for x in range(self.dimension[0]):
            if self.grid[corridor.start.y][x].grid_property.is_obstacle:
                return True
        return False
    
    def reset(self):
        """
        Reset all the direction limitation, every direction is available after the reset.
        """
        for row in self.grid:
            for grid in row:
                grid.grid_property.reset_all_direction()

    def show(self):
        tokens = []
        for row in self.grid:
            row_tokens = []
            for grid in row:
                if(grid.grid_property.is_obstacle):
                    row_tokens.append("X")
                else:
                    if(not grid.grid_property.is_up):
                        row_tokens.append("↓")
                    elif(not grid.grid_property.is_down):
                        row_tokens.append("↑")
                    elif(not grid.grid_property.is_left):
                        row_tokens.append("→")
                    elif(not grid.grid_property.is_right):
                        row_tokens.append("←")
                    else:
                        row_tokens.append("O")
            tokens.append(row_tokens)
        for row in tokens[::-1]:
            for token in row:
                print(token, end=' ')
            print()
    
    def get_distance_map(self, highway_heuristic_w=None):
        distance_map = []
        for y in range(self.dimension[1]):
            row_list = []
            for x in range(self.dimension[0]):
                row_list.append(get_distance_map_from_single_point(self,(x,y), highway_heuristic_w))
            distance_map.append(row_list)
        return distance_map

def make_map(obstacle_x: int, obstacle_y: int, num_x: int, num_y: int, warehouse_form=False, num_line: int=1, num_pad: int=0, only_one_line: bool=False, only_allow_main_direction: bool=False):
    x = num_line + (obstacle_x + num_line) * num_x + num_pad*2
    y = num_line + (obstacle_y + num_line) * num_y + num_pad*2
    map = Map(x, y, only_allow_main_direction)
    corridor = []
    for x_idx in range(num_x):
        for y_idx in range(num_y):
            x_start = num_line + (obstacle_x + num_line) * x_idx + num_pad
            y_start = num_line + (obstacle_y + num_line) * y_idx + num_pad
            for i in range(obstacle_x):
                for j in range(obstacle_y):
                    map.grid[y_start + j][x_start + i].grid_property.is_obstacle = True
    for x_idx in range(num_x+1):
        for y_idx in range(num_y+1):
            x_start = (obstacle_x + num_line) * x_idx + num_pad
            y_start = (obstacle_y + num_line) * y_idx + num_pad
            for l_idx in range(num_line):
                if(x_idx != num_x):
                    corridor.append({'start': [x_start+num_line, y_start+l_idx], 'end': [x_start+num_line-1+obstacle_x, y_start+l_idx], 'reverse': (False if (x_idx+y_idx)%2 == 0 else True) if not warehouse_form else (False if (y_idx)%2 == 0 else True)})
                if(y_idx != num_y):
                    corridor.append({'start': [x_start+l_idx, y_start+num_line], 'end': [x_start+l_idx, y_start+num_line-1+obstacle_y], 'reverse': (False if (x_idx+y_idx)%2 == 1 else True) if not warehouse_form else (False if (x_idx)%2 == 1 else True)})
                if only_one_line:
                    break
    # print(corridor)
    return map, corridor
   
# highway_in_heuristic: strict limitation if highway_in_heuristic is False, moving backward is not allowed in a highway.
#                       otherwise, use soft limitation by increasing cost on the heuristic function
def get_distance_map_from_single_point(map: Map, start, highway_heuristic_w):
    distance_map_from_start = []
    dimension = map.dimension
    for y in range(dimension[1]):
        row_list = []
        for x in range(dimension[0]):
            row_list.append(-1)
        distance_map_from_start.append(row_list)

    if not highway_heuristic_w:
        queue = [start]
        distance_map_from_start[start[1]][start[0]] = 0
        while(queue):
            current = queue.pop(0)
            x, y = current[0], current[1]
            if(x-1 >=0 and not map.grid[y][x-1].grid_property.is_obstacle and distance_map_from_start[y][x-1] == -1 and map.grid[y][x].grid_property.is_left):
                queue.append((x-1, y))
                distance_map_from_start[y][x-1] = distance_map_from_start[y][x] +1
            if(y-1 >=0 and not map.grid[y-1][x].grid_property.is_obstacle and distance_map_from_start[y-1][x] == -1 and map.grid[y][x].grid_property.is_down):
                queue.append((x, y-1))
                distance_map_from_start[y-1][x] = distance_map_from_start[y][x] +1
            if(x+1 < dimension[0] and not map.grid[y][x+1].grid_property.is_obstacle and distance_map_from_start[y][x+1] == -1 and map.grid[y][x].grid_property.is_right):
                queue.append((x+1, y))
                distance_map_from_start[y][x+1] = distance_map_from_start[y][x] +1
            if(y+1 < dimension[1] and not map.grid[y+1][x].grid_property.is_obstacle and distance_map_from_start[y+1][x] == -1 and map.grid[y][x].grid_property.is_up):
                queue.append((x, y+1))
                distance_map_from_start[y+1][x] = distance_map_from_start[y][x] +1
    else:
        queue = PrioritySet()
        queue.add(start, 0)
        distance_map_from_start[start[1]][start[0]] = 0

        while(queue.set):
            current = queue.pop()
            distance = distance_map_from_start[current[1]][current[0]]
            x, y = current[0], current[1]
            # print("current:", current, distance)
            if(x-1 >=0 and not map.grid[y][x-1].grid_property.is_obstacle):
                new_distance = distance_map_from_start[y][x] +1 if map.grid[y][x].grid_property.is_left else distance_map_from_start[y][x] +highway_heuristic_w
                # print("reach:",(x-1, y), new_distance)
                if distance_map_from_start[y][x-1]== -1:
                    queue.add((x-1, y), new_distance)
                    distance_map_from_start[y][x-1] = new_distance
                elif new_distance < distance_map_from_start[y][x-1]:
                    distance_map_from_start[y][x-1] = new_distance
                    if (x-1, y) in queue.set:
                        queue.update((x-1, y), new_distance)
                    else:
                        queue.add((x-1, y), new_distance)
            if(y-1 >=0 and not map.grid[y-1][x].grid_property.is_obstacle):
                new_distance = distance_map_from_start[y][x] +1 if map.grid[y][x].grid_property.is_down else distance_map_from_start[y][x] +highway_heuristic_w
                # print("reach:",(x, y-1), new_distance)
                if distance_map_from_start[y-1][x]== -1:
                    queue.add((x, y-1), new_distance)
                    distance_map_from_start[y-1][x] = new_distance
                elif new_distance < distance_map_from_start[y-1][x]:
                    distance_map_from_start[y-1][x] = new_distance
                    if (x, y-1) in queue.set:
                        queue.update((x, y-1), new_distance)
                    else:
                        queue.add((x, y-1), new_distance)
            if(x+1 < dimension[0] and not map.grid[y][x+1].grid_property.is_obstacle):
                new_distance = distance_map_from_start[y][x] +1 if map.grid[y][x].grid_property.is_right else distance_map_from_start[y][x] +highway_heuristic_w
                # print("reach:",(x+1, y), new_distance)
                if distance_map_from_start[y][x+1]== -1:
                    queue.add((x+1, y), new_distance)
                    distance_map_from_start[y][x+1] = new_distance
                elif new_distance < distance_map_from_start[y][x+1]:
                    distance_map_from_start[y][x+1] = new_distance
                    if (x+1, y) in queue.set:
                        queue.update((x+1, y), new_distance)
                    else:
                        queue.add((x+1, y), new_distance)
            if(y+1 < dimension[1] and not map.grid[y+1][x].grid_property.is_obstacle):
                new_distance = distance_map_from_start[y][x] +1 if map.grid[y][x].grid_property.is_up else distance_map_from_start[y][x] +highway_heuristic_w
                # print("reach:",(x, y+1), new_distance)
                if distance_map_from_start[y+1][x]== -1:
                    queue.add((x, y+1), new_distance)
                    distance_map_from_start[y+1][x] = new_distance
                elif new_distance < distance_map_from_start[y+1][x]:
                    distance_map_from_start[y+1][x] = new_distance
                    if (x, y+1) in queue.set:
                        queue.update((x, y+1), new_distance)
                    else:
                        queue.add((x, y+1), new_distance)
        # import pdb; pdb.set_trace()
    return distance_map_from_start

class Agent:
    def __init__(self, name: str, location: Location, goal: Location):
        self.name = name
        self.location = location
        self.goal = goal
    def __str__(self):
        return self.name + str(self.location) + str(self.goal)

class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, d, pri):
        if not d in self.set:
            heapq.heappush(self.heap, (pri, d))
            self.set.add(d)
    
    def update(self, d, pri):
        if d in self.set:
            for old_pri, old_d in self.heap:
                if d == old_d:
                    old_pri = pri
                    break
                    
            heapq.heapify(self.heap)

    def pop(self):
        pri, d = heapq.heappop(self.heap)
        self.set.remove(d)
        return d
    
def make_agent_dict(num, spaces, seed = 1):
    if(len(spaces) < num):
        return False
    random.seed(seed)
    random.shuffle(spaces)
    agents = []
    for idx in range(num):
        agents.append({'start': list(spaces[idx]), 'goal': list(spaces[idx]), 'name': 'agent'+str(idx)})
    return agents

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
    parser.add_argument("--x_len", type= int, default= 2, help="length of x of obstacles")
    parser.add_argument("--y_len", type= int, default= 2, help="length of y of obstacles")
    parser.add_argument("--x_num", type= int, default= 5, help="number of obstacles on x-axis")
    parser.add_argument("--y_num", type= int, default= 5, help="number of obstacles on y-axis")
    parser.add_argument("--agent_num", type= int, default= 25, help="number of agents")
    parser.add_argument("--warehouse_form", type= str2bool, default=True, help="form like warehouse, not care loops")
    parser.add_argument("--line_num", type= int, default= 1, help="number of corridors in a single column or row")
    parser.add_argument("--line_pad", type= int, default= 0, help="number of corridors padding around")
    parser.add_argument("--only_one_line", type=str2bool, default=False, help="setup only one highway in one shared corridor")
    parser.add_argument("output", type= str, help="output file containing map and obstacles")
    args = parser.parse_args()

    map, corridor = make_map(args.x_len, args.y_len, args.x_num, args.y_num, args.warehouse_form, args.line_num, args.line_pad, args.only_one_line)
    # print(map)
    map.show()

    agents = make_agent_dict(args.agent_num, map.get_spaces())

    if(agents == False):
        print("Too many agents.")
        return

    print("Dimension:", map.dimension)
    print("Space:", len(map.get_spaces()))
    print("Obstacle:", len(map.get_obstacles()))
    print("Obstacle Ratio:", len(map.get_obstacles())/(map.dimension[0]*map.dimension[1]))
    print("Agent Ratio:", len(agents)/len(map.get_spaces()))
    print("Corridors:", len(corridor))

    output = {}
    output["map"] = {}
    output["map"]["dimensions"] = map.dimension
    output["map"]["obstacles"] = map.get_obstacles()
    output["map"]["corridor"] = corridor
    if(agents):
        output["agents"] = agents
    # print(output)

    with open(args.output, 'w') as output_yaml:
        yaml.dump(output, output_yaml, default_flow_style=None)

if __name__ == "__main__":
    main()
