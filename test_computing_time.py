from pbs import *
from cbs import CBS
from map import *
from env import *
from time import time
import multiprocessing
import csv


class TestParameter:
    def __init__(self, solver_type, x_len, y_len, x_num, y_num, line_num, pad_num, agent_num, window_size, buffer_size, worker_num, plan_full_path, only_one_line, only_allow_main_direction):
        self.solver_type = solver_type
        self.x_len = x_len
        self.y_len = y_len
        self.x_num = x_num
        self.y_num = y_num
        self.line_num = line_num
        self.pad_num = pad_num
        self.agent_num = agent_num # num or density
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.worker_num = worker_num
        self.plan_full_path = plan_full_path
        self.only_one_line = only_one_line
        self.only_allow_main_direction = only_allow_main_direction
    
    def save_result(self, directory, result):
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = directory + "/" + f"{self.x_len}_{self.y_len}_{self.x_num}_{self.y_num}_{self.agent_num}_{self.window_size}_{self.buffer_size}" + ("_full" if self.plan_full_path else "") + ("_only_one_highway" if self.only_one_line else "") + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory+"result.txt", 'a+') as f:
            f.write(result + '\n')
    
    def load_result(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            return None
        directory = directory + "/" + f"{self.x_len}_{self.y_len}_{self.x_num}_{self.y_num}_{self.agent_num}_{self.window_size}_{self.buffer_size}" + ("_full" if self.plan_full_path else "")  + ("_only_one_highway" if self.only_one_line else "") + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            return None
        with open(directory+"result.txt", 'r') as f:
            return f.read()

def test_without_control(
    env: Environment, episode_reset_seed=0, n_iterations=50, no_highway=False, output=False
):
    test_reward = 0
    test_reach_nodes = 0
    test_expand_nodes = 0

    env.reset(episode_reset_seed)

    time_start = time()
    for iteration in range(n_iterations):
        finished_tasks, forward_distance, instruction, no_solution_in_time, reach_nodes, expand_nodes  = env.step(
            iteration * env.mapf_solver.buffer_size + 1,
            reset_corridor=None if no_highway else "highway",
        )
        if no_solution_in_time:
            return False
        # finished_tasks, forward_distance, instruction = env.simple_step(iteration*env.mapf_solver.buffer_size+1, reset_corridor="instruction")
        test_reward += finished_tasks
        test_reach_nodes += reach_nodes
        test_expand_nodes += expand_nodes
    
    test_computing_time = time() - time_start

    print("Episode: {} \t Reward: {} \t Computing Time: {}".format(episode_reset_seed, test_reward, test_computing_time))

    if output:
        # Output History
        env.output_yaml_history(
            "history", "episode" + str(episode_reset_seed) + "_output.yaml", not no_highway
        )

    return test_reward, test_computing_time, test_reach_nodes, test_expand_nodes

def init_worker(args):
    global env
    env = args

def job(args):
    global env
    (episode_reset_seed, control_type) = args
    if control_type == "highway":
        results = test_without_control(env, episode_reset_seed, no_highway=False, output=True)
    else:
        results = test_without_control(env, episode_reset_seed, no_highway=True, output=False)

    return results

class Worker:
    def __init__(self, num_workers, env):
        self.pool = self.make_workers(num_workers, env)
        self.num_workers = num_workers

    def make_workers(self, num_workers, env):
        pool = multiprocessing.Pool(num_workers, initializer=init_worker, initargs=[env])
        print("Make Workers", num_workers)
        return pool

    def work(self, i_episodes, control_type):
        episode_reset_seeds = [i for i in range(i_episodes)]
        control_types = [control_type for _ in range(i_episodes)]
        work_results = self.pool.map(job, zip(episode_reset_seeds, control_types))
        rewards, computing_times, fail_cases, reach_nodes, expand_nodes = self.make_results(work_results)
        success_cases = i_episodes - fail_cases
        avg_reward, avg_computing_time = 0 if not success_cases else sum(rewards)/success_cases, 0 if not success_cases else sum(computing_times)/success_cases
        print(rewards, computing_times)
        print(avg_reward, avg_computing_time)
        print("Fails:", fail_cases)
        return avg_reward, avg_computing_time, fail_cases, reach_nodes, expand_nodes
    
    def make_results(self, work_results):
        fail_cases = 0
        test_rewards = []
        test_computing_times = []
        test_reach_nodes = []
        test_expand_nodes = []
        for result in work_results:
            if result:
                test_reward, test_computing_time, test_reach_node, test_expand_node = result
                test_rewards.append(test_reward)
                test_computing_times.append(test_computing_time)
                test_reach_nodes.append(test_reach_node)
                test_expand_nodes.append(test_expand_node)
            else:
                fail_cases += 1
        return test_rewards, test_computing_times, fail_cases, test_reach_nodes, test_expand_nodes


def test(params: TestParameter, control_type, test_episodes, use_highway_heuristic, highway_heuristic_setup=None):
    grid_map, corridor_params = make_map(params.x_len, params.y_len, params.x_num, params.y_num, True, params.line_num, params.pad_num, params.only_one_line, params.only_allow_main_direction)
    corridors = []
    for corridor in corridor_params:
        corridors.append(Corridor(Location(corridor['start'][0], corridor['start'][1]),Location(corridor['end'][0], corridor['end'][1]), corridor['reverse']))
    # print(map)
    # grid_map.show()

    if type(params.agent_num) == int:
        agent_num = params.agent_num
    else:
        agent_num = int(params.agent_num * len(grid_map.get_spaces()))
    agent_params = make_agent_dict(agent_num, grid_map.get_spaces())
    agents = {}
    for agent in agent_params:
        agents[agent['name']] = Agent(agent['name'], Location(agent['start'][0], agent['start'][1]), Location(agent['goal'][0], agent['goal'][1]))


    if(agents == False):
        print("Too many agents.")
        return

    print("Dimension:", grid_map.dimension)
    print("Agent Num:", len(agents))
    print("Corridor Num:", len(corridors))
    print("Obstacle Ratio:", len(grid_map.get_obstacles())/(grid_map.dimension[0]*grid_map.dimension[1]))
    print("Agent Ratio:", len(agents)/len(grid_map.get_spaces()))
    
    if params.solver_type == "CBS":
        mapf_solver = CBS(grid_map, params.window_size, params.buffer_size, False, corridors if use_highway_heuristic else [], highway_heuristic_setup)
        mapf_solver.plan_full_paths = params.plan_full_path
    elif params.solver_type == "PBS":
        mapf_solver = PBS(grid_map, params.window_size, params.buffer_size, False, corridors if use_highway_heuristic else [], highway_heuristic_setup)
        mapf_solver.plan_full_paths = params.plan_full_path

    

    if control_type == "highway":
        mapf_solver.map.fit_corridors(corridors)

    X = "X"
    N = "N"
    env = Environment(
        f"test_diff_highway_w_{params.solver_type}_{params.x_len}_{params.y_len}_{params.x_num}_{params.y_num}_{params.line_num}_{params.pad_num}_a{len(agents)}_window{params.window_size if not params.window_size==10e10 else X}_{control_type}_w{highway_heuristic_setup if not highway_heuristic_setup==None else N}" + ("_only_one_line" if params.only_one_line else ""), [grid_map.dimension[0], grid_map.dimension[1]], mapf_solver, agents, corridors, 1
    )
    workers = Worker(params.worker_num, env)
    avg_reward, avg_computing_time, fail_cases, reach_nodes, expand_nodes = workers.work(test_episodes, control_type)
    print(reach_nodes, expand_nodes)

    # params.save_result("./test/", (str(avg_reward) + " " + str(avg_computing_time) + "\n")
    # params.save_result("./test/", ("Highway: " if highway else "NoLomit: ") + "Avg. Reward = " + str(avg_reward) + ", Avg. Computing Time = " + str(avg_computing_time))
    # print(("Highway: " if highway else "NoLomit: ") + "Avg. Reward = " + str(avg_reward) + ", Avg. Computing Time = " + str(avg_computing_time))

    return avg_reward, avg_computing_time, fail_cases, grid_map.dimension, len(agents), len(agents)/len(grid_map.get_spaces()), reach_nodes, expand_nodes

def test_diff_window_size(params: TestParameter, range_iter, test_episodes = 10, output_path = None):
    avg_finished_tasks = {}
    avg_computing_time = {}
    fail_case = {}
    reach_nodes_list = {}
    expand_nodes_list = {}
    
    test_types = ["highway(directed)", "highway(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        avg_finished_tasks[test_type] = []
        avg_computing_time[test_type] = []
        fail_case[test_type] = []
        reach_nodes_list[test_type] = []
        expand_nodes_list[test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for window_size in range_iter:
        params.window_size = window_size

        task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "highway", test_episodes, use_highway_heuristic=True)
        avg_finished_tasks["highway(directed)"].append(task_num)
        avg_computing_time["highway(directed)"].append(time_compute)
        fail_case["highway(directed)"].append(fail_cases)
        reach_nodes_list["highway(directed)"].append(reach_nodes)
        expand_nodes_list["highway(directed)"].append(expand_nodes)

        # task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "highway", test_episodes, use_highway_heuristic=False)
        # avg_finished_tasks["highway(obsolute)"].append(task_num)
        # avg_computing_time["highway(obsolute)"].append(time_compute)
        # fail_case["highway(obsolute)"].append(fail_cases)
        # reach_nodes_list["highway(obsolute)"].append(reach_nodes)
        # expand_nodes_list["highway(obsolute)"].append(expand_nodes)

        # # task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio = test(params, "policy", test_episodes, use_highway_heuristic=False)
        # # avg_finished_tasks["policy(obsolute)"].append(task_num)
        # # avg_computing_time["policy(obsolute)"].append(time_compute)
        # # fail_case["policy(obsolute)"].append(fail_cases)

        # task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "nolimit", test_episodes, use_highway_heuristic=True)
        # avg_finished_tasks["nolimit(directed)"].append(task_num)
        # avg_computing_time["nolimit(directed)"].append(time_compute)
        # fail_case["nolimit(directed)"].append(fail_cases)
        # reach_nodes_list["nolimit(directed)"].append(reach_nodes)
        # expand_nodes_list["nolimit(directed)"].append(expand_nodes)

        # task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "nolimit", test_episodes, use_highway_heuristic=False)
        # avg_finished_tasks["nolimit(obsolute)"].append(task_num)
        # avg_computing_time["nolimit(obsolute)"].append(time_compute)
        # fail_case["nolimit(obsolute)"].append(fail_cases)
        # reach_nodes_list["nolimit(obsolute)"].append(reach_nodes)
        # expand_nodes_list["nolimit(obsolute)"].append(expand_nodes)

        map_dims.append(map_dim)
        agent_nums.append(agent_num)
        agent_ratios.append(agent_ratio)

    print("Map Dimensions:", map_dims)
    print("Number of Agents:", agent_nums)
    print("Ratio of Agents to Spaces:", agent_ratios)
    for test_type in test_types:
        print(f"Avg Finished Tasks ({test_type}):", avg_finished_tasks[test_type])
        print(f"Avg Computing Time ({test_type}):", avg_computing_time[test_type])
        print(f"Fail Cases ({test_type}):", fail_case[test_type])

    if not output_path == None:
        with open(output_path, 'w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["test_type", "dimension", "window_size", "buffer_size", "agent_num", "agent_ratio", "avg. finished tasks", "avg. computing time", "fail cases"])
            writer.writerow([])
            for test_type in test_types[:1]:
                for i, window_size in enumerate(range_iter):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dims[i])
                    row_data.append(window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_ratios[i])
                    row_data.append(avg_finished_tasks[test_type][i])
                    row_data.append(avg_computing_time[test_type][i])
                    row_data.append(fail_case[test_type][i])
                    row_data.append(reach_nodes_list[test_type][i])
                    row_data.append(expand_nodes_list[test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

def test_diff_highway_w(params: TestParameter, range_iter, test_episodes = 10, output_path = None):
    avg_finished_tasks = {}
    avg_computing_time = {}
    fail_case = {}
    reach_nodes_list = {}
    expand_nodes_list = {}
    
    test_types = ["highway(Strict Limit)", "highway(Strict Limit, Partial Plan)", "nolimit(Soft Limit)"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        avg_finished_tasks[test_type] = []
        avg_computing_time[test_type] = []
        fail_case[test_type] = []
        reach_nodes_list[test_type] = []
        expand_nodes_list[test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for highway_w in range_iter:

        task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "nolimit", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=highway_w)
        avg_finished_tasks["nolimit(Soft Limit)"].append(task_num)
        avg_computing_time["nolimit(Soft Limit)"].append(time_compute)
        fail_case["nolimit(Soft Limit)"].append(fail_cases)
        reach_nodes_list["nolimit(Soft Limit)"].append(reach_nodes)
        expand_nodes_list["nolimit(Soft Limit)"].append(expand_nodes)

        task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "highway", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=highway_w)
        avg_finished_tasks["highway(Strict Limit)"].append(task_num)
        avg_computing_time["highway(Strict Limit)"].append(time_compute)
        fail_case["highway(Strict Limit)"].append(fail_cases)
        reach_nodes_list["highway(Strict Limit)"].append(reach_nodes)
        expand_nodes_list["highway(Strict Limit)"].append(expand_nodes)

        params.plan_full_path = False
        
        task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "highway", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=highway_w)
        avg_finished_tasks["highway(Strict Limit, Partial Plan)"].append(task_num)
        avg_computing_time["highway(Strict Limit, Partial Plan)"].append(time_compute)
        fail_case["highway(Strict Limit, Partial Plan)"].append(fail_cases)
        reach_nodes_list["highway(Strict Limit, Partial Plan)"].append(reach_nodes)
        expand_nodes_list["highway(Strict Limit, Partial Plan)"].append(expand_nodes)

        params.plan_full_path = True

        map_dims.append(map_dim)
        agent_nums.append(agent_num)
        agent_ratios.append(agent_ratio)

    print("Map Dimensions:", map_dims)
    print("Number of Agents:", agent_nums)
    print("Ratio of Agents to Spaces:", agent_ratios)
    for test_type in test_types:
        print(f"Avg Finished Tasks ({test_type}):", avg_finished_tasks[test_type])
        print(f"Avg Computing Time ({test_type}):", avg_computing_time[test_type])
        print(f"Fail Cases ({test_type}):", fail_case[test_type])

    if not output_path == None:
        with open(output_path, 'w+') as csvfile:
            print(output_path)
            writer = csv.writer(csvfile)
            writer.writerow(["test_type", "dimension", "highway_w", "window_size", "buffer_size", "agent_num", "agent_ratio", "avg. finished tasks", "avg. computing time", "fail cases", "reach nodes", "expand nodes"])
            writer.writerow([])
            for test_type in test_types:
                for i, highway_w in enumerate(range_iter):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dims[i])
                    row_data.append(highway_w)
                    row_data.append(params.window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_ratios[i])
                    row_data.append(avg_finished_tasks[test_type][i])
                    row_data.append(avg_computing_time[test_type][i])
                    row_data.append(fail_case[test_type][i])
                    row_data.append(reach_nodes_list[test_type][i])
                    row_data.append(expand_nodes_list[test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

def test_highway(params: TestParameter, test_episodes = 10, output_path = None):
    # params.window_size = params.buffer_size

    avg_finished_tasks = {}
    avg_computing_time = {}
    fail_case = {}
    reach_nodes_list = {}
    expand_nodes_list = {}
    
    test_types = ["highway(Strict Limit)"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        avg_finished_tasks[test_type] = []
        avg_computing_time[test_type] = []
        fail_case[test_type] = []
        reach_nodes_list[test_type] = []
        expand_nodes_list[test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "highway", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=None)
    avg_finished_tasks["highway(Strict Limit)"].append(task_num)
    avg_computing_time["highway(Strict Limit)"].append(time_compute)
    fail_case["highway(Strict Limit)"].append(fail_cases)
    reach_nodes_list["highway(Strict Limit)"].append(reach_nodes)
    expand_nodes_list["highway(Strict Limit)"].append(expand_nodes)

    map_dims.append(map_dim)
    agent_nums.append(agent_num)
    agent_ratios.append(agent_ratio)

    print("Map Dimensions:", map_dims)
    print("Number of Agents:", agent_nums)
    print("Ratio of Agents to Spaces:", agent_ratios)
    for test_type in test_types:
        print(f"Avg Finished Tasks ({test_type}):", avg_finished_tasks[test_type])
        print(f"Avg Computing Time ({test_type}):", avg_computing_time[test_type])
        print(f"Fail Cases ({test_type}):", fail_case[test_type])

    if not output_path == None:
        with open(output_path, 'w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["test_type", "dimension", "highway_w", "window_size", "buffer_size", "agent_num", "agent_ratio", "avg. finished tasks", "avg. computing time", "fail cases", "reach nodes", "expand nodes"])
            writer.writerow([])
            for test_type in test_types:
                for i in range(1):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dims[i])
                    row_data.append("None")
                    row_data.append(params.window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_ratios[i])
                    row_data.append(avg_finished_tasks[test_type][i])
                    row_data.append(avg_computing_time[test_type][i])
                    row_data.append(fail_case[test_type][i])
                    row_data.append(reach_nodes_list[test_type][i])
                    row_data.append(expand_nodes_list[test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

# def compare_fully_highway_and_fully_pbs(params: TestParameter, test_episodes = 10, output_path = None):

#     avg_finished_tasks = {}
#     avg_computing_time = {}
#     fail_case = {}
#     reach_nodes_list = {}
#     expand_nodes_list = {}
    
#     test_types = ["fully highway", "fully pbs"]
#     # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

#     for test_type in test_types:
#         avg_finished_tasks[test_type] = []
#         avg_computing_time[test_type] = []
#         fail_case[test_type] = []
#         reach_nodes_list[test_type] = []
#         expand_nodes_list[test_type] = []

#     map_dims = []
#     agent_nums = []
#     agent_ratios = []

#     params.window_size = 10e10
    
#     task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "nolimit", test_episodes, use_highway_heuristic=False, highway_heuristic_setup=None)
#     avg_finished_tasks["fully pbs"].append(task_num)
#     avg_computing_time["fully pbs"].append(time_compute)
#     fail_case["fully pbs"].append(fail_cases)
#     reach_nodes_list["fully pbs"].append(reach_nodes)
#     expand_nodes_list["fully pbs"].append(expand_nodes)

#     task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "nolimit", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=1)
#     avg_finished_tasks["fully highway"].append(task_num)
#     avg_computing_time["fully highway"].append(time_compute)
#     fail_case["fully highway"].append(fail_cases)
#     reach_nodes_list["fully highway"].append(reach_nodes)
#     expand_nodes_list["fully highway"].append(expand_nodes)

#     map_dims.append(map_dim)
#     agent_nums.append(agent_num)
#     agent_ratios.append(agent_ratio)

#     print("Map Dimensions:", map_dims)
#     print("Number of Agents:", agent_nums)
#     print("Ratio of Agents to Spaces:", agent_ratios)
#     for test_type in test_types:
#         print(f"Avg Finished Tasks ({test_type}):", avg_finished_tasks[test_type])
#         print(f"Avg Computing Time ({test_type}):", avg_computing_time[test_type])
#         print(f"Fail Cases ({test_type}):", fail_case[test_type])

#     if not output_path == None:
#         with open(output_path, 'w+') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(["test_type", "dimension", "highway_w", "window_size", "buffer_size", "agent_num", "agent_ratio", "avg. finished tasks", "avg. computing time", "fail cases", "reach nodes", "expand nodes"])
#             writer.writerow([])
#             for test_type in test_types:
#                 for i in range(1):
#                     row_data = []
#                     row_data.append(test_type)
#                     row_data.append(map_dims[i])
#                     row_data.append("None")
#                     row_data.append(params.window_size)
#                     row_data.append(params.buffer_size)
#                     row_data.append(agent_nums[i])
#                     row_data.append(agent_ratios[i])
#                     row_data.append(avg_finished_tasks[test_type][i])
#                     row_data.append(avg_computing_time[test_type][i])
#                     row_data.append(fail_case[test_type][i])
#                     row_data.append(reach_nodes_list[test_type][i])
#                     row_data.append(expand_nodes_list[test_type][i])
#                     writer.writerow(row_data)
#                 writer.writerow([])

def main():
    # OUTPUT = "/media/NFS/fong/KIva-System-RL/test/test_diff_agent_num_1133_5_10.csv"
    # OUTPUT = "/media/NFS/fong/KIva-System-RL/test/test_diff_window_size_2233_a20_5_50_strict_highway_solve_collision_with_windowsize.csv"
    # OUTPUT = "/media/NFS/fong/KIva-System-RL/test/test1.csv"
    # OUTPUT = "/media/NFS/fong/KIva-System-RL/test/test_diff_map_size_2233_0.05_2.csv"

    WINDOW_SIZE = 5
    BUFFER_SIZE = 5
    WORKER_NUM = 10
    plan_full_path = False

    x_len = 2
    y_len = 2
    x_num = 3
    y_num = 3
    agent_num = 10
    solver = "PBS"

    line_num = 1
    pad_num = 0
    only_one_line = False
    only_allow_main_direction = False

    X = "X"

    # density = 0.15
    # for only_one_line in [False]:
    #     for ob_num in [3, 5, 7]:
    #         x_num, y_num = ob_num, ob_num
    #         params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, WORKER_NUM, True, only_one_line)

    #         grid_map, corridor_params = make_map(params.x_len, params.y_len, params.x_num, params.y_num, True, params.line_num, params.pad_num)

    #         agent_num = len(grid_map.get_spaces()) * density
    #         agent_num = round(agent_num/5) *5 if agent_num < 100 else int(agent_num - agent_num%10)

    #         print(agent_num)
    #         params.agent_num = agent_num

    #         OUTPUT = f"/media/NFS/fong/KIva-System-RL/test/test_diff_highway_w_{x_len}{y_len}{x_num}{y_num}_{line_num}_{pad_num}_a{agent_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    #         test_diff_highway_w(params, [1, 1.2, 1.5, 2, 5, 10, 20, 50, 0], output_path = OUTPUT)
    # params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, WORKER_NUM, True)
    for agent_num in [10]:
        # OUTPUT = f"/media/NFS/fong/KIva-System-RL/test/test_highway_w_{x_len}{y_len}{x_num}{y_num}_{line_num}_{pad_num}_a{agent_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ("_only_allow_main_direction" if only_allow_main_direction else "") + ".csv"
        OUTPUT = f"./test/test.csv"
        params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, WORKER_NUM, plan_full_path, only_one_line, only_allow_main_direction)
        test_highway(params, output_path = OUTPUT)
    # test_diff_highway_w(params, [1, 1.2, 1.5, 2, 5, 10, 20, 50, 0], output_path = OUTPUT)

    # test_diff_agent_num(params, [5, 8, 10], output_path = OUTPUT)
    # test_diff_map_size(params, 2, output_path = OUTPUT)
    # test_diff_window_size(params, [5, 6, 7, 8, 9], output_path = OUTPUT)

    # aa(params, WINDOW_SIZE, BUFFER_SIZE, False)

if __name__ == "__main__":
    main()