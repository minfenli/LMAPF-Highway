from pbs import *
from cbs import CBS
from map import *
from env import *
from time import time
import multiprocessing
import csv
import statistics


class TestParameter:
    def __init__(self, solver_type, x_len, y_len, x_num, y_num, line_num, pad_num, agent_num, window_size, buffer_size, iterations, worker_num, plan_full_path, only_one_line, only_allow_main_direction):
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
        self.iterations = iterations
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
    env: Environment, episode_reset_seed=0, n_iterations=50, output=False
):
    result_reach_nodes = []
    result_expand_nodes = []
    result_computing_times = []
    result_finished_idle_timesteps = []
    result_finished_moving_timesteps = []
    result_finished_detour_distances = []

    result_total_finished_tasks = 0

    env.reset(episode_reset_seed)


    total_time_start = time()
    for iteration in range(n_iterations):
        time_start = time()
        finished_tasks, forward_distance, no_solution_in_time, reach_nodes, expand_nodes, finished_idle_timesteps, finished_moving_timesteps, finished_detour_distances  = env.step(
            iteration * env.mapf_solver.buffer_size + 1
        )
        # print(finished_idle_timesteps, finished_moving_timesteps, finished_detour_distances)
        if no_solution_in_time:
            return False
        result_total_finished_tasks += finished_tasks
        result_reach_nodes.append(reach_nodes)
        result_expand_nodes.append(expand_nodes)
        result_finished_idle_timesteps += finished_idle_timesteps
        result_finished_moving_timesteps += finished_moving_timesteps
        result_finished_detour_distances += finished_detour_distances
    
        result_computing_times.append(time() - time_start)
    
    result_total_computing_time = time() - total_time_start

    print("Episode: {} \t Reward: {} \t Computing Time: {}".format(episode_reset_seed, result_total_finished_tasks, result_total_computing_time))

    if output:
        # Output History
        env.output_yaml_history(
            "history", "episode" + str(episode_reset_seed) + "_output.yaml"
        )

    return result_total_finished_tasks, result_total_computing_time, result_computing_times, result_reach_nodes, result_expand_nodes, result_finished_idle_timesteps, result_finished_moving_timesteps, result_finished_detour_distances

def init_worker(args):
    global env
    env = args

def job(args):
    global env
    (episode_reset_seed, n_iterations) = args
    results = test_without_control(env, episode_reset_seed, n_iterations, output=True)
    return results

class Worker:
    def __init__(self, num_workers, env):
        self.pool = self.make_workers(num_workers, env)
        self.num_workers = num_workers

    def make_workers(self, num_workers, env):
        pool = multiprocessing.Pool(num_workers, initializer=init_worker, initargs=[env])
        print("Make Workers", num_workers)
        return pool

    def work(self, i_episodes, iterations):
        episode_reset_seeds = [i for i in range(i_episodes)]
        n_iterations = [iterations for _ in range(iterations)]
        work_results = self.pool.map(job, zip(episode_reset_seeds, n_iterations))
        test_total_finished_tasks, test_total_computing_times, computing_times_list, reach_nodes, expand_nodes, idle_timesteps, moving_timesteps, detour_distances, fail_cases = self.make_results(work_results)
        success_cases = i_episodes - fail_cases
        avg_finished_task, avg_computing_time = "-" if not success_cases else sum(test_total_finished_tasks)/(success_cases*iterations), "-" if not success_cases else sum(test_total_computing_times)/(success_cases*iterations)
        avg_reach_nodes, avg_expand_nodes = "-" if not success_cases else sum(reach_nodes)/(success_cases*iterations), "-" if not success_cases else sum(expand_nodes)/(success_cases*iterations)
        # print(idle_timesteps)
        avg_idle_timesteps, avg_moving_timesteps = "-" if not success_cases else sum(idle_timesteps)/(success_cases*iterations), "-" if not success_cases else sum(moving_timesteps)/(success_cases*iterations)
        avg_detour_distances = "-" if not success_cases else sum(detour_distances)/(success_cases*iterations)
        computing_time_stdev = statistics.stdev(computing_times_list)
        reach_nodes_stdev = statistics.stdev(reach_nodes)
        expand_nodes_stdev = statistics.stdev(expand_nodes)
        idle_timesteps_stdev = statistics.stdev(idle_timesteps)
        moving_timesteps_stdev = statistics.stdev(moving_timesteps)
        detour_distances_stdev = statistics.stdev(detour_distances)
        print("Fails:", fail_cases)
        # print(avg_finished_task, avg_computing_time, computing_times_list, statistics.stdev(computing_times_list))
        # print(computing_time_stdev, reach_nodes_stdev, expand_nodes_stdev)
        return avg_finished_task, avg_computing_time, fail_cases, avg_reach_nodes, avg_expand_nodes, avg_idle_timesteps, avg_moving_timesteps, avg_detour_distances, computing_time_stdev, reach_nodes_stdev, expand_nodes_stdev, idle_timesteps_stdev, moving_timesteps_stdev, detour_distances_stdev
    
    def make_results(self, work_results):
        fail_cases = 0
        test_total_finished_tasks = []
        test_total_computing_times = []
        test_computing_times_list = [] 
        test_reach_nodes = []
        test_expand_nodes = []
        test_finished_idle_timesteps = []
        test_finished_moving_timesteps = []
        test_finished_detour_distances = []
        for result in work_results:
            if result:
                result_total_finished_tasks, result_total_computing_time, result_computing_times, result_reach_nodes, result_expand_nodes, result_finished_idle_timesteps, result_finished_moving_timesteps, result_finished_detour_distances = result
                test_total_finished_tasks.append(result_total_finished_tasks)
                test_total_computing_times.append(result_total_computing_time)
                test_computing_times_list += result_computing_times
                test_reach_nodes += result_reach_nodes
                test_expand_nodes += result_expand_nodes
                test_finished_idle_timesteps += result_finished_idle_timesteps
                test_finished_moving_timesteps += result_finished_moving_timesteps
                test_finished_detour_distances += result_finished_detour_distances
            else:
                fail_cases += 1
        
        return test_total_finished_tasks, test_total_computing_times, test_computing_times_list, test_reach_nodes, test_expand_nodes, test_finished_idle_timesteps, test_finished_moving_timesteps, test_finished_detour_distances, fail_cases


def test(params: TestParameter, highway_type, test_episodes, use_highway_heuristic, highway_heuristic_setup=None):
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


    s = time()
    init_corridor_for_heuristic = corridors if use_highway_heuristic else []
    if use_highway_heuristic:
        grid_map.fit_corridors(corridors)
        heuristic_distance_map = grid_map.get_distance_map(highway_heuristic_setup)
        abstract_distance_map = grid_map.get_distance_map() if highway_heuristic_setup else heuristic_distance_map
        grid_map.reset()
    else:
        heuristic_distance_map = grid_map.get_distance_map()
        abstract_distance_map = heuristic_distance_map
    print("Finish Heuristic Map in", time() - s, "s")
    
    if params.solver_type == "CBS":
        mapf_solver = CBS(grid_map, params.window_size, params.buffer_size, False, heuristic_distance_map, abstract_distance_map)
        mapf_solver.plan_full_paths = params.plan_full_path
    elif params.solver_type == "PBS":
        mapf_solver = PBS(grid_map, params.window_size, params.buffer_size, False, heuristic_distance_map, abstract_distance_map)
        mapf_solver.plan_full_paths = params.plan_full_path

    X = "X"
    N = "N"
    env = Environment(
        f"test_diff_highway_w_{params.solver_type}_{params.x_len}_{params.y_len}_{params.x_num}_{params.y_num}_{params.line_num}_{params.pad_num}_a{len(agents)}_window{params.window_size if not params.window_size==10e10 else X}_{highway_type}_w{highway_heuristic_setup if not highway_heuristic_setup==None else N}" + ("_only_one_line" if params.only_one_line else ""), [grid_map.dimension[0], grid_map.dimension[1]], mapf_solver, agents, corridors
    )
    env.set_highway_type(highway_type)

    workers = Worker(params.worker_num, env)
    avg_finished_task, avg_computing_time, fail_cases, avg_reach_nodes, avg_expand_nodes, avg_idle_timesteps, avg_moving_timesteps, avg_detour_distances, computing_time_stdev, reach_nodes_stdev, expand_nodes_stdev, idle_timesteps_stdev, moving_timesteps_stdev, detour_distances_stdev = workers.work(test_episodes, params.iterations)
    # print(reach_nodes, expand_nodes)

    # params.save_result("./test/", (str(avg_reward) + " " + str(avg_computing_time) + "\n")
    # params.save_result("./test/", ("Highway: " if highway else "NoLomit: ") + "Avg. Reward = " + str(avg_reward) + ", Avg. Computing Time = " + str(avg_computing_time))
    # print(("Highway: " if highway else "NoLomit: ") + "Avg. Reward = " + str(avg_reward) + ", Avg. Computing Time = " + str(avg_computing_time))
    timestep_per_iteration = params.buffer_size
    return grid_map.dimension, len(agents), len(agents)/len(grid_map.get_spaces()), avg_finished_task/timestep_per_iteration, avg_computing_time, fail_cases, avg_reach_nodes, avg_expand_nodes, avg_idle_timesteps, avg_moving_timesteps, avg_detour_distances, computing_time_stdev, reach_nodes_stdev, expand_nodes_stdev, idle_timesteps_stdev, moving_timesteps_stdev, detour_distances_stdev

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
    data_dict = [{} for _ in range(17)]
    
    test_types = ["Strict Limit", "Strict Limit, Partial Plan", "Soft Limit"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for highway_w in range_iter:

        results = test(params, "soft", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=highway_w)
        for i in range(len(data_dict)):
            data_dict[i][test_types[2]].append(results[i])

        results = test(params, "strict", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=highway_w)
        for i in range(len(data_dict)):
            data_dict[i][test_types[0]].append(results[i])

        params.plan_full_path = False
        
        results = test(params, "strict", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=highway_w)
        for i in range(len(data_dict)):
            data_dict[i][test_types[1]].append(results[i])

        params.plan_full_path = True

        map_dims.append(results[0])
        agent_nums.append(results[1])
        agent_ratios.append(results[2])

    print("Map Dimensions:", map_dims)
    print("Number of Agents:", agent_nums)
    print("Ratio of Agents to Spaces:", agent_ratios)
    for test_type in test_types:
        print(f"Avg Finished Tasks ({test_type}):", data_dict[0][test_type])
        print(f"Avg Computing Time ({test_type}):", data_dict[1][test_type])
        print(f"Fail Cases ({test_type}):", data_dict[2][test_type])

    if not output_path == None:
        with open(output_path, 'w+') as csvfile:
            print(output_path)
            writer = csv.writer(csvfile)
            writer.writerow(["test_type", "dimension", "highway_w", "window_size", "buffer_size", "agent_num", "agent_ratio", "avg. finished tasks", "avg. computing time", "fail cases", "avg. reach nodes", "avg. expand nodes", "avg. idle_timesteps", "avg. moving_timesteps", "avg. detour_distances", "computing time std.", "reach nodes std.", "expand nodes std.", "idle timesteps std.", "moving timesteps std.", "detour distances std."])
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
                    for n in range(3,17):
                        row_data.append(data_dict[n][test_type][i])
                    # row_data.append(avg_finished_tasks[test_type][i])
                    # row_data.append(avg_computing_time[test_type][i])
                    # row_data.append(fail_case[test_type][i])
                    # row_data.append(reach_nodes_list[test_type][i])
                    # row_data.append(expand_nodes_list[test_type][i])
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

    task_num, time_compute, fail_cases, map_dim, agent_num, agent_ratio, reach_nodes, expand_nodes = test(params, "strict", test_episodes, use_highway_heuristic=True, highway_heuristic_setup=None)
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
    ITERATION_NUM = 10
    WORKER_NUM = 10
    plan_full_path = False

    x_len = 2
    y_len = 2
    x_num = 5
    y_num = 5
    agent_num = 25
    solver = "PBS"

    line_num = 1
    pad_num = 0
    only_one_line = False
    only_allow_main_direction = False

    X = "X"

    params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    OUTPUT = f"./test/test_5.csv"
    test_diff_highway_w(params, [1, 1.2, 1.5, 2, 5, 10, 20, 50, 0], output_path = OUTPUT)

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
    # for agent_num in [10]:
    #     # OUTPUT = f"/media/NFS/fong/KIva-System-RL/test/test_highway_w_{x_len}{y_len}{x_num}{y_num}_{line_num}_{pad_num}_a{agent_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ("_only_allow_main_direction" if only_allow_main_direction else "") + ".csv"
    #     OUTPUT = f"./test/test.csv"
    #     params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, WORKER_NUM, plan_full_path, only_one_line, only_allow_main_direction)
    #     test_highway(params, output_path = OUTPUT)
    # test_diff_highway_w(params, [1, 1.2, 1.5, 2, 5, 10, 20, 50, 0], output_path = OUTPUT)

    # test_diff_agent_num(params, [5, 8, 10], output_path = OUTPUT)
    # test_diff_map_size(params, 2, output_path = OUTPUT)
    # test_diff_window_size(params, [5, 6, 7, 8, 9], output_path = OUTPUT)

    # aa(params, WINDOW_SIZE, BUFFER_SIZE, False)

if __name__ == "__main__":
    main()