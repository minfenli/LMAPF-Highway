from pbs import *
from cbs import CBS
from ca import CA
from map import *
from env import *
from time import time
import multiprocessing
import csv
import statistics

OUTPUT_HISTORY = False
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
    result_reroute_agents = []
    result_computing_times = []
    result_finished_idle_timesteps = []
    result_finished_moving_timesteps = []
    result_finished_detour_distances = []

    result_total_finished_tasks = 0

    env.reset(episode_reset_seed)


    total_time_start = time()
    for iteration in range(n_iterations):
        time_start = time()
        finished_tasks, reroute_agents, no_solution_in_time, reach_nodes, expand_nodes, finished_idle_timesteps, finished_moving_timesteps, finished_detour_distances  = env.step(
            iteration * env.mapf_solver.buffer_size + 1
        )
        # print(finished_idle_timesteps, finished_moving_timesteps, finished_detour_distances)
        if no_solution_in_time:
            return False
        result_total_finished_tasks += finished_tasks
        result_reach_nodes.append(reach_nodes)
        result_expand_nodes.append(expand_nodes)
        result_reroute_agents.append(reroute_agents)
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

    return result_total_finished_tasks, result_total_computing_time, result_computing_times, result_reach_nodes, result_expand_nodes, result_reroute_agents, result_finished_idle_timesteps, result_finished_moving_timesteps, result_finished_detour_distances

def init_worker(args):
    global env
    env = args

def job(args):
    global env, OUTPUT_HISTORY
    (episode_reset_seed, n_iterations) = args
    results = test_without_control(env, episode_reset_seed, n_iterations, output=OUTPUT_HISTORY)
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
        n_iterations = [iterations for _ in range(i_episodes)]
        work_results = self.pool.map(job, zip(episode_reset_seeds, n_iterations))
        test_total_finished_tasks, test_total_computing_times, computing_times_list, reach_nodes, expand_nodes, reroute_agents, idle_timesteps, moving_timesteps, detour_distances, fail_cases = self.make_results(work_results)
        success_cases = i_episodes - fail_cases
        avg_finished_task, avg_computing_time = "-" if not success_cases else sum(test_total_finished_tasks)/(success_cases*iterations), "-" if not success_cases else sum(test_total_computing_times)/(success_cases*iterations)
        avg_reach_nodes, avg_expand_nodes, avg_reroute_agents = "-" if not success_cases else sum(reach_nodes)/(success_cases*iterations), "-" if not success_cases else sum(expand_nodes)/(success_cases*iterations), "-" if not success_cases else sum(reroute_agents)/(success_cases*iterations)
        # print(detour_distances)
        avg_idle_timesteps, avg_moving_timesteps = "-" if not idle_timesteps else statistics.mean(idle_timesteps), "-" if not moving_timesteps else statistics.mean(moving_timesteps)
        avg_detour_distances = "-" if not detour_distances else statistics.mean(detour_distances)
        computing_time_stdev = "-" if not len(computing_times_list) >= 2 else statistics.stdev(computing_times_list)
        reach_nodes_stdev = "-" if not len(reach_nodes) >= 2 else statistics.stdev(reach_nodes)
        expand_nodes_stdev = "-" if not len(expand_nodes) >= 2 else statistics.stdev(expand_nodes)
        reroute_agents_stdev = "-" if not len(reroute_agents) >= 2 else statistics.stdev(reroute_agents)
        idle_timesteps_stdev = "-" if not len(idle_timesteps) >= 2 else statistics.stdev(idle_timesteps)
        moving_timesteps_stdev = "-" if not len(moving_timesteps) >= 2 else statistics.stdev(moving_timesteps)
        detour_distances_stdev = "-" if not len(detour_distances) >= 2 else statistics.stdev(detour_distances)
        print("Fails:", fail_cases)
        # print(avg_finished_task, avg_computing_time, computing_times_list, statistics.stdev(computing_times_list))
        # print(computing_time_stdev, reach_nodes_stdev, expand_nodes_stdev)
        return avg_finished_task, avg_computing_time, fail_cases, avg_reach_nodes, avg_expand_nodes, avg_reroute_agents, avg_idle_timesteps, avg_moving_timesteps, avg_detour_distances, computing_time_stdev, reach_nodes_stdev, expand_nodes_stdev, reroute_agents_stdev, idle_timesteps_stdev, moving_timesteps_stdev, detour_distances_stdev
    
    def make_results(self, work_results):
        fail_cases = 0
        test_total_finished_tasks = []
        test_total_computing_times = []
        test_computing_times_list = [] 
        test_reach_nodes = []
        test_expand_nodes = []
        test_reroute_agents = []
        test_finished_idle_timesteps = []
        test_finished_moving_timesteps = []
        test_finished_detour_distances = []
        for result in work_results:
            if result:
                result_total_finished_tasks, result_total_computing_time, result_computing_times, result_reach_nodes, result_expand_nodes, result_reroute_agents, result_finished_idle_timesteps, result_finished_moving_timesteps, result_finished_detour_distances = result
                test_total_finished_tasks.append(result_total_finished_tasks)
                test_total_computing_times.append(result_total_computing_time)
                test_computing_times_list += result_computing_times
                test_reach_nodes += result_reach_nodes
                test_expand_nodes += result_expand_nodes
                test_reroute_agents += result_reroute_agents
                test_finished_idle_timesteps += result_finished_idle_timesteps
                test_finished_moving_timesteps += result_finished_moving_timesteps
                test_finished_detour_distances += result_finished_detour_distances
            else:
                fail_cases += 1
        return test_total_finished_tasks, test_total_computing_times, test_computing_times_list, test_reach_nodes, test_expand_nodes, test_reroute_agents, test_finished_idle_timesteps, test_finished_moving_timesteps, test_finished_detour_distances, fail_cases


def test(params: TestParameter, highway_type, test_episodes, highway_heuristic_setup=None):
    assert(highway_type=="strict" or highway_type=="soft" or highway_type=="none")
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
    if highway_type != "none":
        grid_map.fit_corridors(corridors)
        heuristic_distance_map = grid_map.get_distance_map(highway_heuristic_setup)
        abstract_distance_map = heuristic_distance_map  # grid_map.get_distance_map() if highway_heuristic_setup else heuristic_distance_map
        grid_map.reset()
        # for row in heuristic_distance_map[0][0]: print(row)
        # if not highway_heuristic_setup:
        #     import pdb; pdb.set_trace()
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
    elif params.solver_type == "CA*":
        mapf_solver = CA(grid_map, params.window_size, params.buffer_size, False, heuristic_distance_map, abstract_distance_map)
        mapf_solver.plan_full_paths = params.plan_full_path

    X = "X"
    N = "N"
    env = Environment(
        f"{params.solver_type}_{params.x_len}_{params.y_len}_{params.x_num}_{params.y_num}_{params.line_num}_{params.pad_num}_a{len(agents)}_window{params.window_size if not params.window_size==10e10 else X}" + ("" if params.plan_full_path else "_partial") + f"_{highway_type}_" + (f"w{highway_heuristic_setup}" if highway_heuristic_setup else N) + ("_only_one_line" if params.only_one_line else "") + ("_only_main_direction" if params.only_allow_main_direction else ""), [grid_map.dimension[0], grid_map.dimension[1]], mapf_solver, agents, corridors
    )
    env.set_highway_type(highway_type)

    workers = Worker(params.worker_num, env)
    avg_finished_task, avg_computing_time, fail_cases, avg_reach_nodes, avg_expand_nodes, avg_reroute_agents, avg_idle_timesteps, avg_moving_timesteps, avg_detour_distances, computing_time_stdev, reach_nodes_stdev, expand_nodes_stdev, reroute_agents_stdev, idle_timesteps_stdev, moving_timesteps_stdev, detour_distances_stdev = workers.work(test_episodes, params.iterations)
    # print(reach_nodes, expand_nodes)

    # params.save_result("./test/", (str(avg_reward) + " " + str(avg_computing_time) + "\n")
    # params.save_result("./test/", ("Highway: " if highway else "NoLomit: ") + "Avg. Reward = " + str(avg_reward) + ", Avg. Computing Time = " + str(avg_computing_time))
    # print(("Highway: " if highway else "NoLomit: ") + "Avg. Reward = " + str(avg_reward) + ", Avg. Computing Time = " + str(avg_computing_time))
    timestep_per_iteration = params.buffer_size
    return grid_map.dimension, len(agents), len(agents)/len(grid_map.get_spaces()), avg_finished_task if avg_finished_task=="-" else avg_finished_task/timestep_per_iteration, avg_computing_time, fail_cases, avg_reach_nodes, avg_expand_nodes, avg_reroute_agents, avg_idle_timesteps, avg_moving_timesteps, avg_detour_distances, computing_time_stdev, reach_nodes_stdev, expand_nodes_stdev, reroute_agents_stdev, idle_timesteps_stdev, moving_timesteps_stdev, detour_distances_stdev

# test three different types of highway with different h for calculating heuristic
def test_diff_highway_w(params: TestParameter, range_iter, test_episodes = 10, output_path = None):
    data_dict = [{} for _ in range(19)]
    
    test_types = ["Strict Limit", "Strict Limit, Partial Plan", "Soft Limit"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for highway_w in range_iter:

        results = test(params, "soft", test_episodes, highway_heuristic_setup=highway_w)
        for i in range(len(data_dict)):
            data_dict[i][test_types[2]].append(results[i])

        results = test(params, "strict", test_episodes, highway_heuristic_setup=highway_w)
        for i in range(len(data_dict)):
            data_dict[i][test_types[0]].append(results[i])

        params.plan_full_path = False
        
        results = test(params, "strict", test_episodes, highway_heuristic_setup=highway_w)
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
            writer.writerow([
                "test_type", 
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])
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
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    # row_data.append(avg_finished_tasks[test_type][i])
                    # row_data.append(avg_computing_time[test_type][i])
                    # row_data.append(fail_case[test_type][i])
                    # row_data.append(reach_nodes_list[test_type][i])
                    # row_data.append(expand_nodes_list[test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

# test three different types of highway with different agent numbers
def test_three_highway_diff_agent_num(params: TestParameter, agent_num_list, test_episodes = 10, output_path = None):
    data_dict = [{} for _ in range(19)]
    
    test_types = ["Strict Limit, Partial Plan", "Strict Limit", "Soft Limit", "No Highway"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for agent_num in agent_num_list:

        params.agent_num = agent_num

        results = test(params, "none", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[3]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[2]].append(results[i])

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[1]].append(results[i])

        params.plan_full_path = False
        
        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[0]].append(results[i])

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
            writer.writerow([
                "test_type", 
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])
            writer.writerow([])
            for test_type in test_types:
                for i, agent_num in enumerate(agent_num_list):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dims[i])
                    row_data.append("None")
                    row_data.append(params.window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_ratios[i])
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

# test three different types of highway with different agent densities
def test_three_highway_diff_agent_density(params: TestParameter, agent_density_list, test_episodes = 10, output_path = None):
    data_dict = [{} for _ in range(19)]
    
    test_types = ["Strict Limit, Partial Plan", "Strict Limit", "Soft Limit", "No Highway"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for density in agent_density_list:

        grid_map, corridor_params = make_map(params.x_len, params.y_len, params.x_num, params.y_num, True, params.line_num, params.pad_num, params.only_one_line, params.only_allow_main_direction)
        params.agent_num = round(len(grid_map.get_spaces())*density)

        results = test(params, "none", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[3]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[2]].append(results[i])

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[1]].append(results[i])

        params.plan_full_path = False
        
        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[0]].append(results[i])

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
            writer.writerow([
                "test_type", 
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_density",
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])
            writer.writerow([])
            for test_type in test_types:
                for i, agent_density in enumerate(agent_density_list):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dims[i])
                    row_data.append("None")
                    row_data.append(params.window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_density)
                    row_data.append(agent_ratios[i])
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

# test three different types of highway with different buffer sizes
def test_three_highway_diff_buffer_size(params: TestParameter, buffer_size_list, test_episodes = 10, output_path = None):
    data_dict = [{} for _ in range(19)]
    
    test_types = ["Strict Limit, Partial Plan", "Strict Limit", "Soft Limit", "No Highway"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for buffer_size in buffer_size_list:

        params.buffer_size = buffer_size
        params.window_size = buffer_size

        results = test(params, "none", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[3]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[2]].append(results[i])

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[1]].append(results[i])

        params.plan_full_path = False
        
        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[0]].append(results[i])

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
            writer.writerow([
                "test_type", 
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])
            writer.writerow([])
            for test_type in test_types:
                for i, buffer_size in enumerate(buffer_size_list):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dims[i])
                    row_data.append("None")
                    row_data.append(buffer_size)
                    row_data.append(buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_ratios[i])
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

# test three different types of highway with different window sizes
def test_three_highway_diff_window_size(params: TestParameter, window_size_list, test_episodes = 10, output_path = None):
    data_dict = [{} for _ in range(19)]
    
    test_types = ["Strict Limit, Partial Plan", "Strict Limit", "Soft Limit", "No Highway"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for window_size in window_size_list:
        
        params.window_size = window_size

        results = test(params, "none", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[3]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[2]].append(results[i])

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[1]].append(results[i])

        params.plan_full_path = False
        
        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[0]].append(results[i])

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
            writer.writerow([
                "test_type", 
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])
            writer.writerow([])
            for test_type in test_types:
                for i, window_size in enumerate(window_size_list):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dims[i])
                    row_data.append("None")
                    row_data.append(window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_ratios[i])
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

# test three different types of highway with different map sizes and a constant agent density
def test_three_highway_diff_map_size(params: TestParameter, map_size_list, density, test_episodes = 10, output_path = None):
    data_dict = [{} for _ in range(19)]
    
    test_types = ["Strict Limit, Partial Plan", "Strict Limit", "Soft Limit", "No Highway"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for map_size in map_size_list:
        params.x_num = map_size
        params.y_num = map_size
        grid_map, corridor_params = make_map(params.x_len, params.y_len, params.x_num, params.y_num, True, params.line_num, params.pad_num, params.only_one_line, params.only_allow_main_direction)
        params.agent_num = round(len(grid_map.get_spaces())*density)

        results = test(params, "none", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[3]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[2]].append(results[i])

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[1]].append(results[i])

        params.plan_full_path = False
        
        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[0]].append(results[i])

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
            writer.writerow([
                "test_type", 
                "blocks",
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])
            writer.writerow([])
            for test_type in test_types:
                for i, map_size in enumerate(map_size_list):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(str(map_size)+"x"+str(map_size))
                    row_data.append(map_dims[i])
                    row_data.append("None")
                    row_data.append(params.window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_ratios[i])
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

# test highway with different map size and a constant density
def test_highway_w_diff_map_size(params: TestParameter, highway_w_list, map_size_list, density, test_episodes = 10, output_path = None):
    data_dict = [{} for _ in range(19)]

    for highway_w in highway_w_list:
        for i in range(len(data_dict)):
            data_dict[i][highway_w] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for map_size in map_size_list:
        params.x_num = map_size
        params.y_num = map_size
        grid_map, corridor_params = make_map(params.x_len, params.y_len, params.x_num, params.y_num, True, params.line_num, params.pad_num, params.only_one_line, params.only_allow_main_direction)
        params.agent_num = round(len(grid_map.get_spaces())*density)

        for n, highway_w in enumerate(highway_w_list):
            results = test(params, "soft", test_episodes, highway_heuristic_setup=highway_w)
            for i in range(len(data_dict)):
                data_dict[i][highway_w].append(results[i])
        

        map_dims.append(results[0])
        agent_nums.append(results[1])
        agent_ratios.append(results[2])

    print("Map Dimensions:", map_dims)
    print("Number of Agents:", agent_nums)
    print("Ratio of Agents to Spaces:", agent_ratios)

    if not output_path == None:
        with open(output_path, 'w+') as csvfile:
            print(output_path)
            writer = csv.writer(csvfile)
            writer.writerow([
                "blocks",
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])
            writer.writerow([])
            for highway_w in highway_w_list:
                for i, map_size in enumerate(map_size_list):
                    row_data = []
                    row_data.append(str(map_size)+"x"+str(map_size))
                    row_data.append(map_dims[i])
                    row_data.append(highway_w if highway_w != 0 else "∞")
                    row_data.append(params.window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_ratios[i])
                    for n in range(3,19):
                        row_data.append(data_dict[n][highway_w][i])
                    writer.writerow(row_data)
                writer.writerow([])

# test three different types of highway with different agent densities
def test_highway_line_diff_agent_density(params: TestParameter, agent_density_list, test_episodes = 10, output_path = None):
    data_dict = [{} for _ in range(19)]
    
    test_types = ["Strict Limit, full", "Soft Limit, full", "Strict Limit, full-cross", "Soft Limit, full-cross", "Strict Limit, one", "Soft Limit, one", "Strict Limit, one-cross", "Soft Limit, one-cross", "No Highway"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    for density in agent_density_list:

        grid_map, corridor_params = make_map(params.x_len, params.y_len, params.x_num, params.y_num, True, params.line_num, params.pad_num, params.only_one_line, params.only_allow_main_direction)
        params.agent_num = round(len(grid_map.get_spaces())*density)

        params.only_one_line = False
        params.only_allow_main_direction = True

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[0]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[1]].append(results[i])

        params.only_one_line = False
        params.only_allow_main_direction = False

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[2]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[3]].append(results[i])

        params.only_one_line = True
        params.only_allow_main_direction = True

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[4]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[5]].append(results[i])

        params.only_one_line = True
        params.only_allow_main_direction = False

        results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[6]].append(results[i])

        results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[7]].append(results[i])

        params.only_one_line = False
        params.only_allow_main_direction = False

        results = test(params, "none", test_episodes, highway_heuristic_setup=None)
        for i in range(len(data_dict)):
            data_dict[i][test_types[-1]].append(results[i])
        

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
            writer.writerow([
                "test_type", 
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_density",
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])
            writer.writerow([])
            for test_type in test_types:
                for i, agent_density in enumerate(agent_density_list):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dims[i])
                    row_data.append("None")
                    row_data.append(params.window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_nums[i])
                    row_data.append(agent_density)
                    row_data.append(agent_ratios[i])
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])


# test three different types of highway with the setting of params
def test_three_highway(params: TestParameter, test_episodes = 10, output_path = None):
    # params.window_size = params.buffer_size

    data_dict = [{} for _ in range(19)]
    
    test_types = ["Strict Limit, Partial Plan", "Strict Limit", "Soft Limit", "No Highway"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    results = test(params, "none", test_episodes, highway_heuristic_setup=None)
    for i in range(len(data_dict)):
        data_dict[i][test_types[3]].append(results[i])
    
    results = test(params, "soft", test_episodes, highway_heuristic_setup=None)
    for i in range(len(data_dict)):
        data_dict[i][test_types[2]].append(results[i])

    results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
    for i in range(len(data_dict)):
        data_dict[i][test_types[1]].append(results[i])

    params.plan_full_path = False
    
    results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
    for i in range(len(data_dict)):
        data_dict[i][test_types[0]].append(results[i])

    map_dim = results[0]
    agent_num = results[1]
    agent_ratio = results[2]

    print("Map Dimensions:", map_dim)
    print("Number of Agents:", agent_num)
    print("Ratio of Agents to Spaces:", agent_ratio)
    for test_type in test_types:
        print(f"Avg Finished Tasks ({test_type}):", data_dict[0][test_type])
        print(f"Avg Computing Time ({test_type}):", data_dict[1][test_type])
        print(f"Fail Cases ({test_type}):", data_dict[2][test_type])

    if not output_path == None:
        with open(output_path, 'w+') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "test_type", 
                "dimension", 
                "highway_w", 
                "window_size", 
                "buffer_size", 
                "agent_num", 
                "agent_ratio", 
                "avg. finished tasks", 
                "avg. computing time", 
                "fail cases", 
                "avg. reach nodes", 
                "avg. expand nodes", 
                "avg. reroute_agents",
                "avg. idle_timesteps", 
                "avg. moving_timesteps", 
                "avg. detour_distances", 
                "computing time std.", 
                "reach nodes std.", 
                "expand nodes std.", 
                "reroute_agents std.",
                "idle timesteps std.", 
                "moving timesteps std.", 
                "detour distances std."])            
            writer.writerow([])
            for test_type in test_types:
                for i in range(1):
                    row_data = []
                    row_data.append(test_type)
                    row_data.append(map_dim)
                    row_data.append("None")
                    row_data.append(params.window_size)
                    row_data.append(params.buffer_size)
                    row_data.append(agent_num)
                    row_data.append(agent_ratio)
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

# test strict highway with the setting of params
def test_highway(params: TestParameter, test_episodes = 10, output_path = None):
    # params.window_size = params.buffer_size

    data_dict = [{} for _ in range(19)]
    
    test_types = ["highway(Strict Limit)"]
    # test_types = ["highway(directed)", "highway(obsolute)", "policy(obsolute)", "nolimit(directed)", "nolimit(obsolute)"]

    for test_type in test_types:
        for i in range(len(data_dict)):
            data_dict[i][test_type] = []

    map_dims = []
    agent_nums = []
    agent_ratios = []

    results = test(params, "strict", test_episodes, highway_heuristic_setup=None)
    for i in range(len(data_dict)):
        data_dict[i][test_types[0]].append(results[i])

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
            writer = csv.writer(csvfile)
            writer.writerow(["test_type", "dimension", "highway_w", "window_size", "buffer_size", "agent_num", "agent_ratio", "avg. finished tasks", "avg. computing time", "fail cases", "avg. reach nodes", "avg. expand nodes", "avg. reroute agents", "avg. idle_timesteps", "avg. moving_timesteps", "avg. detour_distances", "computing time std.", "reach nodes std.", "expand nodes std.", "reroute agents std.", "idle timesteps std.", "moving timesteps std.", "detour distances std."])
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
                    for n in range(3,19):
                        row_data.append(data_dict[n][test_type][i])
                    writer.writerow(row_data)
                writer.writerow([])

def main():
    time_start = time()

    WINDOW_SIZE = 5
    BUFFER_SIZE = 5
    ITERATION_NUM = 100
    EPISODE_NUM = 100
    WORKER_NUM = 10
    # plan_full_path = False

    x_len = 10
    y_len = 2
    x_num = 3
    y_num = 3
    agent_num = 10
    solver = "PBS"

    line_num = 1
    pad_num = 0
    only_one_line = False
    only_allow_main_direction = False
    
    # ==================================================================
    # ============== Experient of Different MAPF Solvers ===============
    # ==================================================================

    # solver = "PBS"
    # params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    # OUTPUT = OUTPUT = f"./exp/test_highway_types_diff_map_size_den5%_i{ITERATION_NUM}_e{EPISODE_NUM}_{x_len}{y_len}_{line_num}_{pad_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    # test_three_highway_diff_map_size(params, range(3,16,2), 0.05, test_episodes=EPISODE_NUM, output_path = OUTPUT)

    # solver = "CBS"
    # params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    # OUTPUT = OUTPUT = f"./exp/test_highway_types_diff_map_size_den5%_i{ITERATION_NUM}_e{EPISODE_NUM}_{x_len}{y_len}_{line_num}_{pad_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    # test_three_highway_diff_map_size(params, range(3,16,2), 0.05, test_episodes=EPISODE_NUM, output_path = OUTPUT)

    # [o] FINISH on BRANDY
    # solver = "CA*"
    # params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    # OUTPUT = OUTPUT = f"./exp/test_highway_types_diff_map_size_den5%_i{ITERATION_NUM}_e{EPISODE_NUM}_{x_len}{y_len}_{line_num}_{pad_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    # test_three_highway_diff_map_size(params, range(3,16,2), 0.05, test_episodes=EPISODE_NUM, output_path = OUTPUT)


    # ==================================================================
    # ============= Experient of Different Agent Densities =============
    # ==================================================================

    # solver = "PBS"
    # params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    # OUTPUT = OUTPUT = f"./exp/output_test_highway_types_diff_agent_density_i{ITERATION_NUM}_e{EPISODE_NUM}_{x_len}{y_len}{x_num}{y_num}_{line_num}_{pad_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    # test_three_highway_diff_agent_density(params, [x*0.01 for x in range(5,45,5)], test_episodes=EPISODE_NUM, output_path = OUTPUT)


    # ==================================================================
    # ============ Experient of Different "s" and Map Sizes ============
    # ==================================================================

    # solver = "PBS"
    # params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    # OUTPUT = OUTPUT = f"./exp/test_highway_w_diff_map_size_den5%_i{ITERATION_NUM}_e{EPISODE_NUM}_{x_len}{y_len}{x_num}{y_num}_{line_num}_{pad_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    # test_highway_w_diff_map_size(params, [1, 1.2, 1.5, 2, 5, 10, 50, 0], range(3,12,2), 0.05, test_episodes=EPISODE_NUM, output_path = OUTPUT)

    
    # ==================================================================
    # ============ Experient of Different "h" and Map Sizes ============
    # ==================================================================

    # solver = "PBS"
    # for l_n, agent_n in zip([3, 5, 7,9], [8, 20, 37, 59]):
    #     x_num = l_n
    #     y_num = l_n
    #     agent_num = agent_n
    #     params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    #     OUTPUT = OUTPUT = f"./exp/test_three_highway_diff_buffer_size_i{ITERATION_NUM}_e{EPISODE_NUM}_{x_len}{y_len}{x_num}{y_num}_{line_num}_{pad_num}_a{agent_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    #     test_three_highway_diff_buffer_size(params, range(1,11), test_episodes=EPISODE_NUM, output_path = OUTPUT)


    # ==================================================================
    # ============ Experient of Different "w" and Map Sizes ============
    # ==================================================================
    
    # solver = "PBS"
    # for l_n, agent_n in zip([3, 5, 7,9], [8, 20, 37, 59]):
    #     x_num = l_n
    #     y_num = l_n
    #     agent_num = agent_n
    #     params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    #     OUTPUT = OUTPUT = f"./exp/test_three_highway_diff_window_size_i{ITERATION_NUM}_e{EPISODE_NUM}_{x_len}{y_len}{x_num}{y_num}_{line_num}_{pad_num}_a{agent_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    #     test_three_highway_diff_window_size(params, range(5,11), test_episodes=EPISODE_NUM, output_path = OUTPUT)

    # ==================================================================
    # Experient of Different Agent Densities of different line-settings=
    # ==================================================================

    # solver = "PBS"
    # line_num = 2

    # params = TestParameter(solver,x_len,y_len,x_num,y_num,line_num,pad_num,agent_num, WINDOW_SIZE, BUFFER_SIZE, ITERATION_NUM, WORKER_NUM, True, only_one_line, only_allow_main_direction)
    # OUTPUT = OUTPUT = f"./exp/test_highway_lines_diff_agent_density_i{ITERATION_NUM}_e{EPISODE_NUM}_{x_len}{y_len}{x_num}{y_num}_{line_num}_{pad_num}_{solver}" + ("_only_one_line" if only_one_line else "") + ".csv"
    # test_highway_line_diff_agent_density(params, [x*0.01 for x in range(5,25,5)], test_episodes=EPISODE_NUM, output_path = OUTPUT)

    print("Overall Computing Time:", time()-time_start)

if __name__ == "__main__":
    main()