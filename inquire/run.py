import os
import pdb
import pickle
import time

import numpy as np
from inquire.environments.environment import CachedTask, Task
from inquire.utils.datatypes import Modality
from numpy.random import RandomState

import matplotlib.pyplot as plt
import matplotlib

def run_both(lst):
    vals = np.array([lst[i][0] for i in range(len(lst))])
    names = np.array([lst[i][1] for i in range(len(lst))])
    vals = vals.reshape(5,4)
    names = names.reshape(5,4)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create a single figure with two subplots
    heatmap = ax[0].imshow(vals, cmap='coolwarm_r', interpolation='nearest')
    cbar = plt.colorbar(heatmap, ax=ax[0])
    cbar.set_ticks([np.min(vals), np.max(vals)])
    cbar.ax.set_yticklabels(['More Preferred', 'Less Preferred'])
    ax[0].axis('off')
    ax[0].set_title('Vending Machine Preferences')

    for i in range(5):
        for j in range(4):
            txt = names[i, j].split(" ")
            txt = " \n".join(txt)
            text = ax[0].text(j, i, txt, wrap=True, ha='center', va='center', color='black', fontsize=8)

    lst.sort()
    top_ten = lst[:10]
    values = [item[0] for item in top_ten]
    values = [max(values) - value for value in values]
    names = [item[1].split(" ") for item in top_ten]
    names = [" \n".join(name) for name in names]
    ax[1].bar(names, values, color='skyblue')
    ax[1].set_xlabel('Item Name')
    ax[1].set_title('Top Ten Items (Descending Order)')
    ax[1].tick_params(axis='x', labelrotation=45, labelsize=8)  # Rotate and adjust font size of x-axis labels
    ax[1].tick_params(axis='y', labelsize=8)  # Adjust font size of y-axis labels
    ax[1].get_yaxis().set_visible(False)  # Turn off y-axis
    
    plt.tight_layout()  # Adjust subplot layout to prevent overlap
    plt.show(block=False)

def run_heatmap(lst):
    vals = np.array([lst[i][0] for i in range(len(lst))])
    names = np.array([lst[i][1] for i in range(len(lst))])
    vals = vals.reshape(5,4)
    names = names.reshape(5,4)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(vals, cmap='coolwarm_r', interpolation='nearest')
    cbar = plt.colorbar(heatmap)
    cbar.set_ticks([np.min(vals), np.max(vals)])
    cbar.ax.set_yticklabels(['More Preferred', 'Less Preferred'])
    plt.axis('off')
    plt.title('Vending Machine Preferences')

    for i in range(5):
        for j in range(4):
            txt = names[i, j].split(" ")
            txt = " \n".join(txt)
            text = ax.text(j, i, txt, wrap=True, ha='center', va='center', color='black', fontsize=8)

    plt.show(block=False)

def run_top_ten(lst):
    lst.sort()
    top_ten = lst[:10]
    values = [item[0] for item in top_ten]
    values = [max(values) - value for value in values]
    names = [item[1].split(" ") for item in top_ten]
    names = [" \n".join(name) for name in names]
    plt.bar(names, values, color='skyblue')
    plt.xlabel('Item Name')
    plt.title('Top Ten Items (Descending Order)')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.xticks(fontsize=8)
    plt.show(block=False)

def run(
    domain,
    teacher,
    agent,
    num_tasks,
    num_runs,
    num_queries,
    num_test_states,
    step_size,
    convergence_threshold,
    use_cached_trajectories=False,
    static_state=False,
    reuse_weights=False,
    verbose=False,
):
    test_state_rand = RandomState(0)
    init_w_rand = RandomState(0)
    real_num_queries = num_queries
    perf_mat = np.zeros(
        (num_tasks, num_runs, num_test_states, num_queries + 1)
    )
    dist_mat = np.zeros((num_tasks, num_runs, 1, num_queries + 1))
    query_mat = np.zeros((num_tasks, num_runs, 1, num_queries + 1))
    debug = False

    if static_state:
        query_states = 1
    else:
        query_states = real_num_queries

    if verbose:
        print("Initializing tasks...")
    if use_cached_trajectories:
        tasks = []
        for i in range(num_tasks):
            state_samples = []
            for j in range((num_runs * query_states) + num_test_states):
                f_name = (
                    "cache/"
                    + domain.__class__.__name__
                    + "_task-"
                    + str(i)
                    + "_state-"
                    + str(j)
                    + "_cache.pkl"
                )
                if os.path.isfile(f_name):
                    f = open(f_name, "rb")
                    state_samples.append(pickle.load(f))
                    f.close()
                else:
                    raise FileNotFoundError(f_name)
            tasks.append(
                CachedTask(
                    state_samples, num_runs * query_states, num_test_states
                )
            )
    else:
        # print("reached here: ", domain, num_runs * query_states, num_test_states)
        tasks = [
            Task(
                domain,
                num_runs * query_states,
                num_test_states,
                test_state_rand,
            )
            for _ in range(num_tasks)
        ]

    if static_state:
        for t in tasks:
            repeated_states = []
            for s in t.query_states:
                for _ in range(num_queries):
                    repeated_states.append(s)
            t.query_states = repeated_states

    ## Each task is an instantiation of the domain (e.g., a particular reward function)
    for t in range(num_tasks):
        task_start = time.perf_counter()
        task = tasks[t]
        test_set = []
        state_idx = 0

        ## Establish baseline task performance using optimal trajectories
        if verbose:
            print("Finding optimal and worst-case baselines...")
        for p, test_state in enumerate(task.test_states):
            best_traj = task.optimal_trajectory_from_ground_truth(
                test_state
            )
            # print("best_traj: ", best_traj.phi, best_traj.states, best_traj.actions)
            max_reward = task.ground_truth_reward(best_traj)
            worst_traj = task.least_optimal_trajectory_from_ground_truth(
                test_state
            )
            min_reward = task.ground_truth_reward(worst_traj)
            test_set.append(
                [
                    test_state,
                    (worst_traj, best_traj),
                    (min_reward, max_reward),
                ]
            )
            if debug:
                print(f"Finished {p+1}.")

        ## Reset learning agent for each run and iterate through queries
        if verbose:
            print("Done. Starting queries...")
        agent.reset()
        for r in range(num_runs):
            run_start = time.perf_counter()
            perfs = []
            feedback = []
            if agent.__class__.__name__.lower() == "dempref":
                for _ in range(agent.n_demos):
                    q = agent.generate_demo_query(
                        task.query_states[state_idx], domain,
                    )
                    teacher_fb = teacher.query_response(q, task, verbose)
                    if teacher_fb is not None:
                        feedback.append(teacher_fb)
                agent.seed_with_demonstrations(feedback)
            ## Record performance before first query
            w_dist = agent.initialize_weights(init_w_rand, domain)
            w_mean = np.mean(w_dist, axis=0)
            if debug:
                print(f"w0: {w_mean}")
            for c in range(num_test_states):
                model_traj = domain.optimal_trajectory_from_w(
                    test_set[c][0], w_mean
                )
                reward = task.ground_truth_reward(model_traj)
                min_r, max_r = test_set[c][2]
                perfs.append((reward - min_r) / (max_r - min_r))
                if perfs[-1] < -0.1 or perfs[-1] > 1.1:
                    pdb.set_trace()
                # assert 0 <= perfs[-1] <= 1
            perf_mat[t, r, :, 0] = perfs
            dist_mat[t, r, 0, 0] = task.distance_from_ground_truth(w_mean)
            query_mat[t, r, 0, 0] = Modality.NONE.value
            ## Iterate through queries
            for k in range(num_queries):
                q_start = time.perf_counter()
                print(
                    "\nTask "
                    + str(t + 1)
                    + "/"
                    + str(num_tasks)
                    + ", Run "
                    + str(r + 1)
                    + "/"
                    + str(num_runs)
                    + ", Query "
                    + str(k + 1)
                    + "/"
                    + str(num_queries)
                    + "     ",
                    end="\n",
                )

                ## Generate query and learn from feedback
                q = agent.generate_query(
                    domain, task.query_states[state_idx], w_dist, verbose
                )
                state_idx += 1
                print("teacher: ", teacher)
                teacher_fb = teacher.query_response(q, task, verbose)
                if teacher_fb is not None:
                    feedback.append(teacher_fb)

                w_dist, w_opt = agent.update_weights(
                    (w_dist if reuse_weights else None),
                    domain,
                    feedback,
                    learning_rate=step_size,
                    sample_threshold=convergence_threshold,
                    opt_threshold=1.0e-5,
                )
                # print("w_dist: ", w_dist)
                # print("w_opt: ", w_opt)
                if debug:
                    print(f"w after query {k+1}: {w_opt.mean(axis=0)}")
                ## Get performance metrics for each test-state after
                ## each query and corresponding weight update:
                perfs = []
                for c in range(num_test_states):
                    model_traj = domain.optimal_trajectory_from_w(
                        test_set[c][0], np.mean(w_opt, axis=0)
                    )

                    # NEW
                    lst = []
                    for food in domain.food_items:
                        distance = np.linalg.norm(domain.food_items[food].get_features() - model_traj.phi)
                        lst.append([distance, food])
                        
                    plt.close()
                    # run_heatmap(lst)
                    # run_top_ten(lst)
                    run_both(lst)


                    print("model_traj", model_traj.phi)#, model_traj.states, model_traj.actions)

                    reward = task.ground_truth_reward(model_traj)
                    min_r, max_r = test_set[c][2]
                    if k > 0 and debug:
                        print(
                            f"Reward:\nMin: {min_r}"
                            f"\nMax: {max_r}\nActual: {reward}"
                        )
                    perfs.append((reward - min_r) / (max_r - min_r))
                    # assert 0 <= perf <= 1
                perf_mat[t, r, :, k + 1] = perfs
                latest_dist = task.distance_from_ground_truth(
                    np.mean(w_opt, axis=0)
                )
                print("latest_dist: ", latest_dist)
                if k > 0 and debug:
                    print(f"Latest dist: {latest_dist}.")
                dist_mat[t, r, 0, k + 1] = latest_dist
                query_mat[t, r, 0, k + 1] = q.query_type.value
                q_time = time.perf_counter() - q_start
                if verbose:
                    print(
                        f"Query {k+1} in task {t+1}, run {r+1} took "
                        f"{q_time:.4}s to complete."
                    )
            run_time = time.perf_counter() - run_start
            if verbose:
                print(
                    f"Run {r+1} in task {t+1} took {run_time:.4}s "
                    "to complete."
                )

        task_time = time.perf_counter() - task_start
        if verbose:
            print(f"Task {t+1} took {task_time:.4f}s to complete.")
    return perf_mat, dist_mat, query_mat
