import numpy as np
import multiprocessing

from qlearn.toys.main_nchain import parse, main

num_runs = 10
seeds = np.random.randint(0, 1e8, num_runs)
input_dim_list = [
    # 10,
    # 100,
    # 200,
    # 400,
    # 800,
    1600,
]
agent_list = [
    # 'DQN',
    'MNFDQN',
]


def f(agent, input_dim, run):
    args.agent = agent
    args.input_dim = input_dim
    args.seed = seeds[run]
    return main(args)


if __name__ == '__main__':

    args = parse()
    output = {}
    pool = multiprocessing.Pool(num_runs)
    for agent in agent_list:
        output[agent] = {}
        for input_dim in input_dim_list:

            # res = np.zeros(num_runs)
            # for run in range(num_runs):
            #
            #     args.agent = dqn
            #     args.input_dim = input_dim
            #     args.seed = seeds[run]
            #
            #     res[run] = main(args)

            # agents = iter([agent] * num_runs)
            # input_dims = iter([input_dim] * num_runs)
            # runs = iter(range(num_runs))

            iter_argument = [(agent, input_dim, run) for run in range(num_runs)]

            res = np.asarray(pool.starmap(f, iter_argument))
            output[agent][input_dim] = res

            print(output)

    print("DONE")
    print(output)

