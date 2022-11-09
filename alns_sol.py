import json
import numpy as np
import random
from alns import ALNS, State
from alns.accept import HillClimbing
from alns.stop import MaxIterations
from alns.weights import SimpleWeights
from matplotlib import pyplot as plt
import copy


def read_data(data_path='./dev.jsonl'):
    hosts = {}
    instances = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            d = json.loads(line)
            if d['type'] == 'host':
                hosts[d['host_id']] = d
            elif d['type'] == 'instance':
                instances[d['instance_id']] = d
            line = f.readline()
    return hosts, instances


def dict2vec(d, host=True):
    return np.array([[
        d['host_id'] if host else 0,
        d['host_cpu'],
        d['host_memory'],
        d['host_disk'],
        d['instance_cpu_0'],
        d['instance_memory_0'],
        d['instance_disk_0'],
        d['instance_cpu_1'],
        d['instance_memory_1'],
        d['instance_disk_1'],
        d['instance_cpu_2'],
        d['instance_memory_2'],
        d['instance_disk_2'],
        d['instance_cpu_3'],
        d['instance_memory_3'],
        d['instance_disk_3'],
    ]])


class my_state(State):
    """
        sol: host_id -> {ori_host_dict + "instance_ids": [] + "apps": set}
    """

    def __init__(self, hosts, instances, sol=None):
        self.hosts = hosts
        self.instances = instances
        self.sol = {}
        self.host_matrix = None

    def objective(self):
        score1 = 12000 - len(self.sol)
        score2 = 0
        ratios = np.zeros((4, len(self.sol)))
        idx = 0
        for h in self.sol.values():
            ratio = []
            for i in range(4):
                ratio.append(10000 * h['instance_cpu_{}'.format(i)] /
                             h['host_cpu'])
            ratios[:, idx] = ratio
            idx += 1
        mean_std = np.mean(ratios.std(axis=1))
        if mean_std < 2000:
            score2 = 2000 - mean_std
        score = score1 + score2
        print(score1)
        print(score2)
        return -score

    def output_sol(self, path="./res.jsonl"):
        with open(path, 'w', encoding='utf-8') as f:
            for h in self.sol.values():
                host_id = h['host_id']
                for i in h['instance_ids']:
                    d = {'host_id': host_id, 'instance_id': i}
                    line = json.dumps(d)
                    f.write(line)
                    f.write('\n')

    def copy(self):
        return copy.deepcopy(self)


destroy_ratio = 0.25
resources = ["cpu", "memory", "disk"]


def random_destroy(current, rnd_state):
    """
        random remove some instances' host
        - ins[host_id] = 0
        - sol[res] -= ins[res]
        - sol[apps].remove(ins[app])
        - sol[instances].remove(ins)
    """
    destroyed = current.copy()
    instance_ids = random.sample(list(destroyed.instances.keys()),
                                 int(len(destroyed.instances) * destroy_ratio))
    for ins_id in instance_ids:
        ins = destroyed.instances[ins_id]
        destroyed.sol[ins['host_id']]['instance_ids'].remove(ins_id)
        for res in resources:
            for i in range(4):
                destroyed.sol[ins['host_id']]['instance_{}_{}'.format(
                    res, i)] -= ins['instance_{}_{}'.format(res, i)]
        flag = True
        for i in destroyed.sol[ins['host_id']]['instance_ids']:
            if destroyed.instances[i]['instance_app_name'] == ins[
                    'instance_app_name']:
                flag = False
                break
        if flag:
            destroyed.sol[ins['host_id']]['apps'].remove(
                ins['instance_app_name'])
        host_idx = np.argwhere(
            destroyed.host_matrix[:, 0] == ins['host_id'])[0][0]
        if len(destroyed.sol[ins['host_id']]['instance_ids']) == 0:
            destroyed.host_matrix = np.delete(destroyed.host_matrix,
                                              host_idx,
                                              axis=0)
            destroyed.sol.pop(ins['host_id'])
        else:
            destroyed.host_matrix[host_idx] -= dict2vec(ins, False)[0]
        ins['host_id'] = 0
    return destroyed


def random_repair(current, rnd_state=None):
    for ins_id, ins in current.instances.items():
        if ins['host_id'] == 0:
            while True:
                host_id = random.randint(1, 12000)
                if host_id in current.sol.keys():
                    if ins['instance_anti_affinity_app_name'] and ins[
                            'instance_anti_affinity_app_name'] in current.sol[
                                host_id]['apps']:
                        continue
                    flag = False
                    for res in resources:
                        for i in range(4):
                            if ins['instance_{}_{}'.format(
                                    res, i)] + current.sol[host_id][
                                        'instance_{}_{}'.format(
                                            res, i)] > current.sol[host_id][
                                                'host_{}'.format(res)]:
                                flag = True
                                break
                    if flag:
                        continue
                    current.sol[host_id]['apps'].add(ins['instance_app_name'])
                    for res in resources:
                        for i in range(4):
                            current.sol[host_id]['instance_{}_{}'.format(
                                res,
                                i)] += ins['instance_{}_{}'.format(res, i)]
                    current.sol[host_id]['instance_ids'].append(
                        ins['instance_id'])
                    ins['host_id'] = host_id
                    ins_vec = dict2vec(ins, False)
                    host_idx = np.argwhere(
                        current.host_matrix[:, 0] == host_id)[0][0]
                    current.host_matrix[host_idx] += ins_vec[0]
                    break
                else:
                    flag = False
                    for res in resources:
                        for i in range(4):
                            if ins['instance_{}_{}'.format(
                                    res, i)] > current.hosts[host_id][
                                        'host_{}'.format(res)]:
                                flag = True
                                break
                    if flag:
                        continue
                    host = current.hosts[host_id].copy()
                    host['instance_ids'] = [ins['instance_id']]
                    host['apps'] = set()
                    host['apps'].add(ins['instance_app_name'])
                    for res in resources:
                        for i in range(4):
                            host['instance_{}_{}'.format(
                                res,
                                i)] += ins['instance_{}_{}'.format(res, i)]
                    current.sol[host_id] = host
                    if type(current.host_matrix) is not np.ndarray:
                        current.host_matrix = dict2vec(host)
                    else:
                        current.host_matrix = np.r_[current.host_matrix,
                                                    dict2vec(host)]
                    ins['host_id'] = host_id
                    break
    return current


def random_repair2(current, rnd_state=None):
    for ins_id, ins in current.instances.items():
        if ins['host_id'] == 0:
            if type(current.host_matrix) is np.ndarray:
                host_matrix_ = current.host_matrix.copy()
                ins_vec = dict2vec(ins, False)
                host_matrix_ += ins_vec
                mask = np.ones(host_matrix_.shape[0], dtype=bool)
                for i in range(4):
                    for j in range(3):
                        mask = np.logical_and(
                            mask, host_matrix_[:, 3 * (i + 1) + j + 1] <=
                            host_matrix_[:, j + 1])
                idxs = np.argwhere(mask).reshape(-1)
                idxs = np.random.choice(idxs, len(idxs), False)
                flag = False
                for idx in idxs:
                    host_id = host_matrix_[idx][0]
                    if ins['instance_anti_affinity_app_name'] and ins[
                            'instance_anti_affinity_app_name'] in current.sol[
                                host_id]['apps']:
                        continue
                    current.sol[host_id]['instance_ids'].append(
                        ins['instance_id'])
                    current.sol[host_id]['apps'].add(ins['instance_app_name'])
                    for res in resources:
                        for i in range(4):
                            current.sol[host_id]['instance_{}_{}'.format(
                                res,
                                i)] += ins['instance_{}_{}'.format(res, i)]
                    current.host_matrix[idx] += ins_vec[0]
                    ins['host_id'] = host_id
                    flag = True
                    break
                if flag:
                    continue
            host_ids = set(list(range(1, 12001)))
            host_ids -= set(current.sol.keys())
            assert len(host_ids)
            host_ids = np.random.choice(list(host_ids), len(host_ids), False)
            flag1 = False
            for host_id in host_ids:
                for res in resources:
                    for i in range(4):
                        if ins['instance_{}_{}'.format(
                                res,
                                i)] > hosts[host_id]['host_{}'.format(res)]:
                            flag1 = True
                            break
                    if flag1:
                        break
                if not flag1:
                    break
            assert not flag1
            host = hosts[host_id].copy()
            host['instance_ids'] = [ins['instance_id']]
            ins['host_id'] = host_id
            host['apps'] = set()
            host['apps'].add(ins['instance_app_name'])
            for res in resources:
                for i in range(4):
                    host['instance_{}_{}'.format(
                        res, i)] += ins['instance_{}_{}'.format(res, i)]
            current.sol[host_id] = host
            # print(host_id, end=' ')
            if type(current.host_matrix) is not np.ndarray:
                current.host_matrix = dict2vec(host)
            else:
                current.host_matrix = np.r_[current.host_matrix, dict2vec(host)]

    return current


if __name__ == '__main__':
    hosts, instances = read_data()
    state = my_state(hosts, instances)
    init_sol = random_repair(state)
    print(init_sol.objective())
    init_sol.output_sol()
    alns = ALNS()
    alns.add_destroy_operator(random_destroy)
    alns.add_repair_operator(random_repair)
    alns.add_repair_operator(random_repair2)
    criterion = HillClimbing()
    weight_scheme = SimpleWeights([3, 2, 1, 0.5], 1, 2, 0.8)
    stop = MaxIterations(60)
    result = alns.iterate(init_sol, weight_scheme, criterion, stop)
    solution = result.best_state
    objective = solution.objective()
    solution.output_sol()
    print(objective)
    _, ax = plt.subplots(figsize=(12, 6))
    result.plot_objectives(ax=ax)
    plt.show()