import json
import numpy as np
import random
import copy
import time
from collections import defaultdict

resources = ['cpu', 'memory', 'disk']


class Host:

    def __init__(self, cpu, memory, disk):
        self.cpu = cpu
        self.memory = memory
        self.disk = disk

    def __hash__(self):
        return int('{}{}{}'.format(self.cpu, self.memory, self.disk))

    def __eq__(self, other):
        return self.cpu == other.cpu and self.memory == other.memory and self.disk == other.disk


class Solution:
    """
        hosts
        key: host_id
        value:
            vec_idx
            instance_ids
            apps
    """

    def __init__(self):
        self.hosts = {}
        self.host_matrix = None
        self.ratios = None

    def test_ins(self, ins, host):
        host_id = host['host_id']
        score = 0
        i = np.array([ins['instance_cpu_{}'.format(i)]
                      for i in range(4)]) / host['host_cpu']
        if host_id in self.hosts.keys():
            self.ratios[self.hosts[host_id]['vec_idx']] += i
            score = self.get_score()
            self.ratios[self.hosts[host_id]['vec_idx']] -= i
        else:
            self.ratios = np.r_[self.ratios, [i]]
            score = self.get_score()
            self.ratios = self.ratios[:-1]
        return score

    def get_score(self):
        score1 = 12000 - self.host_matrix.shape[0]
        score2 = 0
        mean_std = np.mean(self.ratios.std(axis=0))
        if mean_std < 2000:
            score2 = 2000 - mean_std
        score = score1 + score2
        # print(score1)
        # print(mean_std)
        return score

    def add_ins(self, ins, host):
        host_id = host['host_id']
        i = np.array([ins['instance_cpu_{}'.format(i)]
                      for i in range(4)]) / host['host_cpu']
        if host_id in self.hosts.keys():
            if type(ins['instance_id']) is int:
                self.hosts[host_id]['instance_ids'].append(ins['instance_id'])
            elif type(ins['instance_id']) is list:
                self.hosts[host_id]['instance_ids'].extend(ins['instance_id'])
            if type(ins['instance_app_name']) is str:
                self.hosts[host_id]['apps'].add(ins['instance_app_name'])
            elif type(ins['instance_app_name']) is set:
                self.hosts[host_id]['apps'] |= ins['instance_app_name']
            self.host_matrix[self.hosts[host_id]['vec_idx']] += dict2vec(
                ins)[0]
            self.ratios[self.hosts[host_id]['vec_idx']] += i
        else:
            self.hosts[host_id] = dict()
            self.hosts[host_id]['vec_idx'] = len(self.hosts) - 1
            if type(ins['instance_id']) is int:
                self.hosts[host_id]['instance_ids'] = [ins['instance_id']]
            elif type(ins['instance_id']) is list:
                self.hosts[host_id]['instance_ids'] = ins['instance_id']
            if type(ins['instance_app_name']) is str:
                self.hosts[host_id]['apps'] = {ins['instance_app_name']}
            elif type(ins['instance_app_name']) is set:
                self.hosts[host_id]['apps'] = ins['instance_app_name']
            host_vec = dict2vec(host) + dict2vec(ins)
            if type(self.host_matrix) is not np.ndarray:
                self.host_matrix = host_vec
                self.ratios = np.array([i])
            else:
                self.host_matrix = np.r_[self.host_matrix, host_vec]
                self.ratios = np.r_[self.ratios, [i]]

    def print_sol(self,
                  path="/home/admin/workspace/job/output/data/results.jsonl"):
        with open(path, 'w', encoding='utf-8') as f:
            for host_id, host in self.hosts.items():
                for i in host['instance_ids']:
                    d = {'host_id': host_id, 'instance_id': i}
                    line = json.dumps(d)
                    f.write(line)
                    f.write('\n')

    def get_possible_host_idxs(self, ins):
        ins_vec = dict2vec(ins)
        self.host_matrix += ins_vec
        mask = np.ones(self.host_matrix.shape[0], dtype=bool)
        for i in range(4):
            for j in range(3):
                mask = np.logical_and(
                    mask, self.host_matrix[:, 3 * (i + 1) + j + 1] <=
                    self.host_matrix[:, j + 1])
        idxs = np.argwhere(mask).reshape(-1)
        self.host_matrix -= ins_vec
        return idxs

    def __len__(self):
        return len(self.hosts)

    def copy(self):
        return copy.deepcopy(self)

    def find_max_idx(self, ins, idxs):
        i = np.array([ins['instance_cpu_{}'.format(i)] for i in range(4)])
        min_std = 2000
        min_idx = -1
        for idx in idxs:
            j = i / self.host_matrix[idx][1]
            self.ratios[idx] += j
            std = np.mean(self.ratios.std(axis=0))
            if std < min_std:
                min_std = std
                min_idx = idx
            self.ratios[idx] -= j
        host_id = self.host_matrix[min_idx][0]
        self.add_ins(ins, hosts[host_id])


def read_data(data_path='/home/admin/workspace/job/input/test.jsonl'):
    hosts = {}
    instances = {}
    hosts_ = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            d = json.loads(line)
            if d['type'] == 'host':
                hosts[d['host_id']] = d
                host = Host(d['host_cpu'], d['host_memory'], d['host_disk'])
                if host in hosts_.keys():
                    hosts_[host].add(d['host_id'])
                else:
                    hosts_[host] = {d['host_id']}
            elif d['type'] == 'instance':
                instances[d['instance_id']] = d
            line = f.readline()
    return hosts, instances, hosts_


def dict2vec(d, group=False):
    return np.array([[
        d['host_id'] if not group else 0,
        d['host_cpu'] if not group else 0,
        d['host_memory'] if not group else 0,
        d['host_disk'] if not group else 0,
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


def epsilon_greedy_sol(hosts, groups, instances, hosts_, epsilon=0.85):
    sol = Solution()
    num = 0
    last_time = time.time()
    instance_ids = []
    for group in groups.values():
        num += 1
        if (num % 1000 == 0):
            time1 = time.time()
            print(num, time1 - last_time)
            print(len(instance_ids))
            last_time = time1
        # if num > 10000 and num % 100 == 0:
        #     print(num, time.time() - last_time)

        # ?????????????????????host
        if len(sol) == 0:
            while True:
                host_id = random.choice(list(range(1, 12001)))
                flag = False
                for res in resources:
                    for i in range(4):
                        if group['instance_{}_{}'.format(
                                res,
                                i)] > hosts[host_id]['host_{}'.format(res)]:
                            flag = True
                            break
                if flag:
                    continue
                sol.add_ins(group, hosts[host_id])
                hosts_[Host(hosts[host_id]['host_cpu'],
                            hosts[host_id]['host_memory'],
                            hosts[host_id]['host_disk'])].remove(host_id)
                break
        else:
            # ???????????????????????????
            idxs = sol.get_possible_host_idxs(group)
            idxs_ = []
            for idx in idxs:
                host_id = sol.host_matrix[idx][0]
                if len(group['instance_anti_affinity_app_name']
                       & sol.hosts[host_id]['apps']):
                    continue
                idxs_.append(idx)
            if len(idxs_) and random.random() <= epsilon:
                # ????????? epsilon??????????????????
                max_score = 0
                max_idx = -1
                for idx in idxs:
                    host_id = sol.host_matrix[idx][0]
                    score = sol.test_ins(group, hosts[host_id])
                    if score > max_score:
                        max_score = score
                        max_idx = idx
                host_id = sol.host_matrix[max_idx][0]
                sol.add_ins(group, hosts[host_id])
            else:
                # ???????????????1-epsilon ?????????????????? ????????????
                max_score = 0
                max_h = None
                for h in hosts_.keys():
                    flag = False
                    for i in range(4):
                        if group['instance_cpu_{}'.format(i)] > h.cpu:
                            flag = True
                            break
                        if group['instance_memory_{}'.format(i)] > h.memory:
                            flag = True
                            break
                        if group['instance_disk_{}'.format(i)] > h.disk:
                            flag = True
                            break
                    if flag or len(hosts_[h]) == 0:
                        continue
                    host_id = random.choice(list(hosts_[h]))
                    score = sol.test_ins(group, hosts[host_id])
                    if score > max_score:
                        max_score = score
                        max_h = h
                if max_h is None:
                    instance_ids += group['instance_id']
                    continue
                host_id = random.choice(list(hosts_[max_h]))
                hosts_[max_h].remove(host_id)
                sol.add_ins(group, hosts[host_id])
    for ins_id in instance_ids:
        ins = instances[ins_id]
        idxs = sol.get_possible_host_idxs(ins)
        idxs_ = []
        for idx in idxs:
            host_id = sol.host_matrix[idx][0]
            if ins['instance_anti_affinity_app_name'] and ins[
                    'instance_anti_affinity_app_name'] in sol.hosts[host_id][
                        'apps']:
                continue
            idxs_.append(idx)
        if len(idxs_) and random.random() <= epsilon:
            # ????????? epsilon??????????????????
            max_score = 0
            max_idx = -1
            for idx in idxs:
                host_id = sol.host_matrix[idx][0]
                score = sol.test_ins(ins, hosts[host_id])
                if score > max_score:
                    max_score = score
                    max_idx = idx
            host_id = sol.host_matrix[max_idx][0]
            sol.add_ins(ins, hosts[host_id])
        else:
            # ???????????????1-epsilon ?????????????????? ????????????
            max_score = 0
            max_h = None
            for h in hosts_.keys():
                flag = False
                for i in range(4):
                    if ins['instance_cpu_{}'.format(i)] > h.cpu:
                        flag = True
                        break
                    if ins['instance_memory_{}'.format(i)] > h.memory:
                        flag = True
                        break
                    if ins['instance_disk_{}'.format(i)] > h.disk:
                        flag = True
                        break
                if flag:
                    continue
                host_id = random.choice(list(hosts_[h]))
                score = sol.test_ins(ins, hosts[host_id])
                if score > max_score:
                    max_score = score
                    max_h = h
            host_id = random.choice(list(hosts_[max_h]))
            hosts_[max_h].remove(host_id)
            sol.add_ins(ins, hosts[host_id])
    return sol



def get_groups(instances, count=5, epsilon=0.85):
    """
        groups:
            group_id -> 
                instance_id
                instance_res_i
                instance_app_name
                instance_anti_affinity_app_name

    """
    groups = {}
    group_id = 0
    ids = []
    ins_ids = list(instances.keys())
    ins_ids = np.random.choice(ins_ids, len(ins_ids), False)
    for i in ins_ids:
        ins = instances[i]
        if len(ids) == 0:
            group = defaultdict(int)
            group["instance_id"] = [ins['instance_id']]
            group["instance_app_name"] = {ins['instance_app_name']}
            group["instance_anti_affinity_app_name"] = set()
            if ins['instance_anti_affinity_app_name']:
                group['instance_anti_affinity_app_name'].add(
                    ins['instance_anti_affinity_app_name'])
            for res in resources:
                for i in range(4):
                    group['{}_{}'.format(res,
                                         i)] += ins['instance_{}_{}'.format(
                                             res, i)]
            groups[group_id] = group
            ids.append(group_id)
            group_id += 1
        else:
            flag = False
            if random.random() < epsilon:
                ids_ = np.random.choice(ids, len(ids), False)
                for id in ids_:
                    if ins['instance_anti_affinity_app_name'] and ins[
                            'instance_anti_affinity_app_name'] in groups[id][
                                'instance_app_name']:
                        continue
                    else:
                        groups[id]["instance_id"].append(ins['instance_id'])
                        groups[id]["instance_app_name"].add(
                            ins['instance_app_name'])
                        if ins['instance_anti_affinity_app_name']:
                            groups[id]['instance_anti_affinity_app_name'].add(
                                ins['instance_anti_affinity_app_name'])
                        for res in resources:
                            for i in range(4):
                                groups[id]['instance_{}_{}'.format(
                                    res,
                                    i)] += ins['instance_{}_{}'.format(res, i)]
                        if len(groups[id]["instance_id"]) == count:
                            ids.remove(id)
                        flag = True
                        break
            if not flag:
                group = defaultdict(int)
                group["instance_id"] = [ins['instance_id']]
                group["instance_app_name"] = {ins['instance_app_name']}
                group["instance_anti_affinity_app_name"] = set()
                if ins['instance_anti_affinity_app_name']:
                    group['instance_anti_affinity_app_name'].add(
                        ins['instance_anti_affinity_app_name'])
                for res in resources:
                    for i in range(4):
                        group['instance_{}_{}'.format(
                            res, i)] += ins['instance_{}_{}'.format(res, i)]
                groups[group_id] = group
                ids.append(group_id)
                group_id += 1

    return groups




if __name__ == '__main__':
    hosts, instances, hosts_ = read_data()
    groups = get_groups(instances)
    sol = epsilon_greedy_sol(hosts, groups, instances, hosts_)
    score = sol.get_score()
    print(score)
    sol.print_sol()