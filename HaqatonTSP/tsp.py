### THIS UNPUBLISHED PYTHON CODE IS MADE BY WENYANG QIAN

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
import scipy

def get_exp_X_minus_Mean(X, power=1):
    X = np.array(X)
    return np.sum((X - np.mean(X))**power)/len(X)
def get_X_skewness(X):
    X = np.array(X)
    return np.sum(((X - np.mean(X))/np.std(X))**3)/len(X)


class TSPGraph:
    def __init__(self, num_nodes, adj_matrix="random", seed=0, edge_freq=1.0, max_weight=100):
        self.seed = seed
        self.nodes = num_nodes
        self.edge_freq = edge_freq
        self.max_weight = max_weight
        self.adj_matrix = adj_matrix
        if self.adj_matrix == "random":
            self.G, self.pos = get_random_graph(self.nodes, self.edge_freq, self.seed, self.max_weight)
            self.adj_matrix = self.get_adj_matrix()
        else:
            assert len(self.adj_matrix) == self.nodes
            self.G, self.pos = get_graph_from_adj_matrix(self.nodes, self.adj_matrix)
        self.sols = None
        self.best_dist = None
        self.alt_path = None

    def get_name(self):
        return f"TSP_G{self.nodes}_seed{self.seed}_maxw{self.max_weight}"

    def get_edge_weight(self, i, j):
        return self.G.get_edge_data(i,j,default={'weight': float('inf')})['weight']

    def get_all_edge_weights(self):
        edge_weights = []
        for i in range(self.nodes):
            for j in range(i+1, self.nodes):
                edge_weights.append(self.get_edge_weight(i, j))
        return edge_weights

    def get_max_weight(self):
        return max(self.G[i][j]['weight'] for i, j in self.G.edges())

    def get_skewness(self):
        return get_X_skewness(self.get_all_edge_weights())
    
    def get_skewness_from_scipy(self):
        return scipy.stats.skew(self.get_all_edge_weights())

    def get_edge_std(self):
        return np.std(self.get_all_edge_weights())
        

    def draw(self, sol=None, save_to_fig=None, title=None):
        return draw_graph(self.G, self.pos, sol=sol, color='c', save_to_fig=save_to_fig, title=title)

    def draw_with_bf_sol(self, save_to_fig=None, title=None):
        sol = self.get_sols()[1][0]
        return draw_graph(self.G, self.pos, sol=sol, color='c', save_to_fig=save_to_fig, title=title)

    def __len__(self):
        return self.nodes

    def __str__(self):
        return str(self.adj_matrix)

    def get_dist(self, route, fix_init=False):
        if fix_init == True:
            route = tuple([0] + [x+1 for x in route])
        assert self.nodes == len(route)
        idxs = list(range(1, self.nodes)) + [0] 
        return sum(self.get_edge_weight(route[i-1], route[i]) for i in idxs)

    def get_alt_path(self, fix_init=False):
        '''for row swap non-optimal start path'''
        _, sols = self.get_sols(fix_init=fix_init)
        for p in list(permutations(list(range(self.nodes-int(fix_init))))):
            # print(p, sols)
            if p not in sols:
                self.alt_path = p
                return self.alt_path

    def get_sols(self, keep_unique=True, fix_init=False):
        ''' Get TSP solutions by Bruteforce all permutations '''
        # if fix_init == True: assert keep_unique == True, "Fix_init is True requires that keep_unique is True."

        best_dist = float('inf'); sols = []; all_dists = []
        for p in list(permutations(list(range(self.nodes)))):
            d = self.get_dist(p)
            if d < best_dist:
                best_dist, sols = d, [p[:]]
            elif d == best_dist:
                sols.append(p[:])

        def reorder_sol_from_0(sol):
            i = sol.index(0)
            return sol[i:] + sol[:i]
        
        if keep_unique:
            sols = map(reorder_sol_from_0, sols)
            sols = map(tuple, sols)
            sols = list(set(sols))
            self.best_dist, self.sols = best_dist, sols     
        else:
            sols = list(map(tuple, sols))   
            self.best_dist, self.sols = best_dist, sols
        if fix_init == True:
            new_sols = [tuple([sol[i]- 1 for i in range(1, len(sol))]) for sol in self.sols if sol[0] == 0]  
            self.sols = new_sols
        
        if self.best_dist == float('inf'):
            return self.best_dist, None
        else:
            return self.best_dist, self.sols

    def get_starting_sol_for_RowSwap(self, fix_init):
        ''' get a single valid initial state but not be the true solution'''
        n_qubit = int((self.nodes - int(fix_init))**2)
        n = int(np.sqrt(n_qubit))
        true_sols = self.get_sols(keep_unique=False, fix_init=fix_init)[1]
        starting_sol = None
        for p in list(permutations(list(range(n)))):
            if p not in true_sols:
                starting_sol = p
                break
        if starting_sol == None: raise ValueError("Inapplication, no valid sol found!")
        return starting_sol
        
    def get_second_best_sol(self):
        best_dist = self.get_sols()[0]
        second_best_dist = float('inf')
        for p in list(permutations(list(range(self.nodes)))):
            d = self.get_dist(p)
            if d != best_dist and d < second_best_dist:
                second_best_dist = d
        return second_best_dist
            

    def get_adj_matrix(self):
        n = self.nodes
        w = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                temp = self.G.get_edge_data(i, j, default=0)
                if temp != 0:
                    w[i,j] = temp['weight']
        return w

    def get_cost(self, x, fix_init=False):
        ''' x is a binary decision variable, i.e., 0101011110'''
        total = 0
        n_city = n_time = len(self.G)
        if fix_init == False:
            n2 = n_city*n_time
            for i in range(n_city):
                for j in range(n_city):
                    if i == j or self.G[i][j]['weight'] == 0: continue
                    w = self.G[i][j]['weight']
                    for t in range(n_time):
                        idx1 = n_time*i+t
                        idx2 = n_time*j+((t+1) % n_time)
                        assert idx1 != idx2
                        total += w*int(x[idx1])*int(x[idx2])

        elif fix_init == True:
            n2 = (n_city-1)*(n_time-1)
            for i in range(1, n_city):
                for j in range(1, n_city):
                    if i == j or self.G[i][j]['weight'] == 0: continue
                    w = self.G[i][j]['weight']
                    # print(i, j, w)
                    for t in range(1, n_time-1):
                        idx1 = (n_time-1)*(i-1)+(t-1)
                        idx2 = (n_time-1)*(j-1)+(t)
                        # print(i, j, w, t, idx1, idx2)
                        total += w*int(x[idx1])*int(x[idx2])
            # print(total)
            for i in range(1, n_city):
                w = self.G[0][i]['weight']
                # print(i, w)
                idx1 = (n_time-1)*(i-1)+1-1
                idx2 = (n_time-1)*(i-1)+n_city-1-1
                total +=w*(int(x[idx1]) + int(x[idx2]))

        return total


def draw_graph(G, pos=None, sol=None, color='c', save_to_fig=None, title=None):
    ''' sol is city visiting order; pos fix graph position '''

    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)

    if pos == None: pos = nx.spring_layout(G)
    if sol == None: 
        # default_axes = plt.axes(frameon=True)
        nx.draw_networkx(G, node_color=color, edge_color='b', node_size=800, alpha=.8, pos=pos, ax=ax)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=pos, font_color='b', font_size=12, edge_labels=edge_labels, ax=ax)
    else:
        assert len(sol) == len(G)
        G2 = nx.DiGraph()
        G2.add_nodes_from(G)
        n = len(sol)
        for i in range(n):
            j = (i + 1) % n
            assert (sol[i], sol[j]) in G.edges()
            G2.add_edge(sol[i], sol[j], weight=G[sol[i]][sol[j]]['weight'])
        # default_axes = plt.axes(frameon=True)
        nx.draw_networkx(G2, node_color=color, edge_color='r', node_size=800, alpha=.8, pos=pos, ax=ax)
        edge_labels = nx.get_edge_attributes(G2, 'weight')
        nx.draw_networkx_edge_labels(G2, pos, font_color='r', font_size=12, edge_labels=edge_labels, ax=ax)

    if title != None and title != "":
        plt.title(title)
    if save_to_fig != None and save_to_fig != "":
        plt.savefig(save_to_fig, dpi=300, bbox_inches='tight')
    plt.show()
    
def get_random_graph(n, edge_freq=1.0, seed=0, max_weight=5):
    np.random.seed(seed)
    elist = []
    for i in range(n-1):
        for j in range(i+1, n):
            if np.random.random() <= edge_freq:
                elist.append((i, j, np.random.randint(1, max_weight+1)))
    G = nx.Graph()
    G.add_weighted_edges_from(elist)
    pos = nx.spring_layout(G)
    return G, pos

def get_graph_from_adj_matrix(n, adj_matrix, seed=0):
    ''' This is correct but plotting graph from adj_matrix is not optimized. (TODO later)
    '''
    np.random.seed(seed)
    elist = []
    for i in range(n-1):
        for j in range(i+1, n):
            if adj_matrix[i][j] != 0:
                elist.append((i, j, adj_matrix[i][j]))
    G = nx.Graph()
    G.add_weighted_edges_from(elist)
    pos = nx.spring_layout(G)
    return G, pos