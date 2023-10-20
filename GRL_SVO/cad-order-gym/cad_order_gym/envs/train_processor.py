import torch
from cad_order_gym.envs.train_parser import Instance, Parser
import itertools
from torch_geometric.data import Data
import copy
import torch.nn.functional as F
import numpy as np


# Processor reads a file and transforms it to an data(graph) with initial embedding.
class Processor(object):
    def __init__(self) -> None:

        # parser 
        self.parser = Parser()
        
        # poly set's vars' features
        self.variables_features = []
        
        # Instance
        self.instance = None
        
        # for mapping: str (name of variable) to int (vertex index)
        self.name2vari = {}
        
        # a list which elements are every poly's vars: [[vars of poly1], [vars of poly2], ...]
        self.every_polys_vars = []

        self.masks = {
                        'train': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        'Chen': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        'england': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        'max': [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                        'sum': [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        'prop':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        'var': [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        'term': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
                        'poly': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        'degree': [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                        'not_degree': [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
                     }
        for i in range(14):
            self.masks['feature_' + str(i + 1)] = [0] * i + [1] + [0] * (13 - i)

    # clear processor, except max_list
    def clear(self):
        self.variables_features.clear()
        self.instance = None
        self.name2vari.clear()
        self.every_polys_vars.clear()

    # normalize the features
    # include all possible norm mode
    #[[1,2,3,3],
    # [1,2,3,6]]
    def normalize(self, tensor, norm_mode):
        if norm_mode == 'sigmoid':
            return torch.sigmoid(tensor)

        elif norm_mode == 'tanh':
            return torch.tanh(tensor)

        elif norm_mode == 'local_maximum':
            # strange, all zero?
            return tensor / (torch.max(torch.abs(tensor), dim=0)[0] + 1e-07)

        elif norm_mode == 'zscore':
            # print((tensor - torch.mean(tensor, dim=0)) / (torch.std(tensor, dim=0) + 1e-07))
            if tensor.shape[0] == 1:
                return tensor / (torch.max(torch.abs(tensor), dim=0)[0] + 1e-07)
            else:
                return (tensor - torch.mean(tensor, dim=0)) / (torch.std(tensor, dim=0) + 1e-07)

        elif norm_mode == 'row':
            return F.normalize(tensor, p=2, dim=1)

        elif norm_mode == 'min_max':
            return (tensor - torch.min(tensor, dim=0)[0]) / (torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0] + 1e-07)

        elif norm_mode == 'max_abs':
            return 2 * ((tensor - torch.min(tensor, dim=0)[0]) / (torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0] + 1e-07)) - 1

        elif norm_mode == 'max_mean':
            return (tensor - torch.mean(tensor, dim=0)) / (torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0] + 1e-07)
        
        elif norm_mode == 'None':
            return tensor

        elif norm_mode == 'zscore_row':
            if tensor.shape[0] == 1:
                return tensor / (torch.max(torch.abs(tensor), dim=0)[0] + 1e-07)
            else:
                return (tensor - torch.mean(tensor, dim=1).unsqueeze(dim=1)) / (torch.std(tensor, dim=1).unsqueeze(dim=1) + 1e-07)

        else:
            raise ValueError('invalid normalize mode!')


    # reads a file and transforms it to an data(graph) with initial embedding
    # embeding has four basic type: 'Chen', 'England', Huang', 'Our', and we can cat some of them
    # default: 'CHO' : means 'Chen' cat 'Huang' cat 'Our' (remove repetition) 
    # all option: 'C', 'E', 'H', 'O', 'CE', 'CH', 'CO', 'EH', 'EO', 'HO', 'CEH', 'CEO', 'CHO', 'EHO', 'CEHO'
    def process_file(self, file, norm_mode, embedding, edge, feature_mode):
        self.clear()
        # parse
        self.instance = copy.deepcopy(self.parser.parse(file))
        # store every poly's vars in a list
        # print(str(self.instance))
        for poly in self.instance.get_polys():
            self.every_polys_vars.append(poly.get_vars())
        # make data
        return self.main(self.instance, norm_mode, embedding, edge, feature_mode)

    # reads polys:transforms it to an data(graph) with initial embedding
    # accepts polys(non_number_ps): list of  !!! non number !!!  str poly(may has repeat str poly)
    def process_polys(self, polys, norm_mode, embedding, edge, feature_mode):
        self.clear()
        # parse
        self.instance = Instance(polys)
        # store every poly's vars in a list
        for poly in self.instance.get_polys():
            self.every_polys_vars.append(poly.get_vars())
        # make data
        return self.main(self.instance, norm_mode, embedding, edge, feature_mode)

    # generate the data(graph) with vertex embeddings
    def main(self, instance, norm_mode, embedding, edge, feature_mode):

        # 1. variable feature vector
        for var in instance.get_vars():
            # map var:str to vertex index:int
            self.name2vari[var] = len(self.variables_features)
            # append vertex's feature
            if embedding:
                self.variables_features.append(self.variable_node(instance, var, feature_mode))
            else:
                self.variables_features.append([1] * 14)
        # normalize
        x = self.normalize(torch.tensor(self.variables_features, dtype=torch.float), norm_mode)
        # x = torch.tensor(self.variables_features, dtype=torch.float)

        # 2. edge index
        # if there exists f belongs to polys, s.t. x_i and x_j both appears in f
        # then there is an undirected edge between x_i and x_j
        
        # pointer: in order to prevent repeating set edges
        pointer = set()
        edge_index = [[], []]
        # for every poly:
        if edge:
            for num in range(len(instance.get_polys())):
                # for every vars pair appear in the same poly:
                for (a, b) in itertools.combinations(self.every_polys_vars[num], 2):
                    # if the edge hasn't been set, set edge
                    # note: vars in self.every_polys_vars[num] arrange in lex order
                    # so just consider if (a, b) appears in pointer is enough 
                    if (a, b) not in pointer:
                        pointer.add((a, b))
                        edge_index[0].append(self.name2vari[a])
                        edge_index[0].append(self.name2vari[b])
                        edge_index[1].append(self.name2vari[b])
                        edge_index[1].append(self.name2vari[a])
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # 3. generate graph
        graph = Data(x=x, edge_index=edge_index)
        # print(x)
        return graph

    # generate vertex's embedding
    def variable_node(self, instance, var, mode):
        # instance: Instance
        # var: str
        # feature: str

        # Chen:
        Chen1 = 0  # 1.  | {{x, x_} | x and x_ appear in one polynomial} |: Chen1
        Chen2 = 0  # 2.  number of polynomials with x: Chen2
        Chen3 = 0  # 3.  max degree of variable x: Chen3
        Chen4 = 0  # 4.  sum of max degree of variable x: Chen4
        Chen5 = 0  # 5.  max total degree of leading coeff of variable x as main variable: Chen5
        Chen6 = 0  # 6.  max number of monomials with x: Chen6
        Chen7 = 0  # 7.  max total degree of monomials with x: Chen7
        Chen8 = 0  # 8.  sum of total degree of monomials with x: Chen8
        Chen9 = 0  # 9.  sum of total degree of leading coeff of variable x as main variable: Chen9
        Chen10 = 0 # 10. sum of number of monomials with x: Chen10
        real_8 = 0 # sum of degree of x of monomials with x: Chen8
        # Our:
        Our1 = 0   # 1.  max number of other variables in the same monomial: Our1
        Our2 = 0   # 2.  max number of other variables in the same polynomial: Our2

        # Huang:
        # 1.  number of polynomials: Huang1 (here can not be used)
        # 2.  maximum total degree of polynomials: Huang2 (here can not be used)
        Huang3 = 0 # 3.  maximum degree of x among all polynomials : Huang3 (same as Chen3)
        Huang4 = 0 # 4.  proportion of polynomials with x to all the polynomials: Huang4
        Huang5 = 0 # 5.  proportion of terms with x to all the terms: Huang5

        # England: TODO


        # calculate process:

        # the set stored (var, var_) that appear in one poly
        Chen1_set = set()
        
        # calculate with every poly:
        idx = 0
        polys = instance.get_polys()
        while idx < len(polys):

            # print(polys[idx])

            # Chen2: number of polynomials with x
            varlist = instance.get_polys()[idx].get_vars()
            if var in varlist: 
                Chen2 += 1
                # add the vars that appear in one polynomial
                for v in varlist:
                    if v != var:
                        Chen1_set.add(v)

            # var's deg in this poly
            deg = polys[idx].get_var_degree(var)

            # Chen3: max degree of variable x
            if deg > Chen3: Chen3 = deg

            # Chen4: sum of max degree of variable x
            Chen4 += deg

            # if var in polynomial, then calculate the leading term's feature, and term's feature, else we pass
            if var in varlist:
                # according to the max_degree to jutisfy which term is leading term
                max_degree = 0
                for t in polys[idx].get_terms():
                    if t.get_var_degree(var) > max_degree:
                        max_degree = t.get_var_degree(var)

                # calculate leading coefficient's degree
                degree_leading_cof = 0
                for t in polys[idx].get_terms():
                    if t.get_var_degree(var) == max_degree and t.get_degree() - max_degree > degree_leading_cof:
                        degree_leading_cof = t.get_degree() - max_degree

                # Chen5: max total degree of leading coeff of variable x as main variable
                if degree_leading_cof > Chen5: Chen5 = degree_leading_cof

                # Chen9: sum of total degree of leading coeff of variable x as main variable
                Chen9 += degree_leading_cof

                # print(Chen5)
                # print(Chen9)

                term_with_var_max_degree = 0  # max total degree of terms with x in this polynomial
                term_num_with_var = 0    # number of terms with x in this polynomial
                term_max_num_vars = 0  # max vars in the same terms in this polynomial

                for t in polys[idx].get_terms():
                    if not t.get_var_degree(var) == 0:
                        term_num_with_var += 1

                        # the term with var's max degree 
                        if t.get_degree() > term_with_var_max_degree: term_with_var_max_degree = t.get_degree()
                        
                        # max number of vars in the term with var
                        if len(t.get_vars()) > term_max_num_vars: term_max_num_vars = len(t.get_vars())
                        
                        # Chen8: sum of total degree of monomials with x
                        Chen8 += t.get_degree()
                        real_8 += t.get_var_degree(var)

                # Chen7: max total degree of terms with x
                if term_with_var_max_degree > Chen7: Chen7 = term_with_var_max_degree
                
                # Chen6: max number of terms with x
                if term_num_with_var > Chen6: Chen6 = term_num_with_var
                
                # Our1: max number of other variables in the same monomial
                if term_max_num_vars - 1 > Our1: Our1 = term_max_num_vars - 1 
                
                # Chen10: sum of number of monomials with x
                Chen10 += term_num_with_var

            # Our2: max number of other variables in the same polynomial
            if (var in varlist) and (len(varlist) - 1 > Our2):
                Our2 = len(varlist) - 1

            idx += 1

        # Chen1: | {{x, x_} | x and x_ appear in one polynomial} |
        Chen1 = len(Chen1_set)
        
        # Huang3: same as Chen3
        Huang3 = Chen3
        
        # Huang4: proportion of polynomials with x to all the polynomials
        Huang4 = Chen2 / len(polys)

        all_terms = 0
        for poly in polys:
            all_terms += len(poly.get_terms())
        
        # Huang5: proportion of terms with x to all the terms
        Huang5 = Chen10 / all_terms

        total_features = np.array([Chen1, Chen2, Chen3, Chen4, Chen5, Chen6, Chen7, Chen8, Chen9, Chen10, Huang4, Huang5, Our1, Our2])
        
        if mode == 'england_real':
            return [Chen3, Huang4, Huang5]
        elif mode == 'Chen_real':
            return [Chen1, Chen2, Chen3, Chen4, Chen5, Chen6, Chen7, real_8, Chen9, Chen10]
        else:
            return (total_features * self.masks[mode]).tolist()


if __name__ == '__main__':
    p = Processor()
    ins = Instance(['69*x1^2*x3^2-77*x1*x2^2-93*x1*x2+7*x2^2+91*x3-3', '79*x1^2*x3^2+61*x1^2*x3+12*x2^2*x3-65*x1*x2-72*x1*x3+61'])
    for i in ['x1', 'x2', 'x3']:
        print(p.variable_node(ins, i))
