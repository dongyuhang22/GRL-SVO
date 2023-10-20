import sympy
from copy import deepcopy
from fractions import Fraction


def is_number(s):
    try:
        Fraction(s)
        return True
    except ValueError:
        pass
    return False


# term is a monomial -> a map from var:str to degree:int
class Term:
    # accepts a t:str
    def __init__(self, t) -> None:
        # coe: term's coefficient
        self.coe = '0'
        # vars: a dictionary, every element is a map of {var:degree}      
        self.vars = {}
        self.str_term = t
        # vars list
        self.str_vars = []
        self.is_number = False
        
        self.process(t)

    # process to get self.vars:map of {var:degree}
    def process(self, t):
        if is_number(t):
            self.coe = t
            self.is_number = True
            if t[0] != '+' and t[0] != '-':
                self.str_term = '+' + t
        else:
            # make term:str into term:sympy.poly
            term_sympy = sympy.poly(t)

            # term_data: ((deg, deg, ... ,deg), coefficient), variables' deg arrange in lex order
            term_data = term_sympy.terms()[0]
            # term_var: a list of variables:str in the term, arrange in lex order 
            term_var = list(term_sympy.free_symbols)
            # make var:sympy.var into var:str
            for num in range(len(term_var)):
                term_var[num] = str(term_var[num])
            # arrange in lex order
            term_var.sort()

            self.str_vars = term_var

            # process
            self.coe = str(term_data[1])

            for num in range(len(term_var)):
                self.vars[term_var[num]] = term_data[0][num]

            raw_str_term = str(term_sympy)
            self.str_term = raw_str_term[5:raw_str_term.find(',')]
            if self.str_term[0] != '-' and self.str_term[0] != '+':
                self.str_term = '+' + self.str_term

    # get the degree of var
    def is_num(self):
        return self.is_number

    def get_var_degree(self, var):
        if var not in self.vars:
            return 0
        else:
            return self.vars[var]

    # get the degree of term
    def get_degree(self):
        tdeg = 0
        for var in self.vars:
            tdeg += self.vars[var]
        return tdeg

    def get_coe(self):
        return self.coe

    # get vars
    def get_vars(self):
        return self.str_vars

    # get the number of vars
    def get_var_size(self):
        return len(self.vars)

    def __str__(self):
        return self.str_term


class Poly:
    # accepts a p:sympy.poly 
    def __init__(self, p) -> None:
        self.data = deepcopy(p)
        # information
        self.vars = []
        self.terms = [] # a list of terms:class Term 
        self.degree = 0
        self.var_degree = {}
        self.constant = '0'
        self.str_terms = []

        # preprocess to get the information
        self.process(self.data)

    def process(self, p):
        # vars: a list of variables:str in the polynomial, arrange in lex order 
        self.vars = list(p.free_symbols)
        # make var:sympy.var into var:str
        for num in range(len(self.vars)):
            self.vars[num] = str(self.vars[num])
        # arrange in lex order
        self.vars.sort()

        # terms:str can be split by '+'/'-' in sympy.poly
        # str_poly: poly in string
        str_poly = str(p)[5:str(p).find(',')]   # eg: term1 + term2 - term3 + constant
        str_poly = str_poly.replace('- ', '-')  # eg: term1 + term2 -term3 + constant
        str_poly = str_poly.replace('+ ', '+')  # eg: term1 +term2 -term3 +constant
        self.terms = str_poly.split(' ')
        
        if self.terms[0][0] != '-' and self.terms[0][0] != '+':
            self.terms[0] = '+' + self.terms[0]
        self.str_terms = deepcopy(self.terms)
        
        # constant
        if is_number(self.terms[-1]):
            self.constant = self.terms[-1]

        # self.terms is a list of terms:class Term
        for num in range(len(self.terms)):
            self.terms[num] = Term(self.terms[num])

        # degree(total_degree)
        self.degree = p.total_degree()
        
        # var_degree
        for num in range(len(self.vars)):
            self.var_degree[self.vars[num]] = p.degree(self.vars[num])

    def get_vars(self):
        # return vars in lex order
        return self.vars

    def get_var_size(self):
        return len(self.vars)

    def get_terms(self):
        return self.terms

    def get_str_terms(self):
        return self.str_terms

    # the size of terms
    def get_term_size(self):
        return len(self.terms)
    
    # the degree(total_degree) of the polynomial
    def get_degree(self):
        return self.degree
    
    # the max degree of var
    def get_var_degree(self, var):
        if var in self.vars:
            return self.var_degree[var]
        else:
            return 0

    # the constant of the polynomial
    def get_constant(self):
        return self.constant

    # get polynomial in string
    def __str__(self):
        return str(self.data)[5:str(self.data).find(',')]


class Instance:
    # accepts polys(non_number_ps): list of  !!! non number !!!  str poly(may has repeat str poly)
    def __init__(self, polys) -> None:
        # a list of Poly
        self.polys = []
        # a list of Vars (str)
        # arrange by dictionary order, and not repeat
        self.vars = []
        self.terms = []
        self.str_terms = []
        self.str_polys = []
        
        self.process(polys)

    # process: polys -> a list of Poly (self.polys)
    def process(self, polys):
        # polys
        str_ps = set()
        for pp in polys:
            ps = sympy.poly(pp)
            str_ps.add((str(ps))[5:str(ps).find(',')])

        str_ps = list(str_ps)
        str_ps.sort()

        for strp in str_ps:
            self.polys.append(Poly(sympy.poly(strp)))

        # vars
        vars = set()
        terms = set()
        
        for poly in self.polys:
            vars = vars | set(poly.get_vars())
            terms = terms | set(poly.get_str_terms())
            # print(str(poly))
            # str_ps = str_ps | {str(poly)}
        
        self.vars = list(vars)
        self.str_terms = list(terms)
        self.str_polys = str_ps
        self.vars.sort()
        self.str_terms.sort()
        # self.str_polys.sort()

        for term in self.str_terms:
            # print(term)
            self.terms.append(Term(term))

    def get_polys(self):
        return self.polys

    def get_str_polys(self):
        return self.str_polys    

    def get_vars(self):
        return self.vars

    def get_terms(self):
        return self.terms

    def get_str_terms(self):
        return self.str_terms

    def __getitem__(self, index):
        return self.polys[index]

    def __setitem__(self, index, value):
        self.polys[index] = value

    def __len__(self):
        return len(self.polys)

    def __str__(self):
        return ', '.join(self.str_polys)
        

# parser reads a file and transforms it to an instance.
class Parser(object):
    def __init__(self) -> None:
        self.data = None

    def clear(self):
        self.data = None
    
    # transform file to an instance
    def parse(self, file):
        self.clear()
        total = ""
        with open(file, 'r') as f:
            lines = f.readlines()

        # combine together
        for i in lines:
            total += i.strip()
        
        if total[0] == '[' or total[0] == '{':
            total = total[1:-1]
        
        # x^2+1,x+y,...
        ps = total.split(',')
        if len(ps) == 0: return None

        # 'x+y' != 'y+x', but sympy.poly('x+y') == sympy.poly('y+x')
        non_number_ps = list()
        for pp in ps:
            if not is_number(pp):
                non_number_ps.append(pp)

        # all polys are constants
        if len(non_number_ps) == 0:
            return None
        
        self.data = Instance(non_number_ps)
        
        return deepcopy(self.data)
