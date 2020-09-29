from nltk import CFG, ChartParser, grammar
import numpy as np

class ContextFreeGrammarProcessor:
  def __init__(self, grammar_string):
    self.grammar = CFG.fromstring(grammar_string)
    self.parser = ChartParser(self.grammar)
    self.tokenizer = self._get_tokenizer()

  @property
  def start_index(self):
    return self.grammar.start()

  @property
  def last_index(self):
    return self.grammar.productions()[-1].lhs()

  @property
  def total_productions(self):
    return len(self.grammar.productions())

  @property
  def lhs(self):
    return list(expression.lhs() for expression in self.grammar.productions())

  @property
  def unique_lhs(self):
    return list(dict.fromkeys(self.lhs))

  @property
  def unique_lhs_dictionary(self):
    return {left_rule: idx for idx, left_rule in enumerate(self.unique_lhs)}
    
  @property
  def production_dictionary(self):
    return {production: idx for idx, production in enumerate(self.grammar.productions())}

  def get_masks(self):
    mask = np.zeros((len(self.unique_lhs), self.total_productions))
    for idx, symbol in enumerate(self.unique_lhs):
      mask[idx] = np.array([symbol == symbol_lhs for symbol_lhs in self.lhs], dtype=int)
    return mask 

  def get_masks_idx(self):
    temp_mask = self.get_masks()
    res = [np.where(temp_mask[:, idx]==1)[0][0] for idx in range(self.total_productions)]
    return np.array(res)
    
  def _get_tokenizer(self):
    #TODO cleanup the function --> improve the logic  
    long_tokens = list(filter(lambda symbol: len(symbol) > 1, self.grammar._lexical_index.keys()))
    replacements = ['$','%','^'] # ,'&']
    assert len(long_tokens) == len(replacements)
    # for token in replacements: 
    #     assert not cfg._lexical_index.has_key(token)
    
    def tokenize(smiles):
        for idx, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[idx])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token) 
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens
    return tokenize

  def smile_to_production_seq(self, smile): 
    production_seq = self.parser.parse(self.tokenizer(smile)).__next__().productions()
    return production_seq
  
  def to_one_hot(self, smile, max_depth=277):
    """
    Args:
      smile: str
        Molecule represented in SMILE grammar
      max_depth: int
        Maximum number of productions used for composition of the SMILE string 
    """
    smile_to_prod_idx = [self.production_dictionary[production] for production in self.smile_to_production_seq(smile)]
    len_production_seq = len(smile_to_prod_idx)
    one_hot = np.zeros((max_depth, self.total_productions))
    one_hot[np.arange(len_production_seq), smile_to_prod_idx] = 1.
    one_hot[np.arange(len_production_seq, max_depth),-1] = 1.
    return one_hot

  def sample_using_masks(self, logit_matrix):
    """
    Implements Algorithm 1 from GrammarVAE paper: https://arxiv.org/abs/1703.01925
    Args: 
      logit_matrix: np.array
    """
    # input: masks for selecting valid production rules 
    masks = self.get_masks()  

    stack = list()
    # initiate stack with the valid production rule (e.g. [smile] for SMILE CFG)
    stack.append(self.start_index)
    res = np.zeros_like(logit_matrix)
    eps = 1e-100
    idx = 0

    def pop_from_stack(stack_):
      try: 
        res_ = stack_.pop()
      except: 
        # the stack is empty, return 'end' production rule: Nothing -> None 
        res_ = self.last_index
      return res_

    while stack is not None and idx < logit_matrix.shape[0]:
      #print('Iteration: {}'.format(idx))
      # 1. given (continuous) logit vector select valid production rule
      # pop the last pushed non-terminal production from the stack
      key = pop_from_stack(stack)
      #print(key)
      next_nonterminal = [self.unique_lhs_dictionary[key]]
      #print('Next nonterminal: {}'.format(next_nonterminal))
      # select mask for mask for the last non-terminal rule
      mask = masks[next_nonterminal]
      #print(mask)
      # mask the logit vector so that only valid right-hand sides can be sampled
      masked_output = np.exp(logit_matrix[idx,:])*mask + eps
      #print(masked_output)
      # given the last non-terminal rule, sample a new valid production rule
      sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
      #print('Sampled output: {}'.format(sampled_output))
      # 2. one_hot encode the new sampled production rule 
      res[idx, sampled_output] = 1.0

      # 3. identify all non-terminals in RHS of selected production
      rhs = list()
      for idx_ in sampled_output: 
        rhs.extend(list(filter(lambda a: (type(a) == grammar.Nonterminal) and (str(a) != 'None'),
                     self.grammar.productions()[idx_].rhs())))
      #print(rhs)
      # 4. push the selected non-terminals onto the stack in reverse order
      stack.extend(rhs[::-1])
      idx += 1
      #print("stack: {}".format(stack))
    return res


  def from_logit_to_production_seq(self, logit):
    one_hot_vec = self.sample_using_masks(logit)
    one_hot_to_production_seq = [self.grammar.productions()[one_hot_vec[idx].argmax()] 
                                 for idx in range(one_hot_vec.shape[0])]
    return one_hot_to_production_seq
