'''
MAB

A python library for perform each multi-armed bandit algorithm

'''

import numpy as np

class Mab(object):
	'''
	multi-armed bandit test class
	'''

	def __init__(self, num_bandits = 10, probs = None, payoff = None, params = None):
		'''
		Parametes
		---------
		num_bandits : int
			default is 10
		probs : np.array of float
			payoff probabilities
		payoff : np.array of float
			each arm's payoff when it's chosen
		params : some parameters of relative algoritnm
		'''
		if not probs:
			probs = [np.random.rand() for x in range(num_bandits)]
		if not payoff:
			payoff = np.ones(num_bandits)

		self.num_bandits = num_bandits
		self.probs = probs
		self.payoff = payoff
		#save each step's choice 
		self.choices = []
		self.naive_index = None

		self.wins = np.zeros(num_bandits)
		self.pulls = np.zeros(num_bandits)

		self.strategies = ['random', 'naive', 'eps_greedy', 'softmax', 'ucb', 'bayes']

		self.bandits = Bandits(probs = self.probs, payoff = self.payoff)


	def run(self, strategy, parameters = None):
		'''
		Run single tiral of a MAB algorithm

		Parameters
		----------
		strategy : str
			the algorithm name
		parameters : dict

		Returns
		-------
		None
		'''

		choice = self.run_strategy(strategy, parameters)
		self.choices.append(choice)
		payoff = self.bandits.pull(choice)
		self.wins[choice] += payoff
		self.pulls[choice] += 1

	def run_strategy(self, strategy, parameters):
		'''
		Run the selected algorithm and return bandit choice

		Parameters
		----------
		strategy : str
			Name of MAB algorithm
		parameters : dict
			Algorithm function parameters

		Returns
		-------
		int 
			Bandit choice index
		'''
		return self.__getattribute__(strategy)(params = parameters)



#######==========================MAB Algorithm============================########
	def ind_max_mean(self):
		'''
		Pick the bandit with the current best observed proportion of winning

		Returns
		-------
		int 
			Index  of chosen bandit
		'''
		return np.argmax(self.wins / (self.pulls + 0.1))

	def random(self, params = None):
		'''
		Run the Random Bandit algorithm

		Parameters
		----------
		params:None
		
		Returns
		-------
		int
		Index of chosen bandit
		'''
		return np.random.randint(self.num_bandits)

	def naive(self, params = None):
		'''
		Run the Naive Bandit algorithm

		Parameters
		---------
		params:None

		Return
		------
		int
		Index of chosen bandit
		'''

		# Handle cold start. Not all bandits tested yet.
		if min(self.pulls) < 3:
			return np.argmin(self.pulls)
		else:
			if not self.naive_index:
				self.naive_index = self.ind_max_mean()
			return self.naive_index

	def eps_greedy(self, params = None):
		'''
		Run epsilon-greedy algorithm and update 

		Parameters
		----------
		params : dict
			epsilon

		Returns
		-------
		int
			Index of chose bandit
		'''

		if params and type(params) == dict:
			eps = params.get('epsilon')
		else:
			eps = 0.1
		r = np.random.rand()
		if r < eps:
			return np.random.randint(len(self.wins))
		else:
			return self.ind_max_mean()

	def softmax(self, params = None):
		'''
		Run the softmax selection strategy.

		Parameters
		----------
		Params : dict
			Tau

		Returns
		-------
		int
			Index of chosen bandit
		'''


		default_tau = 0.1
		if params and type(params) == dict:
			tau = params.get('tau')
			try:
				float(tau)
			except ValueError:
				'mab: softmax: setting tau to default'
				tau = default_tau
		else:
			tau = default_tau

		# Handle cold start. Not all bandits tested yet.
		if min(self.pulls) < 3:
			return np.argmin(self.pulls)
		else:
			payoffs = self.wins / self.pulls
			norm = sum(np.exp(payoffs / tau))
		ps = np.exp(payoffs / tau) / norm

		#Randomly choose index based on CMF
		cmf = [sum(ps[:i+1]) for i in range(len(ps))]

		r = np.random.rand()

		found = False
		found_k = None
		k = 0
		while not found:
			if r < cmf[k]:
				found_k = k
				found = True
			else:
				k += 1
		return found_k 

	def ucb(self, params = None):
		'''
		Run the upper confidence bound MAB selection algorithm

		This is the UCB1 algorithm

		Parametes
		---------
		params : None

		Returns
		-------
		int
			Index of chosen bandit
		'''
    	
		# UCB = argmax_j{ payoff_j + sqrt(2ln(n_tot)/n_j) }

		# Handle cold start. Not all bandits tested yet.
		if min(self.pulls) < 3:
			return np.argmin(self.pulls)
		else:
			n_tot = sum(self.pulls)
			payoffs = self.wins / self.pulls
			ucbs = payoffs + np.sqrt(2 * np.log(n_tot) / self.pulls)
			return np.argmax(ucbs)

	def bayes(self, params = None):
		'''
		Run the Bayes Bandit algorithm which use a beta distribution
		for exploration and exploitation

		Parametes
		---------
		params : None

		Returns 
		-------
		int 
			Index of chose bandit
		'''

		p_bandits = [np.random.beta(self.wins[k] + 1, self.pulls[k] - self.wins[k] + 1) 
					for k in range(len(self.wins))
					]
		return np.array(p_bandits).argmax()



###########==============================================================#############
	def best_payoffs(self):
		'''
		Calculate current estimate of average payout for each bandit

		Returns
		-------
		array of float or None
		'''

		if len(self.choices) < 1:
			print('No trials run so far.')
		else:
			return self.wins / (self.pulls + 0.1)

	def regret(self):
		'''
		Calculate cumulative regret

		cumulative regret = T * max(probs[k]) - sum(probs * pulls)
		Returns
		-------
		float
		'''
		return sum(self.pulls) * np.max(self.probs) - sum(self.probs * self.pulls)

class Bandits():
	'''
	Bandit class
	'''
	def __init__(self, probs, payoff):
		'''
		initiate bandit class
		Parameters
		----------
		probs : array of float
			probability of bandit payoff
		payoff : array of float
			amount of bandit payoff, in Bernoulli distribution, it's 0 or 1
			N length array
		'''
		if len(probs) != len(payoff):
			raise Exception('Bandits.__init__: probability and payoff has different length!')
		self.probs = probs
		self.payoff = payoff

	def pull(self, k):
		'''
		pull the k-th bandit
		Parameters
		----------
		k : int
			index of bandit

		Returns
		-------
		float : the k-th bandit payoff
		'''

		if np.random.rand() < self.probs[k]:
			return self.payoff[k]
		else:
			return 0.0
