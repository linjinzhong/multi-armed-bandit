'''
test some algorithm for multi-armed bandit
'''
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sea
import MAB

#number of experiment
num_exp = 10
#number of selection
num_sel = 10000
# number of bandit
K = 100
#probabily of each bandit
probs = list(np.random.random(K))

#each algorithm and it's regret, label
algorithms = [{'algorithm':'random', 'regret':[], 'label':'random'},
			  {'algorithm':'naive', 'regret':[], 'label':'Naive'},
			  {'algorithm':'eps_greedy', 'regret':[], 'label':'$\epsilon$-greedy($\epsilon$=0.1)'},
			  {'algorithm':'softmax', 'regret':[], 'label':'Softmax($T$=0.1)'},
			  {'algorithm':'ucb', 'regret':[], 'label':'UCB1'},
			  {'algorithm':'bayes', 'regret':[], 'label':'Thompson Sampling'}
			  ]


sumRegret = [np.zeros(num_sel), np.zeros(num_sel), np.zeros(num_sel), 
             np.zeros(num_sel), np.zeros(num_sel), np.zeros(num_sel)]

#run experiment and get regret
for n in range(num_exp):
	print(n)
	# init each algorithm with relative parameters
	for alg in algorithms:
		alg['mab'] = MAB.Mab(num_bandits=K, probs=probs, params=None)

	regret = [[], [], [], [], [], []]
	for t in range(num_sel):
		for a, alg in enumerate(algorithms):
			alg['mab'].run(alg['algorithm'])
			regret[a].append(alg['mab'].regret())
			# alg['regret'].append(alg['mab'].regret())
	for a, alg in enumerate(algorithms):
		sumRegret[a] += np.array(regret[a])

#average regret of each algorithm
for r, alg in enumerate(algorithms):
	alg['regret'] = list(sumRegret[r] / np.float(num_exp))

#color choice
color = ['k', 'b', 'c', 'g', 'y', 'r']

# Pretty plotting
sea.set_style('whitegrid')
sea.set_context('poster')

#plot
fig = plt.figure(figsize=(12,4))
regretFig = fig.add_subplot(111)
regretFig.set_yscale("log")
for i, alg in enumerate(algorithms):
	regretFig.plot(alg['regret'],  color[i], lw = 3, label=alg['label'])
regretFig.grid(False)
regretFig.set_xlim(0,10000)
regretFig.set_ylim(1e-1,1e4)
regretFig.set_xlabel('Trials')
regretFig.set_ylabel('Cumulative Regret')
regretFig.set_title('Multi-armed bandit strategy performance(K=%d)' % K)
regretFig.legend(loc='lower right')
plt.show()


