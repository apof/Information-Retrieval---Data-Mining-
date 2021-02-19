import collections
import matplotlib.pyplot as plt
import numpy as np
import Preprocess
from sklearn.linear_model import LinearRegression


def read_collection(filename):
	with open(filename) as f:
		content = f.readlines()
		# remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content]
		return content

def cosine_distance(v1,v2):
	return 1 - np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


def derive_frequencies_from_collection(passage_collection):

	## build a dictionary storing the frequency of each word appeared on the collection
	word_frequence_dictionary = {}

	total_words = 0

	for passage in  passage_collection:
		for word in passage:
			if(word_frequence_dictionary.get(word) == None):
				word_frequence_dictionary[word] = 1
			else:
				word_frequence_dictionary[word] += 1

			total_words += 1

	## sort the dictionaary according in a decreasing frequency order
	ordered_dictionary = sorted(word_frequence_dictionary.items(), key=lambda kv: kv[1], reverse = True)

	return ordered_dictionary,total_words

def plot_word_frequencies(ordered_dictionary,total_words):

	tuple_list = []

	rank_list = []
	log_ranked_list = []
	frequency_list = []
	log_frequency_list = []

	mult = []

	rank = 1
	for (key,value) in ordered_dictionary:
		## store (rank,frequency,proba)
		tuple_list.append((key,rank,value,float(value/total_words)))
		frequency_list.append(float(value/total_words))
		log_frequency_list.append(np.log(float(value)))
		rank_list.append(rank)
		log_ranked_list.append(np.log(rank))
		mult.append(float(value/total_words)*rank)
		rank += 1


	rank_array = np.array(log_ranked_list).reshape(-1,1)
	rank_frequency = np.array(log_frequency_list).reshape(-1,1)

	### fit a linear regression classifier to the data
	regressor = LinearRegression().fit(rank_array, rank_frequency)

	print("R^2 coefficient is: " + str(regressor.score(rank_array, rank_frequency)))
	print("Linear Model Weight: " + str(regressor.coef_[0][0]))
	print("Linear Model Bias: " + str(regressor.intercept_[0]))

	linear_regression_result = []
	for i in range(len(log_ranked_list)):
		linear_regression_result.append(regressor.coef_[0][0]*log_ranked_list[i] + regressor.intercept_[0])
	

	frequency_plot=plt.plot(rank_list,frequency_list,label="Frequency per Rank")
	plt.legend()
	plt.title("Passage Collection Distribution - Zipfs Law")
	plt.xlabel("rank")
	plt.ylabel("word frequency")
	plt.show()

	frequency_plot=plt.plot(log_ranked_list,log_frequency_list,label="Log Frequency per Rank")
	linearly_fited_plot=plt.plot(log_ranked_list,linear_regression_result,label="Linear Regression Fiting")
	plt.legend()
	plt.title("Passage Collection Distribution - Zipfs Law - Logarithmic Scale")
	plt.xlabel("rank (logarithmic)")
	plt.ylabel("word frequency (logarithmic)")
	plt.show()

	print("Mean Value: " + str(np.mean(np.array(mult))))
	print("Std: " + str(np.std(np.array(mult))))



def get_dictionaries(df1,df2):

	query_dict = {}
	passage_dict = {}
	canidates_dict = {}

	for index, row in df1.iterrows():
		qid = row['qid']
		pid = row['pid']
		query = row['query']
		passage = row['passage']
		## create a query index
		if(query_dict.get(qid) == None):
			query_dict[qid] = query
		## create a passage index
		if(passage_dict.get(pid) == None):
			passage_dict[pid] = passage
		## create a query canidate passages index
		if(canidates_dict.get(qid) == None):
			canidates_dict[qid] = [pid]
		else:
			candidates_list = canidates_dict.get(qid)
			candidates_list.append(pid)
			canidates_dict[qid] = candidates_list

	test_query_dict = {}
	for index, row in df2.iterrows():
		test_query_dict[row['qid']] = row['query']

	return query_dict, test_query_dict, passage_dict, canidates_dict

## create an inverted index given a query id
## the data structure is created based on the candidate passages that have already been retrieved for the query
def inverted_index(query_id,passage_dict,canidates_dict):

	inverted_index_dictionary = {}
	## get the passages ids
	canidate_ids = canidates_dict.get(query_id)

	canidate_passage_list = []
	for passage_id in canidate_ids:
		canidate_passage_list.append(passage_dict.get(passage_id))

	## preprocess the candidate passages
	preprocessed_passages = Preprocess.process_data(canidate_passage_list,rm_stopwords = True)

	token_index = 0
	token_index_dictionary = {}

	## create the inverted index
	for i in range(len(preprocessed_passages)):
		for token in preprocessed_passages[i]:
			## if a passage dictionary does not exist for this token
			if(inverted_index_dictionary.get(token) == None):
				token_index_dictionary[token] = token_index
				token_index += 1
				inside_dict = {}
				inside_dict[canidate_ids[i]] = 1
				inverted_index_dictionary[token] = inside_dict
			## if a passage dictionary already exists for this token update it
			else:
				inside_dict = inverted_index_dictionary.get(token)
				if(inside_dict.get(canidate_ids[i]) == None):
					inside_dict[canidate_ids[i]] = 1
				else:
					inside_dict[canidate_ids[i]] += 1

	candidates_dict = {}
	for i in range(len(canidate_ids)):
		candidates_dict[canidate_ids[i]] = preprocessed_passages[i]


	return inverted_index_dictionary, candidates_dict, token_index_dictionary

def preprocess_queries(test_query_dict):

	queries_list = []
	key_list = []
	for key in test_query_dict:
		value = test_query_dict.get(key)
		queries_list.append(value)
		key_list.append(key)

	## preprocess the candidate queries
	preprocessed_queries = Preprocess.process_data(queries_list,rm_stopwords = True)


	queries_dict = {}
	for i in range(len(key_list)):
		queries_dict[key_list[i]] = preprocessed_queries[i]

	return queries_dict


## implement TF-IDF vectorisation for a collection of passages - given the inverted index data structure
def TFIDF_vectorisation(query_id,query_dict,inverted_index,candidates_dict, token_index_dictionary):

	query_vectorised_representation = np.zeros((len(token_index_dictionary)))

	## find the tfidf representation of the query

	## get the specific query
	query_text = query_dict.get(query_id)

	## compute the frequency of each term of the query
	query_term_freq = {}
	for token in query_text:
		if(query_term_freq.get(token) == None):
			query_term_freq[token] = 1
		else:
			query_term_freq[token] += 1


	for token in query_text:

		## if the token exists in the candidate passage corpus
		if(token_index_dictionary.get(token) != None):

			# calculate term frequency
			tf = query_term_freq.get(token)/len(query_text)

			## compute how many passages contain this term
			term_occur = inverted_index.get(token)
			if(term_occur == None):
				term_occur = 0
			else:
				term_occur = len(term_occur)

			## calculate inverse document frequency (number of docs / how many docs contain the term)
			idf = np.log(len(candidates_dict)/(term_occur + 1))

			## compute TF-IDF for the specific token
			tfidf = tf*idf

			## save it on the respective position of the representation -- each token on the corpus has a unique position
			query_vectorised_representation[token_index_dictionary.get(token)] = tfidf

	
	ids_list = []
	vectorised_representations = []

	## create representations for each of the candidate passages

	## for each passage
	for passage_id in candidates_dict:
		ids_list.append(passage_id)
		passage = candidates_dict.get(passage_id)
		passage_representation = np.zeros((len(token_index_dictionary)))

		## for each token in the passage
		for token in passage:

			## frequency of term in the documnet

			## retrieve how many times does the token appear on the specific passage
			freq = inverted_index.get(token).get(passage_id)
			if(freq == None):
				freq = 0
			## calculate term frequency
			tf = freq/len(passage)

			## compute how many passages contain this term
			term_occur = inverted_index.get(token)
			if(term_occur == None):
				term_occur = 0
			else:
				term_occur = len(term_occur)

			## calculate inverse document frequency (number of docs / how many docs contain the term)
			idf = np.log(len(candidates_dict)/(term_occur + 1))

			tfidf = tf*idf

			passage_representation[token_index_dictionary.get(token)] = tfidf

		vectorised_representations.append(passage_representation)

	return np.array(vectorised_representations), ids_list, query_vectorised_representation