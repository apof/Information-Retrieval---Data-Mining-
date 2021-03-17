import collections
import matplotlib.pyplot as plt
import numpy as np
import Preprocess
from sklearn.linear_model import LinearRegression
import csv

## read the document collection line by line
def read_collection(filename):
	with open(filename) as f:
		content = f.readlines()
		# remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content]
		return content

## cosine distance between two vectors
def cosine_distance(v1,v2):
	return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def write_results(file_name, version, key, model_type, ranking, flag):

	f_name = model_type + ".txt"
	## open a csv file for writing

	if(flag == 1):
		w = csv.writer(open(file_name + f_name, "w"))
	else:
		w = csv.writer(open(file_name + f_name, "a"))

	cols = ['query_id', 'version', 'passage_id', 'rank', 'score', 'method']

	if(flag == 1):
		w.writerow(cols)

	index = 1
	for (passage_id,score) in ranking:
		row = [str(key), version, str(int(passage_id)), str(index) , str(score), model_type]
		w.writerow(row)
		index += 1


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
		rank_list.append(rank)
		log_frequency_list.append(np.log(float(frequency_list[-1])))
		log_ranked_list.append(np.log(rank_list[-1]))
		mult.append(float(frequency_list[-1])*rank_list[-1])
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


## split the top 1000 file in dictionaries
def get_dictionaries(df1,df2):

	## queries dictionary: key-->query id / value --> the raw query text
	query_dict = {}
	## all passage dictionary: key --> passage id / value --> the raw passage text 
	passage_dict = {}
	## contains which passages are already retrieved for each query
	## candidate passage dictionary: key --> passage id / value --> the raw passage text 
	canidates_dict = {}

	## a dictionary of the unique passage ids
	unique_passage_ids_dict = {}
	unique_passage_ids_list = []

	for index, row in df1.iterrows():
		qid = row['qid']
		pid = row['pid']
		query = row['query']
		passage = row['passage']
		## store the query
		if(query_dict.get(qid) == None):
			query_dict[qid] = query
		## store the passage
		if(passage_dict.get(pid) == None):
			passage_dict[pid] = passage
		## store the candidate passage
		if(canidates_dict.get(qid) == None):
			canidates_dict[qid] = [pid]
		else:
			candidates_list = canidates_dict.get(qid)
			candidates_list.append(pid)
			canidates_dict[qid] = candidates_list

		## store the passage id if it does not alreday exist
		if(unique_passage_ids_dict.get(pid) == None):
			unique_passage_ids_dict[pid] = 1
			unique_passage_ids_list.append(pid)

		canidates_dict['all_query'] = unique_passage_ids_list


	## the dictionary of the test queries
	test_query_dict = {}
	for index, row in df2.iterrows():
		test_query_dict[row['qid']] = row['query']

	return test_query_dict, passage_dict, canidates_dict

def preprocess_passages(query_id,passage_dict,candidates_dict,flag):

	if(flag == 'query'):

		## flag specifies if the inverted index is going to be created for a specific query using
		## the already retrieved passages for this query or based on the whole corpus

		## get the already retrieved passages ids
		candidate_ids = candidates_dict.get(query_id)
		## the list of the candidate passages
		canidate_passage_list = []
		for passage_id in candidate_ids:
			canidate_passage_list.append(passage_dict.get(passage_id))
		## preprocess the candidate passages
		preprocessed_canidate_passages = Preprocess.process_data(canidate_passage_list,rm_stopwords = True)

		## a dictionary of the candiadte passages to be reranked
		preprocessed_candidates_dict = {}
		for i in range(len(candidate_ids)):
			preprocessed_candidates_dict[candidate_ids[i]] = preprocessed_canidate_passages[i]

	
		return preprocessed_canidate_passages, candidate_ids, preprocessed_candidates_dict

	else:

		## get the ids of all the unique passages
		all_passage_ids = candidates_dict.get('all_query')
		## the list of the candidate passages
		all_passage_list = []
		for passage_id in all_passage_ids:
			all_passage_list.append(passage_dict.get(passage_id))
		## preprocess the candidate passages
		preprocessed_all_passages = Preprocess.process_data(all_passage_list,rm_stopwords = True)

		return preprocessed_all_passages, all_passage_ids,{}
	

## create an inverted index given a query id
## the data structure is created based on the candidate passages that have already been retrieved for the query
## or the hole corpus as an alternative
def inverted_index(preprocessed_passages, candidate_ids):

	inverted_index_dictionary = {}

	token_index = 0
	token_index_dictionary = {}

	## create the inverted index

	## for every preprocessed passage
	for i in range(len(preprocessed_passages)):
		## for every token of the passage
		for token in preprocessed_passages[i]:
			## if a passage dictionary does not exist for this token
			## a new token found in the corpus
			if(inverted_index_dictionary.get(token) == None):
				token_index_dictionary[token] = token_index
				token_index += 1
				inside_dict = {}
				inside_dict[candidate_ids[i]] = 1
				inverted_index_dictionary[token] = inside_dict
			## if a passage dictionary already exists for this token update it
			else:
				## get the dictionary inside
				inside_dict = inverted_index_dictionary.get(token)
				## if the specific token doesnt exist already in this document
				if(inside_dict.get(candidate_ids[i]) == None):
					inside_dict[candidate_ids[i]] = 1
				## if already exists in this document
				else:
					inside_dict[candidate_ids[i]] += 1

	return inverted_index_dictionary, token_index_dictionary

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

			#print("token " + token + " tf-idf " + str(tf) + "-" + str(idf))

			## save it on the respective position of the representation -- each token on the corpus has a unique position
			query_vectorised_representation[token_index_dictionary.get(token)] = tfidf

	
	#print(query_vectorised_representation)
	#print(query_vectorised_representation.shape)

	ids_list = []
	vectorised_representations = []

	## create representations for each of the candidate passages

	## for each passage
	for passage_id in candidates_dict:
		ids_list.append(passage_id)
		passage = candidates_dict.get(passage_id)
		passage_representation = np.zeros((len(token_index_dictionary)))

		#print("passage id: " + str(passage_id))

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

			#if(passage_id == 8138768):
			#	print("token " + token + " tf-idf " + str(tf) + "-" + str(idf))

			passage_representation[token_index_dictionary.get(token)] = tfidf

		
		#print(passage_representation)
		#print(passage_representation.shape)

		vectorised_representations.append(passage_representation)

		

	return np.array(vectorised_representations), ids_list, query_vectorised_representation