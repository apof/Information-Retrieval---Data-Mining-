import Utils
import numpy as np


def Vector_Model(passage_vectors, query_vector, passage_ids):

	ranking_list = []
	for i in range(len(passage_vectors)):
		## compute thee cosine distance between the query and all the candidate passages
		ranking_list.append((passage_ids[i],Utils.cosine_distance(passage_vectors[i],query_vector)))

	## sort the cosine distance from lower to higher
	sorted_ranking = sorted(ranking_list, key=lambda tup: tup[1], reverse = True)

	## return the top 100 items
	return sorted_ranking[0:100]

def Language_Model(query_key,queries_dict,inverted_index,preprocessed_candidates_dict,smoothing_type,param):

	# the given query
	query = queries_dict.get(query_key)

	ranking = []

	for passage_key in preprocessed_candidates_dict:
		## the length of the passage to compute a score for
		passage_length = len(preprocessed_candidates_dict.get(passage_key))
		score = 0
		for query_token in query:

			## get the passages that contain this query token
			psg_dict = inverted_index.get(query_token)

			## if the query token does not exist in the passage collection then continue
			if(psg_dict == None):
				continue

			## compute the occurencies of the specific query token in the passage
			fqd = psg_dict.get(passage_key)
			if(fqd == None):
				fqd = 0

			## compute cqi --> how many times the query term exists in the entire collection
			cqi = 0
			for k in psg_dict:
				cqi += psg_dict.get(k)

			if(smoothing_type == 'LM-Laplace'):
				score += np.log((fqd + 1)/(len(inverted_index) + passage_length))
			elif(smoothing_type == 'LM-Dirichlet'):
				score += np.log((passage_length/(passage_length + param))*(fqd/passage_length) + (param/(param + passage_length))*(cqi/len(inverted_index)))
			elif(smoothing_type == 'LM-JelineqMercer'):
				score += np.log(param*(fqd/passage_length) + (1-param)*(cqi/len(inverted_index)))
			elif(smoothing_type == 'LM-Lindstone'):
				score += np.log((fqd + param)/(param*len(inverted_index) + passage_length))

		ranking.append((passage_key,score))

	## sort the ranking score from higher to lower
	sorted_ranking = sorted(ranking, key=lambda tup: tup[1], reverse = True)

	return sorted_ranking[0:100]

def BM25_Model(query_key,queries_dict,inverted_index,preprocessed_candidates_dict,k = 1.5, b = 0.75):

	## compute the average document length in the candidates collection
	avgDL = 0
	for passage_key in preprocessed_candidates_dict:
		avgDL += len(preprocessed_candidates_dict.get(passage_key))
	avgDL /= len(preprocessed_candidates_dict)

	## the given query
	query = queries_dict.get(query_key)

	ranking = []

	## BM25 score procedure -- given a query Q create a score for each candidate passage
	for  passage_key in preprocessed_candidates_dict:
		## the length of the passage to compute a score for
		passage_length = len(preprocessed_candidates_dict.get(passage_key))
		BM25_score = 0
		for query_token in query:
			## compute fqd --> frequency of query_token on the specific passage
			## retrieve the passages that contain the specific query token
			psg_dict = inverted_index.get(query_token)
			## if the query toekn does not exist on the collection then continue -- cannot score this token
			if(psg_dict == None):
				continue
			## retrieve the frequency of the specific token on the passage
			fqd = psg_dict.get(passage_key)
			if(fqd == None):
				fqd = 0

			## number of passages containing query token
			qn = len(psg_dict)

			## total passages number
			total_passage_n = len(preprocessed_candidates_dict)

			## compute inverse document frequency
			idf = np.log((total_passage_n - qn + 0.5)/(qn + 0.5) + 1)

			BM25_score += idf * ( fqd * (k + 1)/(fqd + k*(1 - b + b*(passage_length/avgDL)))) 

		ranking.append((passage_key,BM25_score))

	## sort the ranking score from higher to lower
	sorted_ranking = sorted(ranking, key=lambda tup: tup[1], reverse = True)

	return sorted_ranking[0:100]


def Retrieval_Pipeline(queries_dict,passages_dict,query_passage_dict,model_type,hyperparam = None):

	index = 0

	## for each test query
	for key in queries_dict:

		#key = 1112389

		#print("Processing query " + str(key) + " --> " + str(index + 1))

		## get the correct passages related to this query and preprocess them
		## create the inverted index based only on the already retrieved passages
		preprocessed_passages, passage_ids, preprocessed_candidates_dict = Utils.preprocess_passages(key,passages_dict,query_passage_dict,'query')
		## create an inverted index for the specific query based on the candidate passages
		inverted_index, token_index_dictionary = Utils.inverted_index(preprocessed_passages,passage_ids)

		#print(inverted_index)
	
		ranking = None
		if(model_type == 'VS'):
			## create a vector representation for both the query and the candidate passages
			passage_vectors, passage_ids, query_vector = Utils.TFIDF_vectorisation(key,queries_dict,inverted_index,preprocessed_candidates_dict, token_index_dictionary)
			## retrieve documents based on vector model
			ranking = Vector_Model(passage_vectors, query_vector, passage_ids)
		elif(model_type == 'BM25'):
			ranking = BM25_Model(key,queries_dict,inverted_index,preprocessed_candidates_dict)
		else:
			ranking = Language_Model(key,queries_dict,inverted_index,preprocessed_candidates_dict,model_type,hyperparam)

		#print(ranking)

		flag = 0
		if(index == 0):
			flag = 1

		Utils.write_results('../Results/', 'A1', key, model_type, ranking,flag)

		index += 1

		#return