import nltk
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
from nltk.stem import PorterStemmer 
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer



## preprocess the passages tokenising each sentence, convering to lower case, excluding non alphabetic words
## removing stopwords and stemming
def process_data(data,rm_stopwords = False, lemm = True):

	stop_words = set(stopwords.words('english'))

	tokenizer = RegexpTokenizer(r'\w+')


	processed_sentences = []
	for sentence in data:

		## tokenise each sentence
		#tokenised_sentence = sentence.split(" ")

		tokenised_sentence = tokenizer.tokenize(sentence)

		##convert to lower case
		sentence = [w.lower() for w in tokenised_sentence]

		##exclude non alphabetic words
		#only_alpha_sentence = [word for word in sentence if word.isalpha()]

		only_alpha_sentence = sentence
		
		## remove stop words
		if(rm_stopwords == True):
			filtered_sentence = [w for w in only_alpha_sentence if not w in stop_words]
		else:
			filtered_sentence = only_alpha_sentence

		lemmatized_sentence = []

		if(lemm == True):
			## stemming
			## lemmatizer = WordNetLemmatizer()
			stemmer = PorterStemmer()
			for word in filtered_sentence:
				#lemmatized_sentence.append(lemmatizer.lemmatize(word))
				lemmatized_sentence.append(stemmer.stem(word))
		else:
			lemmatized_sentence = filtered_sentence

		processed_sentences.append(lemmatized_sentence)
	return processed_sentences