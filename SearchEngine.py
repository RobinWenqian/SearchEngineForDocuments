from collections import defaultdict
import json
import os
from nltk.stem import PorterStemmer
from math import log

#positional_index_filepath = 'pos_inv_index.json'
positional_index_stem_filepath = 'pos_inv_stem_index_5000.json'
stop_word_filepath = 'englishST.txt'
engSW_list = []
word_pos_dict = defaultdict(dict)

# generate stop word list for TFIDF ranking
with open(stop_word_filepath,'r') as inf:
    for line in inf:
        engSW_list.append(line.strip())

# load word positional index json into dict
with open(positional_index_stem_filepath,'r') as inf:
    word_pos_dict = json.load(inf)
    inf.close()

# boolean search engine
def BooleanSearch(times, word_pos_dict, words):
    if len(words.split(' ')) == 1:
        obj = PorterStemmer()
        words_stm = obj.stem(words)
        try:
            text_id_list = word_pos_dict[words_stm].keys()
            return text_id_list
        except:
            print('Boonlean Search No Results')
    
    elif len(words.split(' ')) == 2:
        obj = PorterStemmer()
        try:
            word_formmer = obj.stem(words.split(' ')[0].split('"')[1])
            word_latter = obj.stem(words.split(' ')[1].split('"')[0])
            text_id_final = []
            for text_id_formmer in word_pos_dict[word_formmer].keys():
                text_id_latter_list = word_pos_dict[word_latter].keys()
                if text_id_formmer in text_id_latter_list:
                    word_formmer_pos_list = word_pos_dict[word_formmer][text_id_formmer]
                    word_latter_pos_list = word_pos_dict[word_latter][text_id_formmer]
                    for pos_latter in word_latter_pos_list:
                        i = 0
                        while i < len(word_formmer_pos_list):
                            pos_deviation = int(pos_latter)-int(word_formmer_pos_list[i])
                            i += 1
                            if pos_deviation == 1:
                                text_id_final.append(text_id_formmer)

            return list(set(text_id_final))
        except:
            print('Boonlean Phrase Search No Results')

# In this version 2.0 code, I put logic judgement outside the search engine relization which is more reasonable.
'''
    elif 'OR' in words.split(' ') and 'NOT' not in words.split(' '):
        obj = PorterStemmer()
        try:
            word_1 = words.split(' OR ')[0]
            word_1_stm = obj.stem(word_1)
            word_2 = words.split(' OR ')[1]
            word_2_stm = obj.stem(word_2)

            text_id_list_wd1 = word_pos_dict[word_1_stm].keys()
            text_id_list_wd2 = word_pos_dict[word_2_stm].keys()
            text_id_list_all = list(set(text_id_list_wd1).union(set(text_id_list_wd2)))
            for text_id in text_id_list_all:
                print(str(times) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')

# -----------if you want to print position as well, use following 6 lines of code-------------- #
            #for text_id in text_id_list_wd1:
                #pos = word_pos_dict[word_1_stm][text_id]
                #print(word_1_stm + '|' + str(text_id) + '|' + str(pos))
            #for text_id in text_id_list_wd2:
                #pos = word_pos_dict[word_2_stm][text_id]
                #print(word_2_stm + '|' + str(text_id) + '|' + str(pos))
        except:
            print('Boonlean Search No Results')
    
    elif 'AND' in words.split(' ') and 'NOT' not in words.split(' '):
        obj = PorterStemmer()
        try:
            word_1 = words.split(' AND ')[0]
            word_1_stm = obj.stem(word_1)
            word_2 = words.split(' AND ')[1]
            word_2_stm = obj.stem(word_2)

            text_id_list_wd1 = word_pos_dict[word_1_stm].keys()
            text_id_list_wd2 = word_pos_dict[word_2_stm].keys()
            InBothList = list(set(text_id_list_wd1).intersection(set(text_id_list_wd2)))

            for text_id in InBothList:
                print(str(times) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')

# -----------if you want to print position as well, use following 4 lines of code-------------- #
                #pos_1 = word_pos_dict[word_1_stm][text_id]
                #pos_2 = word_pos_dict[word_2_stm][text_id]
                #print(word_1_stm + '|' + text_id + '|' + str(pos_1))
                #print(word_2_stm + '|' + text_id + '|' + str(pos_2))
        except:
            print('Boonlean Search No Results')
        
    elif 'AND' in words.split(' ') and 'NOT' in words.split(' '):
        obj = PorterStemmer()
        try:
            word_1 = words.split(' AND NOT ')[0]
            word_1_stm = obj.stem(word_1)
            word_2 = words.split(' AND NOT ')[1]
            word_2_stm = obj.stem(word_2)

            text_id_list_wd1 = word_pos_dict[word_1_stm].keys()
            text_id_list_wd2 = word_pos_dict[word_2_stm].keys()
            In_A_Not_in_B_list = list(set(text_id_list_wd1).difference(set(text_id_list_wd2)))

            for text_id in In_A_Not_in_B_list:
                print(str(times) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')

# -----------if you want to print position as well, use following 2 lines of code-------------- #
                #pos_1 = word_pos_dict[word_1_stm][text_id]
                #print(word_1_stm + '|' + text_id + '|' + str(pos_1))
        except:
            print('Boonlean Search No Results')
'''

# Proximity search engine just for two_word phrase
def ProximitySearch(times, word_pos_dict,phrase):
    obj = PorterStemmer()
    try:
        word_formmer = obj.stem(phrase.split(', ')[0].split('(')[1].strip())
        word_latter = obj.stem(phrase.split(', ')[1].split(')')[0])
        distance = int(phrase.split('(')[0].split('#')[1].strip())
        text_id_final = []
        for text_id_formmer in word_pos_dict[word_formmer].keys():
            text_id_latter_list = word_pos_dict[word_latter].keys()
            if text_id_formmer in text_id_latter_list:
                word_formmer_pos_list = word_pos_dict[word_formmer][text_id_formmer]
                word_latter_pos_list = word_pos_dict[word_latter][text_id_formmer]
                for pos_latter in word_latter_pos_list:
                    i = 0
                    while i < len(word_formmer_pos_list):
                        pos_deviation = int(pos_latter)-int(word_formmer_pos_list[i])
                        i += 1
                        if pos_deviation <= distance and pos_deviation > 0:
                            text_id_final.append(text_id_formmer)
        for text_id in set(text_id_final):
            print(str(times) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')
    except:
        print('Proximity Search No Results')

# TFIDF ranking module
def TFIDF_Ranking(times, word_pos_dict, query, DocQuantity):
    obj = PorterStemmer()
    related_text_list = []
    query_word_list = query.split(' ')
    query_word_without_sw_list = []
    query_word_stem_list = []

    # ignore stop words in query before ranking
    for word in query_word_list:
        if word not in engSW_list:
            query_word_without_sw_list.append(obj.stem(word).lower())

    # ignore words in query which are not in any documents
    for word in query_word_without_sw_list:
        if word in list(word_pos_dict.keys()):
            query_word_stem_list.append(word)
    print(query_word_stem_list)
    document_word_tf_dict = defaultdict(defaultdict)
    word_df_dict = defaultdict(list)

    # resort document_word by document and add df,tf to them 
    for word in query_word_stem_list:
        related_text_this_word = list(word_pos_dict[word].keys())
        word_df_dict[word] = len(related_text_this_word)
        related_text_list.append(related_text_this_word)
        for document in related_text_this_word:
            document_word_tf_dict[document][word] = len(word_pos_dict[word][document])
    related_text_list = list(set(sum(related_text_list,[])))

    # calculate score doc by doc
    doc_weight_list = []
    for doc in related_text_list:
        weight_this_doc_list = []
        related_words_list = list(document_word_tf_dict[doc].keys())
        for word in related_words_list:
            weight = float((1 + log(int(document_word_tf_dict[doc][word]),10))*log((DocQuantity+1)/(word_df_dict[word]+1),10))
            weight_this_doc_list.append(weight)
        whole_weight = sum(weight_this_doc_list)
        doc_weight_list.append((doc,whole_weight))
    # sort the doc_weight turple by weight and print top-ten text_id
    weight_sequence = sorted(doc_weight_list,key = lambda x:x[1],reverse=True)
    top_thousand = weight_sequence[0:1001:1]
    if len(top_thousand) < 1000:
        print('There are just %d documents related to this query'%len(top_thousand))
    for rank,text_weight_turple in enumerate(top_thousand):
        print(str(times) + ' ' + '0' +  ' ' + text_weight_turple[0] + ' ' + '0' + ' ' + str(text_weight_turple[1]) + ' ' + '0')



# Search iteration begins depends on number of user's input
# Version 2.0 code judge logic outside the search engine realization

print('''
-------- Welcome to use search engine created by Wenqian (Bradley) He ---------
###############################################################################
Queries' format should belong to following format:
1:word;  
2:word word word --- Any number of word divided by space;  
3:"word word" --- Double-word phrase
4:#int(number)(word, word)
5:word AND word --- AND can be changed to be OR, AND NOT; word can be changed to "word word"

---------- The right format of qurey is NECESSARY for right answer -------------
''')

Number_of_Terms_for_search = int(input("Enter the number of words you need to search (int):"))
i = 0
while i < Number_of_Terms_for_search:
    WordOrPhrase_for_search = str(input("Enter your query:"))
    # if the query is for proximity search
    if '#' in WordOrPhrase_for_search:
        ProximitySearch(i+1, word_pos_dict, WordOrPhrase_for_search)
        i += 1

    elif 'AND' in WordOrPhrase_for_search or 'OR' in WordOrPhrase_for_search or 'NOT' in WordOrPhrase_for_search:
        # for AND logic
        if 'AND' in WordOrPhrase_for_search and 'NOT' not in WordOrPhrase_for_search:
            query_formmer = WordOrPhrase_for_search.split(' AND ')[0]
            query_latter = WordOrPhrase_for_search.split(' AND ')[1]
            text_id_list_formmer = BooleanSearch(i+1, word_pos_dict, query_formmer)
            text_id_list_latter = BooleanSearch(i+1, word_pos_dict, query_latter)
            InBothList = list(set(text_id_list_formmer).intersection(set(text_id_list_latter)))
            for text_id in InBothList:
                print(str(i+1) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')
        # for AND NOT logic
        elif 'AND' in WordOrPhrase_for_search and 'NOT' in WordOrPhrase_for_search:
            query_formmer = WordOrPhrase_for_search.split(' AND NOT ')[0]
            query_latter = WordOrPhrase_for_search.split(' AND NOT ')[1]
            text_id_list_formmer = BooleanSearch(i+1, word_pos_dict, query_formmer)
            text_id_list_latter = BooleanSearch(i+1, word_pos_dict, query_latter)
            In_A_Not_in_B_list = list(set(text_id_list_formmer).difference(set(text_id_list_latter)))
            for text_id in In_A_Not_in_B_list:
                print(str(i+1) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')
        # for OR logic
        elif 'OR' in 'NOT' not in WordOrPhrase_for_search:
            query_formmer = WordOrPhrase_for_search.split(' OR ')[0]
            query_latter = WordOrPhrase_for_search.split(' OR ')[1]
            text_id_list_formmer = BooleanSearch(i+1, word_pos_dict, query_formmer)
            text_id_list_latter = BooleanSearch(i+1, word_pos_dict, query_latter)
            text_id_list_all = list(set(text_id_list_formmer).union(set(text_id_list_latter)))
            for text_id in text_id_list_all:
                print(str(i+1) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')
        i += 1
    # if the query is for phrase search only
    elif len(WordOrPhrase_for_search.split(' ')) == 2 and '"' in WordOrPhrase_for_search:
        text_id_list = BooleanSearch(i+1, word_pos_dict, WordOrPhrase_for_search)
        for text_id in text_id_list:
            print(str(i+1) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')
        i += 1
    # if the query is for single-word search only
    elif len(WordOrPhrase_for_search.split(' ')) == 1:
        text_id_list = BooleanSearch(i+1, word_pos_dict, WordOrPhrase_for_search)
        for text_id in text_id_list:
            print(str(i+1) + ' ' + ' 0 ' + ' ' + str(text_id) + ' ' + ' 0 ' + ' ' +' 1 ' + ' ' +' 0 ')
        i += 1
    # if the query is for TFIDF ranking
    else:
        TFIDF_Ranking(i+1, word_pos_dict,WordOrPhrase_for_search, 5000)
        i += 1

print('Thank you for your using. Utilize search engine again by typing: python SearchEngine.py')