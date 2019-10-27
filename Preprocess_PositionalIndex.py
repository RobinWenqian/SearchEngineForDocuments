from collections import defaultdict
import string
import re
import json
from nltk.stem import PorterStemmer
import xml.etree.cElementTree as ET

sample_filepath = 'trec.sample.txt'
#converted_filepath = 'converted_sample.txt'
sample_filepath_xml = './CW1collection/trec.5000.xml'
eng_sw = 'englishST.txt'
pos_index_stemming_filepath = 'pos_inv_stem_index_5000.json'

eng_sw_list = []
id_news_dict = defaultdict(list)
id_word_position_dict = defaultdict(lambda:defaultdict(list))


# generate stop-words list
with open(eng_sw,'r') as inf:
    for line in inf:
        if "'" in line:
            #line_refine = line.strip().translate(str.maketrans('', '', string.punctuation))
            eng_sw_list.append(line.strip().split("'")[0])
            eng_sw_list.append(line.strip().split("'")[1])
        else:
            eng_sw_list.append(line.strip())
eng_sw_list = list(set(eng_sw_list))

# Tokenization_Casefolding and sortPreprocessedFile are for parsing txt file
# easy preprocessing for tokenization and casefolding. Then write the results to outputfile
def Tokenization_Casefolding(input_filepath,output_filepath):
    with open(output_filepath,'w') as outf:
        with open(input_filepath, 'r') as infile:
            for line in infile:
                line_Tokenization = line.translate(str.maketrans('', '', string.punctuation))
                #line_sub = re.sub(u"([^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a])","",line)
                line_tokCasefolding = line_Tokenization.lower()
                outf.write(line_tokCasefolding)

# sort the output_file by id
def sortPreprocessedFile(input_filepath):
    with open(input_filepath,'r') as inf:
        id = 0
        for line in inf:
            line_parts = line.strip().split(' ')
            id_flag = line_parts[0]
            if 'id' == id_flag:
                id = line_parts[1]
            else:
                id_news_dict[id].append(line_parts)

# XML_preprocessing is for parsing XML file and replace Tokenization_Casefolding and sortPreprocessedFile functions
def XML_preprocessing(input_filpath):
    doc = ET.parse(input_filpath)
    for element in doc.iterfind('DOC'):
        docNo = element.findtext('DOCNO')
        headline_init = element.findtext('HEADLINE')
        headline_prep = headline_init.translate(str.maketrans('', '', string.punctuation)).lower().strip().split(' ')
        text_init = element.findtext('TEXT')
        text_prep = text_init.translate(str.maketrans('', '', string.punctuation)).lower().replace('\n',' ').replace('\t','').split(' ')
        id_news_dict[docNo].append(headline_prep)
        id_news_dict[docNo].append(text_prep)

# merge headline and text into one list of each text; remove stop words and empty word
def mergeText(text_dict):
    for text_id in id_news_dict.keys():
        id_news_dict[text_id] = sum(id_news_dict[text_id],[])
        while '' in id_news_dict[text_id]:
            id_news_dict[text_id].remove('')
            intermediat_list = [x for x in id_news_dict[text_id] if x not in eng_sw_list]
            id_news_dict[text_id] = intermediat_list


# do Porter Stemming for the text
def PorterStemming(text_dict):
    for text_id in id_news_dict.keys():
        for pos,word in enumerate(id_news_dict[text_id]):
            obj = PorterStemmer()
            id_news_dict[text_id][pos] = obj.stem(word) 
    
# generate inverted position index
def positional_index_generator(id_news_dict):
    for text_id, word_list in id_news_dict.items():
        for pos, word in enumerate(word_list):
            #word_pos = word_list.index(word)
            id_word_position_dict[word][text_id].append(pos)

# dump the positional inverted index dict into json
def dump_to_json(word_postion_dict,json_filepath):
    with open(json_filepath,'w') as outf:
        json.dump(word_postion_dict,outf, indent=4)

#Tokenization_Casefolding(sample_filepath,converted_filepath)
#sortPreprocessedFile(converted_filepath)

XML_preprocessing(sample_filepath_xml)
mergeText(id_news_dict)
PorterStemming(id_news_dict)
positional_index_generator(id_news_dict)
dump_to_json(id_word_position_dict,pos_index_stemming_filepath)
