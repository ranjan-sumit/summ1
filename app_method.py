import torch
from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import spacy
from spacy import displacy
import streamlit as st
import time
from IPython.display import display, HTML
import re
import requests
from bs4 import BeautifulSoup


def scrape_website_info(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Create a BeautifulSoup object with the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the specific elements containing the data you want to scrape
    paragraphs = soup.find_all('p')
    
    # Exclude everything after the occurrence of "Conflict of Interests"
    conflict_of_interests_found = False
    cleaned_paragraphs = []
    for p in paragraphs:
        if "conflict of interests" in p.get_text().lower():
            conflict_of_interests_found = True
        if not conflict_of_interests_found:
            cleaned_paragraphs.append(p)
            
    # Exclude everything after the occurrence of "Review Questions"
    review_questions_found = False
    cleaned_paragraphs2 = []
    for p in cleaned_paragraphs:
        if "Review Questions" in p.get_text():
            review_questions_found = True
        if not review_questions_found:
            cleaned_paragraphs2.append(p.get_text(strip=True))
            
            
    try:
        # Incase Pubmed data - Remove Disclosure and beginning author section:
        last_update_start_index = next((index for index, ele in enumerate(cleaned_paragraphs2) if ele.startswith(("Last Update:", "Published online:"))), None)
        end_index = next((i for i, el in enumerate(cleaned_paragraphs2) if el.startswith('Disclosure:')), None)

        # Extract the final list:
        final_paragraphs_text = cleaned_paragraphs2[(last_update_start_index+1):end_index]
   
    except:        
        #Other Links excluding Pudmed - Extract the final list:
        final_paragraphs_text = cleaned_paragraphs2
       
    
    return final_paragraphs_text


def split_sentences(data):
    # Replace numbers with bullets
    data = re.sub(r'(\d+\. )', '• ', data)

    # Replace sub-bullets with bullets
    data = re.sub(r'(\n\s+[a-z]\.\s)', '\n   • ', data)

    # Split the data into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', data)

    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence.strip()]

    return sentences


def calculate_read_time(text, words_per_minute=200):
    # Remove non-alphanumeric characters and count the words
    words = re.findall(r'\w+', text)
    word_count = len(words)

    # Calculate the read time in seconds
    read_time_seconds = int(word_count / (words_per_minute / 60))

    return read_time_seconds

def color_code_entities(text):
    nlp = spacy.load("en_ner_bc5cdr_md")
    doc = nlp(text)
    html = displacy.render(doc, style="ent", options={"colors": {"DISEASE": "#ff9999", "CHEMICAL": "#99ff99"}})
    return html

tokenizer_bart = BartTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
def summarize_texts(input_texts, model,model_name):
    input_ids = tokenizer_bart.batch_encode_plus(
        input_texts,
        truncation=True,
        max_length=64,
        return_tensors="pt",
        padding="longest"
    )["input_ids"]
    summary_ids = model.generate(input_ids, num_beams= 4, max_length=700, early_stopping=True)
    summaries = tokenizer_bart.batch_decode(summary_ids, skip_special_tokens=True)
    return summaries


