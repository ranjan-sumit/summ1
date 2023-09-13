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
from app_method import scrape_website_info,split_sentences,calculate_read_time, summarize_texts,color_code_entities

# import warnings
# warnings.filterwarnings("ignore")
# st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_resource
def load_model(model_name):
    nlp = spacy.load(model_name)
    return(nlp)


@st.cache_resource
def load_bert_model():
    model_path = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    return tokenizer, model


@st.cache_resource
def load_bart_model():
    model_path_summarization = r"philschmid/bart-large-cnn-samsum"
    model_s = BartForConditionalGeneration.from_pretrained(model_path_summarization)
    return model_s

# Set page title and subtitle
st.set_page_config(page_title="MediVerse", page_icon="sum_logo.png",layout="wide")
st.title("MediVerse")

# Load and display the logo
logo_image = "Mediverse.png"
st.sidebar.image(logo_image, width=200)

# App introduction in the sidebar
st.sidebar.markdown(
    """
    <h1 style='font-size:24px;'>Welcome to MediVerse!</h1>
    <p style='font-size:16px;'>MediVerse is a Biomedical Summarization Tool designed specifically for medical data! Our app simplifies complex medical information into easy-to-understand summaries. Save time and stay informed with efficient insights.</p>
    """
    , unsafe_allow_html=True
)

# About section
st.markdown(
    """
    <h2>About</h2>
    <p>MediVerse is a powerful summarization tool that leverages language models to extract key insights from biomedical text data.</p>
    """
    , unsafe_allow_html=True
)

# model_path = "C:/Users/Yajna/OneDrive - mresult.com/Documents/NLP - Summarization/SapBERT-from-PubMedBERT-fulltext"
tokenizer, model = load_bert_model()
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertModel.from_pretrained(model_path)
#tokenizer = BertTokenizer.from_pretrained(model_path) 

model_name = "philschmid/bart-large-cnn-samsum"
# model_path_summarization = r"C:\Users\Yajna\OneDrive - mresult.com\Documents\NLP - Summarization\bart-large-cnn-samsum"
model_s = load_bart_model()
#model_s = BartForConditionalGeneration.from_pretrained(model_path_summarization)


#loading Annotation model
#nlp = spacy.load("en_ner_bc5cdr_md")
nlp = load_model("en_ner_bc5cdr_md")



# Radio button for summarization

st.subheader("Summarizer")
option = st.radio("Select Input Option", ("Text Summarizer", "URL Summarizer", "PDF Summarizer Beta"))
data = ""

if option == "Text Summarizer":
    
    data = st.text_area("Input Text", value="Enter your text here and press summarize", height=300)
    
    if st.button ("Summarize Text"):
        dataset = split_sentences(data)
        start_time = time.time()
        buffer_time = 10
        st.text("Step 1: Getting the input data...")
        st.text("Step 2: Importing the model and generating embeddings...")
        # Getting model for clustering

        
        encoded_inputs = tokenizer.batch_encode_plus(
        dataset,
        add_special_tokens=True,
        max_length=512,
        padding='longest',
        truncation=True,
        return_tensors='pt')
        
        input_ids = encoded_inputs['input_ids']
        
        attention_mask = encoded_inputs['attention_mask']

        with torch.no_grad():
            model.eval()
            outputs = model(input_ids, attention_mask=attention_mask)
            sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
        # Step 3: Perform hierarchical clustering
        st.text("Step 3: Performing hierarchical clustering...")
        if len(sentence_embeddings) > 1:
            clustering = AgglomerativeClustering(n_clusters=(1 + int(round(len(dataset) / 3))), linkage='average')
            cluster_labels = clustering.fit_predict(sentence_embeddings)
        else:
            cluster_labels = [0]

        cluster_mapping = {}
        new_index = 0
        for i, label in enumerate(cluster_labels):
            if label not in cluster_mapping:
                cluster_mapping[label] = new_index
                new_index += 1
            cluster_labels[i] = cluster_mapping[label]



        distance_matrix = linkage(sentence_embeddings, method='average', metric='cosine')
        
        
        # Step 4 Creating Clusters from above Dendogram:
        st.text("Step 4: Creating clusters from the dendrogram...")
        cluster_values = cluster_labels
        string_values = dataset

        result_dict = {}

        for cluster, value in zip(cluster_values, string_values):
            if cluster in result_dict:
                result_dict[cluster] += '.' + value
            else:
                result_dict[cluster] = value
                
        # Step 5: Function for text summarization
        st.text("Step 5: Performing text summarization...")
        
        # Rank the summaries by cluster
        ranked_summaries = [result_dict[cluster] for cluster in sorted(result_dict.keys())]
        
        # Summarize the ranked summaries
        merged_summary = " ".join(summarize_texts(ranked_summaries, model_s,model_name))
        #merged_summary = " ".join(summary for summary in summaries)
        pattern = r"(Â\s*|Â\.*.*\s*(y|Y))"
        summary = re.sub(pattern, "", merged_summary)
        
        
        # Step 6: Highlight the summaries
        st.text("Step 6: Highlighting the summaries...")
        
        html_output = color_code_entities(summary)
        st.subheader("Summary")

        st.markdown("Highlighted disease & Chemical keywords:")
        st.write(HTML(html_output))
        
         # Calculate the read time
        time_in_seconds_data = calculate_read_time(data)
        time_in_seconds = calculate_read_time(merged_summary)
        st.text("")
        st.text("")
        st.info("The estimated read time of input text is:" + str (time_in_seconds_data + buffer_time) + " seconds and estimated read time of summarized text is: " +  str (time_in_seconds) + " seconds.")
        st.warning("Percentage reduction in read time: " + str((1 - round(time_in_seconds / time_in_seconds_data, 4)) * 100) + "%")

   
    
        end_time = time.time()
        runtime = end_time - start_time
        st.write("")

        st.text("Runtime: " + str(round(runtime,2)) + " seconds")
            
        

elif option == "URL Summarizer":

    website_url = st.text_input("Enter the website URL:")
          
                
                
                
    if st.button("Summarize URL"):
        
        #st.warning("Please click on Summarize URL Button.")
        start_time = time.time()
            
        buffer_time = 10
            
        st.text("Step 1: Scraping Data...")
        dataset = scrape_website_info(website_url)
        #dataset = dataset[:10]
        for paragraph in dataset:
                st.write(paragraph)
                
        st.text("Step 2: Generating and Highlighting the summaries...")
        
        #Getting model for Summarization

#         batch_size = 32  # Adjust batch size as per your system's capacity
#         input_batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
#         summaries = []
#         for batch in input_batches:
#             batch_summaries = summarize_texts(batch, model_s, model_name)
#             summaries.extend(batch_summaries)

            
        # Summarize the ranked summaries
        merged_summary = " ".join(summarize_texts(input_texts=dataset, model=model_s,model_name=model_name))
        #merged_summary = " ".join(summary for summary in summaries)
        pattern = r"(Â\s*|Â\.*.*\s*(y|Y))"
        summary = re.sub(pattern, "", merged_summary)
            
       
            
        html_output = color_code_entities(summary)
        st.subheader("Summary")

        st.markdown("Highlighted disease & Chemical keywords:")
        st.write(HTML(html_output))
        
        over_all_data = " ".join(dataset)
        
            
        # Calculate the read time
        time_in_seconds_data = calculate_read_time(over_all_data)
        time_in_seconds = calculate_read_time(summary)
        st.text("")
        st.text("")
        st.info("The estimated read time of input text is:" + str (time_in_seconds_data + buffer_time) + " seconds and estimated read time of summarized text is: " +  str (time_in_seconds) + " seconds.")
        #st.warning("% reduction in readtime: " + str((1-(round(time_in_seconds/time_in_seconds_data),4)  )*100))
        st.warning("Percentage reduction in read time: " + str((1 - round(time_in_seconds / time_in_seconds_data, 4)) * 100) + "%")

   
    
        end_time = time.time()
        runtime = end_time - start_time
        st.write("")

        st.text("Runtime: " + str(round(runtime,2)) + " seconds")
        

        
elif option == "PDF Summarizer Beta":
    pass
        
    
           
