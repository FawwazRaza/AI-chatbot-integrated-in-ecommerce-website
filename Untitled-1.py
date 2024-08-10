# %%

import chromadb


# %%
import openai
import json
from pprint import pprint
client=chromadb.Client()
collection=client.get_or_create_collection(name="prod")
# print(client.list_collections())
with open('./productList.json', 'r') as data_file:
    json_data = data_file.read()

data = json.loads(json_data)
# pprint(data)

document_to_add=[]
metadata_to_add=[]
embeddings_to_add=[]
ids_to_add=[]

from chromadb.utils import embedding_functions

# Sample JSON data
products = data["product"]

# Initialize lists for documents, metadata, IDs, and keywords
documents = []
metadata = []
ids = []
keywords = []
students_embedd = []

# Initialize your OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="API KEY",
    model_name="text-embedding-3-small"
)

# Iterate through each product and extract the required information
test_arr_2 = []
for product in products:
    doc = {
        "name": product["name"],
        "description": product["description"],
        "price": product["price"],
        "image_path": product["image_path"],
        "company_name": product["company_id"]["name"],
        "category_name": product["category_id"]["name"],
        "subcategory_name": product["subcategory_id"]["name"]
    }
    doc_str = ', '.join([f"'{key}': '{value}'" for key, value in doc.items()])
    test_arr_2.append(doc_str)  # Store the concatenated string instead of list of key-value pairs
    doc_strlower=doc_str.lower()
    documents.append(doc_str)

    meta = {
        "company_description": product["company_id"]["description"],
        "company_contact": product["company_id"]["company_contact"],
        "company_email": product["company_id"]["company_email"],
        # "company_location": product["company_id"]["location_details"][0]["address"],
        "category_description": product["category_id"]["description"],
        "subcategory_description": product["subcategory_id"]["description"]
    }
    meta_str = ', '.join([f"'{key}': '{value}'" for key, value in meta.items()])
    meta_Strlower=meta_str.lower()
    metadata.append(meta_Strlower)

    document_id = product["_id"]
    product_keywords = product["keyword"]
    product_keywordslower=[item.lower() for item in product_keywords]
    document_idlower=document_id.lower()
    ids.append(document_idlower)
    keywords.append(product_keywordslower)

# Obtain embeddings for the concatenated product information
# students_embeddings = openai_ef(test_arr_2)  # Pass a list of strings

# Append embeddings to students_embedd
# students_embedd.append(students_embeddings)

# Adding data to the collection (adjust as per your implementation)
# collection.add(documents=documents, ids=ids, embeddings=students_embedd)

# Output the lists for verification
# print("Documents:", documents)
# print("Metadata:", metadata)
# print("IDs:", ids)
# print("Keywords:", keywords)
# print("Embeddings:", students_embedd)  # Verify embeddings format


# %%
#len(students_embeddings)
# collection.add( ids=ids, embeddings=students_embeddings,documents=documents)
# pprint(students_embeddings)



# %%

# %%


# %%
# embedd=val[0]
# collection.add(
#     documents=documents,
#     embeddings=embedd,
#     ids=ids
# )

# %%
# pprint(val)
# pprint(embedd)

# %%
# import chromadb.utils.embedding_functions as embedding_functions

# # Initialize the OpenAI embedding function
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#     api_key="API KEY",
#     model_name="text-embedding-3-small"
# )

# # Function to generate embeddings for each product
# def generate_product_embeddings(data):
#     product_embeddings = {}
    
#     for product in data:
#         product_id = product["_id"]
        
#         # Combine the relevant text fields
#         text_to_embed = product["name"] + " " + product["description"] + " " + str(product["price"]) + " " + " ".join(product["keyword"])
        
#         # Generate embedding
#         embedding = openai_ef([text_to_embed])[0]
        
#         # Store embedding with product ID
#         product_embeddings[product_id] = embedding
    
#     return product_embeddings

# # Assume products data is already provided
# product_embeddings = generate_product_embeddings(products)

# # Print the generated embeddings
# for product_id, embedding in product_embeddings.items():
#     print(f"Product ID: {product_id}, Embedding: {embedding[:2]}...")  # Printing first 10 values of the embedding for brevity


# %%
# pprint(students_embedd)

# %%
# pprint(students_embeddings)

# %%
# !pip install haversine


# %%
# import haversine as hs   
# from haversine import Unit
 
# loc1=(31.511327, 74.342059)
# loc2=(31.526437, 74.352327)
 
# result=hs.haversine(loc1,loc2,unit=Unit.KILOMETERS)
# print("The distance calculated is:",result)



# %%
import ast
from openai import OpenAI
## Conversational Q&A Chatbot
import streamlit as st

## Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

from dotenv import load_dotenv
load_dotenv()
import os

# Initialize the OpenAI client
client = OpenAI(api_key="API KEY")

api_key = "API KEY"
if not api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

# search_text = "burgers"
# Calculate distances and find closest product that matches search term
def correct_spelling(inp):
    response = client.chat.completions.create( 
                            model="gpt-3.5-turbo", 
                       messages = [
    {"role": "system", "content": "You correct the grammar and spellings of the sentence and after analyzing the query provide suggestions in the form of array of strings of products (or items) for searching. only provide keywords in this form [\"keyword1\",\"keyword2\",\"keyword n\"]"},
    {"role": "user", "content": inp}
]
    )
    # You Extract keywords from the provide sentence after the correction of Spelling and grammar checking and provide the resulted keywords with comma separated.
    corrected_text =response.choices[0].message.content
    # print("corrected_text   :   ",corrected_text)
    return corrected_text

import ast
import re

def extract_list_from_string(s):
    # Regular expression pattern to match strings enclosed with brackets
    pattern = r'\[.*?\]'
    match = re.search(pattern, s)
    if match:
        return match.group(0)
    return None

# corrected_search_text = correct_spelling(search_text)
# print("cor     :      ",corrected_search_text)
def create_filter_dict(inp):
    filter_dict = {"$or": []}
    for search_string in inp:
        filter_dict["$or"].append({"$contains": search_string})
    return filter_dict


from openai.embeddings_utils import get_embedding, cosine_similarity

def search_reviews(df, product_description, n=3, pprint=True):
   embedding = get_embedding(product_description, model='text-embedding-3-small')
   df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res

res = search_reviews(products, 'delicious beans and burgers', n=3)
print("res", res)







# Query the collection
def search_Result(input):
    # Convert the string representation of the list into an actual list
    inp = ast.literal_eval(input)
    
   
    
    # Create the filter dictionary
    filter_diction ={'$or': [{'$contains': 'mobile'}, {'$contains': 'samsung mobile'}]} 
    # create_filter_dict(inp)
    print("filll   :   ",filter_diction)
    # inp_lower=[it.lower() for it in inp]
    # Assuming openai_ef is a function that returns query embeddings
    # print("inp     : ", inp_lower)
    print("inp   :  ",inp)
    query_embeddings = openai_ef(inp)
    
    
    results = collection.query(
        query_embeddings=query_embeddings,  # Chroma will embed this for you
        where=filter_diction,
        n_results=1
    )
    print(results)
    return results
# input=st.text_input("Input: ",key="input")

# response_After_correction=correct_spelling(input)
# print("response_After_correction  : ",response_After_correction)
# searchresult=search_Result(response_After_correction)
# print("searchresult   :  ",searchresult)
# # document_str = searchresult['documents'][0][0]
# # # Check if any results are returned
# # or len(searchresult['documents'][0][0]) == 0
# if  searchresult==None:
#     print("yes  ")
#     st.subheader("The Response is")
#     st.write("No product Found!")
# else:
#     print("no  ")
#     #  Convert the string representation of the dictionary into an actual dictionary
#     document_str=searchresult['documents'][0][0]
#     document_dict = ast.literal_eval("{" + document_str + "}")
#     # Extract and print the name, description, and price
#     name = document_dict.get('name')
#     description = document_dict.get('description')
#     price = document_dict.get('price')
#     search_Result_str = f"Name: {name}\nDescription: {description}\nPrice: {price}"    
#     submit=st.button("Ask the question")

#     if submit:
#         st.subheader("The Response is")
#         st.write(search_Result_str)





# # %%


# # %%



# # %%


# Function to check searchresult structure
def check_searchresult_format(inp):
    if 'documents' in inp and isinstance(inp['documents'], list) and len(inp['documents']) > 0 and isinstance(inp['documents'][0], list) and len(inp['documents'][0]) == 0:
        return True
    return False

# Streamlit input
input_text = st.text_input("Input:", key="input")

# Process input when button is pressed
submit = st.button("Ask the question")

if submit:
    # response_after_correction = correct_spelling(input_text)
    # extract_array= extract_list_from_string(response_after_correction)
    # searchresult = search_Result(extract_array)
    # if check_searchresult_format(searchresult):
      st.write("No documents found.")
    # else:
    #     document_str = searchresult['documents'][0][0]
    #     document_dict = ast.literal_eval("{" + document_str + "}")
    #     name = document_dict.get('name')
    #     description = document_dict.get('description')
    #     price = document_dict.get('price')
    #     search_result_str = f"Name: {name}\nDescription: {description}\nPrice: {price}"
    #     st.subheader("The Response is")
    #     st.write(search_result_str)
    

# client.delete_collection("prod")