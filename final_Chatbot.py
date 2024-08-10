from openai import OpenAI
import pandas as pd
import numpy as np
import json
import os
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   try:
       response = client.embeddings.create(input = [text], model=model)
       return response.data[0].embedding
   except Exception as e:
       print(f"Error in get_embedding: {e}")
       return None

# Load the JSON data
with open('./productList.json', 'r') as data_file:
    json_data = json.load(data_file)

# Convert the JSON data to a DataFrame
df = pd.DataFrame(json_data['product'])

# Create a 'combined' column with all relevant text data
df['combined'] = df.apply(lambda row: f"{row['name']} {row['description']} {' '.join(row['keyword'])}", axis=1)


# Apply the embedding function to the 'combined' column

df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))


# Check if 'ada_embedding' column was created successfully
if 'ada_embedding' in df.columns:
    print("\n'ada_embedding' column created successfully")
    # Convert the 'ada_embedding' column to numpy arrays
    df['ada_embedding'] = df.ada_embedding.apply(lambda x: np.array(x) if x is not None else None)
else:
    print("\nError: 'ada_embedding' column was not created")

# Create the 'output' directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save the DataFrame with embeddings to a CSV file
df.to_csv('output/embedded_products.csv', index=False)


st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")


def search_reviews(df, product_description, n=3, pprint=True):
   embedding = get_embedding(product_description, model='text-embedding-3-small')
   if embedding is None:
       print("Error: Could not generate embedding for search query")
       return None
   df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity([x], [embedding])[0][0] if x is not None else -1)
   res = df.sort_values('similarities', ascending=False).head(n)
   return res



# print("Search results:")
# print(res)

# Streamlit input
input_text = st.text_input("Input:", key="input")

# Process input when button is pressed
submit = st.button("Ask the question")

if submit:
    res = search_reviews(df, input_text, n=1)
    st.subheader("The Response is")
    st.write(res)
else:
    st.subheader("The Response is")
    st.write("error")