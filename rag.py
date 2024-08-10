# import json
# from openai import OpenAI
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from scipy.spatial import distance

# # Initialize the OpenAI client
# client = OpenAI(api_key="API-key")

# # Load the product list from the JSON file
# with open('productList.json', 'r') as file:
#     data = json.load(file)

# # Extract the products list
# products = data.get('product', [])

# # Prepare a list to store embeddings and product names for visualization
# embeddings_list = []
# product_names = []

# # Function to create a concatenated string for embedding
# def create_text_for_embedding(product):
#     name = product.get('name', '')
#     description = product.get('description', '')
#     price = f"Price: ${product.get('price', 'N/A')}"
    
#     # Extract location details, including latitude and longitude
#     location_details = product.get('company_id', {}).get('location_details', [])
#     location_texts = [
#         f"Location: {loc.get('name', '')}, {loc.get('address', '')}, Latitude: {loc.get('latitude', 'N/A')}, Longitude: {loc.get('longitude', 'N/A')}" 
#         for loc in location_details
#     ]
#     location_text = ' | '.join(location_texts)
    
#     # Extract category and subcategory
#     category_name = product.get('category_id', {}).get('name', '')
#     category_description = product.get('category_id', {}).get('description', '')
#     category_text = f"Category: {category_name} - {category_description}"
    
#     subcategory_name = product.get('subcategory_id', {}).get('name', '')
#     subcategory_description = product.get('subcategory_id', {}).get('description', '')
#     subcategory_text = f"Subcategory: {subcategory_name} - {subcategory_description}"
    
#     # Combine all parts into one string
#     combined_text = f"{name}. {description}. {price}. {location_text}. {category_text}. {subcategory_text}."
#     return combined_text

# # Iterate through each product and get embeddings
# for idx, product in enumerate(products):
#     text_to_embed = create_text_for_embedding(product)
#     if text_to_embed:
#         try:
#             # Create the embedding
#             response = client.embeddings.create(model="text-embedding-3-small", input=text_to_embed)
#             embedding = response.data[0].embedding
#             product['embedding'] = embedding  
#             embeddings_list.append(embedding)  
#             product_names.append(product.get('name', 'Unnamed'))  
#             print(f"Processed product {idx+1}/{len(products)}")
#         except Exception as e:
#             print(f"Failed to process product {idx+1}/{len(products)}: {e}")

# # Save the updated product list with embeddings back to a JSON file
# with open('product_list_with_embeddings.json', 'w') as file:
#     json.dump(products, file, indent=2)

# # Calculate distances and find closest product
# search_text = "pizza" 
# search_embedding = client.embeddings.create(model="text-embedding-3-small", input=search_text).data[0].embedding
# distances = []
# for product in products:
#     dist = distance.cosine(search_embedding, product["embedding"])
#     print(dist)
#     distances.append(dist)

# min_dist_ind = np.argmin(distances)
# print(min_dist_ind)
# print("Closest product:", products[min_dist_ind]['name'])


import json
from openai import OpenAI
import numpy as np
from scipy.spatial import distance
# Initialize the OpenAI client
client = OpenAI(api_key="API-key")

# Load the product list from the JSON file
with open('productList.json', 'r') as file:
    data = json.load(file)

# Extract the products list
products = data.get('product', [])

# Prepare a list to store embeddings and product names for visualization
embeddings_list = []
product_names = []

# Function to create a concatenated string for embedding
def create_text_for_embedding(product):
    name = product.get('name', '')
    description = product.get('description', '')
    price = f"Price: ${product.get('price', 'N/A')}"
    
    # Extract location details, including latitude and longitude
    location_details = product.get('company_id', {}).get('location_details', [])
    location_texts = [
        f"Location: {loc.get('name', '')}, {loc.get('address', '')}, Latitude: {loc.get('latitude', 'N/A')}, Longitude: {loc.get('longitude', 'N/A')}" 
        for loc in location_details
    ]
    location_text = ' | '.join(location_texts)
    
    # Extract category and subcategory
    category_name = product.get('category_id', {}).get('name', '')
    category_description = product.get('category_id', {}).get('description', '')
    category_text = f"Category: {category_name} - {category_description}"
    
    subcategory_name = product.get('subcategory_id', {}).get('name', '')
    subcategory_description = product.get('subcategory_id', {}).get('description', '')
    subcategory_text = f"Subcategory: {subcategory_name} - {subcategory_description}"
    
    # Combine all parts into one string
    combined_text = f"{name}. {description}. {price}. {location_text}. {category_text}. {subcategory_text}."
    return combined_text

# Iterate through each product and get embeddings
for idx, product in enumerate(products):
    text_to_embed = create_text_for_embedding(product)
    if text_to_embed:
        try:
            # Create the embedding
            response = client.embeddings.create(model="text-embedding-3-small", input=text_to_embed)
            embedding = response.data[0].embedding
            product['embedding'] = embedding  
            embeddings_list.append(embedding)  
            product_names.append(product.get('name', 'Unnamed'))  
            print(f"Processed product {idx+1}/{len(products)}")
        except Exception as e:
            print(f"Failed to process product {idx+1}/{len(products)}: {e}")

# Save the updated product list with embeddings back to a JSON file
with open('product_list_with_embeddings.json', 'w') as file:
    json.dump(products, file, indent=2)

search_text = "chocolate lava cake"
# Calculate distances and find closest product that matches search term
def correct_spelling(inp):
    response = client.chat.completions.create( 
                            model="gpt-3.5-turbo", 
                            messages=[{"role": "system", "content": "You Extract keywords from the provide sentence after the correction of Spelling and grammar checking and provide the resulted keywords with comma separated."}, 
                                      {"role": "user", "content": inp}] ) 
    
    corrected_text =response.choices[0].message.content
    return corrected_text


corrected_search_text = correct_spelling(search_text)

print("Corrected search text:", corrected_search_text)


search_embedding = np.mean([product['embedding'] for product in products], axis=0)
keywords = [word.strip() for word in corrected_search_text.split(',')]
for i in keywords:
    print(i)
distances = []
valid_products = []

# Compare each keyword with products
for keyword in keywords:
    for product in products:
        if any(keyword.lower() in kw.lower() for kw in product.get('keyword', [])):
            dist = distance.cosine(search_embedding, np.array(product['embedding']))
            distances.append(dist)
            valid_products.append(product)

# Find the closest product based on minimum distance
if valid_products:
    min_dist_ind = np.argmin(distances)
    closest_product = valid_products[min_dist_ind]
    print("Closest product:", closest_product['name'])
else:
    print("No matching product found.")
