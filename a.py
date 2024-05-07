# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd  

df = pd.read_csv('output.csv')
url_to_product = df.set_index('URL')['Product Name'].dropna().to_dict()

df_tax = pd.read_csv('taxonomy.csv')

# Initialize an empty dictionary to store your results
result_dict = {}

# Group the DataFrame by 'Family' which automatically groups by 'Family Name' due to their relationship
for _, group in df_tax.groupby('Family'):
    # Get the unique 'Family' ID (since all will be the same in the group, just take the first one)
    family_id = group['Family'].iloc[0]
    
    # Get the 'Family Name' (again, all are the same in the group, so take the first one)
    family_name = group['Family Name'].iloc[0]
    
    # Combine unique 'Class Name' and 'Commodity Name' from this group
    class_and_commodity_names = list(set(group['Class Name'].tolist() + group['Commodity Name'].tolist()))
    
    # Add to the dictionary
    result_dict[family_name] = (family_id, class_and_commodity_names)

# Now result_dict contains the mapping you wanted


from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import time

# Start the timer
start_time = time.time()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

# Function to generate BERT embeddings for a list of texts
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(input_ids)
            embedding = outputs[0][:, 0, :]  # Extract embedding for [CLS] token
        embeddings.append(embedding.numpy())
    return np.mean(embeddings, axis=0)

# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()

# Function to generate embeddings for a list of product names
def generate_embeddings_multi_threaded(product_names, num_cores=num_cores):
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        embeddings = list(executor.map(generate_embeddings, [[name] for name in product_names]))
    return [embedding[0] for embedding in embeddings]  # Extract the embeddings from the list

# Function to generate embeddings for a list of product names
def generate_embeddings_multi_threaded(product_names):
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(generate_embeddings, [[name] for name in product_names]))
    return [embedding[0] for embedding in embeddings]  # Extract the embeddings from the list


# Function to calculate cosine similarity
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)

def predict_category(url_to_product, result_dict):
    ## Set a counter variable to limit the iteration
    count = 0

    # Precompute embeddings for all categories
    category_embeddings_dict = {}
    for category, (code, keywords) in result_dict.items():
        combined_keywords = ' '.join([category] + keywords)
        category_embedding = generate_embeddings([combined_keywords])
        category_embeddings_dict[category] = category_embedding

        count += 1
        print(count, category)
        # if count == 1:
        #     break

    # Create an empty list to store the results
    results = []

    count = 0
    # Iterate through each product
    for product_url, product_name in url_to_product.items():
        print(product_name)
        product_embedding = generate_embeddings([product_name])
        max_similarity = -1
        predicted_category_name = None
        predicted_category_code = None
        predicted_category_keywords = None

        # Compare product embedding with precomputed category embeddings
        for category, category_embedding in category_embeddings_dict.items():
            similarity = calculate_similarity(product_embedding, category_embedding)
        
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_category_name = category
                predicted_category_code = result_dict[category][0]
                predicted_category_keywords = result_dict[category][1]  # Get all keywords

    
        count += 1
        if count == 101:
            break   

        # Append the results to the list
        results.append({
            'url': product_url,
            'product_name': product_name,
            'predicted_category_name': predicted_category_name,
            'predicted_category_code': predicted_category_code,
            'predicted_category_keywords': predicted_category_keywords
        })

    # Create a DataFrame from the results list
    df = pd.DataFrame(results)
    df.to_csv('predictions.csv', index=False)
    
    # Display the DataFrame
    print(df)

# Combine product names into a list
no_products = 5000
product_names = list(url_to_product.values())[:no_products]
product_urls = list(url_to_product.keys())[:no_products]

# Generate embeddings for product names
product_embeddings = generate_embeddings_multi_threaded(product_names)


# Perform K-means clustering
num_clusters = int(465 / 2) # You can adjust the number of clusters based on your data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(product_embeddings)

# Map product names to cluster labels
product_clusters = {}
for product_name, cluster_label in zip(product_names, cluster_labels):
    if cluster_label not in product_clusters:
        product_clusters[cluster_label] = []
    product_clusters[cluster_label].append(product_name)


# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {int(elapsed_time // 60)} minutes & {int(elapsed_time % 60)} seconds for {no_products} products and {num_clusters} clusters")








from ipywidgets import widgets
from IPython.display import display
from plotly.subplots import make_subplots
from color_palettes import colors

total_clusters = len(set(cluster_labels))

# Create a colorscale manually or use an existing one. Here we create a simple one for demonstration.
# Example: Define colors as a list of CSS-compatible color strings.


# Ensure there are enough colors for the number of clusters
if total_clusters > len(colors):
    import random
    random.seed(42)
    # Extend the color list by generating random colors if needed
    colors += [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(total_clusters - len(colors))]

# Create figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
    subplot_titles=('3D Visualization', '2D Visualization')
)

# Prepare the scatter data for both 3D and 2D
for i in range(total_clusters):
    cluster_visibility = [label == i for label in cluster_labels]
    cluster_points = [(embedding, name, url) for embedding, name, url, visible in zip(product_embeddings, product_names, product_urls, cluster_visibility) if visible]
    
    x = [point[0][0] for point in cluster_points]
    y = [point[0][1] for point in cluster_points]
    z = [point[0][2] if len(point[0]) > 2 else 0 for point in cluster_points]
    hover_texts = [f"{point[1]}<br>{point[2]}" for point in cluster_points]
    
    color = colors[i % len(colors)]  # Use modulo to loop over colors if there are more clusters than colors

    # Add 3D trace
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color=color, opacity=0.8),
        hovertext=hover_texts,
        name=f"Cluster {i}",
        legendgroup=f"group{i}"
    ), row=1, col=1)
    
    # Add 2D trace
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=5, color=color, opacity=0.8),
        hovertext=hover_texts,
        name=f"Cluster {i}",
        legendgroup=f"group{i}",
        showlegend=False
    ), row=1, col=2)

# Function to update visibility based on selected clusters
def update_visibility(change):
    visible_clusters = change['new']
    for i in range(len(fig.data)):
        fig.data[i].visible = (i // 2) in visible_clusters

# Dropdown widget for cluster selection
cluster_dropdown = widgets.SelectMultiple(
    options=[(f'Cluster {i}', i) for i in range(total_clusters)],
    value=list(range(total_clusters)),
    description='Clusters:',
    disabled=False
)
cluster_dropdown.observe(update_visibility, names='value')

# Display dropdown
display(cluster_dropdown)

fig.update_layout(
    width=1200,
    height=600,
    title='Visualization of Product Embeddings',
    margin=dict(l=0, r=0, b=0, t=40)
)

# Display the plot
fig.show()










import csv

# Path to save the CSV file
csv_file_path = "cluster_predictions.csv"

# Sort products by cluster label
sorted_products = sorted(zip(url_to_product.keys(), url_to_product.values(), cluster_labels), key=lambda x: x[2])

# Write results to CSV file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ['url', 'product_name', 'predicted_cluster']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # Write each row
    for product_url, product_name, cluster_label in sorted_products:
        writer.writerow({
            'url': product_url,
            'product_name': product_name,
            'predicted_cluster': cluster_label
        })

print("CSV file has been saved successfully!")



# predict_category(url_to_product, result_dict)


