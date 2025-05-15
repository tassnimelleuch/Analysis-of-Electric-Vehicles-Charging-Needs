import pandas as pd
from sklearn.preprocessing import StandardScaler


from google.colab import drive
drive.mount('/content/drive')
file_path ="/content/drive/MyDrive/Projet Machine Learning /Electric Vehicle Population Data.csv"
df = pd.read_csv(file_path)
print("Original Dataset:")
print(df.head())
rows, columns = df.shape

print("Number of rows:", rows)
print("Number of columns:", columns)
print("Column Names:")
print(df.columns) 
# Create a filtered dataset with only the necessary columns
filtered_df = df[[
    'City', 'State', 'Model Year', 'Electric Vehicle Type',
    'Electric Range', 'Vehicle Location', 'Electric Utility', '2020 Census Tract'
]]

# Display the filtered dataset
print("Filtered Dataset:")
display(filtered_df.head())
rows, columns = filtered_df.shape

print("Number of rows:", rows)
print("Number of columns:", columns) print('columns with missing values')
print(filtered_df.isnull().sum()) import numpy as np
filtered_df['Longitude'] = filtered_df['Vehicle Location'].apply(lambda x: float(x.split(' ')[1].strip('(')) if isinstance(x, str) else np.nan)
filtered_df['Latitude'] = filtered_df['Vehicle Location'].apply(lambda x: float(x.split(' ')[2].strip(')')) if isinstance(x, str) else np.nan)

print("filtered dataset")
print(filtered_df.columns)
# Handle missing values
# Fill 'Electric Range' with the mode
filtered_df['Electric Range'] = filtered_df['Electric Range'].fillna(filtered_df['Electric Range'].mode()[0])


# Fill 'City' with the mode
filtered_df['City'] = filtered_df['City'].fillna(filtered_df['City'].mode()[0])

# Fill '2020 Census Tract' with the mode
filtered_df['2020 Census Tract'] = filtered_df['2020 Census Tract'].fillna(filtered_df['2020 Census Tract'].mode()[0])

#Drop missing values for the longitude and latitude
filtered_df = filtered_df.dropna(subset=['Longitude','Latitude'])


# Verify no missing values remain
display("Missing values after handling:")
display(filtered_df.isnull().sum()) # Encode 'Electric Vehicle Type' as binary (BEV = 1, PHEV = 0)
filtered_df['Electric Vehicle Type'] = filtered_df['Electric Vehicle Type'].apply(
    lambda x: 1 if x == 'Battery Electric Vehicle (BEV)' else 0
)

# Encode 'Electric Utility' as numerical codes
filtered_df['Electric Utility'] = filtered_df['Electric Utility'].astype('category').cat.codes

# Verify the encoded dataset
print("Dataset after encoding:")
print(filtered_df.head()) # Aggregate data by City, State, and Census Tract
city_aggregated = filtered_df.groupby(['City', 'State', '2020 Census Tract']).agg({
    'Electric Range': 'mean',  # Average electric range per city
    'Model Year': 'mean',      # Average model year per city
    'Longitude': 'mean',       # Average longitude per city
    'Latitude': 'mean',        # Average latitude per city
    'Electric Utility': 'nunique',  # Number of unique utilities per city
    'Electric Vehicle Type': 'mean'  # Percentage of BEVs per city
}).reset_index()

# Rename columns
city_aggregated.columns = [
    'City', 'State', '2020 Census Tract', 'Avg_Electric_Range', 'Avg_Model_Year',
    'Avg_Longitude', 'Avg_Latitude', 'Num_Electric_Utilities', 'Pct_BEV'
]

# Display the aggregated dataset
print("Aggregated Dataset:")
display(city_aggregated.head()) display(city_aggregated[['Avg_Electric_Range', 'Avg_Model_Year', 'Num_Electric_Utilities', 'Pct_BEV']].describe()) from sklearn.preprocessing import StandardScaler

# Define the feature matrix X
X = city_aggregated[['Avg_Electric_Range', 'Avg_Model_Year', 'Num_Electric_Utilities', 'Pct_BEV']].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature distributions
plt.figure(figsize=(12, 8))
for i, col in enumerate(['Avg_Electric_Range', 'Avg_Model_Year', 'Num_Electric_Utilities', 'Pct_BEV']):
    plt.subplot(2, 2, i+1)
    sns.histplot(city_aggregated[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show() import numpy as np

# Apply log transformation to highly skewed features
city_aggregated['Log_Electric_Range'] = np.log1p(city_aggregated['Avg_Electric_Range'])

# Update the feature matrix X
X = city_aggregated[['Log_Electric_Range', 'Avg_Model_Year', 'Num_Electric_Utilities', 'Pct_BEV']].values # Compute correlation matrix
corr_matrix = city_aggregated[['Log_Electric_Range', 'Avg_Model_Year', 'Num_Electric_Utilities', 'Pct_BEV']].corr()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show() from sklearn.cluster import KMeans

k = 4
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

# Predict clusters for the data
y_labels = model.predict(X)

# Add cluster labels to the DataFrame
city_aggregated['Cluster'] = y_labels

# Display the upd # Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled) 
# Determine the optimal number of clusters using the Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)
ated DataFrame
print("Dataset with Cluster Labels:")
print(city_aggregated.head()) import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show() 
# Fit K-Means with the optimal number of clusters
k = 4  # Change this based on the Elbow Method
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_pca)
 
# Add cluster labels to the DataFrame
city_aggregated['Cluster'] = kmeans.labels_

# Visualize the clusters in 2D PCA space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=city_aggregated['Cluster'], palette='viridis', s=100)
plt.title('K-Means Clustering with PCA (2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()
 
# Analyze cluster characteristics
cluster_summary = city_aggregated.groupby('Cluster').agg({
    'Avg_Electric_Range': 'mean',
    'Avg_Model_Year': 'mean',
    'Num_Electric_Utilities': 'mean',
    'Pct_BEV': 'mean'
}).reset_index()

print("Cluster Summary:")
print(cluster_summary)
from sklearn.metrics import silhouette_score

# Calculate Silhouette Score
silhouette_avg = silhouette_score(X_pca, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.4f}") # Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns # Define thresholds for each feature
range_threshold = city_aggregated['Avg_Electric_Range'].median()
bev_threshold = city_aggregated['Pct_BEV'].median()
utility_threshold = city_aggregated['Num_Electric_Utilities'].median()
year_threshold = city_aggregated['Avg_Model_Year'].median()  # Median model year

# Create the target variable
city_aggregated['Needs_More_Charging_Stations'] = (
    (city_aggregated['Avg_Electric_Range'] < range_threshold) |  # Low electric range
    (city_aggregated['Pct_BEV'] > bev_threshold) |              # High percentage of BEVs
    (city_aggregated['Num_Electric_Utilities'] < utility_threshold) |  # Few electric utilities
    (city_aggregated['Avg_Model_Year'] < year_threshold)        # Older vehicles
).astype(int)

# Display the target variable distribution
print("Target Variable Distribution:")
print(city_aggregated['Needs_More_Charging_Stations'].value_counts()) 
# Define the feature matrix X and target variable y
X = city_aggregated[['Avg_Electric_Range', 'Avg_Model_Year', 'Num_Electric_Utilities', 'Pct_BEV']]
y = city_aggregated['Needs_More_Charging_Stations']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  from imblearn.over_sampling import SMOTE

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Check the new distribution
print("Resampled Target Variable Distribution:")
print(pd.Series(y_train_res).value_counts()) 
# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show() 
# KNN Model
print("KNN Model:")
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_res, y_train_res)
evaluate_model(knn, X_test_scaled, y_test)
# SVM Model
print("SVM Model:")
svm = SVC(random_state=42, probability=True)
svm.fit(X_train_res, y_train_res)
evaluate_model(svm, X_test_scaled, y_test) # Import Libraries
import folium
from folium.plugins import MarkerCluster

# Predict which cities need more charging stations using XGBoost
city_aggregated['Needs_More_Charging_Stations_Pred'] = xgb.predict(X_scaled)

# Filter cities that need more charging stations
cities_needing_charging = city_aggregated[city_aggregated['Needs_More_Charging_Stations_Pred'] == 1]

# Group by state and count the number of cities needing more charging stations
states_needing_charging = cities_needing_charging.groupby('State').size().reset_index(name='Count')

# Create a base map centered on the US
map_center = [37.0902, -95.7129]  # Center of the US
mymap = folium.Map(location=map_center, zoom_start=4)

# Add a MarkerCluster for cities needing more charging stations
marker_cluster = MarkerCluster().add_to(mymap)

# Add markers for each city needing more charging stations
for idx, row in cities_needing_charging.iterrows():
    folium.Marker(
        location=[row['Avg_Latitude'], row['Avg_Longitude']],
        popup=f"{row['City']}, {row['State']}<br>Avg Electric Range: {row['Avg_Electric_Range']:.2f} miles",
        icon=folium.Icon(color='red', icon='bolt', prefix='fa')
    ).add_to(marker_cluster)

# Add a choropleth layer for states needing more charging stations
folium.Choropleth(
    geo_data='https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json',
    name='choropleth',
    data=states_needing_charging,
    columns=['State', 'Count'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of Cities Needing More Charging Stations'
).add_to(mymap)

# Add layer control
folium.LayerControl().add_to(mymap)

# Display the map
mymap 
