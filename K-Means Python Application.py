# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/ifeol/Downloads/patients.csv")

# Preprocessing
# Select relevant columns
df = df[['BIRTHDATE', 'MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']]

# Convert BIRTHDATE to AGE
df['AGE'] = pd.to_datetime('today').year - pd.to_datetime(df['BIRTHDATE']).dt.year

# Encode categorical variables
df = pd.get_dummies(df, columns=['MARITAL', 'RACE', 'ETHNICITY', 'GENDER'], drop_first=True)

# Normalize numerical data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['AGE', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']])


# Finding the optimal number of clusters using Elbow method
# Calculate inertia for different values of K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(K_range, inertia, 'bx-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()



# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Plot clusters with a legend 
fig, ax = plt.subplots() 
scatter = ax.scatter(df['HEALTHCARE_EXPENSES'], df['HEALTHCARE_COVERAGE'], c=df['Cluster'], cmap='viridis') 

# Adding a legend 
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters") 
ax.add_artist(legend1)

# Visualize the clusters
plt.scatter(df['HEALTHCARE_EXPENSES'], df['HEALTHCARE_COVERAGE'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Healthcare Expenses')
plt.ylabel('Healthcare Coverage')
plt.title('Patient Segmentation')
plt.show()



