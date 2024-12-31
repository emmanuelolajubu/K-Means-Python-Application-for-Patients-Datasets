
# Import required libraries
import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        # Load dataset
        data = pd.read_csv("C:/Users/ifeol/Downloads/patients.csv")
        self.df = pd.DataFrame(data)

    def preprocess_numeric_data(self, df):
        """
        Preprocess the DataFrame to retain only numeric columns for scaling and clustering.
        Cleans any NaN, infinite or -infinite values.
        """
        df_numeric = df.select_dtypes(include=[np.number])
        # Drop rows with NaN, Infinite, or -Infinity
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).dropna()
        return df_numeric

    def test_data_preprocessing(self):
        # Ensure data preprocessing works correctly
        df_numeric = self.preprocess_numeric_data(self.df)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)

        # Check that the scaled data has the same number of rows as the valid numeric rows
        self.assertEqual(df_scaled.shape[0], df_numeric.shape[0])
        self.assertEqual(df_scaled.shape[1], df_numeric.shape[1])

    def test_kmeans_clustering(self):
        # Test K-means clustering
        df_numeric = self.preprocess_numeric_data(self.df)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(df_scaled)

        # Check the number of clusters matches expected values.
        self.assertEqual(len(clusters), df_numeric.shape[0])
        self.assertIn(max(clusters), [0, 1, 2])
        self.assertIn(min(clusters), [0, 1, 2])

    def test_invalid_data(self):
        # Test handling of invalid data
        # Create a DataFrame with non-numeric and invalid data (all NaN values)
        df_invalid = pd.DataFrame({'Invalid Column': [None] * 10})
        df_invalid = self.preprocess_numeric_data(df_invalid)  # This will clear invalid data
        with self.assertRaises(ValueError):
            if df_invalid.empty or df_invalid.shape[1] == 0:
                raise ValueError("Invalid or empty DataFrame passed to scaler.")
            scaler = StandardScaler()
            scaler.fit_transform(df_invalid)


if __name__ == '__main__':
    unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TestKMeansClustering))
