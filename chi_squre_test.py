import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

# Load cleaned dataset
df = pd.read_csv("spotify_songs_cleaned.csv")

# Create categorical mood feature (valence > 0.5 = positive mood, else negative)
df['mood'] = df['valence'].apply(lambda x: "Positive" if x > 0.5 else "Negative")

# Crosstab between playlist genre and mood
contingency_table = pd.crosstab(df['genre'], df['mood'])

# Chi-square test
chi2_stat, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Test Results")
print("Chi2 Stat:", chi2_stat)
print("p-value:", p)

if p < 0.05:
    print("✅ Significant relationship between Genre and Mood")
else:
    print("❌ No significant relationship between Genre and Mood")
#feature selection
# Load cleaned dataset
df = pd.read_csv("spotify_songs_cleaned.csv")

# Create a categorical mood variable based on valence & energy
df['mood'] = df.apply(lambda row: 
                      "Sad" if row['valence'] < 0.3 and row['energy'] < 0.4 
                      else ("Dance" if row['valence'] > 0.6 and row['energy'] > 0.7 
                      else ("Love" if row['valence'] > 0.5 and row['energy'] < 0.6 
                      else "Chill")), axis=1)

# Prepare X and y
X = df[["danceability","energy","valence","tempo","acousticness","instrumentalness","liveness","speechiness","loudness","duration_ms"]]
y = LabelEncoder().fit_transform(df["mood"])

# Scale features (Chi2 works with positive values)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply Chi-Square
chi_scores, p_values = chi2(X_scaled, y)

# Show results
chi_results = pd.DataFrame({"Feature": X.columns, "Chi2": chi_scores, "p-value": p_values})
chi_results = chi_results.sort_values(by="Chi2", ascending=False)
print("🔹 Chi-Square Feature Importance:\n", chi_results)
# Select Top 5 features
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X_scaled, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("✅ Selected Features:", selected_features)
#correlation analysis
plt.figure(figsize=(10,6))
sns.heatmap(df[["danceability","energy","valence","tempo","acousticness","instrumentalness","liveness","speechiness","loudness","duration_ms"]].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()


