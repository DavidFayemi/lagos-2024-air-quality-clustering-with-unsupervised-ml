# Unsupervised Machine Learning Clustering for Lagos Air Quality Pattern Discovery

## Final Year Project Overview

This project implements an operational air quality monitoring system for Lagos, Nigeria. Using unsupervised machine learning clustering, the system automatically processes monthly sensor data from the openAFRICA portal to discover pollution patterns across the city — identifying pollution hotspots, seasonal trends, and emission signatures that can inform environmental management decisions. Unlike a one-time analysis, the system runs continuously, updating its findings each month without manual intervention.

---

## 📊 What is This Project About?

**The Problem:**
Lagos, like many rapidly developing cities, faces significant air quality challenges. Environmental managers need to understand:

- Where are the most polluted areas?
- Are there distinct pollution patterns or zones?
- What conditions characterize different pollution levels?
- How can limited resources be targeted effectively?

**The Solution:**
Using machine learning clustering, we automatically discovered natural groupings in the data without predefined labels. This unsupervised approach reveals patterns that humans might miss and helps prioritize intervention areas.

---

## 🗂️ Data Overview

### Source

Monthly sensor data is automatically downloaded from the openAFRICA data portal at the start of each processing run. Each monthly file contains hourly sensor readings from multiple locations across Lagos, covering PM1, PM2.5, PM10, humidity, and temperature measurements. Data is fetched, processed, and stored without any manual file management.

### Data Structure - Original (Long Format)

The raw data from sensors was organized in **long format** (separate row for each measurement type):

| sensor_id | location | lat  | lon  | timestamp        | value_type | value |
| --------- | -------- | ---- | ---- | ---------------- | ---------- | ----- |
| S001      | Ikoyi    | 6.45 | 3.63 | 2024-01-01 10:30 | P0         | 25.3  |
| S001      | Ikoyi    | 6.45 | 3.63 | 2024-01-01 10:30 | P1         | 45.2  |
| S001      | Ikoyi    | 6.45 | 3.63 | 2024-01-01 10:30 | humidity   | 65    |

**Problem with this format:** Each measurement type is in a different row, making it hard to cluster and analyze relationships between variables.

### Measurements Collected

| Code        | Full Name                      | Meaning                          | Unit  |
| ----------- | ------------------------------ | -------------------------------- | ----- |
| **P0**      | Particulate Matter 1 (PM1)     | Ultra-fine particles             | µg/m³ |
| **P1**      | Particulate Matter 2.5 (PM2.5) | Fine particles (health-critical) | µg/m³ |
| **P2**      | Particulate Matter 10 (PM10)   | Coarse particles (dust)          | µg/m³ |
| humidity    | Relative Humidity              | Moisture in air                  | %     |
| temperature | Temperature                    | Air temperature                  | °C    |

---

## 🔄 Step 1: Data Transformation (Long → Wide Format)

### Why Transform?

The original long format had each measurement type in separate rows. For clustering, we need all measurements for one observation in the same row so we can compare them.

### How We Did It

We used a **pivot operation** to reorganize the data:

**Before (Long Format):**

```
S001, Ikoyi, P0, 25.3
S001, Ikoyi, P1, 45.2
S001, Ikoyi, humidity, 65
S001, Ikoyi, temp, 28.5
```

**After (Wide Format):**

```
S001, Ikoyi, PM1=25.3, PM2.5=45.2, humidity=65, temp=28.5
```

**Result:** Now each row represents one sensor reading at one time, with all measurements in separate columns.

### Why This Matters

- **Enables direct comparison**: Can now compare PM2.5 vs humidity in same row
- **Makes clustering possible**: Algorithms work on rows of features, not separate entries
- **Allows derived metrics**: Can calculate ratios like PM2.5/PM10 directly

---

## 🧹 Step 2: Data Cleaning (Making Data Reliable)

Raw sensor data is messy with errors, gaps, and invalid values. We performed systematic cleaning:

### 2.1 **Convert Timestamps**

- Parsed date-time strings into proper datetime format
- Removed invalid/corrupted timestamps
- Result: Standardized time references

### 2.2 **Remove Records Missing Critical Values**

Removed records where we couldn't identify:

- **Sensor ID**: Which sensor took the reading?
- **Location/Coordinates**: Where was it measured?
- **Timestamp**: When was it measured?
- **Measurement value**: What was the reading?

Records missing any of these → **deleted** (~5% of data)

### 2.3 **Remove Exact Duplicates**

- Some records appeared multiple times (logging errors)
- Removed all duplicates

### 2.4 **Handle Missing Pollution/Weather Values**

For missing PM and weather data, we used **intelligent step-by-step filling**:

1. **Forward fill per sensor**: Use that sensor's last reading (preserves local continuity)
2. **Backward fill per sensor**: Fill remaining gaps looking backward
3. **Sensor median**: If still missing, use average for that specific sensor
4. **Global median**: Final fallback using overall average

**Why this approach?** Each sensor has its own baseline. Using per-sensor values preserves local characteristics better than blanket global values.

### 2.5 **Remove Physically Impossible Values**

- **Negative PM values?** Impossible for particles. Removed.
- **Humidity < 0% or > 100%?** Invalid. Removed.
- **Temperature < -50°C or > 70°C?** Unrealistic for Lagos. Removed.

### 2.6 **Handle Extreme Outliers**

Strategy:

- Kept legitimate extreme values (high pollution days are real)
- Clipped values exceeding **99th percentile** (likely sensor malfunctions)
- Example: If PM2.5 usually maxes around 280 µg/m³, anything beyond that → clipped to 280

**Result:** Clean, reliable dataset with ~95% data retention

---

## 🏗️ Step 3: Feature Engineering (Creating New Insights)

Raw measurements are useful, but we created **derived features** that capture deeper patterns and relationships:

### 3.1 **Temporal Features** (Time-based patterns)

Why? Pollution patterns change throughout the day and seasons.

Features created:

- **Hour**: Extract time of day (0-23) - pollution peaks during rush hours
- **Day of week**: Monday=0 through Sunday=6 - weekdays differ from weekends
- **Month**: Season indicator (1-12) - seasonal variations in pollution
- **Is_weekend**: Binary flag - activity/traffic patterns differ
- **Hour_sin & Hour_cos**: Cyclical encoding - treats hour 23 and hour 0 as neighbors (not opposites)

**Purpose:** Capture daily/weekly/seasonal rhythms in pollution

### 3.2 **Pollutant Ratio Features** (Pollution source indicators)

Why? Ratios reveal what's causing pollution without knowing exact sources

Features created:

- **Fine ratio = PM2.5 / PM10**
  - **>1** = more fine particles = combustion (cars, industry, generators)
  - **<1** = more coarse particles = dust and soil
- **Coarse ratio = PM10 / PM2.5** (inverse view)

**Purpose:** Distinguish pollution types and identify likely sources

### 3.3 **Weather Interaction Features** (Environmental effects)

Why? Pollution and weather don't just add up - they interact

Features created:

- **PM × Humidity**: How moisture affects particle behavior
- **PM × Temperature**: How heat affects atmospheric chemistry and particle formation

**Purpose:** Capture combined environmental effects that simple addition misses

### 3.4 **Daily Aggregation** (Noise reduction)

Why? Hourly data is noisy; daily averages reveal clearer patterns

Process:

- Grouped all hourly readings by **sensor + location + date**
- Calculated **daily average** for each measurement
- Reduced data from ~100,000 hourly rows to ~5,000 daily rows

**Purpose:** Smooth out hourly fluctuations to see bigger patterns

### 3.5 **Pollution Severity Index** (Composite score)

Created a single-number representation of overall pollution:

- Standardized PM1, PM2.5, PM10 (same scale: mean=0, std=1)
- Averaged them → single pollution index
- Result: Comparable metric combining all three PM measurements

---

## 🎯 Step 4: Feature Selection (Choosing What Matters)

### Selected 6 Features for Clustering:

1. **PM1** - Ultra-fine particle mass
2. **PM2.5** - Fine particles (main health impact)
3. **PM10** - Coarse particles (visibility/respiratory)
4. **Fine_ratio** - Combustion vs. dust indicator
5. **Humidity** - Meteorological context
6. **Temperature** - Meteorological context

### Why Only 6?

- **Too many features** → "curse of dimensionality" (algorithms get confused with noise)
- **Too few features** → miss important patterns
- **Golden zone**: 6 features capture essential variation without noise
- Each feature should be independent and informative

---

## ⚙️ Step 5: Data Standardization (Making Fair Comparisons)

### The Problem

Our features have very different scales:

- PM2.5 ranges: 0-300 µg/m³
- Humidity ranges: 0-100 %
- Temperature ranges: 15-35 °C

**Issue:** Algorithms could over-weight PM2.5 just because its numbers are larger!

### The Solution: StandardScaler

Transform each feature to have:

- **Mean = 0** (centered around zero)
- **Standard Deviation = 1** (uniform spread)

**Formula:** `(value - mean) / standard_deviation`

**Example:**

```
Original PM2.5: 150 → Standardized: +1.2
Original Humidity: 70 → Standardized: +0.8
Original Temp: 28 → Standardized: +1.1
```

**Result:** All features on same scale (~-3 to +3 range)

**Why it matters:** Ensures no feature dominates just because the original numbers are larger

---

## 🤖 Step 6: Clustering Analysis (Three Algorithms)

We used **three different algorithms** because each reveals different aspects of the data:

---

### **Algorithm 1: K-Means Clustering**

#### How It Works (Simple Explanation)

Like organizing students into school groups:

1. **Start:** Pick K random group centers
2. **Assign:** Assign each student to nearest group
3. **Move centers:** Calculate average position of each group
4. **Repeat:** Reassign students → update centers → repeat
5. **Stop:** When assignments stop changing

#### How It Works (Technical)

- Minimizes **within-cluster variance** (distances from points to cluster centers)
- Iterative optimization converging to local minimum
- Results depend on initial random placement (we run 30 times, keep best)

#### Finding Optimal K: Three Methods

**Method 1: Elbow Method**

- Try K = 2, 3, 4, 5, ..., 10
- For each K, calculate "inertia" (sum of all distances to nearest center)
- Plot inertia vs K
- Look for "elbow" point where improvement levels off
- Interpretation: Additional clusters beyond elbow add little benefit

**Method 2: Silhouette Score** (most reliable)
For each point, measures:

- Closeness to own cluster (want small)
- Distance to nearest other cluster (want large)
- Score: -1 to +1 where higher is better
- Select K maximizing average silhouette score

**Method 3: Davies-Bouldin Index** (validation)
Compares ratio of within-cluster to between-cluster distances

- Lower is better
- Validates silhouette result

#### Why K-Means?

✅ Fast and computationally efficient  
✅ Works well with spherical (rounded) clusters  
✅ Clear, interpretable results  
✅ Good baseline for comparison

#### Limitations

❌ Must specify cluster count K in advance  
❌ Can't identify outliers (every point assigned)  
❌ Struggles with non-spherical cluster shapes  
❌ Sensitive to initial random placement

---

### **Algorithm 2: DBSCAN** (Density-Based Spatial Clustering)

#### How It Works (Simple Explanation)

Like identifying friend groups at a party:

1. **Start:** Look at a person and their neighbors within distance `eps`
2. **Check density:** If enough people nearby (min_samples), start a cluster
3. **Grow cluster:** Add their neighbors
4. **Expand:** For each new member, check their neighbors
5. **Repeat:** Until cluster stops growing
6. **Handle loners:** People with no dense neighborhood = "noise"/outliers

#### How It Works (Technical)

- **Core point**: Point with ≥min_samples neighbors within eps distance
- **Density-connected**: Points form cluster if connected by density
- **Noise points**: Don't meet density criteria (labeled -1)

#### Finding Optimal Parameters

**Method 1: K-Distance Graph**

- For each point, find distance to 4th-nearest neighbor
- Sort all distances and plot
- Look for "knee" point (sharp bend)
- Suggests good eps value

**Method 2: Parameter Grid Search**

- Try combinations:
  - eps ∈ [0.3, 0.5, 0.7, 1.0, 1.3]
  - min_samples ∈ [3, 5, 10]
  - Total: 15 combinations
- Evaluate each using silhouette on non-noise points
- Select best-performing combination

#### Why DBSCAN?

✅ Finds arbitrary cluster shapes (not just spheres)  
✅ Automatically identifies outliers/anomalies  
✅ No need to specify cluster count K  
✅ Better for real-world messy data

#### Limitations

❌ Very sensitive to parameter selection (eps and min_samples)  
❌ Struggles with varying cluster densities  
❌ Parameter tuning often requires experimentation  
❌ Computationally slower for large datasets

---

### **Algorithm 3: Hierarchical Clustering**

#### How It Works (Simple Explanation)

Like creating a family tree:

1. **Start:** Each person is in their own group (N clusters)
2. **Merge:** Find two closest groups and merge
3. **Record:** Document this merger
4. **Repeat:** Find next closest pair of groups → merge
5. **Result:** Tree showing all mergers from most similar to least
6. **Cut tree:** Decide where to "cut" to get final clusters

#### How It Works (Technical)

- **Agglomerative (bottom-up):** Start with N individual clusters, merge until 1
- **Linkage method:** Defines how to measure distance between groups:
  - **Ward**: Minimize variance of merged cluster (usually best)
  - **Complete**: Use farthest points in two groups
  - **Average**: Use average distance between all pairs
  - **Single**: Use closest points (can create chain effect)

#### Finding Optimal Clusters

**Step 1: Try Different Linkage Methods**

- Generate tree for each: Ward, Complete, Average, Single
- Evaluate each with silhouette score at multiple cluster counts
- Ward typically performs best

**Step 2: Find Optimal Cluster Count**

- Using Ward linkage, test cutting at different heights
- Try n_clusters = 2, 3, 4, ..., 10
- Select count maximizing silhouette score

#### Why Hierarchical?

✅ Interpretable dendrogram shows relationships between clusters  
✅ No parameter tuning needed (just linkage choice)  
✅ Multiple valid solutions at different granularities  
✅ Clear hierarchical structure of similarity

#### Limitations

❌ Slower computation (O(n²) complexity)  
❌ Can't undo previous merges (greedy algorithm)  
❌ Sensitive to outliers affecting initial merges  
❌ Different linkage methods produce different results

---

## 📊 Step 7: Evaluating Cluster Quality

We used **three independent metrics** for robust evaluation:

### Metric 1: Silhouette Score (Range: -1 to +1)

**What it measures:**
For each point, comparing:

- Distance to own cluster center
- Distance to nearest other cluster

**Formula interpretation:**

- **+1**: Point perfectly in own cluster, far from others
- **0**: Point on border between clusters
- **-1**: Point assigned to wrong cluster

**Practical interpretation:**

- **>0.7**: Strong, well-defined clusters
- **0.5-0.7**: Reasonable clusters
- **<0.5**: Weak clusters, likely poor grouping

### Metric 2: Davies-Bouldin Index (Range: 0 to ∞)

**What it measures:**
Ratio of within-cluster to between-cluster distances

- Measures how separated clusters are
- Lower is better

**Practical interpretation:**

- **0**: Perfect (clusters infinitely separated)
- **0.5-1.0**: Good separation
- **1.0-1.5**: Acceptable
- **>2.0**: Poor separation

### Metric 3: Calinski-Harabasz Index (Range: 0 to ∞)

**What it measures:**
Ratio of between-cluster variance to within-cluster variance

- How concentrated clusters are vs. how spread out
- Higher is better

**Practical interpretation:**

- **>50**: Good clustering
- **50-100**: Strong clustering
- **<30**: Weak clustering

**Why three metrics?**

- No single metric is perfect
- Different metrics emphasize different cluster qualities
- If all three agree → high confidence
- If they disagree → algorithm uncertainty

---

## 🔍 Step 8: Cluster Profiling and Interpretation

For each cluster, we calculated detailed characteristics:

### Profile Metrics

- **Size**: Number of observations in cluster
- **Locations**: Which geographic areas represented
- **PM1, PM2.5, PM10**: Mean values and standard deviation (variability)
- **Fine ratio**: Average = pollution source indicator
- **Humidity & Temperature**: Average conditions
- **Air quality category**: EPA classification (Good/Moderate/Unhealthy/etc.)
- **Top contributing locations**: Which areas most represented

### Why Profile Clusters?

- Transforms abstract numbers into understandable characteristics
- Enables domain validation (do clusters match expected air quality zones?)
- Provides actionable groupings for management decisions
- Reveals environmental conditions associated with different pollution levels

---

## 💡 Step 9: Algorithm Comparison

Compared all three algorithms using metrics:

| Algorithm    | Silhouette | Davies-Bouldin | Calinski-Harabasz | Interpretation             |
| ------------ | ---------- | -------------- | ----------------- | -------------------------- |
| K-Means      | Score      | Index          | Index             | Fast, spherical clusters   |
| DBSCAN       | Score      | Index          | Index             | Finds outliers, shapes     |
| Hierarchical | Score      | Index          | Index             | Hierarchical relationships |

**Decision Logic:**

- **Best Silhouette** → Most compact, well-separated clusters
- **Best Davies-Bouldin** → Least overlapping clusters
- **Best Calinski-Harabasz** → Strongest cluster definition

**Recommendation:** Used highest-scoring algorithm for final insights

---

## 📈 Key Findings & Insights

### 1. **Pollution Hotspots**

- Identified specific geographic areas with elevated pollution
- Different clusters have different average pollution levels
- Some areas consistently worse than others

### 2. **Temporal Patterns**

- Which months have worst pollution (seasonality)
- Peak pollution hours (rush hour effects)
- Seasonal variations and trends

### 3. **Pollution Types** (Source Identification)

Using fine_ratio patterns:

- High fine ratio → combustion sources (cars, generators, industry)
- Low fine ratio → dust sources (unpaved areas, construction, Saharan dust)
- Different clusters have different source signatures

### 4. **Meteorological Relationships**

- How humidity affects pollution transport and particle behavior
- Temperature effects on atmospheric chemistry
- Wind/weather conditions that exacerbate pollution

### 5. **Air Quality Distribution**

EPA-based categories:

- **Good**: ≤12 µg/m³ PM2.5
- **Moderate**: 12-35 µg/m³
- **Unhealthy for Sensitive**: 35-55 µg/m³
- **Unhealthy**: 55-150 µg/m³
- **Very Unhealthy**: >150 µg/m³

Percentage of days in each category reveals Lagos air quality situation

---

## 🎯 Environmental Management Recommendations

### Immediate Actions (1 Month)

1. Deploy targeted monitoring equipment in identified pollution hotspots
2. Establish automated alert systems for threshold exceedances
3. Analyze specific emission sources in high fine_ratio clusters

### Short-Term Interventions (1-3 Months)

1. Implement traffic restriction policies in high-pollution clusters
2. Strengthen industrial emission regulation during peak months
3. Issue public health warnings for "Unhealthy" AQI days
4. Educate public on air quality and health impacts

### Long-Term Strategy (3-12 Months)

1. Expand sensor network to unmeasured areas
2. Conduct source apportionment study (quantify vehicle vs. industrial vs. dust)
3. Develop predictive air quality forecast model
4. Set quantifiable targets (e.g., 15% PM2.5 reduction)
5. Implement vehicle emission standards
6. Regulate industrial operations based on cluster analysis

---

## 🔧 Technical Stack

**Languages & Libraries:**

- **Python 3.10+**: Programming language
- **Pandas**: Data manipulation, transformation, grouping
- **NumPy**: Numerical computations and array operations
- **Scikit-learn**: Machine learning (KMeans, DBSCAN, StandardScaler, metrics)
- **SciPy**: Hierarchical clustering (linkage, fcluster, dendrogram)
- **Matplotlib & Seaborn**: Data visualization and plots

**Why these tools?**

- Pandas: Industry standard for data manipulation
- Scikit-learn: Industry standard for ML
- SciPy: Best for hierarchical clustering
- Matplotlib/Seaborn: Publication-quality visualizations

---

## 📚 How to Use This Notebook
## Running the Pipeline

**Prerequisites:** Python 3.10+, a MongoDB Atlas cluster, and access to the repository.

**Setup:**
```bash
# Clone and enter the project
git clone <repository-url>
cd <project-folder>

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root with your MongoDB connection string:
```
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
```

**Running:**
```bash
# Process the previous month automatically
python run_pipeline.py
```

Results are stored in MongoDB Atlas under the `lagos_air_quality` database — cluster metrics and centroids in `processed_months`, and daily sensor observations with cluster assignments in `daily_observations`. Executed notebooks are saved to the `outputs/` folder for reference.

The pipeline runs automatically on the 1st of every month via GitHub Actions, so manual execution is typically only needed for testing or reprocessing a specific month.

---

## 🎓 Key Concepts Explained Simply

| Concept              | Simple Explanation                           | Why It Matters                                    |
| -------------------- | -------------------------------------------- | ------------------------------------------------- |
| **Clustering**       | Grouping similar items together              | Reveals natural patterns humans would miss        |
| **Unsupervised**     | No labels provided; algorithm finds patterns | Most real-world data isn't pre-labeled            |
| **Standardization**  | Make all measurements same scale             | Prevents large numbers dominating results         |
| **Silhouette Score** | How tight and separated clusters are         | Quality metric: >0.7 = good clusters              |
| **Outliers**         | Points that don't fit any pattern            | May represent anomalies, errors, or unique events |

| **Pivot table** | Reorganize data rows into columns | Transform data from unusable to clusterable format |

---

## 📝 Files in This Project

```
lagos-air-quality/
├── README.md                                                        
├── run_pipeline.py
├── Lagos(2024) Air quality clustering using unsupervised ML.ipynb
├── .github/
│   └── workflows/
│       └── pipeline.yml                                             
└── requirements.txt
```

---

## 🚀 Next Steps After Running Notebook

1. **Verify Results**: Do clusters match known geography of Lagos?
2. **Validate Against Reality**: Compare to known industrial/traffic areas
3. **Domain Review**: Have environmental experts review findings
4. **Stakeholder Presentation**: Share findings with Lagos environmental agencies
5. **Implementation**: Start with recommended immediate actions
6. **Monitor Progress**: Track air quality improvements over time

---

## 📞 Project Details

- **Objective**: Final Year Project on Urban Air Quality Analysis
- **Methodology**: Unsupervised Machine Learning (Clustering)
- **Data Points Analyzed**:
  - ~5,000–10,000 hourly readings per monthly run
  - ~58 daily aggregates used for clustering per month
- **Sensor Types**:
  - PMS5003 (Particulate Matter measurements)
  - DHT22 (Humidity and Temperature)

---

## ⚖️ Assumptions & Limitations

### Assumptions Made:

- Sensor readings are accurate (after outlier removal)
- Daily averages are meaningful aggregation level
- EPA PM2.5 standards apply in Lagos context
- 2024 data is representative of typical year
- Forward fill preserves temporal continuity

### Limitations:

- **Feature selection**: Results depend on which features we chose to cluster
- **Algorithm sensitivity**: Different algorithms/parameters → different results
- **Geographic coverage**: Only covers areas with sensors (not all of Lagos)
- **Historical context**: No pre-2024 data for comparison
- **Causality**: Can identify correlation but not determine causes
- **External validation**: No ground-truth clusters to validate against

---

## 📜 License & Attribution

This project uses open-source libraries under appropriate licenses.
Data sourced from OpenAfrica Lagos Air Quality Initiative.

---

**Last Updated:** February 2026  
**Status:** Final Year Project - Complete Analysis  
**Version:** 1.0
