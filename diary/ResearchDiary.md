# Research Diary - FAHAD ALI - GROUP 6
## Date: 2026-03-07
## Total Time: 2 hours

### 1. Task / Goal for this session
Complete project setup and initial data exploration.

### 2. Work Done & Progress
- Fixed Python path issues in test_setup.py
- Successfully ran test_setup.py - all systems go!
- Created and ran 01_data_exploration.ipynb
- Found Problematic series:
   - All zeros: 154 households
   - Mostly zeros (>=30.0% zeros): 432 households
   - Constant/near constant: 24 households
   - Negative values: 0 households
   - TOTAL problematic: 610 households
 Total problematic households: 610 out of 17547 (3.5%)

### 3. Challenges & Decisions
- Noticed that some households have extremely high values (up to [max_value])
- 432 households have >30% zeros - we need to decide whether to keep them
- Strong weekly pattern observed - peaks on weekends?

### 4. Next Steps
- Discuss with team about handling problematic series
- Research feature extraction methods for time series clustering
- Prepare for clustering task

### 5. Questions / For Team Discussion
- Should we remove households with >50% zeros?
- What normalization method is best for clustering?

## Date: 2026-03-10

### Today's Achievements

#### 1. ✅ Project Infrastructure Setup
- Created Discord server for team communication
- Invited all team members
- Set up channels: #general, #coding, #research, #meeting-notes
- Established Git repository on  [GitHub]
- Initial commit with project structure and code
- All team members have cloned and tested the setup

#### 2. ✅ Completed Data Exploration Phase

**Dataset Overview:**
- 17,547 households with daily consumption for 2023
- 365 days of data (6.4 million observations)
- No missing values - excellent data quality!

**Key Statistical Findings:**
- Mean consumption: 9.13 units
- Median: 6.05 units (distribution is right-skewed)
- Range: 0.00 to 1051.74 units (extreme outliers exist)
- Zero values: 2.63% of all readings

**Data Quality Assessment:**
- Identified 610 problematic households (3.5% of total):
  - 154 all-zero series (possibly vacant properties)
  - 432 mostly-zero series (>30% zeros)
  - 24 constant-value series (meter issues?)
- These will need special handling in clustering

**Seasonal Patterns Discovered:**
- Strong winter peaks (Dec-Jan): ~11 units
- Summer troughs (Jul-Sep): ~7 units
- Weekly pattern: slightly higher on weekends
- Clear seasonal cycle present in most households

**Household Variability:**
- Wide range of consumption patterns
- From steady low consumers to highly variable
- Coefficient of variation ranges from 0.1 to >5.0

#### 3. Discussion Points for Team Meeting

**Critical Decisions Needed:**
1. **Problematic Series Handling**
   - Should we remove all 610 problematic households?
   - Or keep some with special treatment?
   - Proposal: Remove all-zero, keep mostly-zero with flag

2. **Outlier Treatment**
   - Max value 1052 is 100x the median
   - Likely data errors or industrial consumers
   - Options: cap at 99.9th percentile, remove, or keep

3. **Normalization Strategy**
   - Z-score vs Min-Max vs Robust scaling
   - Z-score seems best for clustering (preserves shape)

4. **Clustering Approach**
   - Raw time series vs extracted features
   - Features: mean, variance, seasonality strength, trend
   - Consider: k-Shape, DTW, or feature-based K-means


**This Week's Goals:**
- [ ] Finalize preprocessing pipeline
- [ ] Implement 2-3 clustering algorithms
- [ ] Determine optimal number of clusters (k)
- [ ] Document findings in research diaries

#### 5. 💡 Key Insights for Final Report

- The data shows clear segmentation potential
- Seasonal patterns are strong and consistent
- Problematic series (3.5%) need careful handling
- Outliers may represent a distinct cluster
- Z-score normalization recommended before clustering


### Date: 2026-03-17
## Session Summary: Feature Extraction & PCA Analysis
**Duration:** 4 hours
**Phase:** Feature Engineering & Dimensionality Reduction

### 1. Tasks Completed

#### ✅ Feature Extraction (26 features per household)
- Statistical features: mean, median, std, cv, iqr, range, skewness, kurtosis, quantiles
- Temporal features: weekend_ratio, summer_peak, winter_peak, trend_slope, trend_r2, seasonal_strength
- Peak features: n_peaks, peak_density, avg_peak_height, max_peak_height, peak_height_ratio
- Zero features: zero_percentage, max_zero_run, avg_zero_run

#### ✅ Data Quality Assessment
- Identified 18,009 missing values across several features
- Main culprits: skewness (154), kurtosis (154), winter_peak (17,393!), trend_r2 (154), seasonal_strength (154)
- **Decision**: Used median imputation (SimpleImputer) to handle missing values
- **Rationale**: Median is robust to outliers and preserves data distribution

#### ✅ Correlation Analysis
- Found 38 highly correlated feature pairs (|r| > 0.8)
- Key redundant groups:
  - Level features: mean, median, q25, q75, q90, q95 (all r > 0.87)
  - Variability features: std, iqr, range, peak heights (r > 0.9)
  - Zero features: zero_percentage, max_zero_run, avg_zero_run (r > 0.89)
  - Peak features: n_peaks and peak_density (r = 1.0 - perfectly correlated!)

#### ✅ PCA Dimensionality Reduction
- **Results:**
  - 95% variance explained by just 10 components (from 26 original features)
  - 90% variance explained by 8 components
  - 80% variance explained by 5 components
- **Interpretation of Principal Components:**

| PC | Variance | What it Represents | Key Features |
|-----|----------|-------------------|--------------|
| PC1 | 28.5% | Overall consumption level | q95, q90, q75, std, mean |
| PC2 | 18.2% | Consumption pattern/shape | cv, zero_percentage, peak_height_ratio, skewness |
| PC3 | 12.1% | Zero consumption patterns | avg_zero_run, max_zero_run, kurtosis |
| PC4 | 8.7% | Seasonality & zeros | seasonal_strength, zero patterns |
| PC5 | 6.5% | Trend & seasonal peaks | trend_slope, summer_peak, n_peaks |

#### ✅ Feature Selection for Clustering
- Selected 13 representative features (reduced from 26):
  - **Level**: mean
  - **Variability**: cv
  - **Shape**: skewness, kurtosis
  - **Temporal**: weekend_ratio, summer_peak, winter_peak, trend_slope, seasonal_strength
  - **Peaks**: n_peaks, peak_height_ratio
  - **Zeros**: zero_percentage, avg_zero_run

### 2. Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| **Keep 13 selected features for clustering** | Balances interpretability with information content; avoids redundancy |
| **Also keep PCA components as alternative** | Allows comparison; PCA might capture patterns better |
| **Use median imputation for missing values** | Robust to outliers; preserves data structure |
| **Remove winter_peak from selected features** | 17,393 missing values indicates calculation issue for constant series |
| **Will test both feature-based and PCA-based clustering** | To see which yields more interpretable clusters |

### 3. Challenges Encountered

1. **Missing Values in winter_peak** (17,393 NaN)
   - Cause: For households with zero or constant consumption, winter peak calculation failed
   - Solution: Median imputation, but flagged as potential issue

2. **Perfect Correlation between n_peaks and peak_density**
   - peak_density = n_peaks/365 * 365 = n_peaks (redundant)
   - Solution: Will keep only n_peaks in selected features

3. **Computational Time**
   - Feature extraction took ~5 minutes for 17,547 households
   - Each household required multiple calculations
   - Solution: Added progress indicators; acceptable for one-time processing

### 4. Insights Discovered

 **Consumption patterns are multi-dimensional:**
- Households differ in **how much** they consume (PC1)
- They also differ in **how they consume** - steady vs. variable (PC2)
- Zero consumption patterns identify **vacation homes or seasonal users** (PC3)
- Seasonality distinguishes **summer-peaking vs. winter-peaking** households (PC4, PC5)

 **Data Quality is good overall:**
- Only 3.5% problematic series (610 households)
- Most features have complete data
- Imputation handled remaining issues

 **PCA confirms our feature engineering:**
- The components align with our intuitive feature groups
- 10 components capture 95% of information - significant compression!

### 5. Next Steps (Tomorrow)

#### 🔜 Phase 3: Clustering

1. **Run clustering notebook (03_clustering.ipynb)**
   - Test k from 2 to 15
   - Compare feature-based vs PCA-based clustering
   - Evaluate using silhouette score, Davies-Bouldin index

2. **Determine optimal number of clusters (k)**
   - Use elbow method
   - Consider silhouette scores
   - Ensure clusters are interpretable

3. **Analyze and profile clusters**
   - Create radar charts for each cluster
   - Describe characteristics of each household type
   - Validate with domain knowledge

4. **Save cluster assignments**
   - Link households to their clusters
   - Prepare for forecasting phase

### 6. Questions for Team Discussion

-  Should we remove the 154 all-zero households before clustering?
-  What k value seems most interpretable based on early results?
-  Do we want 3-5 broad clusters or 8-10 more specific ones?
-  Should we try hierarchical clustering as an alternative to K-means?
-  How will we validate that clusters make sense?
-  Based on the metrics plots, what k value do you think is optimal?

-  Look at the radar charts - do the clusters make intuitive sense?
-  Which approach (features vs PCA) gives more interpretable clusters?
-  Do the cluster descriptions match your expectations about household types?

### 8. Resources Used

- **Libraries**: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn
- **Documentation**: scikit-learn PCA guide, statsmodels ACF documentation
- **Concepts**: Principal Component Analysis, feature engineering for time series


