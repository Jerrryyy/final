# File Name: housing_funcs.py
# Authors: Jerry Li, Kai Xie, Nargiz Guliyeva
# Emails: 
    # jerbear.li@mail.utoronto.ca, 
    # ky.xie@mail.utoronto.ca, 
    # nargiz.guliyeva@mail.utoronto.ca

# Course: INF1340H (Programming for Data Science)
# Professor: Dr. Maher Elshakankiri
# Assignment: Final Project (Due December 13th, 2023)

# Graduate Unit: Institute of Medical Science
# Degree: Doctor of Philosophy (PhD)
# Specialization: Neurosurgery, Artificial Intelligence, Epigenetics

# Date Created: 2023-12-08, 3:56 am
# Last Modified: 2023-12-12, 6:47 pm

# File Description:

    # Utility file that holds all functions required for the final project's
    # analysis of the US housing market.
    
    # Functions:
    # (1) load_data: Loads .csv file into a pandas DataFrame.
    # (2) clean_data: 
    #       Subsets data for a region and prepares it for statistical analyses.
    # (3) subset_data:
    #       Slices data for 5 variables most correlated with median sale price.
    # (4) describe_data: Produces descriptive statistics for a given dataset.
    # (5) test_assumptions:
    #       Performs Shapiro-Wilk tests for normality.
    # (6) clean_titles:
    #       Removes underscores and corrects capitalization in titles for
    #       plots.
    # (7) plot_descriptive:
    #       Plots violin plots, QQ plots, or histograms of data.
    # (8) code_sig: Converts p-values to significance levels.
    # (9) test_anova:
    #       Performs Kruskal-Wallis analysis of variance (ANOVA).
    # (10) test_post_hoc:
    #       Performs post-hoc Wilcoxon rank-sum tests.
    # (11) check_multi:
    #       Heatmap of correlation between variables highly correlated with
    #       median sale price.
    # (12) fit_regression:
    #       Fits a multiple linear regression model to data.
    # (13) fit_glm:
    #       Fits a generalized linear model to data.
    # (14) build_svr:
    #       Fits a support vector regression model to data.
    # (15) pickle_me:
    #       Either saves/loads a machine learning model to/from a pickle file.
    # (16) predict_new_sales:
    #       Predicts new median sale prices using a pre-trained SVR model.
    # (17) divergence_plot: Plots longitudinal data of various variables.
    # (18) heatmap_by_metrics:
    #       Heatmap of correlation coefficients between variables for each
    #       city.
    # (19) heatmap_by_regions:
    #       Heatmap of correlation coefficients between regions for
    #       each variable.
    # (20) regression_analysis: Performs lasso/ridge regression.
    # (21) removeOutliers: Removes outliers beyond 1.5*IQR.
    # (22) addYear: Adds a year column.
    # (23) volumeSales: Plots longitudinal data of multiple sales variables.
    # (24) offMarket: Plots bar plots of sales within 1-2 weeks of listing.
    # (25) topPrice: Returns top 10 regions with the highest median sale price.
    # (26) dynamicPriceDays:
    #       Scatter plot of median sale price vs. median days to close.
    # (27) plotDays: Histograms of median days to close for each year.
    # (28) priceTrend: Plots trends of prices and homes sold per year.
    # (29) comparisonSales: Scatter plot of total homes sold vs. supply.
    # (30) trainRandomForest: Trains a random forest regression model.

    
# Importing libraries ----
import csv
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Functions ----
# Jerry's Analyses ----
def load_data(filename: str, sep: str = ",") -> pd.DataFrame:
    """Loads .csv file into a pandas DataFrame.
    
    Args:
        filename: Name of delimited file to be loaded.
        sep: Separator/delimiter used in file.
        
    Returns:
        pandas DataFrame containing raw data.
    """
    
    # Reading in .csv and formatting dates properly
    raw_data = pd.read_csv(
        filename,
        parse_dates = [
            "period_begin",
            "period_end",
            "last_updated"
        ],
        sep = sep
    )
    
    return raw_data


def clean_data(
    data: pd.DataFrame, 
    region: str,
    analysis: str = "j"
) -> dict[str, pd.DataFrame]:
    """Subsets data for a region and prepares it for statistical analyses.
    
    Regions include New York City, Seattle, and Miami. Splits data into entries
    before and after August 2023. Variables of interest include listings and
    inventory, sales, and sale prices. Analysis parameters differ between
    Jerry, Kai, and Nargiz.
    
    Args:
        data: Raw housing data from Redfin.
        region: Region of interest; ("n")ew york, ("s")eattle, or ("m")iami.
        analysis: Analysis to be performed; ("j")erry, ("k")ai, or ("n")argiz.
        
    Returns:
        Cleaned data for the specified region, split into data before and
        after August 2023.
    """
    
    # Setting default paths for variables so they are not unbound
    variables = []
    before = pd.DataFrame()
    after = pd.DataFrame()
    
    # Defensive programming to ensure region is case-insensitive
    region = region.lower()
    
    # Query for desired region
    if region in ["n", "nyc", "new york", "new york city"]:
        region = "New York, NY metro area"
    elif region in ["s", "seattle", "wa", "washington"]:
        region = "Seattle, WA metro area"
    elif region in ["m", "miami", "fl", "florida"]:
        region = "Miami, FL metro area"
    
    # Slicing for predictors in Jerry's analyses
    if analysis.lower() in ["j", "jerry"]:   
        variables = [
            "region_name",
            "period_begin",
            "age_of_inventory",
            "total_new_listings",
            "total_active_listings",
            "inventory",
            "total_homes_sold",
            "total_homes_sold_with_price_drops",
            "average_sale_to_list_ratio",
            "percent_homes_sold_with_price_drops",
            "homes_delisted",
            "median_new_listing_price",
            "median_days_on_market",
            "median_days_to_close",
            "median_sale_price",
            "months_of_supply",
            "pending_sales",
            "percent_off_market_in_two_weeks",
            "percent_homes_sold_above_list",
            "price_drops",
        ]
    
    # Slicing for predictors in Kai's analyses
    elif analysis.lower() in ["k", "kai"]:
        variables = [
            "region_name",
            "period_begin",
            "total_homes_sold",
            "total_homes_sold_yoy",
            "total_homes_sold_with_price_drops",
            "total_homes_sold_with_price_drops_yoy",
            "median_sale_price",
            "median_sale_price_yoy",
            "off_market_in_two_weeks",
            "off_market_in_one_week",
            "total_new_listings",
            "total_new_listings_yoy",
            "median_new_listing_price",
            "median_new_listing_price_yoy",
            "inventory",
            "inventory_yoy",
            "age_of_inventory",
            "age_of_inventory_yoy",
            "homes_delisted",
            "homes_delisted_yoy",
            "months_of_supply",
            "months_of_supply_yoy"
        ]
    
    # Slicing for predictors in Nargiz's analyses
    elif analysis.lower() in ["n", "nargiz"]:
        variables = [
            "region_name",
            "period_begin",
            "total_homes_sold",
            "total_homes_sold_with_price_drops",
            "median_sale_price",
            "median_sale_ppsf",
            "median_days_to_close",
            "price_drops",
            "pending_sales",
            "off_market_in_two_weeks",
            "off_market_in_one_week",
            "total_new_listings",
            "median_new_listing_price",
            "inventory",
            "total_active_listings",
            "active_listings",
            "age_of_inventory",
            "homes_delisted",
            "median_active_list_price",
            "average_of_median_list_price_amount",
            "average_of_median_offer_price_amount",
            "median_days_on_market",
            "months_of_supply",
            "average_pending_sales_listing_updates",
            "average_percent_off_market_in_one_week_listing_updates",
            "average_percent_off_market_in_two_weeks_listing_updates",
            "price_drop_percent_of_old_list_price"
        ]
    
    # Further slicing data
    cleaned = (
        data.loc[
            (data["region_name"] == region)
            & (data["duration"] == "1 weeks"),
            variables
        ]
            .reset_index(drop=True)
            .replace(
                {
                    "New York, NY metro area": "New York City",
                    "Seattle, WA metro area": "Seattle",
                    "Miami, FL metro area": "Miami"
                }
            )
            .sort_values(by="period_begin", ascending=True)
    )
    
    # If using Jerry's analysis
    if analysis.lower() in ["j", "jerry"]:
        cleaned = cleaned.rename(
            columns = {
                "region_name": "region",
                "period_begin": "date"
            }
        )
    
        # Sorting by before and after August 2023
        before = cleaned.query("date <= '2023-08-31'")
        after = cleaned.query("date > '2023-08-31'")
    
    # If using Kai or Nargiz's analyses
    if analysis.lower() in ["k", "kai", "n", "nargiz"]:
        before = cleaned.query("period_begin <= '2023-08-31'")
        after = cleaned.query("period_begin > '2023-08-31'")
    
    # Combining results into dictionary
    final = {
        "before": before,
        "after": after
    }
        
    return final


def subset_data(data: pd.DataFrame) -> pd.DataFrame:
    """Slices data for 5 variables most correlated with median sale price.
    
    Data will also be prepared for OLS regression and comparison tests; region 
    will be re-encoded as integers and dates will be dropped.
    
    Args:
        data: Cleaned housing data.
    
    Returns:
        pandas DataFrame.
    """
    
    # Replacing regions with integers
    data["region"] = data["region"].replace(
        {
            "New York City": 0,
            "Seattle": 1,
            "Miami": 2
        }
    )
    data["region"] = data["region"].astype("category")
    
    # Slicing for variables most correlated with median sale price    
    subset = (
        data.loc[
            :,
            [
                "region",
                "median_sale_price",
                "total_homes_sold_with_price_drops",
                "median_new_listing_price",
                "inventory",
                "age_of_inventory",
                "months_of_supply"
            ]
        ]
    )
    
    return subset


def describe_data(
    data: pd.DataFrame,
    save: bool = False,
    path1: str = "",
    path2: str = ""
) -> dict[str, int | pd.DataFrame]:
    """Produces descriptive statistics for a given dataset.
    
    Calculates the total number of days in the dataset, gives an overall
    summary (min, mean, median, max, standard deviation, and standard error),
    and gives a rolling monthly summary (mean) for each variable of interest.
    
    Args:
        data: pandas DataFrame.
        save: Whether to save summaries as .csv files. (True/False)
        path1: Path and file name in which to save first .csv file.
        path2: Path and file name in which to save second .csv file.
        
    Returns:
        Dictionary containing total number of days in dataset, overall summary,
        and monthly summary. If save=True, saves .csv files.
    """
    
    # Defensive programming
    search = "date"
    if "date" not in data.columns:
        search = "period_begin"
    
    # Calculating total number of days in dataset
    days = (data[search].max() - data[search].min()).days + 1
    
    # Calculating overall summary statistics
    total_summary_df = (
        
        # Only including median and excluding region and date columns
        data.describe(percentiles=[0.5], include=[int, float])
            .transpose()
            
            # Calculating standard error of the mean
            .assign(sem = lambda x: x["std"] / np.sqrt(x["count"]))
            
            # Calculating range for each variable
            .assign(range = lambda x: x["max"] - x["min"])
            
            # Removing count
            .drop(columns=["count"])
            
            # Renaming columns to be more descriptive
            .rename(columns={"50%": "median", "std": "stdev"})
    )
    
    # Rearranging order of columns
    total_summary_df = total_summary_df[[
            "min", "mean", "median", "max", "range", "stdev", "sem"
    ]]
    
    # Calculating rolling monthly summary statistics
    monthly_means = (
        data.drop(columns=["region"])
            .groupby(pd.Grouper(key=search, freq="M"))
            .mean()
    )
    
    # Combining all summary statistics
    summary = {
        "days": days,
        "total_summary": total_summary_df,
        "monthly_means": monthly_means
    }
    
    # If user wants to save summary statistics
    if save:
        total_summary_df.to_csv(path1)
        monthly_means.to_csv(path2)
    
    return summary


def test_assumptions(
    data: pd.DataFrame,
    test: str = "n",
    save: bool = False,
    path: str = ""
) -> pd.DataFrame:
    """Performs Shapiro-Wilk tests for normality on given data.
    
    Args:
        data: pandas DataFrame.
        test: Test to perform; (n)ormality, (v)ariance.
        save: Whether to save summary as .csv file. (True/False)
        path: Path and file name in which to save summary.
    
    Returns:
        Data frame with Shapiro-Wilk test results. If save = True, saves a
        .csv file.
    """
    
    # If user specifies variance test, there must be at least 2 regions
    if test in ["v", "var", "variance"] and len(data["region"].unique()) < 2:
        raise Exception("Variance tests require multiple regions.")
    
    # Saving all regions (if applicable) and variables of interest to a list
    if "region" in data.columns.tolist():
        regions = data["region"].unique().tolist()
        n_regions = len(regions)
    cols = data.columns.tolist()
    
    # If region or dates are in the list, remove them
    if ("region" in cols):
        cols.remove("region")
    if ("date" in cols):
        cols.remove("date")
    if ("period_begin" in cols):
        cols.remove("period_begin")
    
    # Initializing dictionary to store results
    results = {}
    
    # Iterating over each variable of interest
    for col in cols:
        
        # If user wants normality tests
        if test in ["n", "norm", "normality"]:
            
            # Saving test statistic and p-value
            stat, pval = stats.shapiro(data[col])
            
            # Interpreting p-values
            sig = "Normal" if pval >= 0.05 else "Non-normal"
            
            # Adding results to dictionary
            results.update(
                {
                    col: {
                        "Statistic": stat,
                        "p-value": pval,
                        "Skewness": stats.skew(data[col]),
                        "Kurtosis": stats.kurtosis(data[col]),
                        "Result": sig
                    }
                }
            )
        
        # If user wants variance tests
        elif test in ["v", "var", "variance"]:
            
            # If there are only 2 regions                
            if n_regions == 2:
                
                # Saving test statistic and p-value
                stat, pval = stats.levene(
                    data.query("region == @regions[0]")[col],
                    data.query("region == @regions[1]")[col]
                )
            
            # If there are 3 regions
            elif n_regions == 3:
                    
                # Saving test statistic and p-value
                stat, pval = stats.levene(
                    data.query("region == @regions[0]")[col],
                    data.query("region == @regions[1]")[col],
                    data.query("region == @regions[2]")[col]
                )
                
            # Interpreting p-values
            sig = "Equal" if pval >= 0.05 else "Unequal"
                
            # Adding results to dictionary
            results.update(
                {
                    col: {
                        "Statistic": stat,
                        "p-value": pval,
                        "Result": sig
                    }
                }
            )
    
    # Converting dictionary to data frame
    summary = pd.DataFrame(results).transpose()
    
    # If user wants to save summary
    if save:
        summary.to_csv(path)
        
    return summary


def clean_titles(titles: list[str]) -> list[str]:
    """Removes underscores and corrects capitalization in titles for plots.
    
    Args:
        titles: List of titles to be cleaned.
    
    Returns:
        List of cleaned titles.
    """
    
    # Looping through each item in list of titles
    for i, name in enumerate(titles):
            
            # Removing underscores
            name = name.replace("_", " ")
            
            # Capitalizing first letter of each word
            name = name.title()
            
            # Fixing capitalization of certain words
            if "Yoy" in name:
                name = name.replace("Yoy", "YoY")
            if "Of" in name and "Off" not in name:
                name = name.replace("Of", "of")
            if "To" in name and "Total" not in name:
                name = name.replace("To", "to")
            if "In" in name and "Inventory" not in name:
                name = name.replace("In", "in")
            
            # Replacing item in list with cleaned title
            titles[i] = name
    
    return titles


def plot_descriptive(
    data: pd.DataFrame,
    plot: str = "v",
    save: bool = False,
    path: str = ""
) -> None:
    """Plots violin plots, QQ plots, or histograms of data.
    
    Args:
        data: pandas DataFrame.
        plot: Plot to produce; ("v")iolin, ("q")q, ("h")istogram.
        save: Whether to save plots. (True/False)
    """
    
    # Setting up plot
    _, ax = plt.subplots(nrows=6, ncols=3, figsize=(20, 40))
    titles = clean_titles(data.columns.to_list()[2:])
    
    # If user specifies violin plots
    if plot.lower() in ["v", "violin"]:
        
        # Plotting figures
        for i, col in enumerate(data.columns[2:]):
            sns.violinplot(
                x = "region",
                y = col,
                data = data,
                ax = ax[i//3, i%3],     # Giving each plot 1/3 of the axis
            )
            
            # Plot aesthetics
            ax[i//3, i%3].set(xticklabels=[])
            ax[i//3, i%3].set_title(titles[i], fontsize=16)
            ax[i//3, i%3].set_xlabel(None)
            ax[i//3, i%3].set_ylabel("Units", fontsize=12)
            ax[i//3, i%3].tick_params(bottom=False)
            
    # If user specifies QQ plots
    elif plot.lower() in ["q", "qq"]:
        for i, col in enumerate(data.columns[2:]):
            sm.qqplot(
                data[col],
                ax = ax[i//3, i%3],     # Giving each plot 1/3 of the axis
                line = "s"
            )
            
            # Plot aesthetics
            ax[i//3, i%3].set_title(titles[i], fontsize=16)
            ax[i//3, i%3].set_xlabel("Theoretical Quantiles", fontsize=12)
            ax[i//3, i%3].set_ylabel("Sample Quantiles", fontsize=12)
    
    # If user specifies histograms
    elif plot.lower() in ["h", "histogram"]:
        for i, col in enumerate(data.columns[2:]):
            sns.histplot(
                x = col,
                hue = "region",
                data = data,
                ax = ax[i//3, i%3],     # Giving each plot 1/3 of the axis
                kde = True,
                kde_kws = {"bw_adjust": 1.5},
                stat = "density",
                legend = False
            )
            
            # Plot aesthetics
            ax[i//3, i%3].set_title(titles[i], fontsize=16)
            ax[i//3, i%3].set_xlabel(None)
            ax[i//3, i%3].set_ylabel("Density", fontsize=12)
    
    # Plot aesthetics
    sns.despine(top=True, right=True)
    
    # If user wants to save plots
    if save:
        plt.savefig(path, bbox_inches="tight")


def code_sig(pval: float) -> str:
    """Converts p-values to significance levels.
    
    Args:
        pval: p-value from statistical test.
        
    Returns:
        Significance level; "ns" = not significant, *p < 0.05, **p < 0.01, and
        ***p < 0.001.
    """
    
    sig = "ns"
    if pval < 0.05 and pval >= 0.01:
        sig = "*"
    elif pval < 0.01 and pval >= 0.001:
        sig = "**"
    elif pval < 0.001:
        sig = "***"
    
    return sig


def test_anova(
    data: pd.DataFrame, 
    save: bool = False, 
    path: str = ""
) -> pd.DataFrame:
    """Performs Kruskal-Wallis analysis of variance (ANOVA) on given data.
    
    KW ANOVA is a non-parametric alternative to one-way ANOVA.
    
    Args:
        data: pandas DataFrame.
        save: Whether to save summary as .csv file. (True/False)
        path: Path and file name in which to save summary.
    
    Returns:
        pandas DataFrame with results. If save=True, saves a .csv file.
    """
    
    # Saving variables to test to a list
    vars = data.columns.tolist()
    if "region" in vars:
        vars.remove("region")
    
    # Initializing dictionary to store results
    results = {}
    
    # Performing ANOVA for each variable of interest
    for var in vars:
        stat, pval = stats.kruskal(
            data.query("region == 0")[var],
            data.query("region == 1")[var],
            data.query("region == 2")[var]
        )
        results[var] = {
            "Statistic": stat,
            "p-value": pval
        }
        sig = code_sig(pval)
        post_hoc = "Required" if pval < 0.05 else "Not Required"
    
    # Converting dictionary to data frame and adding interpretation
    summary = pd.DataFrame(results).transpose()
    summary["Result"] = sig
    summary["Post-Hoc"] = post_hoc
    
    # If user wants to save summary
    if save:
        summary.to_csv(path)
    
    return summary


def test_post_hoc(
    data: pd.DataFrame,
    save: bool = False,
    path: str = ""
) -> dict[str, pd.DataFrame]:
    """Performs post-hoc Wilcoxon rank-sum tests on given data.
    
    Wilcoxon rank-sum or Mann-Whitney U tests are non-parametric alternatives 
    to independent t-tests.
    
    Args:
        data: pandas DataFrame.
        save: Whether to save summary as .csv file. (True/False)
        path: Path and file name in which to save summary.
    
    Returns:
        Dictionary with results. If save=True, saves a .csv file.
    """
    
    # Reformatting data for more descriptive region names
    data["region"] = data["region"].replace(
        {
            0: "NYC",
            1: "Seattle",
            2: "Miami"
        }
    )
    
    # Saving post-hoc variables to a list and initializing results dictionary
    vars = data.columns.tolist()
    vars.remove("region")
    results = {}
    
    # Performing post-hoc tests for each variable of interest
    for var in vars:
        results.update(
            {
                var: sp.posthoc_mannwhitney(
                    data,
                    group_col = "region",
                    val_col = var,
                    p_adjust = "fdr_bh"
                )
            }
        )
    
    # If user wants to save summary
    if save:
        
        # Opening new file
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Iterating through dictionary
            for key, value in results.items():
                
                # Unpacking key and column names
                writer.writerow([key, *value.columns.to_list()])
                
                # Unpacking row names and values
                for row in value:
                    writer.writerow([row, *value[row].to_list()])
    
    return results


def check_multi(
    data: pd.DataFrame,
    save: bool = False,
    path: str = ""
) -> None:
    """Plots heatmap to check for multicollinearity between predictors.
    
    Args:
        data: pandas DataFrame
        save: Whether to save plot. (True/False)
        path: Path and file name in which to save plot.
    
    Returns:
        Optionally saves a heatmap of the correlation between predictors if
        save=True and a path is given.
    """
    
    # Preparing data for plotting
    cleaned = data.drop(columns=["region"])
    vars = dict(zip(cleaned.columns, clean_titles(cleaned.columns.to_list())))
    
    # Figure dimensions
    _, ax = plt.subplots(figsize=(6, 6))
    
    # Plotting
    fig = sns.heatmap(
        cleaned.corr().rename(columns=vars, index=vars),
        cmap = "RdBu",
        annot = True,
        ax = ax
    )
    
    # Plot aesthetics
    fig.set_title("Correlation Between Predictors", fontsize=16, pad=10)
    plt.xticks(rotation=75, fontsize=12, ha="right")
    plt.yticks(fontsize=12)

    # If user wants to save plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def fit_regression(
    data: pd.DataFrame,
    plot: bool = False,
    summary: bool = False,
    save: bool = False,
    path: str = ""
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fits a multiple linear regression model to predict sale price.
    
    Scales and centres data before fitting and plots predicted vs. 
    actual sale price.
    
    Args:
        data: pandas DataFrame.
        plot: Whether to plot predicted vs. actual sale price. (True/False)
        summary: Whether to print model summary. (True/False)
        save: Whether to save plot. (True/False)
        path: Path and file name in which to save plot.
        
    Returns:
        statsmodels OLS regression object.
    """
    
    # Copying data to prevent modifications to original data
    regression = data.copy()
    
    # Preparing for regression
    regression = regression.drop(columns=["region"])
    features = regression.columns.to_list()
    features.remove("median_sale_price")
    
    # Scaling data before regression
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    features = regression[features]
    target = regression["median_sale_price"]
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))
    
    # Combining scaled features and target into a data frame
    scaled = pd.concat(
        [
            pd.DataFrame(features_scaled, columns=features.columns),
            pd.DataFrame(target_scaled, columns=["median_sale_price"])
        ],
        axis = "columns"
    )
    regression = scaled
    
    # Setting up model
    model = smf.ols(
        data = regression,
        formula = (
            "median_sale_price ~"
            " total_homes_sold_with_price_drops"
            " + median_new_listing_price"
            " + inventory"
            " + age_of_inventory"
            " + months_of_supply"
        )
    )
    
    # Fitting model and making predictions
    fitted = model.fit()
    preds = scaler_y.inverse_transform(
        fitted.predict(regression).values.reshape(-1, 1)
    )
    
    # If user wants to print model summary
    if summary:
        print(fitted.summary())
    
    # If user wants to plot model fit
    if plot:
        _, ax = plt.subplots(figsize=(6, 6))
        fig = sns.regplot(
            x = data["median_sale_price"],
            y = preds,
            ax = ax,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"}
        )

        # Adding fit metrics to plot
        r2 = fitted.rsquared_adj
        ax.text(
            x = 0.04,
            y = 0.98,
            s = r"$R^{2} = $" + f"{r2:.3f}",
            transform = ax.transAxes,
            fontsize = 14,
            verticalalignment = "top"
        )
        
        # Plot aesthetics
        fig.set_title("Predicted vs. Actual Sale Price", fontsize=16, pad=10)
        fig.set_xlabel("Actual Sale Price", fontsize=12)
        fig.set_ylabel("Predicted Sale Price", fontsize=12)
        sns.despine(top=True, right=True)
        
        # If user wants to save plot
        if save:
            plt.savefig(path, bbox_inches="tight")
    
    return fitted


def fit_glm(
    data: pd.DataFrame,
    plot: bool = True,
    summary: bool = False,
    save: bool = False,
    path: str = ""
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fits a generalized linear model (GLM) to predict sale price.
    
    Uses a mixed effects model (which had lower AIC and BIC than fixed effects
    models), with a gamma error distribution and log link function.
    
    Args:
        data: pandas DataFrame.
        plot: Whether to plot predicted vs. actual sale price. (True/False)
        summary: Whether to print model summary. (True/False)
        save: Whether to save plot. (True/False)
        path: Path and file name in which to save plot.
        
    Returns:
        statsmodels GLM regression object.
    """
    
    # Cleaning input data
    regression = data.copy()
    regression = regression.drop(columns=["region"])
    
    # Setting up model
    model = smf.glm(
        data = regression,
        formula = (
            "median_sale_price ~"
            " total_homes_sold_with_price_drops"
            " + median_new_listing_price"
            " * inventory"
            " * age_of_inventory"
            " * months_of_supply"
        ),
        family = sm.families.Gamma(sm.families.links.Log())
    )
    
    # Fitting model and making predictions
    fitted = model.fit()
    preds = model.predict(fitted.params)
    
    # If user wants to print model summary
    if summary:
        print(fitted.summary())
    
    # If user wants to plot model fit
    if plot:
        _, ax = plt.subplots(figsize=(6, 6))
        fig = sns.regplot(
            x = data["median_sale_price"],
            y = preds,
            ax = ax,
            scatter_kws = {
                "alpha": 0.5,
                "edgecolor": "black",
                "s": 100
            },
            line_kws = {"color": "red"}
        )

        # Adding fit metrics to plot
        r, pval = stats.pearsonr(data["median_sale_price"], preds)
        pval = "< 0.001" if pval < 0.001 else f"{pval:.3f}"
        pr = fitted.pseudo_rsquared()
        aic = fitted.aic
        bic = fitted.bic
        ax.text(
            x = 0.04,
            y = 0.98,
            s = (
                r"$r = $" + f"{r:.3f}\n"
                r"$p$" + f" {pval}\n"
                r"Pseudo-$R^{2} = $" + f"{pr:.3f}\n"
                r"$AIC = $" + f"{aic:.3f}\n"
                r"$BIC = $" + f"{bic:.3f}"
            ),
            transform = ax.transAxes,
            fontsize = 14,
            verticalalignment = "top"
        )
        
        # Plot aesthetics
        fig.set_title(
            "GLM: Predicted vs. Actual Sale Price", fontsize=16, pad=10
        )
        fig.set_xlabel("Actual Sale Price", fontsize=12)
        fig.set_ylabel("Predicted Sale Price", fontsize=12)
        sns.despine(top=True, right=True)
        
        # If user wants to save plot
        if save:
            plt.savefig(path, bbox_inches="tight")
    
    return fitted


def build_svr(
    data: pd.DataFrame,
    target: str = "median_sale_price",
    iters: int = 1,
    summary: bool = True,
    plot: bool = True,
    save: bool = False,
    path: str = ""
) -> SVR:
    """Fits a support vector regression model to data for a given target.
    
    Uses a radial basis function kernel. Performs 10-fold cross-validation and
    optionally prints model results. Train-test split of 80-20%.
    
    Args:
        data: pandas DataFrame.
        target: Target variable for the model.
        iters: Number of iterations to fit models and calculate fit metrics.
        summary: Print model fit metrics (True/False).
        plot: Plot model predictions against actual values (True/False).
        save: Save plot (True/False).
        path: Path and file name in which to save plot.
        
    Returns:
        Support vector regression model. Optionally plots model performance
        and saves plot in a given path if save=True.
    """

    # Preparing features and target
    X = data.drop(columns = ["region", target])
    X["date"] = X["date"].dt.strftime("%m").astype(int)
    y = data[target]
    svr = SVR(gamma=1e-8, C=1000, epsilon=0.0001)

    # Running model fitting a specified number of times (default once)
    for _ in range(iters):

        # Splitting into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        # Scaling and centering data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = (
            scaler_y.fit_transform(y_train.values.reshape(-1, 1))
                .ravel()
        )
        
        # Fitting model to data
        svr = SVR().fit(X_train_scaled, y_train_scaled)
        y_pred = (
            scaler_y.inverse_transform(
                svr.predict(X_test_scaled)
                    .reshape(-1, 1)
                )
                .ravel()
        )

        # Running 10-fold cross validation
        scores = cross_val_score(svr, X_train_scaled, y_train_scaled, cv=10)

        # Fit metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Printing results if user desires
        if summary:
            print(
                f"Model Results:"
                f"\nAvg CV Score: {np.mean(scores):.2f}"
                f"\nRMSE: {rmse:.2f}"
                f"\nMAE: {mae:.2f}"
                f"\nR-squared: {r2:.2f}\n"
            )

    # If user wants to plot model fit
    if plot:
        
        # Calculating correlation
        corr, pval = stats.pearsonr(y_pred, y_test)
        if pval < 0.001:
            pval = "< 0.001"
        
        # Setting figure dimensions
        _, ax = plt.subplots(figsize=(6, 6))

        # Plotting correlation line
        sns.regplot(
            x = y_test,
            y = y_pred,
            scatter_kws = {
                "alpha": 0.5,
                "edgecolor": "black",
                "s": 100,
                },
            line_kws = {"color": "red"}
        )
        
        # Adding fit metrics and correlation to plot
        plt.text(
            x = 0.03,
            y = 0.88,
            s = (
                r"$R^{2} = $" + f"{r2:.3f}\n"
                r"$r = $" + f"{corr:.3f}\n"
                r"$p$" + f" {pval}"
            ),
            fontsize = 12,
            transform = ax.transAxes
        )

        # Setting plot aesthetics
        ax.set_title(
            "SVR: Actual vs. Predicted Median Sale Price", fontsize=16, pad=10
        )
        ax.set_xlabel("Actual", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        sns.despine(top=True, right=True)
        
        # If user wants to save plot        
        if save:
            plt.savefig(path, bbox_inches="tight")

    return svr


def pickle_me(path: str = "", usage: str = "s", **kwargs: SVR) -> SVR | None:
    """Either saves or loads a machine learning model saved in a pickle file.
    
    Args:
        path: Path or file name to save model to or to open model from.
        usage: Whether to save or load model; ("s")ave or ("l")oad.
        model: Machine learning model to be saved (optional).
        
    Returns:
        Either saves a machine learning model to a pickle file or loads a model
        from a pickle file.
    """
    
    # If user is saving a model, model will be in kwargs
    model = kwargs.get("model", None)
    
    # If user is saving a model
    if usage.lower() in ["s", "save"]:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    
    # If user is loading a model
    elif usage.lower() in ["l", "load"]:
        with open(path, "rb") as f:
            model = pickle.load(f)
        
        return model


def predict_new_sales(
    data: pd.DataFrame,
    model: SVR,
    summary: bool = True,
    plot: bool = True,
    save: bool = False,
    path: str = ""
) -> dict[str, pd.DataFrame]:
    """Predicts new median sale prices using a pre-trained SVR model.
    
    Args:
        data: pandas DataFrame.
        model: Pre-trained SVR model.
        summary: Print model fit metrics. (True/False)
        plot: Plot model predictions against actual values. (True/False)
        save: Save plot. (True/False)
        path: Path and file name in which to save plot.
    
    Returns:
        pandas DataFrame with predicted median sale prices. Optionally returns
        model predictions vs. actual values, fit metrics, feature weights,
        and saves a plot within a desired directory and file name.
    """
    
    # Saving features and target
    X = data.drop(columns = ["region", "median_sale_price"])
    X["date"] = X["date"].dt.strftime("%m")
    y = data["median_sale_price"]
    
    # Scaling features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Predicting on new data
    predictions = (
        scaler_y.inverse_transform(
            model.predict(X_scaled)
                .reshape(-1, 1)
            )
            .ravel()
    )
    
    # If user wants to print model summary
    if summary:
    
        # Calculating residuals and prediction correlation
        residuals = y - predictions
        res_mean = residuals.mean()
        res_stderr = residuals.std() / np.sqrt(len(residuals))
        res_corr, res_pval = stats.pearsonr(y, predictions)

        # Printing fit metrics
        print(
            f"Mean of Residuals: {res_mean:.2f}"
            f"\nResidual Standard Error: {res_stderr:.2f}"
            f"\nCorrelation: {res_corr:.2f}"
            f"\np-value: {res_pval:.2f}"
        )
    
    # If user wants to plot predictions vs. actual values
    if plot:
        _, _ = plt.subplots(figsize=(6, 6))
        sns.regplot(
            x = y / 1000,
            y = predictions / 1000,
            scatter_kws = {
                "alpha": 0.5,
                "edgecolor": "black",
                "s": 100,
                },
            line_kws = {"color": "red"}
        )
        
        # Annotating correlation and p-value
        plt.text(
            x = 0.03,
            y = 0.88,
            s = (
                r"$R^{2} = $" + f"{res_corr:.3f}\n"
                r"$p = $" + f"{res_pval:.2f}"
            ),
            fontsize = 12,
            transform = plt.gca().transAxes
        )
        
        # Plot aesthetics
        plt.title(
            "SVR: Actual vs. Predicted Median Sale Price",
            fontsize=16,
            pad=10
        )
        plt.xlabel("Actual Median Sale Price ($, thousands)", fontsize=12)
        plt.ylabel("Predicted Median Sale Price ($, thousands)", fontsize=12)
        sns.despine(top=True, right=True)
        
        # If user wants to save plot
        if save:
            plt.savefig(path, bbox_inches="tight")
            
    # Identifying most important features in predicting sale prices
    model_coefs = SelectFromModel(model, prefit=True).estimator._get_coef()
    coefficients = (
        pd.DataFrame(model_coefs, columns=X.columns)
            .transpose()
            .rename(columns={0: "Coefficient"})
    )
    coefficients.index = clean_titles(coefficients.index.to_list())

    # Returning results as a dictionary    
    actual_pred = (
        pd.concat([y, pd.Series(predictions)], axis="columns")
            .rename(columns={0: "predicted_price"})
    )
    results = {
        "fit": actual_pred,
        "coefs": coefficients
    }
    
    return results


# Kai's Analyses ----
def divergence_plot(
    city_data: pd.DataFrame, 
    all_data: pd.DataFrame, 
    save: bool = False, 
    path: str = ""
) -> None:
    """Plots longitudinal data for a variable vs. the national median.
    
    Args: 
        city_data: Data for a specific city.
        all_data: Data for all cities.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
        
    Returns:
        Plots of longitudinal data for a variable vs. the national median.
        Optionally saves plots within a given directory and file name.
    """
    x = "period_begin"
    y = (
        city_data.drop(columns=["region_name", "period_begin"])
            .columns
            .to_list()
    )
    titles = clean_titles(y.copy())
    
    # Plotting two series on the same x-axis; first creating a figure with a 
    # single y-axis (ax1)
    _, ax1 = plt.subplots(len(y), figsize=(10, 5*len(y)))
    
    # Looping through each variable of interest
    for i, col in enumerate(y):

        # Setting data to be plotted
        y_data = all_data.groupby(all_data[x])[col].median()
        seattle = city_data[[x, col]]
        seattle = seattle.set_index(x)
        
        # Plotting national median
        _ = ax1[i].plot(y_data, label="Overall", color="tab:blue")

        # Plotting city data
        _ = ax1[i].plot(seattle[col], label="Seattle", color="tab:orange")
        ax1[i].set_ylabel(titles[i])

        # Creating a second y-axis (ax2) for a difference calculation
        ax2 = ax1[i].twinx()

        # Ploting differences
        difference = seattle[col] - y_data
        _ = ax2.plot(
            difference, 
            label = "Difference (RHS)",
            linestyle = "--", 
            color = "tab:red"
        )
        ax2.set_ylabel("Difference", rotation=-90, labelpad=15)

        # Adding labels and legends
        plt.title(titles[i] + " Comparison for Seattle vs. National Median")
        lines, labels = ax1[i].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1[i].legend(lines + lines2, labels + labels2, loc="upper left")
    
    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def heatmap_by_metrics(
    data: pd.DataFrame,
    strong_only: bool = False,
    threshold: float = 0.6,
    save: bool = False,
    path: str = ""
) -> None:
    """Plots correlation heatmaps between each variable of interest.
    
    Args:
        data: Data for a specific region.
        strong_only: Whether to only plot strong correlations. (True/False)
        threshold: Threshold for strong correlations.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
        
    Returns:
        Correlation heatmaps between each variable of interest. Optionally
        saves the plots within a given directory and file name.
    """

    # Generating list of metrics
    regions = data["region_name"].unique()
    ys_smaller = (
        data.drop(
            columns = [
                "region_name",
                "period_begin", 
                "homes_delisted", 
                "homes_delisted_yoy"
            ]
        )
        .columns
        .to_list()
    )
    cleaned_ys = clean_titles(ys_smaller.copy())
    
    # Setting up plots
    _, ax = plt.subplots(len(regions), figsize=(14, 10*len(regions) + 2))
    
    # Looping through each city
    for i, region in enumerate(regions):
    
        # Calculating correlations
        matrix = (
            data.loc[data["region_name"] == region, ys_smaller]
                .corr()
                .rename(
                    columns = dict(zip(ys_smaller, cleaned_ys)),
                    index = dict(zip(ys_smaller, cleaned_ys))
                )
        )
        
        # If only plotting strong correlations
        if strong_only:
            mask = (matrix >= threshold) | (matrix <= -threshold)
            
            # Greying out non-strong correlations and colouring strong ones
            sns.heatmap(
                matrix,
                cmap = ["lightgray"],
                ax = ax[i],
                mask = mask,
                cbar = False
            )
            sns.heatmap(
                matrix, 
                annot = True, 
                cmap = "coolwarm",
                vmin = -1,
                vmax = 1,
                ax = ax[i],
                mask = ~mask
            )
            
            # Adding dashed gridlines to help viewer identify intersections
            ax[i].hlines(
                [range(0, len(matrix), 1)], *ax[i].get_xlim(), 
                colors = "gray",
                linestyles = "dashed"
            )
            ax[i].vlines(
                [range(0, len(matrix), 1)], *ax[i].get_ylim(), 
                colors = "gray",
                linestyles = "dashed"
            )
        
        # If plotting all correlations
        else:
            sns.heatmap(
                matrix, 
                annot = True, 
                cmap = "coolwarm",
                vmin = -1,
                vmax = 1,
                ax = ax[i]
            )
        ax[i].set_title(f"Correlation Heatmap for {region}")
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
            
    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def heatmap_by_regions(
    combined_data: pd.DataFrame,
    all_data: pd.DataFrame,
    save: bool = False,
    path: str = ""
) -> None:
    """Plots correlation heatmaps for regions across all variables.
    
    Args:
        combined_data: Data for our 3 regions, combined.
        all_data: Data for all regions.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
    
    Returns:
        Correlation heatmaps for region across all variables. Optionally saves
        the plots within a given directory and file name.
    """
    
    # Saving variables for plotting
    metrics = (
        combined_data.drop(columns=["region_name", "period_begin"])
            .columns
            .to_list()
    )
    cleaned_metrics = clean_titles(metrics.copy())
    
    # Setting up plots
    _, ax = plt.subplots(5, 4, figsize=(15, 15))
    
    # Looping through each metric
    for i, metric in enumerate(metrics):
    
        # Calculating national medians
        median_data = (
            all_data.groupby(all_data["period_begin"])
            [metric]
                .median()
        )
        
        # Combining cities and national medians
        data = (
            pd.concat(
                [
                    combined_data.pivot(
                        index = "period_begin",
                        columns = "region_name",
                        values = metric
                    ),
                    median_data,
                ],
                axis = "columns"
            )
                .rename(columns={metric: "National Median"})
        )

        # Calculating correlations
        correlation_matrix = data.corr()
    
        # Plotting
        sns.heatmap(
            correlation_matrix, 
            annot = True, 
            cmap = "coolwarm", 
            vmin = -1, 
            vmax = 1,
            ax = ax[i//4, i%4]
        )
        plt.setp(ax[i//4, i%4].get_xticklabels(), rotation=45, ha="right")
        ax[i//4, i%4].set_title(cleaned_metrics[i])
        plt.tight_layout()
    
    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def regression_analysis(
    data: pd.DataFrame,
    type: str = "lasso",
    summary: bool = False,
    save: bool = False,
    path: str = ""
) -> float:
    """Performs either Lasso or Ridge regression on given data.
    
    Train-test split is 7-3. Model is fitted with alpha = 1.0. Model accuracy
    is given in terms train and test errors and mean squared error (MSE).
    
    Args:
        data: Data to be used in regression (city data).
        type: Model to be used in regression; "l"asso or "r"idge.
        summary: Whether to print a summary of the model. (True/False)
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
    
    Returns:
        Model test score. Prints model metrics and plotted predicted points
        against actual points. Optionally saves plot in a given directory and
        file name.
    """
    
    # Setting targets and features
    X = sm.add_constant(
        data.drop(
            columns = [
                "region_name",
                "period_begin",
                "homes_delisted",
                "homes_delisted_yoy",
                "median_sale_price",
                "median_sale_price_yoy"
            ]
        )
    )
    y = data["median_sale_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Fitting model
    model = Lasso()
    message = "Lasso" if type.lower() in ["l", "lasso"] else "Ridge"
    if type.lower() in ["l", "lasso"]:
        model = Lasso(alpha=1.0)
    elif type.lower() in ["r", "ridge"]:
        model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Predicting data
    y_pred = model.predict(X_test)

    # Calculating fit metrics
    train_score = model.score(X_train, y_train)
    test_score = float(model.score(X_test, y_test))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # If user wants a summary of the model
    if summary:
        print(
            f"Fitting with a {message} Regression Model...\n"
            f"\n---- Model Details ----"
            f"\nIntercept: {model.intercept_:.2f}"
            "\n\nCoefficients:"
        )
        
        cleaned = clean_titles(X.columns.to_list())
        
        for feature, coefficient in zip(cleaned, model.coef_):
            print(f"{feature}: {coefficient:.2f}")
            
        print(
            f"\n---- Model Accuracy ----"
            f"\nTrain Score: {train_score:.2f}"
            f"\nTest Score: {test_score:.2f}"
            f"\nRMSE on Test Data: {rmse:.2f}"
        )

    # Plotting predicted vs. actual data
    _, _ = plt.subplots(figsize=(6, 6))
    sns.regplot(
        x = y_test / 1000,
        y = y_pred / 1000,
        label="Test Data",
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "tab:red"}
    )
    plt.title(
        f"{message} Regression: Predicted vs. Actual Median Sale Price"
    )
    plt.xlabel("Actual Data ($, thousands)")
    plt.ylabel("Predicted Data ($, thousands)")
    plt.legend()
    
    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")
        
    return test_score


# Nargiz's Analyses ----
def removeOutliers(
    data: pd.DataFrame,
    plot: bool = False,
    save: bool = False,
    path: str = ""
) -> pd.DataFrame:
    """Removes outliers from data using the cap (1.5*IQR) method.
    
    Outliers are considered points beyond 1.5 times the interquartile range
    (IQR) from the first and third quartiles. Outliers are replaced with the
    respective upper and lower bounds. Boxplots are generated before and after
    removing outliers.
    
    Args:
        data: Data to be used in regression (city data).
        plot: Whether to create boxplots. (True/False)
        save: Whether to save the plots. (True/False)
        path: Path to save the plots.
    
    Returns:
        Data with outliers removed. Optionally saves boxplots in a given
        directory and file name.
    """

    # Preparing data for cleaning
    raw = data.copy()
    columns = data.columns.to_list()
    columns.remove("region_name")
    columns.remove("period_begin")
    cleaned_columns = clean_titles(columns.copy())
    raw["status"] = "raw"
    cleaned = data.copy()
    cleaned["status"] = "cleaned"
    
    # Calculating quartiles and IQR and defining bounds
    q1 = raw[columns].quantile(0.25)
    q3 = raw[columns].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    
    # Replacing values outside bounds with respective bounds
    for col in columns:
        cleaned[col] = cleaned[col].mask(cleaned[col] > upper[col], upper[col])
        cleaned[col] = cleaned[col].mask(cleaned[col] < lower[col], lower[col])

    # If user wants to plot
    if plot:
    
        # Setting up plots with 5 plots per row
        _, ax = plt.subplots(5, 5, figsize=(30, 30))
        
        # Combining raw and cleaned data for plotting
        final = pd.concat([raw, cleaned], axis="index").reset_index(drop=True)

        # Creating boxplots for each variable to compare removed outliers
        for i, col in enumerate(columns):
            sns.boxplot(
                data = final,
                x = col,
                hue = "status",
                ax = ax[i//5, i%5],
                gap = 0.1
            )
            
            # Reformatting axis and legend labels
            ax[i//5, i%5].set_xlabel(cleaned_columns[i])
            handles, labels = ax[i//5, i%5].get_legend_handles_labels()
            ax[i//5, i%5].legend(
                handles,
                [label.title() for label in labels],
                loc = "upper right"
            )
            
        # If saving the plot
        if save:
            plt.savefig(path, bbox_inches="tight")
    
    return cleaned


def addYear(data: pd.DataFrame) -> pd.DataFrame:
    """Adds a column for years to a given DataFrame.
    
    Args:
        data: DataFrame to be used for analyses.
    
    Returns:
        DataFrame with an additional column for years.
    """
    
    # Adding a column for years
    data["Year"] = data["period_begin"].dt.year
    
    return data


def volumeSales(
    data: pd.DataFrame,
    save: bool = False,
    path: str = ""
) -> None:
    """Plots longitudinal data for multiple variables.
    
    Variables include total homes sold, total homes sold with price drops,
    new listings, and delisted homes. Data is grouped by year.
    
    Args:
        data: DataFrame to be plotted.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
    
    Returns:
        Plots of longitudinal data for multiple variables. Optionally saves
        plots in a given directory and file name.
    """

    # Grouping data and calculating sums
    total_sold = (
        data.groupby("Year")["total_homes_sold"]
            .sum()
            .sort_index()
    )
    sold_drop = (
        data.groupby("Year")["total_homes_sold_with_price_drops"]
            .sum()
    )
    new_listings = data.groupby("Year")["total_new_listings"].sum()
    delisted = data.groupby("Year")["homes_delisted"].sum()

    # Plotting
    plt.figure(figsize=(7, 5))
    plt.plot(total_sold, marker="o", label="Total Homes Sold")
    plt.plot(sold_drop, marker="o", label="Total Homes Sold with Price Drops")
    plt.plot(new_listings, marker="o", label="New Listings")
    plt.plot(delisted, marker="o", label="Delisted")

    # Plot aesthetics
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel("Year")
    plt.ylabel("Number of Homes")
    plt.title("Real Estate Market Trends in Miami (2017-2023)")
    
    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def offMarket(
    data: pd.DataFrame,
    save: bool = False,
    path: str = ""
) -> None:
    """Creates grouped bar plots for homes sold within 1-2 weeks of listing.
    
    Args:
        data: DataFrame to be plotted.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
    
    Returns:
        Grouped bar plots for homes sold within 1-2 weeks of listing.
        Optionally saves plots in a given directory and file name.
    """

    # Grouping by years and calculating means for each weekly dataset
    two_weeks = (
        pd.DataFrame(
            data.groupby("Year")["off_market_in_two_weeks"].mean()
        )
            .rename(columns={"off_market_in_two_weeks": "sold"})
    )
    one_week = (
        pd.DataFrame(
            data.groupby("Year")["off_market_in_one_week"].mean()
        )
            .rename(columns={"off_market_in_one_week": "sold"})
    )
    
    # Providing categorical labels for plotting
    two_weeks["Time Frame"] = "2 Weeks"
    one_week["Time Frame"] = "1 Week"
    
    # Combining datasets for plotting
    grouped = pd.concat([two_weeks, one_week], axis="index")

    # Plotting
    sns.catplot(
        data = grouped,
        kind = "bar",
        x = "Year",
        y = "sold",
        hue = "Time Frame",
        palette = "tab10",
        gap = 0.1
    )

    # Plot aesthetics
    plt.xlabel("Year")
    plt.title(
        "Annual Average Number of Homes Sold "
        "\nin Miami Within 1-2 Weeks of Listing"
    )
    plt.ylabel("Number of Houses Sold")
    
    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def topPrice(
    data: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """Identifies the top n states with the highest median sale price.
    
    Args:
        data: DataFrame to be used for analyses (all data).
        n: Number of states to be returned.
        
    Returns:
        DataFrame containing the top n states with the highest median sale
        price.
    """

    # Sorting by median sales price in descending order
    regions = (
        data.sort_values("median_sale_price", ascending=False)
            .loc[:, ["region_name", "period_begin", "median_sale_price"]]
            
            # Removing regions that appear multiple times
            .drop_duplicates(subset=["region_name"])
            .head(n)
    )
    
    return regions


def dynamicPriceDays(
    data: pd.DataFrame,
    save: bool = False,
    path: str = ""
) -> None:
    """Creates scatter plot of median sale price vs. median days to close.
    
    Args:
        data: DataFrame to be plotted.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
    
    Returns:
        Scatter plot of median sale price vs. median days to close. Optionally
        saves plot in a given directory and file name.
    """

    # Setting up plot
    _, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    sns.scatterplot(
        data = data,
        x = "median_sale_price", 
        y = "median_days_to_close",
        hue = "Year",
        palette = "tab10",
        ax = ax,
        alpha = 0.75
    )

    # Plot aesthetics
    plt.title("Sale Price vs. Days to Close in Miami (2017-2023)")
    plt.xlabel("Median Sale Price")
    plt.ylabel("Median Days to Close")
    ax.grid(True, linestyle="--")
    
    # Moving markers after labels and decreasing space between them
    plt.legend(markerfirst=False, handletextpad=0)
    
    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def plotDays(
    data: pd.DataFrame,
    save: bool = False,
    path: str = ""
) -> None:
    """Plots histograms of median days to close for each year.
    
    Args:
        data: DataFrame to be plotted.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
        
    Returns:
        Histograms of median days to close for each year. Optionally saves
        plots in a given directory and file name.
    """

    # Setting up plot
    _, ax = plt.subplots(2, 4, figsize=(20, 10))
    
    # Plotting for all years
    sns.histplot(
        data = data,
        x = "median_days_to_close",
        bins = 20,
        ax = ax[0, 0],
        color = "lightcoral"
    )
    ax[0, 0].set_title("Median Days to Close in Miami (2017-2023)")
    ax[0, 0].set_xlabel("")
    
    # Plotting for each year
    for i, year in enumerate(data["Year"].unique()):
        sns.histplot(
            data = data.loc[data["Year"] == year],
            x = "median_days_to_close",
            bins = 20,
            ax = ax[(i + 1)//4, (i + 1)%4],
            color = "tab:blue"
        )
        ax[(i + 1)//4, (i + 1)%4].set_title(f"Median Days to Close in {year}")
        ax[(i + 1)//4, (i + 1)%4].set_xlabel("")

    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def priceTrend(
    data: pd.DataFrame,
    save: bool = False,
    path: str = ""
) -> None:
    """Plots trends of median sale prices and total homes sold.
    
    Args:
        data: DataFrame to be plotted.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
    
    Returns:
        Trends of median sale prices and total homes sold. Optionally saves
        plots in a given directory and file name.
    """

    # Calculating medians for each year and sum of sales, then sorting by years
    median_price = (
        data.groupby("Year")["median_sale_price"]
            .median()
            .sort_index()
    )
    price_new_listing = (
        data.groupby("Year")["median_new_listing_price"]
            .median()
    )
    total_sold = data.groupby("Year")["total_homes_sold"].sum()

    # Plotting line plots above bar plot
    _, _ = plt.subplots(figsize=(7, 5))
    plt.plot(
        median_price.index,
        median_price.values, 
        marker = "o", 
        label = "Median Sales Price (MSP)"
    )
    plt.plot(
        price_new_listing.index, 
        price_new_listing.values, 
        marker = "o", 
        label = "MSP of New Listings"
    )
    plt.bar(
        total_sold.index, 
        total_sold.values, 
        label = "Total Homes Sold",
        color = "lightcoral"
    )

    # Plot aesthetics
    plt.legend(loc='upper left')
    plt.xlabel("Year")
    plt.ylabel("Number of Homes")
    plt.title("Trends of Price vs. Homes Sold in Miami (2017-2023)")
    
    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def comparisonSales(
    data: pd.DataFrame,
    comparison: int = 1,
    save: bool = False,
    path: str = ""
) -> None:
    """Creates scatter plot of total homes sold vs. months of supply.
    
    User can choose to compare total homes sold vs. months of supply (1) or 
    median sale price vs. median days to close (2).
    
    Args:
        data: DataFrame to be plotted.
        comparison: Variables to be compared; 1 or 2.
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
    
    Returns:
        Scatter plot of variables of interest. Optionally saves plot in a 
        given directory and file name.
    """

    # Setting up plot
    _, ax = plt.subplots(figsize=(10, 6))

    # Plotting for total homes sold vs. months of supply
    if comparison == 1:
        sns.scatterplot(
            data = data,
            y = "months_of_supply",
            x = "total_homes_sold",
            hue = "region_name",
            palette = "deep",
            ax = ax,
            alpha = 0.75
        )
        
        # Plot aesthetics
        plt.title("Total Homes Sold vs. Months of Supply")
        plt.xlabel("Total Homes Sold")
        plt.ylabel("Months of Supply")
        plt.legend(loc="upper right")
        plt.ylim(-10, 100)
    
    # Plotting for median sale price vs. median days to close
    elif comparison == 2:
        sns.scatterplot(
            data = data,
            x = "median_sale_price",
            y = "median_days_to_close",
            hue = "region_name",
            palette = "deep",
            ax = ax,
            alpha = 0.75
        )
        
        # Plot aesthetics
        plt.title("Median Sale Price vs. Median Days to Close")
        plt.xlabel("Median Sale Price ($)")
        plt.ylabel("Median Days to Close")
        plt.legend(loc="upper right")

    # If saving the plot
    if save:
        plt.savefig(path, bbox_inches="tight")


def trainRandomForest(
    data: pd.DataFrame,
    summary: bool = True,
    plot: bool = True,
    save: bool = False,
    path: str = ""
) -> None:
    """Trains a random forest model on given data.
    
    Uses a train-test split of 7-3 and random state of 42 for both train-test
    splits and random forest model. Uses 100 estimators.
    
    Args:
        data: DataFrame to be used for training the model.
        summary: Whether to print model fit metrics. (True/False)
        plot: Whether to plot model predictions. (True/False)
        save: Whether to save the plot. (True/False)
        path: Path to save the plot.
    
    Returns:
        Prints model fit metrics and plots model predictions. Optionally saves
        plots in a given directory and file name.
    """
    
    # Setting features and target
    X = data.drop(
        columns = [
            "median_sale_price",
            "region_name",
            "period_begin",
            "average_of_median_list_price_amount",
            "average_of_median_offer_price_amount",
            "Year"
        ]
    )
    if "status" in X.columns:
        X = X.drop(columns="status")
    y = data["median_sale_price"]
    
    # Train-test splits and scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        
    # Fitting model
    regRF = RandomForestRegressor(random_state=42, n_estimators=100)
    regRF.fit(X_train_scaled, y_train_scaled)
    
    # If user wants a summary of the model's fit metrics
    if summary:
        y_pred = scaler_y.inverse_transform(
            regRF.predict(X_test_scaled)
                .reshape(-1, 1)
        )
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Printing fit metrics
        print(
            f"Random Forest Regression Performance:"
            f"\nRMSE: {rmse:.2f}"
            f"\nR-squared: {r2:.2f}"
        )
    
    # If user wants to plot model predictions
    if plot:
        _, ax = plt.subplots(figsize=(6, 6))
        sns.regplot(
            x = y_test / 1000,
            y = y_pred / 1000,
            label = "Test Data",
            ax = ax,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "tab:red"}
        )
        plt.title(
            "Random Forest Regression Performance"
        )
        plt.xlabel("Actual Data ($, thousands)")
        plt.ylabel("Predicted Data ($, thousands)")
        
        # If saving the plot
        if save:
            plt.savefig(path, bbox_inches="tight")
