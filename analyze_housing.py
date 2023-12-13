# File Name: housing_funcs.ipynb
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
# Last Modified: 2023-12-12, 8:59 pm

# File Description:

    # Driver script for the analysis of US housing market data for the final
    # project. Imports functions from the utility file housing_funcs.py.


# Importing libraries and utilities ----
import os
import sys
import pandas as pd
import warnings

from gdown import download

from housing_funcs import (
    load_data, clean_data, subset_data, describe_data, test_assumptions,
    clean_titles, plot_descriptive, test_anova, test_post_hoc, fit_regression,
    build_svr, pickle_me, predict_new_sales, check_multi, fit_glm,
    heatmap_by_metrics, regression_analysis, removeOutliers, addYear,
    volumeSales, offMarket, dynamicPriceDays, plotDays, priceTrend,
    comparisonSales, trainRandomForest
)

# Turning off deprecation warnings from scikit-learn
warnings.filterwarnings("ignore")


# Main ----
def main():

    # Defensive programming
    # Creating a list of whether inputted data file exists
    exists = [os.path.isfile(file) for file in sys.argv[1:]]
    
    # If data file is not provided or does not exist or is not a .csv file
    if (
        len(sys.argv) != 2
        or False in exists
        or not sys.argv[1].endswith(".csv")
    ):
        sys.exit(
            "\nPlease provide a .csv file to be analyzed."
            "\nUsage: py analyze_housing.py data/housing.csv"
        )
    
    # Reading in data
    raw = load_data(sys.argv[1])
    cities = ["nyc", "seattle", "miami"]
    members = ["j", "k", "n"]
    analytics = ["descriptive", "diagnostic", "predictive", "prescriptive"]
    
    # If /results/ does not exist
    if not os.path.isdir("./results"):
        root = "./results"
        os.mkdir(root)
        os.makedirs(os.path.join(root, "all_cities"), exist_ok=True)
        for city in cities:
            subdirs = os.path.join(root, city)
            os.makedirs(subdirs, exist_ok=True)
            for section in analytics:
                os.makedirs(os.path.join(subdirs, section), exist_ok=True)
    
    # If /results/ exists, but not all subdirectories
    elif (
        os.path.isdir("./results")
        and (
            not os.path.isdir("./results/nyc")
            or not os.path.isdir("./results/seattle")
            or not os.path.isdir("./results/miami")
            or not os.path.isdir("./results/all_cities")
        )
    ):
        root = "./results"
        os.makedirs(os.path.join(root, "all_cities"), exist_ok=True)
        for city in cities:
            subdirs = os.path.join(root, city)
            os.makedirs(subdirs, exist_ok=True)
            for section in analytics:
                os.makedirs(os.path.join(subdirs, section), exist_ok=True)
    
    # Status message
    print(
        "\nAnalyzing the housing market data of New York City, "
        "\nSeattle, and Miami, United States, from 2017-2023. "
        "\nData source: Redfin (redfin.com/news/data-center/)..."
    )
    
    # Preparing data for each members' functions and analyses
    data = {
        
        # Jerry
        "nyc_j": clean_data(raw, "nyc", "j")["before"],
        "seattle_j": clean_data(raw, "seattle", "j")["before"],
        "miami_j": clean_data(raw, "miami", "j")["before"],
        
        # Kai
        "nyc_k": clean_data(raw, "nyc", "k")["before"],
        "seattle_k": clean_data(raw, "seattle", "k")["before"],
        "miami_k": clean_data(raw, "miami", "k")["before"],
        
        # Nargiz
        "nyc_n": clean_data(raw, "nyc", "n")["before"],
        "seattle_n": clean_data(raw, "seattle", "n")["before"],
        "miami_n": clean_data(raw, "miami", "n")["before"]
    }
    for member in members:
        data[f"all_cities_{member}"] = pd.concat(
            [
                data[f"nyc_{member}"],
                data[f"seattle_{member}"],
                data[f"miami_{member}"]
            ]
        )
    
    
    # Descriptive analytics ----
    # Status message
    print(
        "---------------------------------------------------------------"
        "\nSaving descriptive analytics to /results/..."
    )
    
    # Saving list of top 10 cities with highest median sales price
    # Since this uses the ENTIRE 3.5 GB dataset, it is instead downloaded from
    # a pre-generated figure from our weekly update notebook to significantly
    # reduce script runtime. The code for generating the figure is also
    # available in the notebook, under Nargiz's Analyses.
    download(
        "https://drive.google.com/uc?id=18DoSQveu7ee1B-EAkMzlHNRrtoTk0Agy",
        output = "./results/all_cities/top10.png",
        quiet = True
    )
    
    # Saving descriptive analytics to .csv files
    days = describe_data(data["nyc_j"])["days"]
    for city in cities:
        
        # Descriptive statistics
        describe_data(
            data[f"{city}_j"],
            save = True,
            path1 = f"./results/{city}/descriptive/summary.csv",
            path2 = f"./results/{city}/descriptive/monthly_means.csv"
        )
        
        # Normality and variance tests
        test_assumptions(
            data[f"{city}_j"],
            test = "n",
            save = True,
            path = f"./results/{city}/descriptive/normality.csv"
        )
        test_assumptions(
            data["all_cities_j"],
            test = "v",
            save = True,
            path = f"./results/{city}/descriptive/variance.csv"
        )

    # Distributional plots to investigate normality in NYC variables
    # These functions can also be put into the for loop above to save plots
    # for all cities; doing it only for NYC is just to reduce script runtime.
    plot_descriptive(
        data["nyc_j"],
        plot = "v",
        save = True,
        path = "./results/nyc/descriptive/violinplots.svg"
    )
    plot_descriptive(
        data["nyc_j"],
        plot = "q",
        save = True,
        path = "./results/nyc/descriptive/qqplots.svg"
    )
    plot_descriptive(
        data["nyc_j"],
        plot = "h",
        save = True,
        path = "./results/nyc/descriptive/histograms.svg"
    )
    
    # Longitudinal plots to investigate trends in Seattle variables
    # Similar to above, this plot requires the ENTIRE 3.5 GB dataset, so it is
    # instead downloaded from a pre-generated plot to significantly reduce
    # script runtime. The code for generating the plot is also available in
    # the notebook, under Kai's Analyses.
    download(
        "https://drive.google.com/uc?id=1JJ0TXhkH58X-fPRWMQN-xiPGPxdOq08d",
        output = "./results/seattle/descriptive/longitudinal.svg",
        quiet = True
    )
    
    # Distributional plots for variables in Miami
    cleaned_nargiz = removeOutliers(
        data["miami_n"],
        plot = True,
        save = True,
        path = "./results/miami/descriptive/outliers.svg"
    )
    
    # Number of sales per year in Miami
    miami_nargiz = addYear(cleaned_nargiz)
    volumeSales(
        miami_nargiz,
        save = True,
        path = "./results/miami/descriptive/annual_sales.svg"
    )
    
    # Number of houses sold in 1 and 2 weeks for each year in Miami
    offMarket(
        miami_nargiz,
        save = True,
        path = "./results/miami/descriptive/weekly_sales.svg"
    )
    
    # Status message
    print(
        f"\nA total of {days} days of data entries were analyzed."
        f"\nPlease note that variance.csv should be the same for all cities."
    )

    
    # Diagnostic analytics ----
    # Status message
    print(
        "---------------------------------------------------------------"
        "\nSaving diagnostic analytics to /results/..."
    )
    
    # A few variables that highly correlated with median sales price were saved
    # from Kai's first week update heatmaps. These are then plotted against
    # each other to see multicollinearity for ordinary least squares (OLS)
    # regression. # However, functions are also coded to be able to analyze ALL
    # variables; this is simply to reduce script runtime.
    nyc_ols = subset_data(data["nyc_j"])
    check_multi(
        nyc_ols,
        save = True,
        path = "./results/nyc/diagnostic/multicollinearity.svg"
    )
    
    # Non-parametric analysis of variance (ANOVA) of variables for each city
    # The above variables are used for these subsequent comparison tests.
    all_cities_tests = subset_data(data["all_cities_j"])
    test_anova(
        all_cities_tests,
        save = True,
        path = "./results/all_cities/anova.csv"
    )
    
    # Post-hoc non-parametric comparison tests for significantly differing
    # variables between cities
    test_post_hoc(
        all_cities_tests,
        save = True,
        path = "./results/all_cities/post_hoc.csv"
    )
    
    # OLS and GLMs to see impact of each variable on median sales price
    ols_results = fit_regression(
        nyc_ols,
        plot = True,
        save = True,
        path = "./results/nyc/diagnostic/ols.svg"
    )
    glm_results = fit_glm(
        nyc_ols,
        save = True,
        path = "./results/nyc/diagnostic/glm.svg"
    )
    
    # Saving summary tables to a pandas DataFrame
    ols_html = ols_results.summary().tables[1].as_html()
    ols_df = (
        pd.read_html(ols_html, header=0, index_col=0)[0]
            .rename(columns={"P>|t|": "pval"})
    )
    ols_n_preds = len(ols_df)
    ols_stats_html = ols_results.summary().tables[0].as_html()
    ols_stats_df = pd.read_html(ols_stats_html, header=None, index_col=None)[0]
    ols_r = ols_stats_df.iloc[1, 3]
    glm_html = glm_results.summary().tables[0].as_html()
    glm_df = pd.read_html(glm_html, header=None, index_col=None)[0]
    glm_r = glm_df.iloc[7, 3]
    
    # Saving significant positive and negative coefficients
    ols_pos = clean_titles(
        ols_df.query("coef > 0 and pval < 0.05")
            .index
            .to_list()
    )
    ols_neg = clean_titles(
        ols_df.query("coef < 0 and pval < 0.05")
            .index
            .to_list()
    )
    compare = "GLM" if glm_r > ols_r else "OLS"
    
    # OLS regression coefficient summary
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        "\nLinear regression summary of New York City:"
        f"\n\nOut of {ols_n_preds} predictors..."
        "\nSignificant positive coefficients:"
    )
    for i, coef in enumerate(ols_pos, start=1):
        print(f"{i}. {coef}")
    print("\nSignificant negative coefficients:")
    for i, coef in enumerate(ols_neg, start=1):
        print(f"{i}. {coef}")
    
    # GLM regression comparison with OLS
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        "\nAfter fitting a generalized linear model "
        f"\nto the same data, the {compare} had a higher "
    )
    if compare == "GLM":
        print(f"pseudo-R-squared value of {glm_r:.2f}.")
    else:
        print(f"R-squared value of {ols_r:.2f}.")
    
    # Heatmaps of correlation between variables of each city
    heatmap_by_metrics(
        data["all_cities_k"],
        save = True,
        
        # Although this is technically for all 3 cities, the original function
        # was made by Kai, so I placed results in his city's directory.
        path = "./results/seattle/diagnostic/correlations.svg"
    )
    heatmap_by_metrics(
        data["all_cities_k"],
        strong_only = True,
        save = True,
        
        # Similarly, this functionality was originally made by Nargiz, so I
        # place the results in her directory, despite it being slightly less
        # user-friendly.
        path = "./results/miami/diagnostic/correlations_strong.svg"
    )
    
    # Similar to previously done, I download a pre-generated heatmap from
    # my weekly update notebook to reduce script runtime, as the function draws
    # from the whole dataset. Relevant code is under Kai's Analyses section in
    # my third week update notebook.
    download(
        "https://drive.google.com/uc?id=1YZWakuGdFzZoLxfC59rlf3C6UQbDxKfN",
        output = "./results/all_cities/heatmaps.svg",
        quiet = True
    )
    
    # Relationship between median sales price and days to close in Miami
    dynamicPriceDays(
        miami_nargiz,
        save = True,
        path = "./results/miami/diagnostic/sale_price_days_to_close.svg"
    )
    
    # Distributions of annual median days to close in Miami
    plotDays(
        miami_nargiz,
        save = True,
        path = "./results/miami/diagnostic/distributions.svg"
    )
    
    # Annual median sales price compared to total homes sold
    priceTrend(
        miami_nargiz,
        save = True,
        path = "./results/miami/diagnostic/price_trends.svg"
    )
    
    # Total homes sold vs. months of supply in Miami
    combined_nargiz = removeOutliers(data["all_cities_n"])
    comparisonSales(
        combined_nargiz,
        comparison = 1,
        save = True,
        path = "./results/miami/diagnostic/homes_sold_months_supply.svg"
    )
    
    # Median sale price vs. median days to close
    comparisonSales(
        combined_nargiz,
        comparison = 2,
        save = True,
        path = "./results/miami/diagnostic/sale_price_median_days_to_close.svg"
    )
    
    
    # Predictive analytics ----
    # Status message
    print(
        "---------------------------------------------------------------"
        "\nSaving predictive analytics to /results/..."
    )
    
    # Support vector regression (SVR) to see impact of each variable on median
    # sales price for all 3 cities
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        "\nSupport vector regression summary of New York City:\n"
    )
    svr_nyc = build_svr(
        data["nyc_j"],
        iters = 1,
        summary = True,
        save = True,
        path = "./results/nyc/predictive/svr.svg"
    )
    
    # Saving SVR model to a pickle file
    pickle_me("./results/nyc/predictive/svr_nyc.pickle", model=svr_nyc)
    print("SVR model saved to results/nyc/predictive/svr_nyc.pickle")
    
    # SVR model for all three cities
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        "\nSupport vector regression summary of all three cities:\n"
    )
    svr_all = build_svr(
        data["all_cities_j"],
        iters = 1,
        summary = True,
        save = True,
        path = "./results/all_cities/svr.svg"
    )
    pickle_me("./results/all_cities/svr_all.pickle", model=svr_all)
    
    # Saving SVR model fits of Seattle and Miami
    print("Saving model fit performance plots...")
    more_cities = ["seattle", "miami"]
    for city in more_cities:
        build_svr(
            data[f"{city}_j"],
            iters = 1,
            summary = False,
            save = True,
            path = f"./results/{city}/predictive/svr.svg"
        )
    
    # Predicting median sales prices for new NYC entries after August 2023
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        "\nPredicting median sales prices for new entries "
        "\nafter August 2023 in New York City:"
        "\n\nPrediction Accuracy:"
    )
    nyc_new = (
        clean_data(
            load_data("./data/new_housing.csv"),
            "nyc",
            "j"
        )
        ["after"]
    )
    coefs = predict_new_sales(
        nyc_new,
        svr_nyc,
        save = True,
        path = "./results/nyc/predictive/predictions.svg"
    )["coefs"]
    
    # Lasso and Ridge regression of median sale price in Seattle
    lasso = regression_analysis(
        data["seattle_k"],
        "l",
        save = True,
        path = "./results/seattle/predictive/lasso.svg"
    )
    ridge = regression_analysis(
        data["seattle_k"],
        "r",
        save = True,
        path = "./results/seattle/predictive/ridge.svg"
    )
    better = "Lasso" if lasso > ridge else "Ridge"
    
    # Printing comparison of results
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        "\nLasso and Ridge regression of median sales price in Seattle:"
        f"\n\nAfter fitting both models, the {better} "
    )
    if better == "Lasso":
        print(f"model had a higher test score of {lasso:.2f}.")
    else:
        print(f"model had a higher test score of {ridge:.2f}.")
    
    # Random forest regressor of median sale price in Miami
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        "\nRandom forest regression of median sales price in Miami:\n"
    )
    trainRandomForest(
        miami_nargiz,
        summary = True,
        plot = True,
        save = True,
        path = "./results/miami/prescriptive/forest.svg"
    )
    
    
    # Prescriptive analytics ----
    print(
        "---------------------------------------------------------------"
        "\nSaving prescriptive analytics to /results/..."
    )
    
    # Displaying only positively-weighted features of the SVR model
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        "\nSupport vector regression of New York City used the following:"
    )
    n_coefs = len(coefs)
    n_pos_coefs = len(coefs.query("Coefficient > 0"))
    pos_coefs = zip(
        coefs.query("Coefficient > 0").index,
        coefs.query("Coefficient > 0")["Coefficient"]
    )
    print(
        f"\n{n_coefs} features, of which only {n_pos_coefs} were "
        "positively-weighted "
        "\nin terms of predicting median sales price:"
    )
    for i, coef in enumerate(pos_coefs, start=1):
        print(f"{i}. {coef[0]}: {coef[1]:.2f}")
    
    # Downloading pre-generated figure for Kai's prescriptive analysis of
    # Seattle. Code for figure is under Kai's Analyses in my third weekly
    # update notebook.
    download(
        "https://drive.google.com/uc?id=1i9jMrZJhmYLykBMABJJ0QVBIXa31lGB8",
        output = "./results/seattle/prescriptive/regression_results.png",
        quiet = True
    )
    
    # Again, downloading pre-generated figure from my third weekly update
    # notebook, at the end of Jerry's Analyses, since pd.styler objects cannot
    # be displayed in terminal.
    download(
        "https://drive.google.com/uc?id=1T94gpeNfBsrDzdnxXTCUxvx5kAxk8SO5",
        output = "./results/nyc/prescriptive/coefficients.png",
        quiet = True
    )

    # Final status message
    print(
        "---------------------------------------------------------------"
        "\nAnalysis complete. Results saved to /results/."
        "\nThank you for using our program!"
        "\n- Jerry, Kai, Nargiz, 2023"
    )


# Checking if this file is being run directly or imported
if __name__ == "__main__":
    main()
