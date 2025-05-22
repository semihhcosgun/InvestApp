

import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import json
import matplotlib.pyplot as plt # Matplotlib still imported, though Plotly is primary for charts
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# PDF Generation Libraries
from fpdf import FPDF
import io # To handle image data in memory

# Ensure FPDF can handle UTF-8 characters - requires a font supporting them
# Using a default UTF-8 font like DejaVuSansCondensed
# Need to download this font file and place it in a directory accessible by the script
# Or rely on fpdf2's built-in handling which might be limited without specifying fonts
# For simplicity, let's hope basic chars are enough or user installs font if needed for special chars
# Or, you can specify a font path if you include the font file:
# pdf.add_font('DejaVuSansCondensed', '', 'DejaVuSansCondensed.ttf', uni=True)
# pdf.set_font('DejaVuSansCondensed', '', 12)

# Define the stocks to analyze
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-A', 'V', 'JPM', 'JNJ']

# Data loading function
@st.cache_data  # Updated caching method
def load_data(stocks, start_date_yf, end_date_yf):
    # Download stock data from Yahoo Finance
    try:
        stock_data = yf.download(stocks, start=start_date_yf, end=end_date_yf, progress=False)['Close']
    except Exception as e:
        st.error(f"Error downloading stock data: {e}")
        return None, None

    # Check if stock_data is empty or contains NaNs
    if stock_data.empty:
        # Check if the date range might be the issue (e.g., weekend, future)
        if start_date_yf > datetime.now() or end_date_yf > datetime.now():
             st.warning("Selected date range is in the future.")
        else:
             st.error("Could not download data for the specified date range. Please try different dates or check stock symbols.")
        return None, None

    initial_cols = stock_data.shape[1]
    if stock_data.isnull().sum().sum() > 0:
         st.warning("Downloaded data contains missing values. Attempting to handle them.")
         stock_data = stock_data.ffill().bfill() # Forward and backward fill NaNs
         # After fill, drop columns that still have NaNs (e.g., if entire column was NaN)
         stock_data = stock_data.dropna(axis=1)
         if stock_data.shape[1] < initial_cols:
              st.warning(f"Dropped {initial_cols - stock_data.shape[1]} stock(s) due to insufficient data after filling.")


    # Check if stock_data has enough rows and columns after cleaning
    if len(stock_data) < 2:
         st.error("Not enough data points after cleaning to calculate returns. Please select a wider date range.")
         return stock_data, None
    if stock_data.shape[1] == 0:
         st.error("No valid stock data columns remaining after handling missing values.")
         return stock_data, None


    # Calculate daily returns
    # Ensure there are at least 2 data points per stock after cleaning to calculate pct_change
    # dropna() on returns will remove rows/columns where returns couldn't be calculated
    daily_returns = stock_data.pct_change().dropna(how='all').dropna(axis=1) # drop rows where all are NaN, then cols with any NaN

    # Check if daily_returns is empty after dropna
    if daily_returns.empty:
         st.error("Could not calculate valid returns for the specified date range for any stock. This can happen if data is too sparse. Please try different dates or check stock symbols.")
         return stock_data, None # Return stock_data even if returns fail

    # Filter stock_data to only include stocks for which returns were calculated
    valid_stocks_for_returns = daily_returns.columns.tolist()
    stock_data_filtered = stock_data[valid_stocks_for_returns]

    return stock_data_filtered, daily_returns

# --- Streamlit UI ---

st.sidebar.title("Investment Application")
page = st.sidebar.radio("Select Page:", ["Portfolio Analysis", "Stock Data"])

# Ask user to input start and end date (placed outside page logic for consistency)
start_date_input = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01").date())
end_date_input = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-03-01").date())

# Get today's date
today = datetime.today().date()

# If the end date is in the future, set it to today's date
if end_date_input > today:
    st.warning(f"End date cannot be in the future. The end date has been set to {today}.")
    end_date_input = today

# Convert back to datetime for load_data. yfinance 'end' parameter is exclusive.
# Add one day to include the selected end_date's data.
start_date_yf = datetime.combine(start_date_input, datetime.min.time())
# Add 1 day to the end_date_input to make yfinance inclusive up to the selected date
end_date_yf = datetime.combine(end_date_input, datetime.min.time()) + pd.Timedelta(days=1)


# Check if start_date is before end_date before loading (using input dates for user logic)
if start_date_input >= end_date_input:
     st.error("Start date must be before the end date. Please adjust the dates.")
     stock_data, daily_returns = None, None # Prevent loading data and subsequent errors
else:
    # Load data based on the selected date range
    stock_data, daily_returns = load_data(stocks, start_date_yf, end_date_yf)


# Only proceed if data was successfully loaded and returns calculated
if stock_data is not None and daily_returns is not None:
    # Compute mean returns and covariance matrix using the potentially filtered daily_returns
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Display the actual range of data loaded
    st.write(f"Analyzing data from: **{stock_data.index[0].date()}** to **{stock_data.index[-1].date()}**")


    def optimal_portfolio(selected_stocks_input, method='SLSQP', num_simulations=10000):
        # Filter selected_stocks_input to only include those with valid data (present in mean_returns/cov_matrix)
        available_stocks = [s for s in selected_stocks_input if s in mean_returns.index and s in cov_matrix.columns]

        if len(available_stocks) < 1: # Need at least one stock with data
             st.warning(f"No data available for selected stocks ({', '.join(selected_stocks_input)}) in the specified date range.")
             # Return zeroed results for consistent structure, but indicate no valid stocks
             return np.zeros(len(selected_stocks_input)), 0, 0, 0, None, None, None, [] # Return empty list for valid_available_stocks

        if method == 'SLSQP' and len(available_stocks) < 2:
             st.warning(f"SLSQP optimization requires at least two stocks with data. Only found: {', '.join(available_stocks)}. Cannot perform optimization.")
             # Return zeroed results, but indicate failure for SLSQP
             return np.zeros(len(selected_stocks_input)), 0, 0, 0, None, None, None, available_stocks # Return available_stocks


        num_available_assets = len(available_stocks)
        selected_returns = mean_returns[available_stocks]
        selected_cov_matrix = cov_matrix.loc[available_stocks, available_stocks]


        optimal_weights_available = np.zeros(num_available_assets)
        optimal_return = 0.0
        optimal_stddev = 0.0
        optimal_sharpe = 0.0
        ret_arr = None # For Monte Carlo simulation returns
        vol_arr = None # For Monte Carlo simulation volatilities
        sharpe_arr = None # For Monte Carlo simulation sharpe ratios

        if method == 'SLSQP':
            def negative_sharpe(weights):
                # Ensure weights are non-negative and sum to 1 for calculation
                # The optimizer handles constraints, but internal calculation needs valid weights
                w = np.maximum(0, weights)
                # Add a tiny epsilon before summing and normalizing to avoid sum being exactly zero if weights are tiny
                sum_w = np.sum(w)
                w = w / sum_w if sum_w > 1e-9 else np.ones(num_available_assets) / num_available_assets

                portfolio_return = np.sum(w * selected_returns)
                try:
                    # Ensure cov_matrix is not singular for np.dot
                    # Add a small diagonal to covariance matrix for numerical stability in case of perfect correlation
                    # A common technique is shrinkage or adding a small identity matrix proportional to variance
                    # For simplicity here, let's just catch the LinAlgError. More robust would be:
                    # cov_matrix_stable = selected_cov_matrix + np.eye(num_available_assets) * 1e-6 * np.trace(selected_cov_matrix) / num_available_assets
                    portfolio_stddev = np.sqrt(np.dot(w.T, np.dot(selected_cov_matrix, w)))
                except np.linalg.LinAlgError:
                     # st.warning("Covariance matrix is singular during optimization.") # Avoid spamming warnings during internal optimization steps
                     return np.inf # Treat as worst case for optimizer

                if portfolio_stddev < 1e-9: # Use a small tolerance for zero stddev
                    # If risk is near zero, maximize return (minimize negative return)
                    return -portfolio_return
                return - (portfolio_return / portfolio_stddev) # Maximize Sharpe

            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = tuple((0, 1) for _ in range(num_available_assets))
            initial_weights = np.ones(num_available_assets) / num_available_assets

            try:
                # Set a higher maxiter or different options if optimization frequently fails
                # disp=False prevents optimization details from printing to console
                optimized = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False, 'maxiter': 1000})

                if not optimized.success:
                     st.warning(f"SLSQP optimization failed: {optimized.message}. Results may not be perfectly optimal.")
                     optimal_weights_raw = optimized.x
                else:
                    optimal_weights_raw = optimized.x

                # Post-process weights: ensure non-negative and sum to 1
                optimal_weights_available = np.maximum(0, optimal_weights_raw)
                sum_optimal_weights = np.sum(optimal_weights_available)
                if sum_optimal_weights > 1e-9:
                    optimal_weights_available /= sum_optimal_weights
                else:
                     optimal_weights_available = np.zeros(num_available_assets) # Fallback if normalization fails, set weights to zero

                # Recalculate return, stddev, sharpe with the final processed weights
                optimal_return = np.sum(optimal_weights_available * selected_returns)
                # Recalculate stddev with processed weights, handle LinAlgError again
                try:
                     optimal_stddev = np.sqrt(np.dot(optimal_weights_available.T, np.dot(selected_cov_matrix, optimal_weights_available)))
                except np.linalg.LinAlgError:
                     optimal_stddev = np.nan # Indicate error in stddev calculation

                optimal_sharpe = optimal_return / optimal_stddev if optimal_stddev is not None and optimal_stddev > 1e-9 and not np.isnan(optimal_stddev) else 0


            except Exception as e:
                st.error(f"An unexpected error occurred during SLSQP optimization: {e}")
                # Return zeroed results on exception
                optimal_weights_available = np.zeros(num_available_assets)
                optimal_return = 0.0
                optimal_stddev = 0.0
                optimal_sharpe = 0.0


        elif method == 'Monte Carlo':
            if num_available_assets == 0:
                 # This case is handled by the initial check, but keeping here for completeness
                 return np.zeros(len(selected_stocks_input)), 0, 0, 0, None, None, None, available_stocks

            all_weights = np.zeros((num_simulations, num_available_assets))
            ret_arr = np.zeros(num_simulations)
            vol_arr = np.zeros(num_simulations)
            sharpe_arr = np.zeros(num_simulations)

            for i in range(num_simulations):
                weights = np.random.random(num_available_assets)
                weights /= np.sum(weights)
                all_weights[i, :] = weights

                ret = np.sum(weights * selected_returns)
                try:
                    vol = np.sqrt(np.dot(weights.T, np.dot(selected_cov_matrix, weights)))
                except np.linalg.LinAlgError:
                    vol = np.nan # Handle error case

                ret_arr[i] = ret
                vol_arr[i] = vol
                sharpe_arr[i] = ret / vol if vol is not None and vol > 1e-9 and not np.isnan(vol) else 0 # Use tolerance for zero stddev, handle NaN


            # Find the portfolio with the maximum Sharpe Ratio
            # Use nanargmax to handle potential NaNs in sharpe_arr
            # nanargmax returns the index of the first NaN if all are NaN, check for this.
            if np.all(~np.isfinite(sharpe_arr)): # Check if all sharpe ratios are non-finite (NaN, Inf, -Inf)
                 max_sharpe_idx = -1 # Indicate no valid optimal portfolio found
                 st.warning("No finite Sharpe Ratios found in Monte Carlo simulations.")
            else:
                 # np.nanargmax ignores NaNs but can return 0 if the first element is the only non-NaN
                 # A robust check would be to ensure there's at least one finite value before calling nanargmax
                 finite_sharpe_indices = np.where(np.isfinite(sharpe_arr))[0]
                 if len(finite_sharpe_indices) > 0:
                     max_sharpe_idx = finite_sharpe_indices[np.argmax(sharpe_arr[finite_sharpe_indices])]
                 else:
                      max_sharpe_idx = -1 # Should be covered by the np.all(~np.isfinite) check, but double check

            optimal_weights_available = None
            optimal_return = None
            optimal_stddev = None
            optimal_sharpe = None

            if max_sharpe_idx != -1:
                optimal_weights_available = all_weights[max_sharpe_idx]
                optimal_return = ret_arr[max_sharpe_idx]
                optimal_stddev = vol_arr[max_sharpe_idx]
                optimal_sharpe = sharpe_arr[max_sharpe_idx]
            else:
                 # If no valid sharpe found, use zeroed values
                 optimal_weights_available = np.zeros(num_available_assets)
                 optimal_return = 0.0
                 optimal_stddev = 0.0
                 optimal_sharpe = 0.0


            # Filter out any non-finite values from MC arrays for plotting and histogram stats
            valid_indices = np.isfinite(ret_arr) & np.isfinite(vol_arr) & np.isfinite(sharpe_arr)
            ret_arr_filtered = ret_arr[valid_indices]
            vol_arr_filtered = vol_arr[valid_indices]
            sharpe_arr_filtered = sharpe_arr[valid_indices]
            # We return the filtered arrays for plotting
            ret_arr = ret_arr_filtered
            vol_arr = vol_arr_filtered
            sharpe_arr = sharpe_arr_filtered


        # Create the full optimal weights array including stocks that had no data
        full_optimal_weights = np.zeros(len(selected_stocks_input))
        if optimal_weights_available is not None:
             for i, stock in enumerate(selected_stocks_input):
                 if stock in available_stocks:
                     full_optimal_weights[i] = optimal_weights_available[available_stocks.index(stock)]


        # Return full_optimal_weights (for all selected_stocks_input),
        # optimal metrics (calculated from available_stocks),
        # and MC plot data arrays (if MC method)
        # Also return the list of stocks that actually had data
        return full_optimal_weights, optimal_return, optimal_stddev, optimal_sharpe, ret_arr, vol_arr, sharpe_arr, available_stocks


    def predict_stock_price(stock_data_series, future_days=30):
        # Use the specific stock data series directly, assume NaNs handled upstream
        stock_prices = stock_data_series.values
        dates_index = np.array(range(len(stock_prices))).reshape(-1, 1)

        if len(stock_prices) < 2:
             # Should be handled upstream, but safety check
             st.warning("Not enough data points for linear regression.")
             return np.array([]), pd.DatetimeIndex([]), None, None, None


        model = LinearRegression()
        model.fit(dates_index, stock_prices)

        future_dates_index = np.array(range(len(stock_prices), len(stock_prices) + future_days)).reshape(-1, 1)
        predicted_prices = model.predict(future_dates_index)

        last_date = stock_data_series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

        intercept = model.intercept_
        slope = model.coef_[0]

        return predicted_prices, future_dates, intercept, slope, model

    # --- PDF Report Generation Function ---
    # Pass only necessary information to the PDF function
    def create_portfolio_report(report_data):
        pdf = FPDF()
        # Add a Unicode font for wider character support if needed (e.g., if stock symbols have special chars)
        # Ensure this font file exists or skip this if only basic ASCII is expected
        # pdf.add_font('DejaVuSansCondensed', '', 'DejaVuSansCondensed.ttf', uni=True)
        # pdf.add_page()
        # pdf.set_font("DejaVuSansCondensed", size=12)

        # Using default Arial for simplicity, might not support all chars
        pdf.add_page()
        pdf.set_font("Arial", size=12)


        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Portfolio Analysis Report", ln=True, align='C')
        pdf.ln(10) # Add some space

        pdf.set_font("Arial", size=12)

        # Parameters
        # Check if selected_stocks list is not empty before joining
        selected_stocks_text = ', '.join(report_data['selected_stocks']) if report_data['selected_stocks'] else "None Selected"
        pdf.cell(200, 10, txt=f"Selected Stocks: {selected_stocks_text}", ln=True)

        pdf.cell(200, 10, txt=f"Data Range: {report_data['start_date_input']} to {report_data['end_date_input']}", ln=True)
        pdf.cell(200, 10, txt=f"Investment Amount: ${report_data['investment_amount']:,.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Analysis Period: {report_data['time_period']}", ln=True)
        pdf.cell(200, 10, txt=f"Optimization Method: {report_data['optimization_method']}", ln=True)
        if report_data['optimization_method'] == 'Monte Carlo' and report_data['num_simulations'] is not None:
             pdf.cell(200, 10, txt=f"Number of Simulations: {report_data['num_simulations']}", ln=True)

        pdf.ln(10)

        # Check if optimization results are available
        if report_data['adjusted_return'] is not None:
            # Optimal Portfolio Metrics
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Optimal Portfolio Metrics", ln=True)
            pdf.set_font("Arial", size=12)

            # Use adjusted metrics from report_data
            pdf.cell(200, 10, txt=f"Portfolio {report_data['time_period']} Return: {report_data['adjusted_return'] * 100:.2f}%", ln=True)
            pdf.cell(200, 10, txt=f"Portfolio {report_data['time_period']} Volatility (Risk): {report_data['adjusted_stddev'] * 100:.2f}%", ln=True)
            # Use report_data['time_period'] directly for Sharpe Ratio text
            pdf.cell(200, 10, txt=f"Portfolio {report_data['time_period']} Sharpe Ratio: {report_data['adjusted_sharpe']:.4f}", ln=True)


            pdf.ln(10)

            # Stock Allocation Table
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Optimal Stock Allocation", ln=True)
            pdf.set_font("Arial", size=10) # Smaller font for table

            # Headers
            # Calculate column widths to fit within page (approx 190mm usable width)
            col_width = 190 / 3 # Divide available width by number of columns

            pdf.cell(col_width, 10, txt="Stock", border=1)
            pdf.cell(col_width, 10, txt="Weight (%)", border=1)
            pdf.cell(col_width, 10, txt="Investment Amount ($)", border=1, ln=True)

            # Data rows
            pdf.set_font("Arial", size=10)
            allocation_df = report_data['allocation_df']
            if not allocation_df.empty:
                for index, row in allocation_df.iterrows():
                    pdf.cell(col_width, 10, txt=str(row['Stock']), border=1)
                    pdf.cell(col_width, 10, txt=f"{row['Weight (%)']:.2f}", border=1) # Ensure formatting
                    pdf.cell(col_width, 10, txt=f"{row['Investment Amount ($)']:.2f}", border=1, ln=True) # Ensure formatting
            else:
                 pdf.cell(col_width * 3, 10, txt="No significant allocation.", border=1, ln=True, align='C')

            pdf.ln(10)

            # Add Plot (if available) - Only for Monte Carlo
            if report_data['plotly_fig'] is not None and report_data['optimization_method'] == 'Monte Carlo':
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Monte Carlo Simulation: Risk vs Return", ln=True)
                pdf.ln(5)

                # Save plotly figure to a bytes object
                try:
                    img_bytes = report_data['plotly_fig'].to_image(format="png", width=800, height=500) # Adjust size as needed
                    img = io.BytesIO(img_bytes)

                    # Add image to PDF
                    img_width_mm = 180 # Desired width in mm
                    # Calculate position to center the image (page width 210mm, margins approx 10mm each side)
                    x_pos = (210 - img_width_mm) / 2

                    # Check if adding image exceeds page height, add new page if necessary
                    # Assuming a reasonable aspect ratio, calculate approximate image height in mm
                    # Original height/width ratio = 500/800 = 0.625
                    img_height_mm_approx = img_width_mm * (500/800)

                    if pdf.get_y() + img_height_mm_approx > pdf.eph - 10: # eph is effective page height, -10 for bottom margin
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(200, 10, txt="Monte Carlo Simulation: Risk vs Return (cont.)", ln=True)
                        pdf.ln(5)


                    # Add image. fpdf will calculate height based on width if h is not provided, maintaining aspect ratio
                    pdf.image(img, x=x_pos, w=img_width_mm)

                except Exception as e:
                    # Print error to console where the script running Streamlit is executed
                    print(f"Error adding scatter plot to PDF: {e}. Make sure 'kaleido' is installed (`pip install kaleido`).")
                    # You could potentially add a text message to the PDF instead of the image
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="[Error: Could not render Risk vs Return plot in PDF. Ensure 'kaleido' is installed]", ln=True, align='C')

            # Add Sharpe Ratio Histogram Plot (if available) - Only for Monte Carlo
            if report_data['sharpe_hist_fig'] is not None and report_data['optimization_method'] == 'Monte Carlo':
                pdf.ln(10) # Space before histogram section
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Monte Carlo Simulation: Sharpe Ratio Distribution", ln=True)
                pdf.ln(5)

                try:
                    img_bytes_hist = report_data['sharpe_hist_fig'].to_image(format="png", width=800, height=400) # Adjust size as needed
                    img_hist = io.BytesIO(img_bytes_hist)

                    img_width_mm = 180 # Desired width in mm
                    x_pos = (210 - img_width_mm) / 2
                    img_height_mm_approx = img_width_mm * (400/800) # Approx height based on aspect ratio

                    if pdf.get_y() + img_height_mm_approx > pdf.eph - 10:
                         pdf.add_page()
                         pdf.set_font("Arial", 'B', 14)
                         pdf.cell(200, 10, txt="Monte Carlo Simulation: Sharpe Ratio Distribution (cont.)", ln=True)
                         pdf.ln(5)


                    pdf.image(img_hist, x=x_pos, w=img_width_mm)

                    # Add histogram stats below the plot
                    pdf.ln(5) # Small space below image
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"Sharpe Ratio Distribution Stats:", ln=True)
                    pdf.cell(200, 10, txt=f"  Mean: {report_data['sharpe_mean']:.4f}", ln=True)
                    pdf.cell(200, 10, txt=f"  Standard Deviation: {report_data['sharpe_std']:.4f}", ln=True)


                except Exception as e:
                    print(f"Error adding histogram plot to PDF: {e}. Make sure 'kaleido' is installed (`pip install kaleido`).")
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="[Error: Could not render Sharpe Ratio histogram in PDF. Ensure 'kaleido' is installed]", ln=True, align='C')
                    # Add stats text even if plot fails
                    pdf.cell(200, 10, txt=f"Sharpe Ratio Distribution Stats (Plot Failed):", ln=True)
                    pdf.cell(200, 10, txt=f"  Mean: {report_data['sharpe_mean']:.4f}", ln=True)
                    pdf.cell(200, 10, txt=f"  Standard Deviation: {report_data['sharpe_std']:.4f}", ln=True)
        else:
             # Message if no results calculated
             pdf.set_font("Arial", size=12)
             pdf.cell(200, 10, txt="Portfolio analysis results could not be calculated for the selected options and date range.", ln=True, align='C')



        # Output PDF as bytes
        # This is the correct way to get bytes from fpdf2
        pdf_output_bytes = pdf.output(dest='S')

        # Wrap the bytes output in io.BytesIO
        return io.BytesIO(pdf_output_bytes)


    # --- Portfolio Analysis Page ---
    if page == "Portfolio Analysis":
        st.title("Create Portfolio Scenarios and Risk Analysis")

        col1, col2 = st.columns(2)
        with col1:
            # Filter stock options to only those with valid data
            available_stocks_for_select = daily_returns.columns.tolist() if daily_returns is not None else []
            # Pre-select default stocks only if they are in the available list
            default_selected_stocks = [s for s in stocks[:5] if s in available_stocks_for_select]
            selected_stocks = st.multiselect("Select stocks to add to your portfolio:", available_stocks_for_select, default=default_selected_stocks)

        with col2:
             investment_amount = st.number_input("Enter Investment Amount ($):", min_value=100.0, value=1000.0, step=100.0, format="%.2f")


        col3, col4 = st.columns(2)
        with col3:
            time_period = st.radio("Adjust results for:", ["Daily", "Monthly", "Yearly"])
        with col4:
            optimization_method = st.radio("Optimization Method:", ["SLSQP", "Monte Carlo"])
            num_simulations = None
            if optimization_method == "Monte Carlo":
                 # Adjust max slider value based on performance needs/expectations
                 num_simulations = st.slider("Number of Monte Carlo simulations:", 1000, 200000, 50000, step=1000)

        # --- Display Mathematical Model for SLSQP ---
        if optimization_method == "SLSQP":
             st.subheader("SLSQP Mathematical Model")
             st.write("This optimization method aims to find the portfolio weights that maximize the Sharpe Ratio based on historical data.")

             st.write("**Objective Function:** Minimize the negative Sharpe Ratio.")
             # Attempting another variation: Use 	ext{-} for the minus sign
             st.latex(r"$$ 	ext{Minimize: } - rac{R_p - R_f}{\sigma_p} $$") # Formula for minimizing negative Sharpe

             st.write("Where:")
             st.latex(r" R_p = \sum_{i=1}^{N} w_i R_i ") # Formula for Rp
             st.write(r"($R_p$: Portfolio Expected Return, $R_i$: Historical Average Return of Asset $i$, $w_i$: Weight of Asset $i$)") # Use r-string for dollar signs
             st.latex(r" \sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}} ") # Formula for sigma_p
             st.write(r"($\sigma_p$: Portfolio Volatility, $\mathbf{w}$: Vector of Weights, $\mathbf{\Sigma}$: Covariance Matrix of Historical Returns)") # Use r-string

             st.write("$R_f$: Risk-Free Rate (assumed to be 0 in this implementation)")


             st.write("**Constraints:**")
             st.latex(r" \sum_{i=1}^{N} w_i = 1 ") # Sum of weights
             st.write("(All weights must sum up to 1)")
             st.latex(r" w_i \ge 0 \quad 	ext{for all } i=1, \ldots, N ") # Non-negativity
             st.write("(No short selling allowed)")
             st.write("*(N is the number of selected stocks with available data)*") # Clarify N
             st.markdown("---") # Separator


        # Dictionary to hold data for the PDF report
        # Define report_data structure outside the analysis block
        report_data = {
            'selected_stocks': selected_stocks,
            'start_date_input': start_date_input,
            'end_date_input': end_date_input,
            'investment_amount': investment_amount,
            'time_period': time_period,
            'optimization_method': optimization_method,
            'num_simulations': num_simulations,
            'optimal_weights': None, # Will store daily optimal weights (full array)
            'optimal_return': None, # Will store daily optimal return (from available stocks)
            'optimal_stddev': None, # Will store daily optimal stddev (from available stocks)
            'optimal_sharpe': None, # Will store daily optimal sharpe (from available stocks)
            'adjusted_return': None, # Will store scaled return
            'adjusted_stddev': None, # Will store scaled stddev
            'adjusted_sharpe': None, # Will store scaled sharpe
            'allocation_df': pd.DataFrame(), # Will store allocation DataFrame
            'plotly_fig': None, # Will store Plotly Risk vs Return figure object
            'sharpe_hist_fig': None, # Will store Plotly Sharpe Histogram figure object
            'sharpe_mean': None, # Will store mean of simulated Sharpe ratios
            'sharpe_std': None, # Will store std dev of simulated Sharpe ratios
        }

        # Check if there are enough selected stocks with data for the chosen method
        # Also check if mean_returns is not None before accessing its index
        # This check is now done within optimal_portfolio function more robustly
        # available_selected_stocks = [s for s in selected_stocks if s in mean_returns.index] if (mean_returns is not None) else []


        if not selected_stocks:
             st.info("Please select stocks to analyze.")
        else:
            # Perform optimization
            # The optimal_portfolio function now returns available_stocks list as the 8th item
            optimal_weights, optimal_return, optimal_stddev, optimal_sharpe, ret_arr, vol_arr, sharpe_arr, available_selected_stocks_actual = optimal_portfolio(selected_stocks, method=optimization_method, num_simulations=num_simulations)

            # Check again after calling optimal_portfolio if there were enough valid stocks for the method
            # This handles cases where selected_stocks had items, but none or too few had valid data
            perform_calculation_and_display = True
            if optimization_method == 'SLSQP' and len(available_selected_stocks_actual) < 2:
                 perform_calculation_and_display = False
            elif optimization_method == 'Monte Carlo' and len(available_selected_stocks_actual) < 1:
                 perform_calculation_and_display = False
            # If no valid stocks were returned at all by optimal_portfolio
            elif optimal_weights is None or len(optimal_weights) != len(selected_stocks):
                 perform_calculation_and_display = False
                 st.error("Portfolio optimization could not be performed with the selected stocks and dates.")


            if perform_calculation_and_display:

                # Store raw optimal results in report_data if successful (check if not None)
                # optimal_weights is the full list including zeros for unavailable stocks
                # optimal_return/stddev/sharpe are based on the available stocks only
                report_data['optimal_weights'] = optimal_weights
                report_data['optimal_return'] = optimal_return
                report_data['optimal_stddev'] = optimal_stddev
                report_data['optimal_sharpe'] = optimal_sharpe


                # Calculate allocation for *all* selected_stocks, using the returned weights
                allocation = {selected_stocks[i]: optimal_weights[i] * investment_amount for i in range(len(selected_stocks))}

                # Calculate adjusted metrics for display and report
                period_factors = {"Daily": 1, "Monthly": 21, "Yearly": 252}
                period_factor = period_factors.get(time_period, 1)

                # Check if optimal_return/stddev are not None/NaN before scaling
                adjusted_return = optimal_return * period_factor if (optimal_return is not None and not np.isnan(optimal_return)) else 0.0
                adjusted_stddev = optimal_stddev * np.sqrt(period_factor) if (optimal_stddev is not None and not np.isnan(optimal_stddev)) else 0.0
                # Recalculate adjusted Sharpe using adjusted return and stddev
                # Sharpe Ratio is typically annualized by sqrt(252), but user asked to adjust per period.
                # Let's scale it consistently with volatility: optimal_sharpe (daily) * sqrt(period_factor)
                # Note: A truly 'Monthly Sharpe' or 'Daily Sharpe' calculated this way isn't standard,
                # the standard is *annualized* Sharpe using annualized return/volatility.
                # But for consistency with the user's request to "Adjust results for", we'll scale it.
                adjusted_sharpe = optimal_sharpe * np.sqrt(period_factor) if (optimal_sharpe is not None and not np.isnan(optimal_sharpe)) else 0.0 # Scale daily sharpe


                # Store adjusted metrics in report_data
                report_data['adjusted_return'] = adjusted_return
                report_data['adjusted_stddev'] = adjusted_stddev
                report_data['adjusted_sharpe'] = adjusted_sharpe


                st.header("Optimal Portfolio Analysis Results")
                st.write(f"Optimization Method: **{optimization_method}**")

                st.subheader(f"{time_period} Performance Metrics")
                # Check if metrics are zeroed out due to lack of available data
                if adjusted_return == 0 and adjusted_stddev == 0 and adjusted_sharpe == 0 and len(available_selected_stocks_actual) < (2 if optimization_method == 'SLSQP' else 1):
                     st.info("Results could not be calculated as not enough selected stocks had available data for the analysis period.")
                else:
                    st.write(f"**Portfolio {time_period} Return:** {adjusted_return * 100:.2f}%")
                    st.write(f"**Portfolio {time_period} Volatility (Risk):** {adjusted_stddev * 100:.2f}%")
                    st.write(f"**Portfolio {time_period} Sharpe Ratio:** {adjusted_sharpe:.4f}") # Display scaled sharpe


                st.subheader("Optimal Stock Allocation")
                allocation_data = []
                # Iterate through ALL originally selected stocks
                # Use optimal_weights (full array) for display and allocation calculation
                for i in range(len(selected_stocks)):
                    # Use a small threshold for display, or display all if preferred
                    # Only add stocks with non-zero weight
                    if optimal_weights[i] > 1e-9: # Display stocks with weight > a tiny threshold
                         allocation_data.append({
                             "Stock": selected_stocks[i],
                             "Weight (%)": (optimal_weights[i] * 100), # Keep float for PDF formatting
                             "Investment Amount ($)": (allocation[selected_stocks[i]]) # Keep float for PDF formatting
                         })
                allocation_df = pd.DataFrame(allocation_data)
                report_data['allocation_df'] = allocation_df # Store DataFrame in report_data

                if not allocation_df.empty:
                   # Sort for display
                   allocation_df_display = allocation_df.copy() # Use a copy for display formatting
                   allocation_df_display['Weight (%)'] = allocation_df_display['Weight (%)'].map('{:.2f}'.format)
                   allocation_df_display['Investment Amount ($)'] = allocation_df_display['Investment Amount ($)'].map('{:,.2f}'.format) # Add comma for thousands
                   # Sort by Investment Amount or Weight
                   allocation_df_display = allocation_df_display.sort_values(by="Weight (%)", ascending=False)

                   st.dataframe(allocation_df_display, use_container_width=True)
                else:
                   # This case is covered if metrics are zeroed out message is shown
                   # or if weights are zero but enough stocks existed for optimization
                   if not (adjusted_return == 0 and adjusted_stddev == 0 and adjusted_sharpe == 0):
                       st.info("Optimal allocation resulted in near-zero weights for all selected stocks.")


                # --- Monte Carlo Plotting ---
                # Plot if Monte Carlo was selected AND we have simulation results
                # Ensure sharpe_arr is not None and has length > 0 for plotting
                if optimization_method == 'Monte Carlo' and sharpe_arr is not None and len(sharpe_arr) > 0:
                    st.subheader("Monte Carlo Simulation Results")

                    # Create DataFrame for scatter plot (Risk vs Return)
                    mc_data_scatter = pd.DataFrame({
                        'Volatility (Risk)': vol_arr * np.sqrt(period_factor), # Scaled volatility
                        'Return': ret_arr * period_factor, # Scaled return
                        'Sharpe Ratio': sharpe_arr # Daily Sharpe (used for coloring)
                    })

                    st.write("#### Portfolio Risk vs Return")

                    # Find the scaled optimal point for plotting (calculated based on original data, then scaled)
                    # Ensure optimal_stddev and optimal_return are not None/NaN before scaling for plotting
                    scaled_optimal_stddev_plot = optimal_stddev * np.sqrt(period_factor) if (optimal_stddev is not None and not np.isnan(optimal_stddev)) else None
                    scaled_optimal_return_plot = optimal_return * period_factor if (optimal_return is not None and not np.isnan(optimal_return)) else None


                    fig_scatter = px.scatter(mc_data_scatter,
                                     x='Volatility (Risk)',
                                     y='Return',
                                     color='Sharpe Ratio', # Color by daily Sharpe
                                     color_continuous_scale=px.colors.sequential.Viridis, # Optional: change color scale
                                     hover_data={'Volatility (Risk)': ':.4f', 'Return': ':.4f', 'Sharpe Ratio': ':.4f'},
                                     title=f'Monte Carlo Simulation: Portfolio Risk vs Return ({time_period} Basis)')

                    # Add the optimal portfolio point (Max Sharpe) if calculated successfully
                    if scaled_optimal_stddev_plot is not None and scaled_optimal_return_plot is not None and optimal_sharpe is not None:
                        fig_scatter.add_trace(go.Scatter(
                            x=[scaled_optimal_stddev_plot],
                            y=[scaled_optimal_return_plot],
                            mode='markers',
                            marker=dict(color='red', size=12, symbol='star', line=dict(width=1, color='DarkSlateGrey')),
                            name=f'Max Sharpe ({optimal_sharpe:.4f})', # Show daily sharpe in name
                            hoverinfo='name+x+y'
                        ))


                    fig_scatter.update_layout(
                        xaxis_title=f'{time_period} Volatility (Risk)', # Percentage formatting handled by tickformat
                        yaxis_title=f'{time_period} Return', # Percentage formatting handled by tickformat
                        hovermode='closest'
                    )

                    fig_scatter.update_xaxes(tickformat=".2%") # Format x-axis as percentage
                    fig_scatter.update_yaxes(tickformat=".2%") # Format y-axis as percentage

                    st.plotly_chart(fig_scatter, use_container_width=True)
                    report_data['plotly_fig'] = fig_scatter # Store scatter figure object in report_data for PDF

                    # --- Sharpe Ratio Histogram ---
                    st.write("#### Sharpe Ratio Distribution")

                    # Calculate mean and standard deviation of the simulated Sharpe Ratios
                    sharpe_mean = np.mean(sharpe_arr) if len(sharpe_arr) > 0 else 0
                    sharpe_std = np.std(sharpe_arr) if len(sharpe_arr) > 0 else 0

                    report_data['sharpe_mean'] = sharpe_mean
                    report_data['sharpe_std'] = sharpe_std


                    st.write(f"Mean of Simulated Sharpe Ratios: **{sharpe_mean:.4f}**")
                    st.write(f"Standard Deviation of Simulated Sharpe Ratios: **{sharpe_std:.4f}**")


                    # Create histogram of Sharpe Ratios
                    # Use the daily sharpe_arr for the histogram
                    fig_hist = px.histogram(sharpe_arr, nbins=50, title='Distribution of Simulated Sharpe Ratios') # Adjust nbins as needed
                    fig_hist.update_layout(xaxis_title="Sharpe Ratio (Daily)", yaxis_title="Frequency")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    report_data['sharpe_hist_fig'] = fig_hist # Store histogram figure object for PDF

                elif optimization_method == 'Monte Carlo':
                    # Show this message if MC was selected but plotting failed (e.g., len(sharpe_arr) is 0 after filtering)
                    st.warning("Monte Carlo simulation results could not be generated for plotting or no valid portfolios found with non-zero standard deviation and finite Sharpe ratio.")


                # --- PDF Download Button ---
                # Only show download button if analysis was successful AND we have allocation data (implies some result was obtained)
                # Checking allocation_df.empty is a good proxy for "meaningful result"
                if not allocation_df.empty:
                    st.markdown("---") # Separator

                    # Generate the PDF content using the populated report_data dictionary
                    # Pass the report_data dictionary to the function
                    pdf_output_io = create_portfolio_report(report_data)

                    # Provide the PDF as a download button
                    # Use a dynamic filename based on date/time
                    file_name = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                         label="Download Portfolio Report (PDF)",
                         data=pdf_output_io, # Pass the io.BytesIO object
                         file_name=file_name,
                         mime="application/pdf"
                    )
                else:
                     # This else corresponds to the 'if not allocation_df.empty:' check for the download button
                     # A message about no significant allocation is already shown above.
                     pass


            # else:
                 # This block is reached if perform_calculation_and_display was False
                 # The specific warning/error message is shown above where perform_calculation_and_display is set
                 # pass # No need for another else block here


    # --- Stock Data Page ---
    elif page == "Stock Data":
        st.title("Daily Stock Data and Price Prediction")

        # Filter stock options to only those with valid data
        available_stocks_for_select_stock = stock_data.columns.tolist() if stock_data is not None else []

        if not available_stocks_for_select_stock:
             st.warning("No stock data available for the selected date range to display.")
        else:
            # Pre-select the first available stock
            selected_stock = st.selectbox("Select the stock you want to view:", available_stocks_for_select_stock)

            # Check if selected_stock data is available in the loaded data (redundant if selectbox is filtered, but safe)
            if selected_stock in stock_data.columns:
                st.subheader(f"Performance Analysis for {selected_stock}")

                # Use daily returns for daily stats - check if stock is in returns index/cov matrix
                if selected_stock in mean_returns.index and selected_stock in cov_matrix.columns:
                     avg_daily_return = mean_returns[selected_stock] * 100
                     daily_std_dev = np.sqrt(cov_matrix.loc[selected_stock, selected_stock]) * 100
                     st.write(f"**Average Daily Return:** {avg_daily_return:.2f}%")
                     st.write(f"**Daily Volatility (Risk):** {daily_std_dev:.2f}%")
                     # Daily Sharpe Ratio (assuming Rf=0)
                     daily_sharpe = avg_daily_return / daily_std_dev if daily_std_dev > 1e-9 else 0
                     st.write(f"**Daily Sharpe Ratio:** {daily_sharpe:.4f}") # Display daily sharpe

                else:
                     st.warning(f"Could not calculate performance metrics for {selected_stock} with the selected date range (data might be incomplete or missing for return calculation).")


                st.subheader(f"Historical Price Data for {selected_stock}")
                fig_hist = px.line(stock_data, y=selected_stock, title=f'{selected_stock} Closing Price History ({stock_data.index[0].date()} to {stock_data.index[-1].date()})')
                fig_hist.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig_hist, use_container_width=True)


                st.subheader(f"Linear Regression Price Prediction for {selected_stock}")
                prediction_days = st.slider("Number of days to predict:", 1, 365, 30)

                # Check if there is enough data for prediction (need at least 2 non-NaN points)
                stock_data_for_pred = stock_data[selected_stock].dropna()
                if len(stock_data_for_pred) >= 2:
                    predicted_prices, future_dates, intercept, slope, model = predict_stock_price(stock_data_for_pred, future_days=prediction_days)

                    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})

                    st.write(f"Predicted stock prices for {selected_stock} for the next {prediction_days} days:")

                    # Combine historical and predicted data for a single plot
                    historical_plot_data = stock_data_for_pred.reset_index()
                    historical_plot_data.columns = ['Date', 'Price']
                    historical_plot_data['Type'] = 'Historical'

                    prediction_plot_data = prediction_df.copy()
                    prediction_plot_data.columns = ['Date', 'Price']
                    prediction_plot_data['Type'] = 'Predicted'

                    combined_plot_data = pd.concat([historical_plot_data, prediction_plot_data])
                    combined_plot_data['Date'] = pd.to_datetime(combined_plot_data['Date'])
                    combined_plot_data = combined_plot_data.sort_values('Date')

                    fig_pred = px.line(combined_plot_data, x='Date', y='Price', color='Type',
                                       title=f'{selected_stock} Price: Historical vs. Predicted')
                    fig_pred.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
                    st.plotly_chart(fig_pred, use_container_width=True)

                    st.subheader("Regression Equation")
                    regression_equation_desc = f"**y = {slope:.4f}x + {intercept:.2f}**"
                    st.write(f"Using a simple linear regression model where 'x' represents the number of days since the start of the historical data period ({stock_data_for_pred.index[0].date()}), the price ('y') is modeled as: {regression_equation_desc}")
                    st.write("*(Note: This is a very basic prediction model based on past trends and may not be accurate for forecasting future stock prices.)*")

                    st.subheader("Prediction Table")
                    prediction_df_display = prediction_df.copy()
                    prediction_df_display['Predicted Price'] = prediction_df_display['Predicted Price'].map('{:.2f}'.format)
                    prediction_df_display['Date'] = prediction_df_display['Date'].dt.date
                    st.dataframe(prediction_df_display, use_container_width=True)

                else:
                    st.warning(f"Not enough valid historical data available for {selected_stock} to perform linear regression prediction ({len(stock_data_for_pred)} data points). Need at least 2.")

            # Else for selected_stock not in stock_data.columns - handled by selectbox filtering now
            # pass

# This block is reached if the initial data load failed completely before any page logic
elif stock_data is None or daily_returns is None:
     # Use st.info here as st.error might persist across reruns depending on cause
     st.info("Please select a valid date range and stocks to load data.")


