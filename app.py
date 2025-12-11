# trading_journal_app.py
# Updated for new Excel CSV format
# Example format:
# Symbol,Opening Direction,Closing Time (UTC-6),Entry price,Closing Price,Closing Quantity,Net USD,Balance USD

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import re

# Page configuration
st.set_page_config(
    page_title="Trading Journal Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header
    st.title("📊 Trading Journal Analyzer")
    st.markdown("""
    Upload your trading history CSV file to analyze your performance.
    *Now compatible with Excel CSV exports with columns: Symbol, Opening Direction, Closing Time, Entry price, Closing Price, Closing Quantity, Net USD, Balance USD*
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Process data
            df, success = process_data(uploaded_file)
            
            if success:
                display_analysis(df)
            else:
                st.error("Please check your CSV format. Required columns: Symbol, Opening Direction, Net USD")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample format expectations
        st.info("""
        **Expected CSV Format (Excel export):**
        - Symbol, Opening Direction, Closing Time (UTC-6), Entry price, Closing Price, Closing Quantity, Net USD, Balance USD
        - Example: `EURGBP,Sell,45:51.8,0.87459,0.87619,0.09 Lots,-19.32,1,795.23`
        """)

def process_data(uploaded_file):
    """Process and clean the uploaded CSV data for new format"""
    try:
        # Read CSV - handle different encodings
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        # Clean column names (remove extra spaces, special characters)
        df.columns = df.columns.str.strip()
        
        # Check if this is the new format
        expected_cols = ['Symbol', 'Opening Direction', 'Net USD']
        new_format = all(col in df.columns for col in expected_cols)
        
        if not new_format:
            # Try with alternative column names
            col_mapping = {
                'Symbol': ['Symbol', 'Instrument', 'Ticker'],
                'Opening Direction': ['Opening Direction', 'Action', 'Side'],
                'Net USD': ['Net USD', 'Net P&L', 'NetPL', 'Profit/Loss']
            }
            
            for standard_name, possible_names in col_mapping.items():
                for name in possible_names:
                    if name in df.columns:
                        df = df.rename(columns={name: standard_name})
                        break
        
        # Check again after renaming
        if not all(col in df.columns for col in expected_cols):
            return None, False
        
        # Data cleaning for new format
        df = df.copy()
        
        # Parse Closing Time to create Date column
        # First, check if there's a date column, otherwise use current date
        if 'Date' not in df.columns and 'Closing Time' not in df.columns:
            # If no date/time columns, add a placeholder
            df['Date'] = datetime.now()
        else:
            try:
                # Try to parse date if available
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                elif 'Closing Time' in df.columns:
                    # Use closing time with today's date
                    df['Date'] = pd.to_datetime(df['Closing Time'].astype(str), errors='coerce')
                    # Fill NaT with current datetime
                    df['Date'] = df['Date'].fillna(pd.Timestamp.now())
            except:
                df['Date'] = pd.Timestamp.now()
        
        # Clean numeric columns
        def clean_numeric(value):
            if pd.isna(value):
                return 0.0
            if isinstance(value, str):
                # Remove currency symbols, commas, and the Â character
                value = str(value).replace('$', '').replace(',', '').replace('Â', '')
                value = value.strip()
                
                # Handle negative numbers in parentheses
                if '(' in value and ')' in value:
                    value = '-' + value.replace('(', '').replace(')', '')
                
                # Remove any remaining non-numeric characters except minus and period
                value = re.sub(r'[^\d.-]', '', value)
                
                try:
                    return float(value)
                except:
                    return 0.0
            return float(value)
        
        # Clean Closing Quantity column (remove "Lots" text)
        if 'Closing Quantity' in df.columns:
            df['Quantity'] = df['Closing Quantity'].apply(
                lambda x: float(str(x).replace('Lots', '').replace('Â', '').strip()) 
                if pd.notna(x) else 0.0
            )
        
        # Clean P&L columns
        pnl_columns = ['Net USD', 'Balance USD']
        for col in pnl_columns:
            if col in df.columns:
                df[f'{col}_Clean'] = df[col].apply(clean_numeric)
        
        # Map Opening Direction to standard Action terms
        if 'Opening Direction' in df.columns:
            df['Action'] = df['Opening Direction'].map({
                'Buy': 'Buy',
                'Sell': 'Sell',
                'BUY': 'Buy',
                'SELL': 'Sell'
            }).fillna(df['Opening Direction'])
        
        # Use Symbol as Instrument
        if 'Symbol' in df.columns:
            df['Instrument'] = df['Symbol']
            df['Instrument_Base'] = df['Symbol']
        
        # Calculate cumulative P&L
        if 'Net USD_Clean' in df.columns:
            df['Cumulative_NetPL'] = df['Net USD_Clean'].cumsum()
        
        # Add trade result
        if 'Net USD_Clean' in df.columns:
            df['Result'] = df['Net USD_Clean'].apply(
                lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'BreakEven'
            )
        
        # Sort by date
        if 'Date' in df.columns:
            df = df.sort_values('Date')
        
        # Add useful calculated columns
        if all(col in df.columns for col in ['Entry price', 'Closing Price']):
            df['Price_Change'] = df.apply(
                lambda row: row['Closing Price'] - row['Entry price'] 
                if pd.notna(row['Entry price']) and pd.notna(row['Closing Price']) else 0,
                axis=1
            )
        
        return df, True
        
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None, False

def display_analysis(df):
    """Display all analysis components"""
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Instrument filter
    if 'Instrument_Base' in df.columns:
        instruments = ['All'] + sorted(df['Instrument_Base'].unique().tolist())
        selected_instrument = st.sidebar.selectbox("Filter by Instrument:", instruments)
    else:
        selected_instrument = 'All'
    
    # Date range filter
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        date_range = st.sidebar.date_input(
            "Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = None
    
    # Apply filters
    filtered_df = df.copy()
    if selected_instrument != 'All' and 'Instrument_Base' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Instrument_Base'] == selected_instrument]
    
    if date_range and len(date_range) == 2 and 'Date' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= start_date) & 
            (filtered_df['Date'].dt.date <= end_date)
        ]
    
    # Main dashboard
    st.header("📈 Performance Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if 'Net USD_Clean' in filtered_df.columns:
        total_net_pl = filtered_df['Net USD_Clean'].sum()
        total_trades = len(filtered_df)
        winning_trades = len(filtered_df[filtered_df['Net USD_Clean'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade = filtered_df['Net USD_Clean'].mean()
        
        with col1:
            st.metric("Total Net P&L", f"${total_net_pl:,.2f}")
        with col2:
            st.metric("Total Trades", total_trades)
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("Avg. Trade P&L", f"${avg_trade:.2f}")
    else:
        st.warning("Net USD column not found in data")
    
    # Equity Curve
    if 'Cumulative_NetPL' in filtered_df.columns and 'Date' in filtered_df.columns:
        st.subheader("Equity Curve")
        fig_equity = px.line(
            filtered_df, 
            x='Date', 
            y='Cumulative_NetPL',
            title="Cumulative P&L Over Time"
        )
        fig_equity.update_layout(height=400)
        st.plotly_chart(fig_equity, use_container_width=True)
    
    # P&L Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Net USD_Clean' in filtered_df.columns:
            st.subheader("P&L Distribution")
            fig_hist = px.histogram(
                filtered_df,
                x='Net USD_Clean',
                nbins=20,
                title="Distribution of Trade P&L"
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        if 'Instrument_Base' in filtered_df.columns and 'Net USD_Clean' in filtered_df.columns:
            st.subheader("Performance by Instrument")
            inst_pl = filtered_df.groupby('Instrument_Base')['Net USD_Clean'].sum().sort_values()
            fig_bar = px.bar(
                inst_pl,
                orientation='h',
                title="Total P&L by Instrument"
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Trade Details
    st.subheader("Trade History")
    
    # Summary statistics
    if 'Net USD_Clean' in filtered_df.columns:
        st.write("**Trade Summary:**")
        win_df = filtered_df[filtered_df['Net USD_Clean'] > 0]
        loss_df = filtered_df[filtered_df['Net USD_Clean'] < 0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Winning Trades", len(win_df))
        with col2:
            st.metric("Losing Trades", len(loss_df))
        with col3:
            avg_win = win_df['Net USD_Clean'].mean() if len(win_df) > 0 else 0
            st.metric("Avg. Win", f"${avg_win:.2f}")
        with col4:
            avg_loss = loss_df['Net USD_Clean'].mean() if len(loss_df) > 0 else 0
            st.metric("Avg. Loss", f"${avg_loss:.2f}")
    
    # Raw data table
    st.subheader("Raw Trade Data")
    
    # Determine which columns to display
    display_cols = []
    possible_cols = [
        'Date', 'Symbol', 'Instrument', 'Action', 'Opening Direction',
        'Entry price', 'Closing Price', 'Quantity', 'Closing Quantity',
        'Net USD', 'Net USD_Clean', 'Balance USD', 'Result'
    ]
    
    for col in possible_cols:
        if col in filtered_df.columns:
            display_cols.append(col)
    
    if display_cols:
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True
        )
    else:
        st.write("No displayable columns found in the data")

if __name__ == "__main__":
    main()
