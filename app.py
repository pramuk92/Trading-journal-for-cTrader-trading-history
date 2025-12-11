# trading_journal_app.py
# Updated for new Excel CSV format with better CSV parsing

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
            st.info("""
            **Common issues:**
            1. The CSV may have commas within numbers (e.g., 1,795.23)
            2. There may be extra header rows
            3. Try saving your Excel file as CSV UTF-8 format
            """)
    
    else:
        # Show sample format expectations
        st.info("""
        **Expected CSV Format (Excel export):**
        - Symbol, Opening Direction, Closing Time (UTC-6), Entry price, Closing Price, Closing Quantity, Net USD, Balance USD
        - Example: `EURGBP,Sell,45:51.8,0.87459,0.87619,0.09 Lots,-19.32,1795.23`
        """)

def read_csv_with_flexibility(uploaded_file):
    """Read CSV file with multiple attempts for different formats"""
    attempts = [
        # Try reading with proper quoting and European decimal format
        lambda: pd.read_csv(uploaded_file, encoding='utf-8', quotechar='"', 
                           decimal=',', thousands='.', engine='python'),
        lambda: pd.read_csv(uploaded_file, encoding='utf-8', quotechar='"', 
                           decimal='.', thousands=',', engine='python'),
        # Try without quoting
        lambda: pd.read_csv(uploaded_file, encoding='utf-8', engine='python'),
        # Try with latin-1 encoding
        lambda: pd.read_csv(uploaded_file, encoding='latin-1', engine='python'),
        # Try reading the entire file and manually parsing
        lambda: manual_csv_parse(uploaded_file),
    ]
    
    for i, attempt in enumerate(attempts):
        try:
            uploaded_file.seek(0)  # Reset file pointer
            df = attempt()
            if not df.empty and len(df.columns) > 1:
                st.success(f"CSV read successfully (method {i+1})")
                return df
        except Exception as e:
            continue
    
    # If all attempts fail, raise error
    raise ValueError("Could not read CSV file. Please check the format.")

def manual_csv_parse(uploaded_file):
    """Manually parse CSV to handle irregular formats"""
    uploaded_file.seek(0)
    content = uploaded_file.read().decode('utf-8', errors='ignore')
    
    # Split into lines
    lines = content.strip().split('\n')
    
    # Find header (look for Symbol or similar)
    header_line = None
    data_start = 0
    
    for i, line in enumerate(lines):
        if 'Symbol' in line and ('Opening' in line or 'Direction' in line):
            header_line = i
            data_start = i + 1
            break
    
    if header_line is not None:
        # Get header
        header = lines[header_line].strip().split(',')
        header = [h.strip() for h in header]
        
        # Parse data rows
        data = []
        for line in lines[data_start:]:
            if line.strip() and not line.strip().startswith('Deals'):
                # Handle commas in numbers by using regex split
                # Split by commas not preceded by a digit and followed by a digit
                parts = re.split(r',(?=\s*[^0-9.,])', line.strip())
                if len(parts) >= len(header):
                    data.append(parts[:len(header)])
                elif len(parts) == len(header) - 1:
                    # Handle case where last column might be merged
                    data.append(parts + [''])
                else:
                    # Skip malformed lines
                    continue
        
        df = pd.DataFrame(data, columns=header)
        return df
    
    return pd.DataFrame()

def process_data(uploaded_file):
    """Process and clean the uploaded CSV data for new format"""
    try:
        # Read CSV with flexible parsing
        df = read_csv_with_flexibility(uploaded_file)
        
        if df.empty:
            return None, False
        
        # Clean column names
        df.columns = [str(col).strip().replace('\ufeff', '') for col in df.columns]
        
        # Show columns for debugging
        st.write(f"Columns found: {list(df.columns)}")
        st.write(f"First few rows:", df.head())
        
        # Map column names to expected names
        column_mapping = {}
        
        # Try to identify columns
        for col in df.columns:
            col_lower = str(col).lower()
            
            if any(x in col_lower for x in ['symbol', 'instrument', 'pair']):
                column_mapping[col] = 'Symbol'
            elif any(x in col_lower for x in ['opening', 'direction', 'side', 'action']):
                column_mapping[col] = 'Action'
            elif any(x in col_lower for x in ['closing time', 'time', 'date']):
                column_mapping[col] = 'Time'
            elif any(x in col_lower for x in ['entry', 'open']):
                column_mapping[col] = 'Entry'
            elif any(x in col_lower for x in ['closing', 'close']):
                column_mapping[col] = 'Close'
            elif any(x in col_lower for x in ['quantity', 'size', 'lot']):
                column_mapping[col] = 'Quantity'
            elif any(x in col_lower for x in ['net', 'pnl', 'profit']):
                column_mapping[col] = 'Net'
            elif any(x in col_lower for x in ['balance']):
                column_mapping[col] = 'Balance'
        
        # Apply column mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Show mapped columns
        st.write(f"Mapped columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['Symbol', 'Action', 'Net']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}")
            st.write("Available columns:", list(df.columns))
            
            # Try to use what we have
            if 'Symbol' not in df.columns and len(df.columns) > 0:
                df['Symbol'] = df.iloc[:, 0]  # Use first column as Symbol
            
            if 'Net' not in df.columns:
                # Look for any numeric column that could be P&L
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    df['Net'] = df[numeric_cols[-1]]
                else:
                    return None, False
        
        # Data cleaning
        df = df.copy()
        
        # Clean the data
        def clean_value(value):
            if pd.isna(value):
                return ''
            value = str(value)
            # Remove unwanted characters
            value = value.replace('Â', '').replace('€', '').replace('$', '').replace('£', '')
            value = value.replace('(', '-').replace(')', '')  # Handle negative parentheses
            value = re.sub(r'\s+', ' ', value.strip())  # Normalize whitespace
            return value
        
        # Apply cleaning to all columns
        for col in df.columns:
            df[col] = df[col].apply(clean_value)
        
        # Create Date column (use today's date if no date available)
        if 'Time' in df.columns:
            try:
                # Try to parse time
                df['Date'] = pd.to_datetime(df['Time'], errors='coerce')
            except:
                df['Date'] = pd.Timestamp.now()
        else:
            df['Date'] = pd.Timestamp.now()
        
        # Fill NaT dates
        df['Date'] = df['Date'].fillna(pd.Timestamp.now())
        
        # Parse numeric columns
        def parse_numeric(value):
            try:
                if pd.isna(value) or value == '':
                    return 0.0
                
                value = str(value)
                # Remove thousands separators and clean up
                value = value.replace(',', '').replace(' ', '')
                
                # Handle negative numbers
                if value.startswith('-'):
                    return -float(value[1:]) if value[1:] else 0.0
                
                return float(value)
            except:
                return 0.0
        
        # Parse quantity (handle "Lots" suffix)
        if 'Quantity' in df.columns:
            df['Quantity_Clean'] = df['Quantity'].apply(
                lambda x: parse_numeric(str(x).replace('Lots', '').replace('lots', ''))
            )
        
        # Parse Net P&L
        if 'Net' in df.columns:
            df['Net_Clean'] = df['Net'].apply(parse_numeric)
        
        # Parse Balance
        if 'Balance' in df.columns:
            df['Balance_Clean'] = df['Balance'].apply(parse_numeric)
        
        # Parse Entry and Close prices
        for price_col in ['Entry', 'Close']:
            if price_col in df.columns:
                df[f'{price_col}_Clean'] = df[price_col].apply(parse_numeric)
        
        # Map Action to standard terms
        if 'Action' in df.columns:
            df['Action'] = df['Action'].apply(
                lambda x: 'Buy' if str(x).lower() in ['buy', 'b', 'long'] 
                else 'Sell' if str(x).lower() in ['sell', 's', 'short'] 
                else str(x)
            )
        
        # Set Instrument columns
        if 'Symbol' in df.columns:
            df['Instrument'] = df['Symbol']
            df['Instrument_Base'] = df['Symbol']
        
        # Calculate cumulative P&L
        if 'Net_Clean' in df.columns:
            df['Cumulative_NetPL'] = df['Net_Clean'].cumsum()
        
        # Add trade result
        if 'Net_Clean' in df.columns:
            df['Result'] = df['Net_Clean'].apply(
                lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'BreakEven'
            )
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Clean up the dataframe
        # Keep only essential columns
        keep_cols = []
        essential_cols = [
            'Date', 'Symbol', 'Instrument', 'Instrument_Base', 'Action',
            'Entry_Clean', 'Close_Clean', 'Quantity_Clean',
            'Net_Clean', 'Balance_Clean', 'Cumulative_NetPL', 'Result'
        ]
        
        for col in essential_cols:
            if col in df.columns:
                keep_cols.append(col)
        
        # Add any original columns that might be useful
        original_cols_to_keep = ['Time', 'Entry', 'Close', 'Quantity', 'Net', 'Balance']
        for col in original_cols_to_keep:
            if col in df.columns and col not in keep_cols:
                keep_cols.append(col)
        
        df = df[keep_cols]
        
        return df, True
        
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
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
    
    # Show data summary
    st.write(f"**Total trades loaded:** {len(filtered_df)}")
    
    # Main dashboard
    st.header("📈 Performance Dashboard")
    
    # Key Metrics
    if 'Net_Clean' in filtered_df.columns:
        col1, col2, col3, col4 = st.columns(4)
        
        total_net_pl = filtered_df['Net_Clean'].sum()
        total_trades = len(filtered_df)
        winning_trades = len(filtered_df[filtered_df['Net_Clean'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade = filtered_df['Net_Clean'].mean()
        
        with col1:
            st.metric("Total Net P&L", f"${total_net_pl:,.2f}")
        with col2:
            st.metric("Total Trades", total_trades)
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("Avg. Trade P&L", f"${avg_trade:.2f}")
    else:
        st.warning("Net P&L data not available")
    
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
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Net_Clean' in filtered_df.columns:
            st.subheader("P&L Distribution")
            fig_hist = px.histogram(
                filtered_df,
                x='Net_Clean',
                nbins=20,
                title="Distribution of Trade P&L"
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        if 'Instrument_Base' in filtered_df.columns and 'Net_Clean' in filtered_df.columns:
            st.subheader("Performance by Instrument")
            inst_pl = filtered_df.groupby('Instrument_Base')['Net_Clean'].sum().sort_values()
            fig_bar = px.bar(
                inst_pl,
                orientation='h',
                title="Total P&L by Instrument"
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Trade Details
    st.subheader("Trade Details")
    
    if 'Net_Clean' in filtered_df.columns:
        st.write("**Trade Summary:**")
        win_df = filtered_df[filtered_df['Net_Clean'] > 0]
        loss_df = filtered_df[filtered_df['Net_Clean'] < 0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Winning Trades", len(win_df))
        with col2:
            st.metric("Losing Trades", len(loss_df))
        with col3:
            avg_win = win_df['Net_Clean'].mean() if len(win_df) > 0 else 0
            st.metric("Avg. Win", f"${avg_win:.2f}")
        with col4:
            avg_loss = loss_df['Net_Clean'].mean() if len(loss_df) > 0 else 0
            st.metric("Avg. Loss", f"${avg_loss:.2f}")
    
    # Raw data table
    st.subheader("Trade History")
    
    # Prepare display dataframe
    display_df = filtered_df.copy()
    
    # Rename columns for display
    column_display_names = {
        'Date': 'Date',
        'Symbol': 'Symbol',
        'Instrument': 'Instrument',
        'Action': 'Action',
        'Entry_Clean': 'Entry Price',
        'Close_Clean': 'Close Price',
        'Quantity_Clean': 'Quantity',
        'Net_Clean': 'Net P&L',
        'Balance_Clean': 'Balance',
        'Cumulative_NetPL': 'Cumulative P&L',
        'Result': 'Result'
    }
    
    # Select and rename columns
    display_cols = []
    for col, display_name in column_display_names.items():
        if col in display_df.columns:
            display_cols.append(col)
            # Rename in the dataframe
            display_df = display_df.rename(columns={col: display_name})
    
    # Show the data
    if display_cols:
        st.dataframe(
            display_df[[column_display_names[col] for col in display_cols]],
            use_container_width=True,
            height=400
        )
        
        # Add download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name="processed_trades.csv",
            mime="text/csv"
        )
    else:
        st.write("No data available for display")

if __name__ == "__main__":
    main()
