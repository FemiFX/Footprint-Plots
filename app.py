import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import mysql.connector

# Initialize the Dash app once
app = dash.Dash(__name__)
server = app.server  # Required for running with Gunicorn

# Wrap the graph in a div with a class for better control
app.layout = html.Div(className="app-container", children=[
    html.Div(className="graph-container", children=[
        dcc.Graph(id='live-graph', config={'scrollZoom': True})
    ]),
    dcc.Interval(id='graph-update', interval=10000),
    dcc.Store(id='zoom-store')  # Used to store zoom settings
])

# Database connection setup
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="footprint",
        password="***",
        database="***",
        connection_timeout=300
    )

# Fetch candlestick data
def fetch_candlestick_data(symbol):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        query = """
        SELECT id, open_time, open, high, low, close, volume 
        FROM candlesticks_15m WHERE symbol = %s ORDER BY open_time DESC LIMIT 50
        """
        cursor.execute(query, (symbol,))
        rows = cursor.fetchall()
        
        columns = ['CandleID', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(rows, columns=columns)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.iloc[::-1]  # Reverse the DataFrame to show the most recent candles last
        
        cursor.close()
        db_conn.close()
        
        return df
    except mysql.connector.errors.OperationalError as e:
        print(f"Database error: {e}")
        return pd.DataFrame()

# Fetch footprint data
def fetch_footprint_data(symbol, candle_id):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        query = """
        SELECT price_range_start, price_range_end, buy_volume, sell_volume
        FROM footprints_15m
        WHERE symbol = %s AND candle_id = %s
        """
        cursor.execute(query, (symbol, int(candle_id)))
        result = cursor.fetchall()
        
        cursor.close()
        db_conn.close()
        
        return result
    except Exception as e:
        print(f"Error fetching footprint data: {e}")
        return []

# Overlay footprint data and calculate delta and total volume for each candle
def process_footprint_data(symbol, df):
    price_volume_dict = {}
    footprint_info = []

    for idx, (timestamp, row) in enumerate(df.iterrows()):
        candle_id = row['CandleID']
        footprint_data = fetch_footprint_data(symbol, candle_id)

        for footprint in footprint_data:
            price_start = float(footprint[0])
            price_end = float(footprint[1])
            buy_volume = float(footprint[2])
            sell_volume = float(footprint[3])
            total_volume = buy_volume + sell_volume
            midpoint = (price_start + price_end) / 2  # Calculate midpoint

            if total_volume > 0:
                imbalance = (buy_volume - sell_volume) / total_volume
            else:
                imbalance = 0  # Neutral if there's no volume

            footprint_info.append({
                "price_start": price_start,
                "price_end": price_end,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "imbalance": imbalance,
                "midpoint": midpoint,  # Include midpoint here
                "candle_idx": idx
            })

            if midpoint not in price_volume_dict:
                price_volume_dict[midpoint] = total_volume
            else:
                price_volume_dict[midpoint] += total_volume

    return price_volume_dict, footprint_info


# Identify high-volume price levels based on a volume threshold or top percentage
def detect_high_volume_levels(price_volume_dict, volume_threshold):
    high_volume_levels = {}

    for price, volume in price_volume_dict.items():
        if volume >= volume_threshold:
            high_volume_levels[price] = volume

    return high_volume_levels

# Detect high buy and sell volumes
def detect_high_buy_sell_levels(footprint_info, buy_threshold, sell_threshold):
    high_buy_levels = {}
    high_sell_levels = {}

    for footprint in footprint_info:
        midpoint = footprint["midpoint"]
        buy_volume = footprint["buy_volume"]
        sell_volume = footprint["sell_volume"]

        if buy_volume >= buy_threshold:
            high_buy_levels[midpoint] = buy_volume

        if sell_volume >= sell_threshold:
            high_sell_levels[midpoint] = sell_volume

    return high_buy_levels, high_sell_levels

# Define the six shades for blue (bullish) and red (bearish)
blue_shades = ['#0C008F', '#1400F5', '#5747FF', '#7C70FF', '#B0B3FF', '#D9DBFF']
red_shades = ['#42101A', '#731C2D', '#A42841', '#CF3A58', '#D75B74', '#F0A6B5']

# Function to get the color based on the imbalance
def get_color_for_imbalance(imbalance):
    # For bullish imbalance (positive), use blue shades
    if imbalance > 0:
        if imbalance >= 0.8:
            return blue_shades[0]  # Deepest blue for high positive imbalance
        elif imbalance >= 0.6:
            return blue_shades[1]
        elif imbalance >= 0.4:
            return blue_shades[2]
        elif imbalance >= 0.2:
            return blue_shades[3]
        elif imbalance >= 0.1:
            return blue_shades[4]
        else:
            return blue_shades[5]  # Lightest blue for lowest positive imbalance
    
    # For bearish imbalance (negative), use red shades
    elif imbalance < 0:
        if imbalance <= -0.8:
            return red_shades[0]  # Deepest red for high negative imbalance
        elif imbalance <= -0.6:
            return red_shades[1]
        elif imbalance <= -0.4:
            return red_shades[2]
        elif imbalance <= -0.2:
            return red_shades[3]
        elif imbalance <= -0.1:
            return red_shades[4]
        else:
            return red_shades[5]  # Lightest red for lowest negative imbalance
    
    # For neutral imbalance (zero)
    return 'rgba(128, 128, 128, 0.5)'  # Gray for neutral

@app.callback(
    [Output('live-graph', 'figure'),
     Output('zoom-store', 'data')],
    [Input('graph-update', 'n_intervals')],
    [State('zoom-store', 'data'),
     State('live-graph', 'relayoutData')]  # Track zoom/scroll changes
)
def update_graph(n, zoom_data, relayout_data):
    symbol = "btcusdt"
    df = fetch_candlestick_data(symbol)

    # Process footprint data and accumulate volumes for price levels
    price_volume_dict, footprint_info = process_footprint_data(symbol, df)
    volume_threshold = 1500  # Set a threshold for high volume detection
    high_volume_levels = detect_high_volume_levels(price_volume_dict, volume_threshold)

    # Set buy/sell thresholds for identifying high buy/sell volumes
    buy_threshold = 200  # Define your buy volume threshold
    sell_threshold = 200  # Define your sell volume threshold
    high_buy_levels, high_sell_levels = detect_high_buy_sell_levels(footprint_info, buy_threshold, sell_threshold)

    # Plot the candlestick chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlesticks',
        increasing=dict(line=dict(color='green'), fillcolor='rgba(0, 0, 0, 0)'),  # Hollow green for increasing
        decreasing=dict(line=dict(color='red'), fillcolor='rgba(0, 0, 0, 0)')     # Hollow red for decreasing
    ))

    # Plot the footprint with color based on imbalance using go.Scatter instead of fig.add_shape
    for footprint in footprint_info:
        candle_idx = footprint["candle_idx"]
        price_start = footprint["price_start"]
        price_end = footprint["price_end"]
        buy_volume = footprint["buy_volume"]
        sell_volume = footprint["sell_volume"]
        imbalance = footprint["imbalance"]

        # Get the color for this imbalance
        color = get_color_for_imbalance(imbalance)

        # Use go.Scatter to create a filled rectangle (as a polygon)
        fig.add_trace(go.Scatter(
            x=[df.index[candle_idx] - pd.Timedelta(minutes=7), df.index[candle_idx] + pd.Timedelta(minutes=7),
            df.index[candle_idx] + pd.Timedelta(minutes=7), df.index[candle_idx] - pd.Timedelta(minutes=7)],
            y=[price_start, price_start, price_end, price_end],
            fill='toself',
            fillcolor=color,
            opacity=0.65,
            line=dict(width=0),
            mode='lines',
            showlegend=False
        ))

        # Optionally, plot the volume inside the rectangle
        fig.add_trace(go.Scatter(
            x=[df.index[candle_idx]],
            y=[(price_start + price_end) / 2],
            text=[f"{buy_volume:.2f} x {sell_volume:.2f}"],
            mode='text',
            textfont=dict(color='white', size=9),
            showlegend=False
        ))



    # Plot volume and delta for each candle
    for idx, row in df.iterrows():
        total_volume = row['Volume']  # Assuming total volume is in the 'Volume' column
        candle_id = row['CandleID']
        footprint_data = fetch_footprint_data(symbol, candle_id)
        
        total_buy_volume = 0
        total_sell_volume = 0
        
        for footprint in footprint_data:
            buy_volume = float(footprint[2])
            sell_volume = float(footprint[3])
            total_buy_volume += buy_volume
            total_sell_volume += sell_volume

        delta = total_buy_volume - total_sell_volume

        # Convert Decimal to float before performing calculations
        low_price = float(row['Low'])
        high_price = float(row['High'])

        # Plot total volume below the candle
        fig.add_trace(go.Scatter(
            x=[row.name],  # Use 'row.name' to get the actual timestamp from df.index
            y=[low_price - (high_price - low_price) * 0.1],  # Slightly below the candle
            text=[f"Vol: {total_volume:.2f}"],
            mode='text',
            textfont=dict(color='white', size=9),
            showlegend=False
        ))

        # Plot delta above the candle
        fig.add_trace(go.Scatter(
            x=[row.name],  # Use 'row.name' to get the actual timestamp from df.index
            y=[high_price + (high_price - low_price) * 0.1],  # Slightly above the candle
            text=[f"Delta: {delta:.2f}"],
            mode='text',
            textfont=dict(color='yellow', size=9),
            showlegend=False
        ))

    # Overlay high-volume levels as dashed lines
    for price, volume in high_volume_levels.items():
        fig.add_shape(type="line",
                      x0=df.index[0], x1=df.index[-1],  # Stretching across the plot
                      y0=price, y1=price,
                      line=dict(color="blue", width=2, dash="dash"))

        # Adjust the text to always appear slightly beyond the rightmost candle
        fig.add_trace(go.Scatter(
            x=[df.index[-1] + pd.Timedelta(hours=2)],  # Place label 5 hours beyond the latest candle
            y=[price],
            text=[f"Vol: {volume:.2f}"],
            mode='text',
            textfont=dict(color='white', size=12),
            showlegend=False
        ))

    # Plot the high buy volume levels
    for price, buy_volume in high_buy_levels.items():
        fig.add_shape(type="line",
                      x0=df.index[0], x1=df.index[-1],  # Stretching across the plot
                      y0=price, y1=price,
                      line=dict(color="green", width=2, dash="dash"))

        fig.add_trace(go.Scatter(
            x=[df.index[-1] + pd.Timedelta(hours=4)],  # Place label 5 hours beyond the latest candle
            y=[price],
            text=[f"Buy: {buy_volume:.2f}"],
            mode='text',
            textfont=dict(color='green', size=12),
            showlegend=False
        ))

    # Plot the high sell volume levels
    for price, sell_volume in high_sell_levels.items():
        fig.add_shape(type="line",
                      x0=df.index[0], x1=df.index[-1],  # Stretching across the plot
                      y0=price, y1=price,
                      line=dict(color="red", width=2, dash="dash"))

        fig.add_trace(go.Scatter(
            x=[df.index[-1] + pd.Timedelta(hours=6)],  # Place label 5 hours beyond the latest candle
            y=[price],
            text=[f"Sell: {sell_volume:.2f}"],
            mode='text',
            textfont=dict(color='red', size=12),
            showlegend=False
        ))

    # Apply the zoom state if available
    if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]))
        zoom_data = relayout_data
    elif zoom_data:
        fig.update_layout(xaxis=dict(range=[zoom_data['xaxis.range[0]'], zoom_data['xaxis.range[1]']]))

    # Adjust y-axis to match visible range (optional)
    if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        visible_start = pd.to_datetime(relayout_data['xaxis.range[0]'])
        visible_end = pd.to_datetime(relayout_data['xaxis.range[1]'])
        visible_df = df[(df.index >= visible_start) & (df.index <= visible_end)]

        if not visible_df.empty:
            min_visible_price = visible_df['Low'].min()
            max_visible_price = visible_df['High'].max()

            fig.update_layout(yaxis=dict(range=[min_visible_price, max_visible_price]))

    fig.update_layout(
        title='BTCUSDT Candlestick and Footprint Chart (Real-Time)',
        xaxis_title='Date',
        yaxis_title='Price',
        template="plotly_dark",
        showlegend=False,
        height=1000,
        xaxis_rangeslider_visible=False
    )

    return fig, zoom_data

if __name__ == '__main__':
    app.run_server(debug=True)
