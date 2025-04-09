import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import mysql.connector
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
import plotly.io as pio
import os
import traceback
from dash.exceptions import PreventUpdate

# ---------------------------------------------------------------------------
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Define custom Plotly template
# Replace the current template definition with this more comprehensive one

# Define custom Plotly template
pio.templates["custom_dark"] = pio.templates["plotly_dark"].update(
    layout=dict(
        font=dict(
            family="Roboto, Arial, sans-serif",
            color="#FFFFFF",
            size=12
        ),
        paper_bgcolor="#1E1E1E",
        plot_bgcolor="#1E1E1E",
        colorway=["#00D774", "#FF4858", "#408EFF", "#FFB400", "#9D7DFF", "#FF7DFF"],
        title=dict(
            font=dict(size=24, color="#FFFFFF", family="Roboto, Arial, sans-serif"),
            x=0.5,
            xanchor="center"
        ),
        legend=dict(
            font=dict(size=14, color="#FFFFFF"),
            bgcolor="rgba(30,30,30,0.7)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            gridcolor="rgba(80, 80, 80, 0.3)",
            linecolor="rgba(80, 80, 80, 0.6)",
            tickfont=dict(size=12),
            tickcolor="rgba(80, 80, 80, 0.6)",
            showgrid=True,
            zeroline=False,
            showline=True,
            mirror=True
        ),
        yaxis=dict(
            gridcolor="rgba(80, 80, 80, 0.3)",
            linecolor="rgba(80, 80, 80, 0.6)",
            tickfont=dict(size=12),
            tickcolor="rgba(80, 80, 80, 0.6)",
            showgrid=True,
            zeroline=False,
            showline=True,
            mirror=True
        ),
        hoverlabel=dict(
            bgcolor="#2D2D2D",
            font=dict(size=12, color="#FFFFFF", family="Roboto, Arial, sans-serif"),
            bordercolor="rgba(255, 255, 255, 0.3)"
        ),
        # Smooth animation for updates
        transition=dict(
            duration=300,
            easing="cubic-in-out"
        )
    )
)

# Set as default template
pio.templates.default = "custom_dark"

# ---------------------------------------------------------------------------
# Define color schemes and constants
BLUE_SHADES = ['#0C008F', '#1400F5', '#5747FF', '#7C70FF', '#B0B3FF', '#D9DBFF']
RED_SHADES = ['#42101A', '#731C2D', '#A42841', '#CF3A58', '#D75B74', '#F0A6B5']
DEFAULT_SYMBOL = "btcusdt"
VOLUME_THRESHOLD = 1500
BUY_THRESHOLD = 200
SELL_THRESHOLD = 200
CANDLE_LIMIT = 50

# ---------------------------------------------------------------------------
# Define dataclass for footprint data
@dataclass
class FootprintData:
    """Class to store footprint data for better type hinting and clarity."""
    price_start: float
    price_end: float
    buy_volume: float
    sell_volume: float
    imbalance: float
    midpoint: float
    candle_idx: int

# ---------------------------------------------------------------------------
# Database and Processing Classes

class DatabaseManager:
    """Database connection and query management."""
    
    @staticmethod
    def get_db_connection() -> mysql.connector.connection.MySQLConnection:
        """Create and return a MySQL database connection."""
        try:
            host = os.getenv("DB_HOST", "localhost")
            user = os.getenv("DB_USER", "footprint")
            password = os.getenv("DB_PASSWORD", ".........")
            database = os.getenv("DB_NAME", "...........")
            logger.info(f"Connecting to database: {database} at {host} as {user}")
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                connection_timeout=300
            )
            logger.info("Database connection successful")
            return connection
        except mysql.connector.Error as err:
            logger.error(f"Database connection error: {err}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def fetch_candlestick_data(symbol: str) -> pd.DataFrame:
        """Fetch candlestick data for the specified symbol."""
        try:
            logger.info(f"Fetching candlestick data for {symbol}")
            db_conn = DatabaseManager.get_db_connection()
            cursor = db_conn.cursor()
            query = """
            SELECT id, open_time, open, high, low, close, volume 
            FROM candlesticks_15m 
            WHERE symbol = %s 
            ORDER BY open_time DESC LIMIT %s
            """
            cursor.execute(query, (symbol, CANDLE_LIMIT))
            rows = cursor.fetchall()
            logger.info(f"Query returned {len(rows)} rows")
            
            # If no rows, create dummy data for debugging
            if not rows:
                logger.warning("No data returned from query, creating dummy data for debugging")
                dummy_time = datetime.now()
                rows = []
                for i in range(10):
                    time_offset = i * 15  # 15 minute candles
                    dummy_time_str = (dummy_time - pd.Timedelta(minutes=time_offset)).strftime("%Y-%m-%d %H:%M:%S")
                    rows.append((
                        i,  # id
                        dummy_time_str,  # open_time
                        65000 + i * 100,  # open
                        65000 + i * 100 + 200,  # high
                        65000 + i * 100 - 100,  # low
                        65000 + i * 100 + 50,  # close
                        1000 + i * 10  # volume
                    ))
                logger.info(f"Created {len(rows)} dummy candles for testing")
            
            columns = ['CandleID', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = pd.DataFrame(rows, columns=columns)
            if df.empty:
                logger.warning("DataFrame is empty after processing")
                return pd.DataFrame()
            
            # Convert numeric columns to proper types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df.isna().any().any():
                logger.warning(f"DataFrame contains NaN values: {df.isna().sum()}")
                df = df.fillna(method='ffill')
            df.set_index('Date', inplace=True)
            df = df.iloc[::-1]  # Reverse to have oldest first
            logger.info(f"DataFrame shape: {df.shape}")
            cursor.close()
            db_conn.close()
            return df
        except Exception as err:
            logger.error(f"Error fetching candlestick data: {err}")
            logger.error(traceback.format_exc())
            # Create dummy data for testing purposes
            dummy_data = {
                'CandleID': list(range(10)),
                'Date': [datetime.now() - pd.Timedelta(minutes=i*15) for i in range(10)],
                'Open': [65000 + i * 100 for i in range(10)],
                'High': [65000 + i * 100 + 200 for i in range(10)],
                'Low': [65000 + i * 100 - 100 for i in range(10)],
                'Close': [65000 + i * 100 + 50 for i in range(10)],
                'Volume': [1000 + i * 10 for i in range(10)]
            }
            df = pd.DataFrame(dummy_data)
            df.set_index('Date', inplace=True)
            logger.info("Using dummy data for chart display")
            return df

    @staticmethod
    def fetch_footprint_data(symbol: str, candle_id: int) -> List[Tuple]:
        """Fetch footprint data for a specific candle ID."""
        try:
            logger.info(f"Fetching footprint data for symbol={symbol}, candle_id={candle_id}")
            db_conn = DatabaseManager.get_db_connection()
            cursor = db_conn.cursor()
            query = """
            SELECT price_range_start, price_range_end, buy_volume, sell_volume
            FROM footprints_15m
            WHERE symbol = %s AND candle_id = %s
            """
            cursor.execute(query, (symbol, int(candle_id)))
            result = cursor.fetchall()
            logger.info(f"Fetched {len(result)} footprint rows for candle_id {candle_id}")
            cursor.close()
            db_conn.close()
            if not result:
                logger.warning(f"No footprint data found for candle {candle_id}, creating dummy data")
                price_base = 65000 + candle_id * 100
                dummy_footprints = []
                for i in range(5):
                    price_start = price_base - 200 + i * 100
                    price_end = price_start + 100
                    buy_vol = 100 + (candle_id % 5) * 20 + i * 10
                    sell_vol = 80 + ((candle_id + 2) % 5) * 15 + i * 8
                    dummy_footprints.append((price_start, price_end, buy_vol, sell_vol))
                return dummy_footprints
            return result
        except Exception as err:
            logger.error(f"Error fetching footprint data: {err}")
            logger.error(traceback.format_exc())
            price_base = 65000 + candle_id * 100
            dummy_data = [
                (price_base - 200, price_base - 100, 120, 80),
                (price_base - 100, price_base, 150, 70),
                (price_base, price_base + 100, 100, 90),
                (price_base + 100, price_base + 200, 80, 110)
            ]
            return dummy_data


class FootprintProcessor:
    """Process footprint data and calculate volume profiles."""
    
    @staticmethod
    def process_footprint_data(symbol: str, df: pd.DataFrame) -> Tuple[Dict[float, float], List[FootprintData]]:
        """Process footprint data and calculate volume profiles."""
        price_volume_dict = {}
        footprint_info = []
        logger.info(f"Processing footprint data for {len(df)} candles")
        for idx, (timestamp, row) in enumerate(df.iterrows()):
            candle_id = row['CandleID']
            footprint_data = DatabaseManager.fetch_footprint_data(symbol, candle_id)
            for footprint in footprint_data:
                price_start = float(footprint[0])
                price_end = float(footprint[1])
                buy_volume = float(footprint[2])
                sell_volume = float(footprint[3])
                total_volume = buy_volume + sell_volume
                midpoint = (price_start + price_end) / 2
                imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
                footprint_info.append(FootprintData(
                    price_start=price_start,
                    price_end=price_end,
                    buy_volume=buy_volume,
                    sell_volume=sell_volume,
                    imbalance=imbalance,
                    midpoint=midpoint,
                    candle_idx=idx
                ))
                if midpoint not in price_volume_dict:
                    price_volume_dict[midpoint] = total_volume
                else:
                    price_volume_dict[midpoint] += total_volume
        logger.info(f"Processed {len(footprint_info)} footprint data points")
        return price_volume_dict, footprint_info

    @staticmethod
    def detect_high_volume_levels(price_volume_dict: Dict[float, float], volume_threshold: float) -> Dict[float, float]:
        """Identify high-volume price levels based on a volume threshold."""
        high_vol_levels = {price: volume for price, volume in price_volume_dict.items() if volume >= volume_threshold}
        logger.info(f"Detected {len(high_vol_levels)} high volume levels")
        return high_vol_levels

    @staticmethod
    def detect_high_buy_sell_levels(footprint_info: List[FootprintData],
                                    buy_threshold: float, sell_threshold: float) -> Tuple[Dict[float, float], Dict[float, float]]:
        """Detect high buy and sell volumes."""
        high_buy_levels = {}
        high_sell_levels = {}
        for footprint in footprint_info:
            midpoint = footprint.midpoint
            if footprint.buy_volume >= buy_threshold:
                high_buy_levels[midpoint] = footprint.buy_volume
            if footprint.sell_volume >= sell_threshold:
                high_sell_levels[midpoint] = footprint.sell_volume
        logger.info(f"Detected {len(high_buy_levels)} high buy levels and {len(high_sell_levels)} high sell levels")
        return high_buy_levels, high_sell_levels

    @staticmethod
    def get_color_for_imbalance(imbalance: float) -> str:
        """Return a color based on the buy/sell imbalance."""
        if imbalance > 0:
            if imbalance >= 0.8:
                return BLUE_SHADES[0]
            elif imbalance >= 0.6:
                return BLUE_SHADES[1]
            elif imbalance >= 0.4:
                return BLUE_SHADES[2]
            elif imbalance >= 0.2:
                return BLUE_SHADES[3]
            elif imbalance >= 0.1:
                return BLUE_SHADES[4]
            else:
                return BLUE_SHADES[5]
        elif imbalance < 0:
            if imbalance <= -0.8:
                return RED_SHADES[0]
            elif imbalance <= -0.6:
                return RED_SHADES[1]
            elif imbalance <= -0.4:
                return RED_SHADES[2]
            elif imbalance <= -0.2:
                return RED_SHADES[3]
            elif imbalance <= -0.1:
                return RED_SHADES[4]
            else:
                return RED_SHADES[5]
        return 'rgba(128, 128, 128, 0.5)'

class ChartCreator:
    """Create and manage chart visualizations with improved styling."""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, footprint_info: List[FootprintData], symbol: str,
                     high_volume_levels: Dict[float, float],
                     high_buy_levels: Dict[float, float],
                     high_sell_levels: Dict[float, float]) -> go.Figure:
        logger.info("Creating chart")
        fig = go.Figure()
        ChartCreator._add_candlestick(fig, df)
        logger.info("Added candlestick data")
        ChartCreator._add_footprints(fig, df, footprint_info)
        logger.info("Added footprint data")
        ChartCreator._add_volume_delta_annotations(fig, df, symbol)
        logger.info("Added volume and delta annotations")
        ChartCreator._add_high_volume_lines(fig, df, high_volume_levels)
        ChartCreator._add_buy_sell_levels(fig, df, high_buy_levels, high_sell_levels)
        logger.info("Added support/resistance levels")
        ChartCreator._apply_layout(fig, symbol)
        logger.info("Applied chart layout")
        return fig

    @staticmethod
    def _add_candlestick(fig: go.Figure, df: pd.DataFrame) -> None:
        try:
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlesticks',
                increasing=dict(
                    line=dict(color='#00D774', width=1.5), 
                    fillcolor='rgba(0, 215, 116, 0.15)'
                ),
                decreasing=dict(
                    line=dict(color='#FF4858', width=1.5), 
                    fillcolor='rgba(255, 72, 88, 0.15)'
                ),
                hoverinfo='all',
                hovertemplate=(
                    '<b>Date:</b> %{x}<br>' +
                    '<b>Open:</b> %{open:.2f}<br>' +
                    '<b>High:</b> %{high:.2f}<br>' +
                    '<b>Low:</b> %{low:.2f}<br>' +
                    '<b>Close:</b> %{close:.2f}<extra></extra>'
                )
            ))
        except Exception as e:
            logger.error(f"Error adding candlestick: {e}")
            logger.error(traceback.format_exc())
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Price',
                line=dict(color='#00D774', width=2)
            ))

    @staticmethod
    def _add_footprints(fig: go.Figure, df: pd.DataFrame, footprint_info: List[FootprintData]) -> None:
        for footprint in footprint_info:
            try:
                candle_idx = footprint.candle_idx
                if candle_idx >= len(df):
                    continue
                price_start = footprint.price_start
                price_end = footprint.price_end
                buy_volume = footprint.buy_volume
                sell_volume = footprint.sell_volume
                imbalance = footprint.imbalance
                color = FootprintProcessor.get_color_for_imbalance(imbalance)
                
                # Adjustments for better visualization
                time_width = pd.Timedelta(minutes=7)
                if len(df) > 0 and candle_idx < len(df):
                    fig.add_trace(go.Scatter(
                        x=[
                            df.index[candle_idx] - time_width,
                            df.index[candle_idx] + time_width,
                            df.index[candle_idx] + time_width,
                            df.index[candle_idx] - time_width
                        ],
                        y=[price_start, price_start, price_end, price_end],
                        fill='toself',
                        fillcolor=color,
                        opacity=0.7,  # Slightly more transparent
                        line=dict(width=0),
                        mode='lines',
                        showlegend=False,
                        hovertemplate=(
                            f'<b>Price Range:</b> {price_start:.2f} - {price_end:.2f}<br>' +
                            f'<b>Buy Volume:</b> {buy_volume:.2f}<br>' +
                            f'<b>Sell Volume:</b> {sell_volume:.2f}<br>' +
                            f'<b>Imbalance:</b> {imbalance:.2f}<extra></extra>'
                        )
                    ))
                    
                    # Add volume text with background for better readability
                    fig.add_trace(go.Scatter(
                        x=[df.index[candle_idx]],
                        y=[(price_start + price_end) / 2],
                        text=[f"{buy_volume:.1f} × {sell_volume:.1f}"],
                        mode='text',
                        textfont=dict(
                            color='white', 
                            size=9, 
                            family='Roboto Mono, monospace'
                        ),
                        showlegend=False,
                        # Add a subtle rectangle behind the text
                        marker=dict(
                            opacity=0.6,
                            color='rgba(0,0,0,0.7)',
                            size=10,
                            line=dict(width=0)
                        )
                    ))
            except Exception as e:
                logger.error(f"Error adding footprint at index {candle_idx}: {e}")
                continue

    @staticmethod
    def _add_volume_delta_annotations(fig: go.Figure, df: pd.DataFrame, symbol: str) -> None:
        for idx, (timestamp, row) in enumerate(df.iterrows()):
            try:
                total_volume = row['Volume']
                candle_id = row['CandleID']
                footprint_data = DatabaseManager.fetch_footprint_data(symbol, candle_id)
                total_buy_volume = sum(float(fp[2]) for fp in footprint_data)
                total_sell_volume = sum(float(fp[3]) for fp in footprint_data)
                delta = total_buy_volume - total_sell_volume
                delta_color = '#00D774' if delta >= 0 else '#FF4858'
                low_price = float(row['Low'])
                high_price = float(row['High'])
                
                # Add volume annotation with subtle background
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[low_price - (high_price - low_price) * 0.1],
                    text=[f"Vol: {total_volume:.1f}"],
                    mode='text',
                    textfont=dict(
                        color='white', 
                        size=9, 
                        family='Roboto Mono, monospace'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
                
                # Add delta annotation with colored text
                fig.add_trace(go.Scatter(
                    x=[timestamp],
                    y=[high_price + (high_price - low_price) * 0.1],
                    text=[f"Δ: {delta:.1f}"],
                    mode='text',
                    textfont=dict(
                        color=delta_color, 
                        size=9, 
                        family='Roboto Mono, monospace'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
            except Exception as e:
                logger.error(f"Error adding volume/delta annotation: {e}")
                continue

    @staticmethod
    def _add_high_volume_lines(fig: go.Figure, df: pd.DataFrame, high_volume_levels: Dict[float, float]) -> None:
        try:
            if len(df) == 0:
                logger.warning("Empty dataframe, skipping high volume lines")
                return
            for price, volume in high_volume_levels.items():
                # Add a more subtle line with better styling
                fig.add_shape(
                    type="line",
                    x0=df.index[0],
                    x1=df.index[-1],
                    y0=price,
                    y1=price,
                    line=dict(
                        color="#408EFF", 
                        width=1.5, 
                        dash="dash"
                    ),
                    opacity=0.7,
                    layer="below"
                )
                
                # Add text annotation with better positioning and styling
                fig.add_trace(go.Scatter(
                    x=[df.index[-1] + pd.Timedelta(hours=2)],
                    y=[price],
                    text=[f"Vol: {volume:.1f}"],
                    mode='text',
                    textfont=dict(
                        color='#408EFF', 
                        size=10, 
                        family='Roboto, sans-serif'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
        except Exception as e:
            logger.error(f"Error adding high volume lines: {e}")

    @staticmethod
    def _add_buy_sell_levels(fig: go.Figure, df: pd.DataFrame,
                             high_buy_levels: Dict[float, float],
                             high_sell_levels: Dict[float, float]) -> None:
        try:
            if len(df) == 0:
                logger.warning("Empty dataframe, skipping buy/sell lines")
                return
                
            # Add buy levels with improved styling
            for price, buy_volume in high_buy_levels.items():
                fig.add_shape(
                    type="line",
                    x0=df.index[0],
                    x1=df.index[-1],
                    y0=price,
                    y1=price,
                    line=dict(
                        color="#00D774", 
                        width=1.5, 
                        dash="dot"
                    ),
                    opacity=0.7,
                    layer="below"
                )
                fig.add_trace(go.Scatter(
                    x=[df.index[-1] + pd.Timedelta(hours=4)],
                    y=[price],
                    text=[f"Buy: {buy_volume:.1f}"],
                    mode='text',
                    textfont=dict(
                        color='#00D774', 
                        size=10, 
                        family='Roboto, sans-serif'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
                
            # Add sell levels with improved styling
            for price, sell_volume in high_sell_levels.items():
                fig.add_shape(
                    type="line",
                    x0=df.index[0],
                    x1=df.index[-1],
                    y0=price,
                    y1=price,
                    line=dict(
                        color="#FF4858", 
                        width=1.5, 
                        dash="dot"
                    ),
                    opacity=0.7,
                    layer="below"
                )
                fig.add_trace(go.Scatter(
                    x=[df.index[-1] + pd.Timedelta(hours=6)],
                    y=[price],
                    text=[f"Sell: {sell_volume:.1f}"],
                    mode='text',
                    textfont=dict(
                        color='#FF4858', 
                        size=10, 
                        family='Roboto, sans-serif'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
        except Exception as e:
            logger.error(f"Error adding buy/sell levels: {e}")

    @staticmethod
    def _apply_layout(fig: go.Figure, symbol: str) -> None:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.update_layout(
                title=f'{symbol.upper()} Candlestick & Footprint Chart (Updated: {current_time})',
                xaxis_title='Date',
                yaxis_title='Price (USDT)',
                template="custom_dark",
                showlegend=False,
                height=800,
                xaxis_rangeslider_visible=False,
                hovermode="closest",
                modebar=dict(
                    orientation='v',
                    bgcolor='rgba(30,30,30,0.7)',
                    color='white'
                ),
                # Add smooth animations to prevent jarring updates
                transition_duration=300,
                transition_easing="cubic-in-out"
            )
            
            # Make sure axes are properly configured
            fig.update_xaxes(
                showspikes=True,
                spikecolor="rgba(255,255,255,0.3)",
                spikethickness=1,
                spikedash="solid"
            )
            
            fig.update_yaxes(
                showspikes=True,
                spikecolor="rgba(255,255,255,0.3)",
                spikethickness=1,
                spikedash="solid",
                tickformat=",.2f"  # Format price with commas and 2 decimal places
            )
            
        except Exception as e:
            logger.error(f"Error applying layout: {e}")
            fig.update_layout(
                template="plotly_dark",
                showlegend=False,
                height=800,
            )

# ---------------------------------------------------------------------------
# Define the App Layout

def create_app_layout() -> html.Div:
    """Create the application layout with modern styling."""
    return html.Div(className="trading-app-container", children=[
        # Header section
        html.Header(className="app-header", children=[
            html.H1("Crypto Footprint Trading Dashboard", className="app-title"),
            html.Div(className="app-controls", children=[
                dcc.Dropdown(
                    id='symbol-selector',
                    options=[
                        {'label': 'BTC/USDT', 'value': 'btcusdt'},
                        {'label': 'ETH/USDT', 'value': 'ethusdt'},
                        {'label': 'SOL/USDT', 'value': 'solusdt'},
                    ],
                    value=DEFAULT_SYMBOL,
                    clearable=False,
                    className="symbol-dropdown"
                ),
                dcc.Dropdown(
                    id='timeframe-selector',
                    options=[
                        {'label': '15m', 'value': '15m'},
                        {'label': '1h', 'value': '1h'},
                        {'label': '4h', 'value': '4h'},
                        {'label': '1d', 'value': '1d'},
                    ],
                    value='15m',
                    clearable=False,
                    className="timeframe-dropdown"
                ),
                # Add a refresh button
                html.Button('Refresh', id='refresh-button', className="refresh-button"),
            ]),
        ]),
        # Debug info (visible only in development mode)
        html.Div(id='debug-info', className="debug-info", children=[
            html.H3("Debug Information"),
            html.Pre(id='debug-output', children="Waiting for data...")
        ], style={'display': 'none'}),
        # Main chart container
        html.Div(className="chart-container", children=[
            dcc.Graph(
                id='live-graph',
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    # Add pan configuration to make pan the default
                    'dragmode': 'pan',
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'modeBarButtonsToRemove': ['lasso2d']
                }
            )
        ]),
        # Changed to a much higher value to essentially disable auto-refresh
        # Only refresh when button is clicked
        dcc.Interval(id='graph-update', interval=3600000),  # 1 hour
        # Store components for state management
        dcc.Store(id='zoom-store'),
        dcc.Store(id='chart-data-store'),
        # Footer
        html.Footer(className="app-footer", children=[
            html.P("© 2025 Trading Data Analytics • Last Updated: " +
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ])
    ])

# ---------------------------------------------------------------------------
# Initialize the Dash app and set layout

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
)
server = app.server  # For Gunicorn deployment
app.layout = create_app_layout()

# ---------------------------------------------------------------------------
# Register Callbacks

@app.callback(
    Output('debug-info', 'style'),
    [Input('graph-update', 'n_intervals')],
    [State('live-graph', 'figure')]
)
def toggle_debug_info(n, figure):
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    return {'display': 'block'} if debug_mode else {'display': 'none'}

@app.callback(
    Output('debug-output', 'children'),
    [Input('graph-update', 'n_intervals')],
    [State('symbol-selector', 'value')]
)
def update_debug_info(n, symbol):
    try:
        db_status = "Unknown"
        try:
            conn = DatabaseManager.get_db_connection()
            db_status = "Connected" if conn else "Failed"
            if conn:
                conn.close()
        except Exception as e:
            db_status = f"Error: {str(e)}"
        data_status = "Unknown"
        try:
            df = DatabaseManager.fetch_candlestick_data(symbol)
            data_status = f"OK - {len(df)} rows" if not df.empty else "No data"
        except Exception as e:
            data_status = f"Error: {str(e)}"
        debug_text = f"""
Timestamp: {datetime.now()}
Symbol: {symbol}

Database Status: {db_status}
Data Status: {data_status}

Memory Usage: {os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()} KB
"""
        return debug_text
    except Exception as e:
        return f"Error collecting debug info: {str(e)}"

# Add a new callback for the refresh button
@app.callback(
    Output('graph-update', 'n_intervals'),
    [Input('refresh-button', 'n_clicks'),
     Input('symbol-selector', 'value'),
     Input('timeframe-selector', 'value')],
    [State('graph-update', 'n_intervals')]
)
def refresh_graph(n_clicks, symbol_value, timeframe_value, n_intervals):
    # This will trigger the graph update callback
    return (n_intervals or 0) + 1

@app.callback(
    [Output('live-graph', 'figure'),
     Output('zoom-store', 'data')],
    [Input('graph-update', 'n_intervals'),
     Input('symbol-selector', 'value')],
    [State('zoom-store', 'data'),
     State('live-graph', 'relayoutData')]
)
def update_graph(n_intervals, symbol, zoom_data, relayout_data):
    try:
        # Don't update if n_intervals is None (initial load)
        if n_intervals is None:
            raise PreventUpdate
            
        symbol = symbol or DEFAULT_SYMBOL
        logger.info(f"Updating graph for {symbol}")
        df = DatabaseManager.fetch_candlestick_data(symbol)
        if df.empty:
            logger.warning("Empty dataframe returned")
            return (ChartCreator.create_chart(
                        pd.DataFrame(), [], symbol, {}, {}, {}
                    ) if False else  # This branch is unreachable; fallback below
                    create_fallback_chart("No data available for the selected symbol")), zoom_data
        price_volume_dict, footprint_info = FootprintProcessor.process_footprint_data(symbol, df)
        high_volume_levels = FootprintProcessor.detect_high_volume_levels(price_volume_dict, VOLUME_THRESHOLD)
        high_buy_levels, high_sell_levels = FootprintProcessor.detect_high_buy_sell_levels(footprint_info, BUY_THRESHOLD, SELL_THRESHOLD)
        fig = ChartCreator.create_chart(df, footprint_info, symbol, high_volume_levels, high_buy_levels, high_sell_levels)
        if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            fig.update_layout(xaxis=dict(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]))
            zoom_data = {'xaxis.range[0]': relayout_data['xaxis.range[0]'], 'xaxis.range[1]': relayout_data['xaxis.range[1]']}
            if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
                fig.update_layout(yaxis=dict(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]))
                zoom_data.update({'yaxis.range[0]': relayout_data['yaxis.range[0]'], 'yaxis.range[1]': relayout_data['yaxis.range[1]']})
        elif zoom_data:
            fig.update_layout(xaxis=dict(range=[zoom_data['xaxis.range[0]'], zoom_data['xaxis.range[1]']]))
            if 'yaxis.range[0]' in zoom_data and 'yaxis.range[1]' in zoom_data:
                fig.update_layout(yaxis=dict(range=[zoom_data['yaxis.range[0]'], zoom_data['yaxis.range[1]']]))
        return fig, zoom_data
    except Exception as e:
        logger.error(f"Error updating graph: {e}")
        logger.error(traceback.format_exc())
        return create_fallback_chart(f"Error updating chart: {str(e)}"), zoom_data

def create_fallback_chart(message="No data available or error processing data"):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="white")
    )
    fig.update_layout(template="custom_dark", height=800)
    return fig

# ---------------------------------------------------------------------------
# Run the app

if __name__ == '__main__':
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    port = int(os.getenv("PORT", 8050))
    host = os.getenv("HOST", "0.0.0.0")
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        app.layout = create_app_layout()  # Ensure layout is built fresh in debug mode
    logger.info(f"Starting server on {host}:{port} with debug={debug_mode}")
    app.run_server(debug=debug_mode, host=host, port=port)
