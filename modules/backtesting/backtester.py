import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from datetime import datetime, timedelta
from ..ml_models.price_predictor import PricePredictor
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import TimeSeriesSplit
from ml_models.hrl import HRLManager, HRLWorker, NeuralHRL
import torch
import random
import matplotlib.pyplot as plt

class AlertSystem:
    """Real-time alerting system for trading strategies"""
    
    def __init__(self):
        self.alerts = {}
        self.triggered_alerts = []
        self.notification_methods = []
        
    def add_alert(self, 
                 name: str, 
                 condition: callable, 
                 message_template: str,
                 priority: str = 'medium') -> None:
        """
        Add a new alert condition.
        
        Args:
            name: Unique alert name
            condition: Function that returns True when alert should trigger
            message_template: Template string for alert message
            priority: 'low', 'medium', or 'high'
        """
        self.alerts[name] = {
            'condition': condition,
            'message_template': message_template,
            'priority': priority,
            'trigger_count': 0
        }
        
    def add_notification_channel(self, method: callable) -> None:
        """
        Add a notification channel for alerts.
        
        Args:
            method: Function to handle notifications (e.g., email, SMS, webhook)
        """
        self.notification_methods.append(method)
        
    def check_alerts(self, current_state: Dict) -> List[Dict]:
        """
        Check all registered alerts against current market state.
        
        Args:
            current_state: Dictionary containing current market data and portfolio state
            
        Returns:
            List of triggered alerts
        """
        triggered = []
        for name, alert in self.alerts.items():
            try:
                if alert['condition'](current_state):
                    alert['trigger_count'] += 1
                    message = alert['message_template'].format(**current_state)
                    alert_data = {
                        'timestamp': datetime.now(),
                        'name': name,
                        'message': message,
                        'priority': alert['priority'],
                        'data': current_state.copy()
                    }
                    triggered.append(alert_data)
                    self.triggered_alerts.append(alert_data)
                    
                    # Send notifications
                    for notify in self.notification_methods:
                        notify(alert_data)
                        
            except Exception as e:
                logger.error(f"Error checking alert {name}: {e}")
                
        return triggered

    def add_bollinger_alert(self,
                           name: str,
                           window: int = 20,
                           num_std: float = 2,
                           direction: str = 'below',
                           priority: str = 'medium') -> None:
        """
        Add Bollinger Band alert.
        
        Args:
            name: Alert name
            window: Moving average window
            num_std: Number of standard deviations for bands
            direction: 'below' (lower band) or 'above' (upper band)
            priority: Alert priority
        """
        def condition(state):
            prices = state['data']['close']
            if len(prices) < window:
                return False
            sma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            current_price = prices[-1]
            if direction == 'below':
                return current_price < lower[-1]
            return current_price > upper[-1]
            
        message = f"Price {direction} Bollinger Band ({window},{num_std}) at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def add_fibonacci_alert(self,
                           name: str,
                           swing_high: float,
                           swing_low: float,
                           level: float = 0.618,
                           direction: str = 'below',
                           priority: str = 'medium') -> None:
        """
        Add Fibonacci retracement alert.
        
        Args:
            name: Alert name
            swing_high: Swing high price level
            swing_low: Swing low price level
            level: Fibonacci level (0.236, 0.382, 0.5, 0.618, 0.786)
            direction: 'below' or 'above' the level
            priority: Alert priority
        """
        def condition(state):
            current_price = state['price']
            fib_level = swing_high - (swing_high - swing_low) * level
            if direction == 'below':
                return current_price < fib_level
            return current_price > fib_level
            
        message = f"Price {direction} Fibonacci {level*100}% level at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def add_anomaly_alert(self,
                         name: str,
                         model: Any,  # Should be a trained anomaly detection model
                         lookback: int = 60,
                         threshold: float = 3.0,
                         priority: str = 'high') -> None:
        """
        Add machine learning-based anomaly detection alert.
        
        Args:
            name: Alert name
            model: Pre-trained anomaly detection model
            lookback: Data window for anomaly detection
            threshold: Z-score threshold for anomalies
            priority: Alert priority
        """
        def condition(state):
            features = self._create_anomaly_features(state['data'], lookback)
            if len(features) < lookback:
                return False
            score = model.decision_function(features.reshape(1, -1))
            return abs(score) > threshold
            
        message = f"Anomaly detected (score: {threshold}+) at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def _create_anomaly_features(self, data: pd.DataFrame, lookback: int) -> np.ndarray:
        """Create features for anomaly detection"""
        features = []
        # Price features
        features.append(data['close'].pct_change().values[-lookback:])
        # Volatility features
        features.append(data['close'].rolling(20).std().values[-lookback:])
        # Volume features
        features.append(data['volume'].pct_change().values[-lookback:])
        return np.nan_to_num(np.concatenate(features))

    def add_confirmed_alert(self,
                           name: str,
                           base_alert: str,
                           confirmations: int = 3,
                           within_period: int = 5,
                           priority: str = 'high') -> None:
        """
        Add alert that requires multiple confirmations.
        
        Args:
            name: Alert name
            base_alert: Name of existing alert to confirm
            confirmations: Number of required triggers
            within_period: Number of periods for confirmation window
            priority: Alert priority
        """
        def condition(state):
            # Check if base alert is in recent triggers
            recent_triggers = [a for a in self.triggered_alerts[-within_period:] 
                              if a['name'] == base_alert]
            return len(recent_triggers) >= confirmations
            
        message = f"Confirmed {base_alert} ({confirmations}x in {within_period} periods) at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def add_head_shoulders_alert(self,
                                name: str,
                                window: int = 30,
                                sensitivity: float = 0.9,
                                pattern_type: str = 'head_shoulders',
                                priority: str = 'high') -> None:
        """
        Add Head & Shoulders pattern detection alert.
        
        Args:
            name: Alert name
            window: Lookback window for pattern detection
            sensitivity: Peak detection sensitivity (0-1)
            pattern_type: 'head_shoulders' or 'inverse_head_shoulders'
            priority: Alert priority
        """
        def condition(state):
            prices = state['data']['close'][-window:]
            peaks = self._find_peaks(prices, sensitivity)
            if len(peaks) < 4:
                return False
                
            # Check pattern sequence
            if pattern_type == 'head_shoulders':
                return (peaks[-4] < peaks[-3] and 
                        peaks[-3] > peaks[-2] and
                        peaks[-2] < peaks[-1])
            else:  # Inverse
                return (peaks[-4] > peaks[-3] and 
                        peaks[-3] < peaks[-2] and
                        peaks[-2] > peaks[-1])
                        
        message = f"{pattern_type.replace('_', ' ').title()} detected at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def add_double_top_bottom_alert(self,
                                   name: str,
                                   window: int = 20,
                                   retracement_pct: float = 0.1,
                                   pattern_type: str = 'double_top',
                                   priority: str = 'medium') -> None:
        """
        Add Double Top/Bottom pattern detection alert.
        
        Args:
            name: Alert name
            window: Lookback window for pattern detection
            retracement_pct: Minimum retracement percentage
            pattern_type: 'double_top' or 'double_bottom'
            priority: Alert priority
        """
        def condition(state):
            prices = state['data']['close'][-window:]
            peaks = self._find_peaks(prices, 0.8)
            if len(peaks) < 3:
                return False
                
            if pattern_type == 'double_top':
                return (peaks[-3] < peaks[-2] and
                        abs(peaks[-2] - peaks[-1]) < 0.02*peaks[-2] and
                        prices[-1] < peaks[-1]*(1-retracement_pct))
            else:
                return (peaks[-3] > peaks[-2] and
                        abs(peaks[-2] - peaks[-1]) < 0.02*peaks[-2] and
                        prices[-1] > peaks[-1]*(1+retracement_pct))
                        
        message = f"{pattern_type.replace('_', ' ').title()} detected at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def _find_peaks(self, prices: pd.Series, sensitivity: float) -> List[float]:
        """Find local maxima/minima in price series"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(prices, prominence=sensitivity*np.std(prices))
        return prices.iloc[peaks].tolist()

    def add_triangle_pattern_alert(self,
                                  name: str,
                                  pattern_type: str = 'symmetrical',
                                  min_swings: int = 4,
                                  convergence_threshold: float = 0.03,
                                  priority: str = 'medium') -> None:
        """
        Add triangle pattern detection alert.
        
        Args:
            name: Alert name
            pattern_type: 'symmetrical', 'ascending', or 'descending'
            min_swings: Minimum number of swing points required
            convergence_threshold: Price convergence threshold
            priority: Alert priority
        """
        def condition(state):
            prices = state['data']['close']
            swings = self._find_swing_points(prices, 0.5)
            
            if len(swings) < min_swings:
                return False
                
            # Calculate trendline slopes
            upper_slope, lower_slope = self._calculate_triangle_slopes(swings)
            
            if pattern_type == 'symmetrical':
                return (abs(upper_slope + lower_slope) < convergence_threshold and
                        upper_slope < 0 and lower_slope > 0)
            elif pattern_type == 'ascending':
                return (lower_slope > 0 and abs(upper_slope) < convergence_threshold)
            else:  # descending
                return (upper_slope < 0 and abs(lower_slope) < convergence_threshold)
                        
        message = f"{pattern_type.title()} triangle pattern detected at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def add_flag_pattern_alert(self,
                             name: str,
                             min_ratio: float = 0.5,
                             consolidation_bars: int = 15,
                             priority: str = 'medium') -> None:
        """
        Add flag/pennant pattern detection alert.
        
        Args:
            name: Alert name
            min_ratio: Minimum flagpole to consolidation ratio
            consolidation_bars: Maximum bars for consolidation period
            priority: Alert priority
        """
        def condition(state):
            prices = state['data']['close']
            swings = self._find_swing_points(prices, 0.7)
            
            if len(swings) < 2:
                return False
                
            # Find flagpole move
            flagpole_height = abs(swings[-2] - swings[-1])
            consolidation_range = prices[-consolidation_bars:].max() - prices[-consolidation_bars:].min()
            
            return (flagpole_height > 0 and 
                    consolidation_range / flagpole_height <= min_ratio)
                        
        message = f"Flag/pennant pattern detected at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def _find_swing_points(self, prices: pd.Series, sensitivity: float) -> List[float]:
        """Identify swing highs/lows in price series"""
        from scipy.signal import argrelextrema
        highs = argrelextrema(prices.values, np.greater, order=2)[0]
        lows = argrelextrema(prices.values, np.less, order=2)[0]
        return sorted([(i, prices[i], 'high') for i in highs] + 
                      [(i, prices[i], 'low') for i in lows], key=lambda x: x[0])

    def _calculate_triangle_slopes(self, swings: List[Tuple]) -> Tuple[float, float]:
        """Calculate upper and lower trendline slopes"""
        upper_swings = [s for s in swings if s[2] == 'high']
        lower_swings = [s for s in swings if s[2] == 'low']
        
        if len(upper_swings) < 2 or len(lower_swings) < 2:
            return 0.0, 0.0
            
        # Upper trendline (connect swing highs)
        x = [s[0] for s in upper_swings[-2:]]
        y = [s[1] for s in upper_swings[-2:]]
        upper_slope = (y[1] - y[0]) / (x[1] - x[0]) if x[1] != x[0] else 0
        
        # Lower trendline (connect swing lows)
        x = [s[0] for s in lower_swings[-2:]]
        y = [s[1] for s in lower_swings[-2:]]
        lower_slope = (y[1] - y[0]) / (x[1] - x[0]) if x[1] != x[0] else 0
        
        return upper_slope, lower_slope

    def add_cup_handle_alert(self,
                            name: str,
                            cup_duration: int = 20,
                            handle_duration: int = 10,
                            depth_threshold: float = 0.3,
                            priority: str = 'high') -> None:
        """
        Add Cup & Handle pattern detection alert.
        
        Args:
            name: Alert name
            cup_duration: Minimum duration for cup formation (bars)
            handle_duration: Maximum duration for handle formation (bars)
            depth_threshold: Minimum cup depth percentage
            priority: Alert priority
        """
        def condition(state):
            prices = state['data']['close']
            if len(prices) < cup_duration + handle_duration:
                return False
                
            # Find cup formation
            cup_start = -cup_duration - handle_duration
            cup_prices = prices[cup_start:-handle_duration]
            cup_low = cup_prices.min()
            cup_depth = (cup_prices[0] - cup_low) / cup_prices[0]
            
            # Check handle formation
            handle_prices = prices[-handle_duration:]
            handle_high = handle_prices.max()
            handle_low = handle_prices.min()
            
            return (cup_depth >= depth_threshold and
                    handle_high < cup_prices[-1] and
                    handle_low > cup_low)
                    
        message = f"Cup & Handle pattern detected at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def add_wolfe_wave_alert(self,
                            name: str,
                            swing_points: int = 5,
                            channel_tolerance: float = 0.02,
                            priority: str = 'high') -> None:
        """
        Add Wolfe Wave pattern detection alert.
        
        Args:
            name: Alert name
            swing_points: Number of required swing points (5 or 6)
            channel_tolerance: Price channel alignment tolerance
            priority: Alert priority
        """
        def condition(state):
            prices = state['data']['close']
            swings = self._find_swing_points(prices, 0.7)
            if len(swings) < swing_points:
                return False
                
            # Check Wolfe Wave structure
            points = [s[1] for s in swings[-swing_points:]]
            
            # Validate channel alignment
            trendline1 = self._calculate_trendline(points[0], points[2], points[4])
            trendline2 = self._calculate_trendline(points[1], points[3], points[5] if swing_points ==6 else points[4])
            
            return (abs(trendline1 - trendline2) <= channel_tolerance and
                    points[2] < points[0] and
                    points[3] > points[1])
                    
        message = f"Wolfe Wave ({swing_points}-point) detected at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def _calculate_trendline(self, p1: float, p2: float, p3: float) -> float:
        """Calculate trendline slope consistency"""
        slope1 = (p2 - p1) / 2  # Assuming equal time intervals
        slope2 = (p3 - p2) / 2
        return abs(slope1 - slope2)

    def add_harmonic_pattern_alert(self,
                                 name: str,
                                 pattern_type: str = 'shark',
                                 tolerance: float = 0.06,
                                 priority: str = 'high') -> None:
        """
        Add advanced harmonic pattern detection with probability scoring.
        """
        advanced_ratios = {
            'shark': {'XA': 1.0, 'AB': 1.13, 'BC': 1.618, 'CD': 2.24},
            'cypher': {'XA': 1.0, 'AB': 0.382, 'BC': 1.272, 'CD': 1.414},
            'deep_crab': {'XA': 1.0, 'AB': 0.382, 'BC': 0.886, 'CD': 2.618}
        }

        def condition(state):
            prices = state['data']['close']
            swings = self._find_swing_points(prices, 0.75)
            if len(swings) < 5:
                return False

            X, A, B, C, D = [s[1] for s in swings[-5:]]
            ratios = advanced_ratios[pattern_type]
            
            # Validate advanced harmonic ratios
            valid_XA = self._validate_ratio(A-X, 0, 1.0, tolerance)  # XA should be 100% move
            valid_AB = self._validate_ratio(B-A, X-A, ratios['AB'], tolerance)
            valid_BC = self._validate_ratio(C-B, B-A, ratios['BC'], tolerance)
            valid_CD = self._validate_ratio(D-C, C-B, ratios['CD'], tolerance)
            
            # Pattern-specific validation
            if pattern_type == 'shark':
                return valid_XA and valid_AB and valid_BC and valid_CD and D > X
            elif pattern_type == 'cypher':
                return valid_XA and valid_AB and valid_BC and valid_CD and D < C
            return valid_XA and valid_AB and valid_BC and valid_CD
            
            # Calculate pattern confidence
            actual_ratios = {
                'AB': (B-A)/(X-A) if (X-A) != 0 else 0,
                'BC': (C-B)/(B-A) if (B-A) != 0 else 0,
                'CD': (D-C)/(C-B) if (C-B) != 0 else 0
            }
            confidence = self._calculate_pattern_score(actual_ratios, ratios)
            state['pattern_confidence'] = confidence  # Store in market state
            
            return valid and confidence >= 0.7  # Minimum confidence threshold
            
        message = f"{pattern_type.title()} harmonic pattern detected at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def add_elliott_wave_alert(self,
                             name: str,
                             wave_type: str = 'impulse',
                             degree: str = 'intermediate',
                             priority: str = 'high') -> None:
        """
        Add Elliott Wave pattern detection alert.
        
        Args:
            name: Alert name
            wave_type: 'impulse' or 'corrective'
            degree: Wave degree ('primary', 'intermediate', 'minor')
            priority: Alert priority
        """
        def condition(state):
            prices = state['data']['close']
            swings = self._find_swing_points(prices, 0.8)
            if len(swings) < 5:
                return False

            # Elliott Wave rules
            wave1, wave2, wave3, wave4, wave5 = [s[1] for s in swings[-5:]]
            
            # Basic impulse wave rules
            valid = (
                wave3 > wave1 and  # Wave 3 not the shortest
                wave4 > wave1 and  # Wave 4 doesn't enter Wave 1 territory
                wave5 > wave3 and  # Wave 5 makes new high
                (wave2 - wave1) > (wave4 - wave3)  # Corrective wave relationships
            )
            
            if wave_type == 'corrective':
                # ABC correction pattern
                a, b, c = swings[-3:]
                valid = (
                    (c[1] - a[1]) > 0.618 * (b[1] - a[1]) and
                    (c[1] - a[1]) < 1.618 * (b[1] - a[1])
                )
            
            return valid
            
        message = f"{degree.title()} {wave_type} wave detected at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def _validate_ratio(self, move: float, reference: float, target: float, tolerance: float) -> bool:
        """Validate Fibonacci ratio with tolerance"""
        if reference == 0:
            return False
        ratio = abs(move / reference)
        return abs(ratio - target) <= tolerance

    def _calculate_pattern_score(self, 
                               actual_ratios: Dict[str, float], 
                               target_ratios: Dict[str, float]) -> float:
        """Calculate pattern confidence score (0-1)"""
        scores = []
        for key in ['AB', 'BC', 'CD']:
            target = target_ratios[key]
            actual = actual_ratios.get(key, 0)
            scores.append(1 - min(abs(actual - target)/target, 1))
        return np.mean(scores)

    def add_confidence_filtered_alert(self,
                                    name: str,
                                    base_alert: str,
                                    confidence_threshold: float = 0.7,
                                    lookback_period: int = 30,
                                    priority: str = 'high') -> None:
        """
        Add alert filtered by historical confidence scores.
        
        Args:
            name: Alert name
            base_alert: Name of alert to filter
            confidence_threshold: Minimum average confidence
            lookback_period: Days to consider for confidence calculation
            priority: Alert priority
        """
        def condition(state):
            # Get historical confidence for base alert
            history = [a for a in self.triggered_alerts 
                      if a['name'] == base_alert][-lookback_period:]
            if not history:
                return False
                
            avg_confidence = np.mean([a['data'].get('pattern_confidence', 0) for a in history])
            current_confidence = state.get('pattern_confidence', 0)
            
            return current_confidence >= avg_confidence and avg_confidence >= confidence_threshold
            
        message = f"High-confidence {base_alert} (≥{confidence_threshold}) at {{timestamp}}"
        self.add_alert(name, condition, message, priority)

    def calculate_confidence_position(self,
                                    confidence: float,
                                    max_allocation: float = 0.2,
                                    base_size: float = 0.1) -> float:
        """
        Calculate position size based on confidence score.
        
        Args:
            confidence: Pattern confidence score (0-1)
            max_allocation: Maximum portfolio percentage per trade
            base_size: Base position size at minimum confidence
            
        Returns:
            Position size as percentage of portfolio
        """
        sigmoid = 1 / (1 + np.exp(-10*(confidence - 0.5)))
        return base_size + (max_allocation - base_size) * sigmoid

class CompressedAttentionHRL:
    """Hierarchical RL with Compressed Attention Patterns"""
    
    class ManagerNetwork(torch.nn.Module):
        def __init__(self, d_model=256, n_head=4, compress_ratio=4):
            super().__init__()
            self.local_attention = torch.nn.MultiheadAttention(d_model, n_head)
            self.compressed_memory = torch.nn.Linear(d_model, d_model//compress_ratio)
            self.fc = torch.nn.Linear(d_model + d_model//compress_ratio, 4)
            
        def forward(self, x):
            # Local attention
            local_ctx, _ = self.local_attention(x[-16:], x[-16:], x[-16:])
            
            # Compressed memory
            compressed = torch.relu(self.compressed_memory(x[:-16]))
            compressed = compressed.mean(dim=0)
            
            return torch.softmax(self.fc(torch.cat([local_ctx[-1], compressed])), dim=-1)

    class WorkerNetwork(torch.nn.Module):
        def __init__(self, d_model=256, n_head=4):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(d_model, n_head)
            self.compressed_proj = torch.nn.Linear(d_model, d_model//2)
            self.action_predictor = torch.nn.Linear(d_model//2, 8)
            
        def forward(self, x):
            compressed = torch.relu(self.compressed_proj(x))
            attn_out, _ = self.attention(compressed, compressed, compressed)
            return self.action_predictor(attn_out.mean(dim=0))

    def visualize_compressed_attention(self, historical_data: pd.DataFrame) -> None:
        """Visualize compressed attention patterns"""
        import matplotlib.pyplot as plt
        
        with torch.no_grad():
            states = self._preprocess_data(historical_data)
            local, compressed = self.manager(states)
            
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(local.squeeze().cpu().numpy(), cmap='viridis', aspect='auto')
        plt.title("Local Attention Patterns")
        
        plt.subplot(1, 2, 2) 
        plt.imshow(compressed.squeeze().cpu().numpy(), cmap='plasma', aspect='auto')
        plt.title("Compressed Memory Patterns")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

class TransformerXLHRL:
    """Transformer-XL based Hierarchical RL"""
    
    class ManagerNetwork(torch.nn.Module):
        def __init__(self, d_model=256, n_head=4, mem_len=100):
            super().__init__()
            self.mem_len = mem_len
            self.transformer = torch.nn.TransformerXL(
                d_model=d_model,
                n_head=n_head,
                mem_len=mem_len
            )
            self.goal_predictor = torch.nn.Linear(d_model, 4)
            
        def forward(self, x, mems=None):
            h, new_mems = self.transformer(x, mems)
            return torch.softmax(self.goal_predictor(h[:, -1]), dim=-1), new_mems
    
    class WorkerNetwork(torch.nn.Module):
        def __init__(self, d_model=256, n_head=4):
            super().__init__()
            self.transformer = torch.nn.TransformerXL(
                d_model=d_model,
                n_head=n_head
            )
            self.action_predictor = torch.nn.Linear(d_model, 8)
            
        def forward(self, x, goal, mems=None):
            h, new_mems = self.transformer(torch.cat([x, goal], dim=-1), mems)
            return self.action_predictor(h[:, -1]), new_mems

    def __init__(self, alert_system: AlertSystem):
        self.manager = self.ManagerNetwork()
        self.worker = self.WorkerNetwork()
        self.memory = self.MemoryBank(capacity=1000)
        
    class MemoryBank:
        """Stores long-term context for Transformer-XL"""
        def __init__(self, capacity=1000):
            self.capacity = capacity
            self.mems = []
            
        def store(self, mem):
            self.mems = (self.mems + mem)[-self.capacity:]
            
        def sample(self, size):
            return random.sample(self.mems, min(size, len(self.mems)))

class Backtester:
    """System for historical performance testing of trading models"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.commission_rate = 0.001  # 0.1% commission per trade
        self.slippage_rate = 0.0005   # 0.05% slippage per trade
        self.alert_system = AlertSystem()
        self.hrl_manager = HRLManager(self.alert_system)
        self.hrl_worker = HRLWorker(self.hrl_manager)
        self.neural_hrl = NeuralHRL(self.alert_system)
        
    def run_backtest(self, 
                    model: PricePredictor,
                    historical_data: pd.DataFrame,
                    lookback_period: int = 60,
                    test_period: int = 365) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            model: Price prediction model
            historical_data: DataFrame containing historical price data
            lookback_period: Number of days for model training
            test_period: Number of days to test
            
        Returns:
            Dictionary containing backtest results and performance metrics
        """
        try:
            # Prepare data
            historical_data = historical_data.sort_index()
            start_date = historical_data.index[0] + timedelta(days=lookback_period)
            end_date = start_date + timedelta(days=test_period)
            
            # Initialize portfolio
            portfolio = {
                'cash': self.initial_capital,
                'shares': 0,
                'value': self.initial_capital,
                'positions': [],
                'returns': []
            }
            
            # Walk forward testing
            current_date = start_date
            alerts_history = []
            while current_date <= end_date:
                # Get training data
                train_data = historical_data[historical_data.index < current_date]
                if len(train_data) < lookback_period:
                    current_date += timedelta(days=1)
                    continue
                    
                # Train model
                model.train_models(train_data[-lookback_period:])
                
                # Get prediction for next day
                test_data = historical_data[historical_data.index == current_date]
                if len(test_data) == 0:
                    current_date += timedelta(days=1)
                    continue
                    
                prediction = model.predict(test_data)
                actual_price = test_data['close'].values[0]
                
                # Execute trading strategy
                self._execute_trade(prediction, actual_price, portfolio)
                
                # Update portfolio value
                portfolio['value'] = portfolio['cash'] + portfolio['shares'] * actual_price
                portfolio['returns'].append((portfolio['value'] - self.initial_capital) / self.initial_capital)
                
                # Check alerts
                market_state = {
                    'timestamp': current_date,
                    'price': actual_price,
                    'portfolio_value': portfolio['value'],
                    'positions': portfolio['positions'],
                    'returns': portfolio['returns'],
                    'indicators': self._calculate_technical_indicators(historical_data[:current_date])
                }
                triggered = self.alert_system.check_alerts(market_state)
                alerts_history.extend(triggered)
                
                current_date += timedelta(days=1)
                
            # Calculate performance metrics
            results = self._calculate_performance_metrics(portfolio)
            
            # Add alerts to results
            results['alerts'] = {
                'all_alerts': alerts_history,
                'alert_stats': self._calculate_alert_statistics(alerts_history)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def _execute_trade(self, prediction: float, actual_price: float, portfolio: Dict) -> None:
        """Updated trade execution with confidence-based sizing"""
        try:
            confidence = self.alert_system.triggered_alerts[-1]['data'].get('pattern_confidence', 0.5)
            position_size = self.alert_system.calculate_confidence_position(confidence)
            
            if prediction > actual_price * 1.01:
                self._buy(actual_price, portfolio, position_size)
            elif prediction < actual_price * 0.99:
                self._sell(actual_price, portfolio, position_size)
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    def _buy(self, price: float, portfolio: Dict, position_size: float) -> None:
        """Confidence-scaled buy order"""
        try:
            commission = price * self.commission_rate
            slippage = price * self.slippage_rate
            total_cost = price + commission + slippage
            
            max_investment = portfolio['cash'] * position_size
            shares_to_buy = max_investment // total_cost
            portfolio['cash'] -= shares_to_buy * total_cost
            portfolio['shares'] += shares_to_buy
            portfolio['positions'].append({
                'type': 'buy',
                'price': price,
                'shares': shares_to_buy,
                'timestamp': datetime.now()
            })
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
    
    def _sell(self, price: float, portfolio: Dict, position_size: float) -> None:
        """Confidence-scaled sell order"""
        try:
            if portfolio['shares'] > 0:
                commission = price * self.commission_rate
                slippage = price * self.slippage_rate
                total_cost = price - commission - slippage
                
                shares_to_sell = int(portfolio['shares'] * position_size)
                portfolio['cash'] += shares_to_sell * total_cost
                portfolio['positions'].append({
                    'type': 'sell',
                    'price': price,
                    'shares': shares_to_sell,
                    'timestamp': datetime.now()
                })
                portfolio['shares'] -= shares_to_sell
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
    
    def _calculate_performance_metrics(self, portfolio: Dict) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            portfolio: Portfolio state after backtest
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            returns = np.array(portfolio['returns'])
            total_return = returns[-1]
            annualized_return = (1 + total_return) ** (365 / len(returns)) - 1
            max_drawdown = (returns.max() - returns.min()) / returns.max()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'final_value': portfolio['value'],
                'num_trades': len(portfolio['positions']),
                'portfolio_history': portfolio
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def plot_backtest_results(self, results: Dict) -> None:
        """
        Plot backtest results.
        
        Args:
            results: Backtest results from run_backtest()
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot portfolio value
            plt.subplot(2, 1, 1)
            plt.plot(results['portfolio_history']['returns'], label='Portfolio Value')
            plt.title('Portfolio Performance')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            
            # Plot trades
            plt.subplot(2, 1, 2)
            for trade in results['portfolio_history']['positions']:
                color = 'g' if trade['type'] == 'buy' else 'r'
                plt.axvline(x=trade['timestamp'], color=color, alpha=0.3)
            plt.title('Trade Execution')
            plt.xlabel('Time')
            plt.ylabel('Trade')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}")

    def _analyze_trade_reasons(self, portfolio: Dict) -> Dict[str, int]:
        """Analyze reasons for trades"""
        try:
            reasons = {}
            for trade in portfolio['positions']:
                reason = trade.get('reason', 'unknown')
                reasons[reason] = reasons.get(reason, 0) + 1
            return reasons
        except Exception as e:
            logger.error(f"Error analyzing trade reasons: {e}")
            return {}

    def plot_enhanced_results(self, results: Dict) -> None:
        """
        Plot enhanced backtest results with additional visualizations.
        
        Args:
            results: Backtest results from run_backtest()
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with multiple subplots
            plt.figure(figsize=(15, 10))
            
            # Portfolio Value
            plt.subplot(3, 2, 1)
            plt.plot(results['portfolio_history']['returns'], label='Portfolio Value')
            plt.title('Portfolio Performance')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            
            # Trade Reasons
            plt.subplot(3, 2, 2)
            reasons = results['trade_reasons']
            plt.bar(reasons.keys(), reasons.values())
            plt.title('Trade Reasons')
            plt.xlabel('Reason')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Drawdown
            plt.subplot(3, 2, 3)
            returns = np.array(results['portfolio_history']['returns'])
            drawdown = (returns - returns.cummax()) / returns.cummax()
            plt.plot(drawdown, label='Drawdown', color='r')
            plt.title('Drawdown')
            plt.xlabel('Time')
            plt.ylabel('Drawdown')
            plt.legend()
            
            # Risk Metrics
            plt.subplot(3, 2, 4)
            metrics = results['risk_metrics']
            plt.bar(metrics.keys(), metrics.values())
            plt.title('Risk Metrics')
            plt.xlabel('Metric')
            plt.ylabel('Value')
            
            # Transaction Costs
            plt.subplot(3, 2, 5)
            costs = results['transaction_costs']
            plt.pie([costs['total'], self.initial_capital - costs['total']],
                   labels=['Transaction Costs', 'Remaining Capital'],
                   autopct='%1.1f%%')
            plt.title('Transaction Cost Analysis')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting enhanced results: {e}")

    def optimize_strategy_parameters(self,
                                    model: PricePredictor,
                                    historical_data: pd.DataFrame,
                                    parameter_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            model: Price prediction model
            historical_data: Historical price data
            parameter_grid: Dictionary of parameters to optimize
            
        Returns:
            Dictionary containing best parameters and performance
        """
        try:
            from itertools import product
            
            best_params = {}
            best_performance = -np.inf
            
            # Generate all parameter combinations
            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())
            combinations = list(product(*param_values))
            
            # Test each combination
            for combination in combinations:
                # Set parameters
                params = dict(zip(param_names, combination))
                self.stop_loss = params.get('stop_loss', self.stop_loss)
                self.take_profit = params.get('take_profit', self.take_profit)
                self.max_position_size = params.get('max_position_size', self.max_position_size)
                
                # Run backtest
                results = self.run_backtest(model, historical_data)
                
                # Evaluate performance
                performance = results['annualized_return'] / results['risk_metrics']['max_drawdown']
                
                # Update best parameters
                if performance > best_performance:
                    best_performance = performance
                    best_params = params
                    
            return {
                'best_params': best_params,
                'best_performance': best_performance
            }
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return {}

    def bayesian_optimization(self,
                             model: PricePredictor,
                             historical_data: pd.DataFrame,
                             param_space: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Bayesian optimization with Gaussian Processes.
        
        Args:
            model: Price prediction model instance
            historical_data: DataFrame containing historical price data with features
            param_space: List of tuples defining parameter spaces:
                - Each tuple: (name, type, min_value, max_value)
                - type: 'real' for continuous, 'integer' for discrete
                - Example: [('stop_loss', 'real', 0.01, 0.1), 
                           ('take_profit', 'real', 0.05, 0.2)]
        
        Returns:
            Dictionary containing:
                - best_params: Dictionary of optimized parameters
                - best_performance: Best achieved performance score
                - parameter_importance: Dictionary of parameter importance scores
        """
        try:
            # Define parameter space
            space = []
            for param in param_space:
                if param[1] == 'real':
                    space.append(Real(param[2], param[3], name=param[0]))
                elif param[1] == 'integer':
                    space.append(Integer(param[2], param[3], name=param[0]))
            
            @use_named_args(space)
            def objective(**params):
                # Set parameters
                self.stop_loss = params.get('stop_loss', self.stop_loss)
                self.take_profit = params.get('take_profit', self.take_profit)
                self.max_position_size = params.get('max_position_size', self.max_position_size)
                
                # Run backtest
                results = self.run_backtest(model, historical_data)
                
                # Return negative performance for minimization
                return - (results['annualized_return'] / results['risk_metrics']['max_drawdown'])
            
            # Run optimization
            res = gp_minimize(objective, space, n_calls=50, random_state=42)
            
            # Get best parameters
            best_params = {param.name: value for param, value in zip(space, res.x)}
            
            # Get parameter importance
            importance = self._calculate_parameter_importance(res)
            
            return {
                'best_params': best_params,
                'best_performance': -res.fun,
                'parameter_importance': importance
            }
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return {}

    def _calculate_parameter_importance(self, optimization_result) -> Dict[str, float]:
        """
        Calculate parameter importance from Bayesian optimization results.
        
        Args:
            optimization_result: Result object from gp_minimize
            
        Returns:
            Dictionary of parameter importance scores
        """
        try:
            importance = {}
            for i, param in enumerate(optimization_result.space):
                # Calculate importance based on how much the objective changes
                # when this parameter is varied
                values = [x[i] for x in optimization_result.x_iters]
                obj_values = optimization_result.func_vals
                correlation = np.corrcoef(values, obj_values)[0, 1]
                importance[param.name] = abs(correlation)
            return importance
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return {}

    def walk_forward_optimization(self,
                                 model: PricePredictor,
                                 historical_data: pd.DataFrame,
                                 param_space: Dict[str, List[Any]],
                                 n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform walk-forward optimization with time series cross-validation.
        
        Args:
            model: Price prediction model instance
            historical_data: DataFrame containing historical price data
            param_space: Dictionary of parameters to optimize:
                - Key: parameter name
                - Value: list of possible values
                - Example: {'stop_loss': [0.01, 0.02, 0.03],
                            'take_profit': [0.05, 0.1, 0.15]}
            n_splits: Number of time series splits (default: 5)
            
        Returns:
            Dictionary containing:
                - most_consistent_params: Parameters that performed best across folds
                - performance_history: List of performance metrics for each fold
                - param_counts: Frequency of each parameter combination
                - parameter_stability: Dictionary of parameter stability scores
        """
        try:
            # Initialize time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            performance_history = []
            best_params_history = []
            
            for train_index, test_index in tscv.split(historical_data):
                # Split data
                train_data = historical_data.iloc[train_index]
                test_data = historical_data.iloc[test_index]
                
                # Optimize parameters on training data
                opt_result = self.optimize_strategy_parameters(model, train_data, param_space)
                
                # Test on validation data
                self.stop_loss = opt_result['best_params']['stop_loss']
                self.take_profit = opt_result['best_params']['take_profit']
                self.max_position_size = opt_result['best_params']['max_position_size']
                
                test_results = self.run_backtest(model, test_data)
                
                # Store results
                performance_history.append({
                    'train_performance': opt_result['best_performance'],
                    'test_performance': test_results['annualized_return'] / test_results['risk_metrics']['max_drawdown'],
                    'params': opt_result['best_params']
                })
                best_params_history.append(opt_result['best_params'])
                
            # Find most consistent parameters
            param_counts = {}
            for params in best_params_history:
                param_key = tuple(params.items())
                param_counts[param_key] = param_counts.get(param_key, 0) + 1
            most_consistent_params = max(param_counts, key=param_counts.get)
            
            # Calculate parameter stability
            stability = self._calculate_parameter_stability(best_params_history)
            
            return {
                'most_consistent_params': dict(most_consistent_params),
                'performance_history': performance_history,
                'param_counts': param_counts,
                'parameter_stability': stability
            }
        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            return {}

    def _calculate_parameter_stability(self, best_params_history: List[Dict]) -> Dict[str, float]:
        """
        Calculate parameter stability across walk-forward folds.
        
        Args:
            best_params_history: List of best parameters from each fold
            
        Returns:
            Dictionary of parameter stability scores
        """
        try:
            stability = {}
            param_names = best_params_history[0].keys()
            
            for param in param_names:
                values = [params[param] for params in best_params_history]
                # Calculate stability as 1 - (std / mean)
                mean_val = np.mean(values)
                std_val = np.std(values)
                stability[param] = 1 - (std_val / mean_val) if mean_val != 0 else 0
            return stability
        except Exception as e:
            logger.error(f"Error calculating parameter stability: {e}")
            return {}

    def genetic_optimization(self,
                           model: PricePredictor,
                           historical_data: pd.DataFrame,
                           param_space: Dict[str, List[Any]],
                           population_size: int = 20,
                           generations: int = 10) -> Dict[str, Any]:
        """
        Optimize strategy parameters using genetic algorithm with evolutionary approach.
        
        Args:
            model: Price prediction model instance
            historical_data: DataFrame containing historical price data
            param_space: Dictionary of parameters to optimize:
                - Key: parameter name
                - Value: list of possible values
            population_size: Size of the population (default: 20)
            generations: Number of generations to evolve (default: 10)
            
        Returns:
            Dictionary containing:
                - best_params: Optimized parameters
                - best_performance: Best achieved performance
                - parameter_evolution: Dictionary of parameter evolution statistics
        """
        try:
            from deap import base, creator, tools, algorithms
            import random
            
            # Define fitness and individual
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Initialize toolbox
            toolbox = base.Toolbox()
            
            # Define parameter attributes
            param_names = list(param_space.keys())
            for i, (name, values) in enumerate(param_space.items()):
                toolbox.register(f"attr_{i}", random.choice, values)
            
            # Create individual and population
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           (getattr(toolbox, f"attr_{i}") for i in range(len(param_space))), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Define evaluation function
            def evaluate(individual):
                # Set parameters
                params = dict(zip(param_names, individual))
                self.stop_loss = params['stop_loss']
                self.take_profit = params['take_profit']
                self.max_position_size = params['max_position_size']
                
                # Run backtest
                results = self.run_backtest(model, historical_data)
                
                # Return performance
                return (results['annualized_return'] / results['risk_metrics']['max_drawdown'],)
            
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutUniformInt, low=[min(v) for v in param_space.values()],
                           up=[max(v) for v in param_space.values()], indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Create population
            population = toolbox.population(n=population_size)
            
            # Run genetic algorithm
            algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2,
                                ngen=generations, verbose=False)
            
            # Get best individual
            best_individual = tools.selBest(population, k=1)[0]
            best_params = dict(zip(param_names, best_individual))
            best_performance = evaluate(best_individual)[0]
            
            # Analyze parameter evolution
            evolution = self._analyze_parameter_evolution(population, param_names)
            
            return {
                'best_params': best_params,
                'best_performance': best_performance,
                'parameter_evolution': evolution
            }
        except Exception as e:
            logger.error(f"Error in genetic optimization: {e}")
            return {}

    def _analyze_parameter_evolution(self, population, param_names) -> Dict[str, Any]:
        """
        Analyze how parameters evolved during genetic optimization.
        
        Args:
            population: Final population from genetic algorithm
            param_names: List of parameter names
            
        Returns:
            Dictionary containing:
                - mean_values: Mean value of each parameter in final population
                - std_values: Standard deviation of each parameter
                - diversity: Measure of population diversity for each parameter
        """
        try:
            evolution = {
                'mean_values': {},
                'std_values': {},
                'diversity': {}
            }
            
            for i, param in enumerate(param_names):
                values = [ind[i] for ind in population]
                evolution['mean_values'][param] = np.mean(values)
                evolution['std_values'][param] = np.std(values)
                # Diversity measure: 1 - (max - min) / range
                param_range = max(values) - min(values)
                evolution['diversity'][param] = 1 - (param_range / (max(values) - min(values))) if param_range != 0 else 1
            return evolution
        except Exception as e:
            logger.error(f"Error analyzing parameter evolution: {e}")
            return {}

    def sensitivity_analysis(self,
                            model: PricePredictor,
                            historical_data: pd.DataFrame,
                            base_params: Dict[str, Any],
                            param_ranges: Dict[str, Tuple[float, float]],
                            num_samples: int = 100) -> Dict[str, Any]:
        """
        Perform sensitivity analysis to understand parameter impact on performance.
        
        Args:
            model: Price prediction model instance
            historical_data: DataFrame containing historical price data
            base_params: Dictionary of base parameter values
            param_ranges: Dictionary of parameter ranges to test:
                - Key: parameter name
                - Value: tuple of (min_value, max_value)
            num_samples: Number of samples to generate per parameter
            
        Returns:
            Dictionary containing:
                - sensitivity_scores: Dictionary of parameter sensitivity scores
                - parameter_effects: Dictionary of parameter effect sizes
                - interaction_effects: Dictionary of parameter interaction effects
        """
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance
            
            # Generate parameter samples
            samples = []
            for param, (min_val, max_val) in param_ranges.items():
                samples.append(np.linspace(min_val, max_val, num_samples))
            
            # Create parameter combinations
            param_names = list(param_ranges.keys())
            param_combinations = np.array(np.meshgrid(*samples)).T.reshape(-1, len(param_names))
            
            # Evaluate performance for each combination
            performances = []
            for combination in param_combinations:
                # Set parameters
                params = dict(zip(param_names, combination))
                for param, value in base_params.items():
                    if param not in params:
                        params[param] = value
                
                # Run backtest
                self.stop_loss = params.get('stop_loss', self.stop_loss)
                self.take_profit = params.get('take_profit', self.take_profit)
                self.max_position_size = params.get('max_position_size', self.max_position_size)
                
                results = self.run_backtest(model, historical_data)
                performance = results['annualized_return'] / results['risk_metrics']['max_drawdown']
                performances.append(performance)
            
            # Train surrogate model to analyze parameter importance
            surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
            surrogate.fit(param_combinations, performances)
            
            # Calculate sensitivity scores
            importance = permutation_importance(surrogate, param_combinations, performances, n_repeats=10)
            sensitivity_scores = {param: score for param, score in zip(param_names, importance.importances_mean)}
            
            # Calculate parameter effects
            param_effects = {}
            for i, param in enumerate(param_names):
                param_values = param_combinations[:, i]
                effect_size = np.corrcoef(param_values, performances)[0, 1]
                param_effects[param] = effect_size
            
            # Calculate interaction effects
            interaction_effects = {}
            for i, param1 in enumerate(param_names):
                for j, param2 in enumerate(param_names):
                    if i < j:
                        interaction = param_combinations[:, i] * param_combinations[:, j]
                        interaction_effect = np.corrcoef(interaction, performances)[0, 1]
                        interaction_effects[f"{param1}_{param2}"] = interaction_effect
            
            return {
                'sensitivity_scores': sensitivity_scores,
                'parameter_effects': param_effects,
                'interaction_effects': interaction_effects
            }
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            return {}

    def plot_sensitivity_analysis(self, results: Dict) -> None:
        """
        Visualize sensitivity analysis results.
        
        Args:
            results: Sensitivity analysis results from sensitivity_analysis()
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure
            plt.figure(figsize=(15, 10))
            
            # Sensitivity Scores
            plt.subplot(2, 2, 1)
            sns.barplot(x=list(results['sensitivity_scores'].keys()),
                       y=list(results['sensitivity_scores'].values()))
            plt.title('Parameter Sensitivity Scores')
            plt.xlabel('Parameter')
            plt.ylabel('Sensitivity Score')
            plt.xticks(rotation=45)
            
            # Parameter Effects
            plt.subplot(2, 2, 2)
            sns.barplot(x=list(results['parameter_effects'].keys()),
                       y=list(results['parameter_effects'].values()))
            plt.title('Parameter Effect Sizes')
            plt.xlabel('Parameter')
            plt.ylabel('Effect Size')
            plt.xticks(rotation=45)
            
            # Interaction Effects
            plt.subplot(2, 2, 3)
            sns.heatmap(pd.DataFrame.from_dict(results['interaction_effects'], orient='index').T,
                       annot=True, cmap='coolwarm', center=0)
            plt.title('Parameter Interaction Effects')
            plt.xlabel('Parameter Pairs')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting sensitivity analysis: {e}")

    def sensitivity_guided_optimization(self,
                                        model: PricePredictor,
                                        historical_data: pd.DataFrame,
                                        sensitivity_results: Dict[str, Any],
                                        base_params: Dict[str, Any],
                                        optimization_method: str = 'bayesian',
                                        top_n: int = 3,
                                        objectives: List[str] = ['return', 'risk'],
                                        constraints: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Enhanced optimization with multi-objective support and constraints.
        
        Args:
            model: Price prediction model instance
            historical_data: Historical price data
            sensitivity_results: From sensitivity_analysis()
            base_params: Base parameter values
            optimization_method: 'bayesian' or 'genetic'
            top_n: Number of top parameters to optimize
            objectives: List of objectives ['return', 'risk', 'sharpe']
            constraints: Dictionary of parameter constraints {param: (min, max)}
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Get top parameters and create adjusted ranges
            sorted_params = sorted(sensitivity_results['sensitivity_scores'].items(),
                                 key=lambda x: x[1], reverse=True)[:top_n]
            top_params = [param[0] for param in sorted_params]
            
            # Sophisticated range adjustment using interaction effects
            param_ranges = {}
            for param in top_params:
                sensitivity = sensitivity_results['sensitivity_scores'][param]
                interaction_effect = np.mean([v for k,v in sensitivity_results['interaction_effects'].items()
                                             if param in k])
                
                # Dynamic range adjustment based on sensitivity and interactions
                range_factor = 0.1 + 0.3 * sensitivity + 0.2 * abs(interaction_effect)
                base_value = base_params[param]
                min_val = base_value * (1 - range_factor)
                max_val = base_value * (1 + range_factor)
                
                # Apply constraints if provided
                if constraints and param in constraints:
                    min_val = max(min_val, constraints[param][0])
                    max_val = min(max_val, constraints[param][1])
                
                param_ranges[param] = (min_val, max_val)
            
            # Multi-objective optimization
            if len(objectives) > 1:
                return self._multi_objective_optimization(
                    model, historical_data, param_ranges, 
                    objectives, optimization_method
                )
            
            # Single objective optimization
            if optimization_method == 'bayesian':
                param_space = [
                    (param, 'real', param_ranges[param][0], param_ranges[param][1])
                    for param in top_params
                ]
                return self.bayesian_optimization(model, historical_data, param_space)
            
            elif optimization_method == 'genetic':
                param_space = {
                    param: np.linspace(param_ranges[param][0], param_ranges[param][1], 20).tolist()
                    for param in top_params
                }
                return self.genetic_optimization(model, historical_data, param_space)
            
            else:
                raise ValueError(f"Unsupported method: {optimization_method}")
                
        except Exception as e:
            logger.error(f"Error in enhanced optimization: {e}")
            return {}
    
    def _multi_objective_optimization(self,
                                      model: PricePredictor,
                                      historical_data: pd.DataFrame,
                                      param_ranges: Dict[str, Tuple[float, float]],
                                      objectives: List[str],
                                      method: str = 'genetic') -> Dict[str, Any]:
        """
        Perform multi-objective optimization using NSGA-II algorithm.
        
        Args:
            model: Price prediction model
            historical_data: Historical price data
            param_ranges: Parameter ranges to optimize
            objectives: List of objectives to optimize
            method: Optimization method (genetic only for multi-objective)
            
        Returns:
            Dictionary containing Pareto front solutions
        """
        try:
            from deap import algorithms, base, creator, tools
            import random
            
            # Define multi-objective fitness
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))
            creator.create("Individual", list, fitness=creator.FitnessMulti)
            
            # Parameter setup
            param_names = list(param_ranges.keys())
            toolbox = base.Toolbox()
            
            # Attribute generator
            for i, (param, (min_val, max_val)) in enumerate(param_ranges.items()):
                toolbox.register(f"attr_{i}", random.uniform, min_val, max_val)
            
            # Structure initializers
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           (getattr(toolbox, f"attr_{i}") for i in range(len(param_ranges))), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Evaluation function
            def evaluate(individual):
                params = dict(zip(param_names, individual))
                self.stop_loss = params.get('stop_loss', self.stop_loss)
                self.take_profit = params.get('take_profit', self.take_profit)
                self.max_position_size = params.get('max_position_size', self.max_position_size)
                
                results = self.run_backtest(model, historical_data)
                
                obj_values = []
                for obj in objectives:
                    if obj == 'return':
                        obj_values.append(results['annualized_return'])
                    elif obj == 'risk':
                        obj_values.append(results['risk_metrics']['max_drawdown'])
                    elif obj == 'sharpe':
                        obj_values.append(results['sharpe_ratio'])
                
                return tuple(obj_values)
            
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                           low=[r[0] for r in param_ranges.values()],
                           up=[r[1] for r in param_ranges.values()], eta=20.0)
            toolbox.register("mutate", tools.mutPolynomialBounded,
                           low=[r[0] for r in param_ranges.values()],
                           up=[r[1] for r in param_ranges.values()], eta=20.0, indpb=0.1)
            toolbox.register("select", tools.selNSGA2)
            
            # Run optimization
            population = toolbox.population(n=100)
            algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=100,
                                      cxpb=0.7, mutpb=0.3, ngen=50, verbose=False)
            
            # Extract Pareto front
            pareto_front = tools.sortNondominated(population, len(population))[0]
            
            # Process results
            solutions = []
            for ind in pareto_front:
                params = dict(zip(param_names, ind))
                performance = ind.fitness.values
                solutions.append({
                    'parameters': params,
                    'objectives': dict(zip(objectives, performance))
                })
            
            return {
                'pareto_front': solutions,
                'objectives': objectives,
                'optimization_method': method
            }
        except Exception as e:
            logger.error(f"Error in multi-objective optimization: {e}")
            return {}
    
    def plot_pareto_front(self, optimization_results: Dict) -> None:
        """
        Visualize Pareto front from multi-objective optimization.
        
        Args:
            optimization_results: Results from multi-objective optimization
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            solutions = optimization_results['pareto_front']
            objectives = optimization_results['objectives']
            
            if len(objectives) == 2:
                # 2D plot
                plt.figure(figsize=(10, 6))
                x = [s['objectives'][objectives[0]] for s in solutions]
                y = [s['objectives'][objectives[1]] for s in solutions]
                plt.scatter(x, y)
                plt.xlabel(objectives[0])
                plt.ylabel(objectives[1])
                plt.title('Pareto Front')
                
            elif len(objectives) == 3:
                # 3D plot
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                x = [s['objectives'][objectives[0]] for s in solutions]
                y = [s['objectives'][objectives[1]] for s in solutions]
                z = [s['objectives'][objectives[2]] for s in solutions]
                ax.scatter(x, y, z)
                ax.set_xlabel(objectives[0])
                ax.set_ylabel(objectives[1])
                ax.set_zlabel(objectives[2])
                plt.title('3D Pareto Front')
                
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting Pareto front: {e}")

    def calculate_risk_adjusted_metrics(self, 
                                       returns: np.ndarray, 
                                       benchmark_returns: np.ndarray = None,
                                       risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted return metrics.
        
        Args:
            returns: Strategy returns array
            benchmark_returns: Benchmark returns array (optional)
            risk_free_rate: Annualized risk-free rate
            
        Returns:
            Dictionary of risk-adjusted performance metrics
        """
        try:
            metrics = {}
            excess_returns = returns - risk_free_rate/252
            
            # Basic metrics
            metrics['annualized_return'] = np.mean(returns) * 252
            metrics['annualized_volatility'] = np.std(returns) * np.sqrt(252)
            
            # Downside risk metrics
            downside_returns = returns[returns < risk_free_rate/252]
            metrics['downside_deviation'] = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Common ratios
            metrics['sharpe_ratio'] = self._calculate_sharpe(returns, risk_free_rate)
            metrics['sortino_ratio'] = self._calculate_sortino(returns, risk_free_rate)
            metrics['calmar_ratio'] = self._calculate_calmar(returns)
            
            if benchmark_returns is not None:
                # Benchmark-relative metrics
                metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns, risk_free_rate))
            
            # Advanced metrics
            metrics['omega_ratio'] = self._calculate_omega_ratio(returns, risk_free_rate)
            metrics['k_ratio'] = self._calculate_k_ratio(returns)
            metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
            metrics['gain_to_pain_ratio'] = self._calculate_gain_to_pain(returns)
            metrics['m2_ratio'] = self._calculate_m2_ratio(returns, metrics['sharpe_ratio'])
            
            # New advanced metrics
            metrics['upside_potential_ratio'] = self._calculate_upside_potential_ratio(returns, risk_free_rate)
            metrics['sterling_ratio'] = self._calculate_sterling_ratio(returns)
            metrics['probabilistic_var'] = self._calculate_probabilistic_var(returns)
            metrics['expected_shortfall'] = self._calculate_expected_shortfall(returns)
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}

    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate annualized Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate annualized Sortino ratio"""
        downside_returns = returns[returns < risk_free_rate/252]
        if len(downside_returns) == 0:
            return np.nan
        downside_risk = np.std(downside_returns) * np.sqrt(252)
        excess_return = np.mean(returns) * 252 - risk_free_rate
        return excess_return / downside_risk

    def _calculate_calmar(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative.max() - cumulative.min()) / cumulative.max()
        return np.mean(returns) * 252 / max_drawdown if max_drawdown != 0 else np.nan

    def _calculate_benchmark_metrics(self, 
                                    returns: np.ndarray, 
                                    benchmark_returns: np.ndarray,
                                    risk_free_rate: float) -> Dict[str, float]:
        """Calculate benchmark-relative metrics"""
        metrics = {}
        excess_returns = returns - benchmark_returns
        
        # Beta and Alpha
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        alpha = (np.mean(returns) - risk_free_rate/252) - beta * (np.mean(benchmark_returns) - risk_free_rate/252)
        
        metrics['beta'] = beta
        metrics['alpha'] = alpha * 252  # Annualized alpha
        metrics['treynor_ratio'] = (np.mean(returns) * 252 - risk_free_rate) / beta if beta not in [0, np.nan] else np.nan
        metrics['information_ratio'] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        metrics['jensens_alpha'] = alpha * 252
        metrics['tracking_error'] = np.std(excess_returns) * np.sqrt(252)
        
        return metrics

    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        threshold = threshold/252  # Convert annual to daily
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        return np.sum(gains) / np.sum(losses) if np.sum(losses) != 0 else np.nan

    def _calculate_k_ratio(self, returns: np.ndarray) -> float:
        """Calculate K-ratio (performance persistence)"""
        from scipy.stats import linregress
        time_periods = np.arange(len(returns))
        slope, _, _, _, std_err = linregress(time_periods, returns)
        return slope / std_err if std_err != 0 else np.nan

    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate Tail Ratio (95th percentile / 5th percentile)"""
        return np.percentile(returns, 95) / abs(np.percentile(returns, 5))

    def _calculate_gain_to_pain(self, returns: np.ndarray) -> float:
        """Calculate Gain to Pain Ratio"""
        total_gain = np.sum(returns[returns > 0])
        total_loss = abs(np.sum(returns[returns < 0]))
        return total_gain / total_loss if total_loss != 0 else np.nan

    def _calculate_m2_ratio(self, returns: np.ndarray, sharpe: float) -> float:
        """Calculate Modigliani-Modigliani ratio"""
        benchmark_vol = np.std(returns) * np.sqrt(252)
        return sharpe * benchmark_vol + (np.mean(returns) * 252)

    def _calculate_upside_potential_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Upside Potential Ratio"""
        try:
            excess_returns = returns - risk_free_rate/252
            upside_returns = excess_returns[excess_returns > 0]
            downside_returns = excess_returns[excess_returns <= 0]
            
            if len(downside_returns) == 0 or len(upside_returns) == 0:
                return np.nan
                
            up_ratio = np.mean(upside_returns) / np.sqrt(np.mean(np.square(downside_returns)))
            return up_ratio * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating Upside Potential Ratio: {e}")
            return np.nan

    def _calculate_sterling_ratio(self, returns: np.ndarray, lookback: int = 12) -> float:
        """Calculate Sterling Ratio using average drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.rolling(lookback, min_periods=1).max()
            drawdowns = (rolling_max - cumulative) / rolling_max
            avg_drawdown = drawdowns.mean()
            return np.mean(returns) * 252 / avg_drawdown if avg_drawdown != 0 else np.nan
        except Exception as e:
            logger.error(f"Error calculating Sterling Ratio: {e}")
            return np.nan

    def _calculate_probabilistic_var(self, returns: np.ndarray) -> float:
        """Calculate Probabilistic VaR (95%)"""
        try:
            var_95 = np.percentile(returns, 5)
            return var_95
        except Exception as e:
            logger.error(f"Error calculating Probabilistic VaR: {e}")
            return np.nan

    def _calculate_expected_shortfall(self, returns: np.ndarray) -> float:
        """Calculate Expected Shortfall (95%)"""
        try:
            var_95 = np.percentile(returns, 5)
            es_95 = returns[returns <= var_95].mean()
            return es_95
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return np.nan

    def scenario_analysis(self, 
                         model: PricePredictor,
                         historical_data: pd.DataFrame,
                         scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform scenario-based risk analysis.
        
        Args:
            model: Price prediction model
            historical_data: Historical price data
            scenarios: Dictionary of scenarios to test:
                Example:
                {
                    'bull_market': {'filter': {'return_threshold': 0.005}},
                    'high_volatility': {'filter': {'volatility_threshold': 0.02}}
                }
                
        Returns:
            Dictionary of scenario analysis results
        """
        try:
            results = {}
            for scenario_name, params in scenarios.items():
                filtered_data = self._filter_scenario_data(historical_data, params.get('filter', {}))
                
                if len(filtered_data) < 100:  # Minimum data check
                    logger.warning(f"Not enough data for scenario: {scenario_name}")
                    continue
                
                # Run backtest with scenario data
                result = self.run_backtest(model, filtered_data)
                metrics = self.calculate_risk_adjusted_metrics(result['portfolio_history']['returns'])
                
                results[scenario_name] = {
                    'metrics': metrics,
                    'periods': len(filtered_data),
                    'start_date': filtered_data.index[0],
                    'end_date': filtered_data.index[-1]
                }
            
            return results
        except Exception as e:
            logger.error(f"Error in scenario analysis: {e}")
            return {}

    def _filter_scenario_data(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Filter historical data based on scenario parameters"""
        try:
            filtered = data.copy()
            
            if 'return_threshold' in filters:
                returns = data['close'].pct_change().dropna()
                filtered = filtered[returns.abs() > filters['return_threshold']]
                
            if 'volatility_threshold' in filters:
                volatility = data['close'].pct_change().rolling(20).std().dropna()
                filtered = filtered[volatility > filters['volatility_threshold']]
                
            return filtered.dropna()
        except Exception as e:
            logger.error(f"Error filtering scenario data: {e}")
            return pd.DataFrame()

    def probabilistic_risk_modeling(self,
                                   returns: np.ndarray,
                                   num_simulations: int = 1000,
                                   days_forward: int = 252,
                                   model_type: str = 'gbm',
                                   **model_params) -> Dict[str, Any]:
        """
        Enhanced probabilistic risk modeling with multiple distribution types.
        
        Args:
            returns: Historical returns array
            num_simulations: Number of simulations to run
            days_forward: Number of days to project forward
            model_type: 'gbm', 'student_t', or 'garch'
            model_params: Additional parameters for specific models
            
        Returns:
            Dictionary containing risk modeling results
        """
        try:
            from scipy.stats import t, norm
            from arch import arch_model
            
            simulations = np.zeros((num_simulations, days_forward))
            last_price = 100  # Normalized starting price
            
            if model_type == 'gbm':
                # Geometric Brownian Motion
                mu = np.mean(returns)
                sigma = np.std(returns)
                for i in range(num_simulations):
                    prices = [last_price]
                    for d in range(days_forward):
                        drift = mu - 0.5 * sigma**2
                        shock = sigma * norm.ppf(np.random.rand())
                        prices.append(prices[-1] * np.exp(drift + shock))
                    simulations[i] = prices[1:]
                    
            elif model_type == 'student_t':
                # Student's t-distribution model
                df, mu, sigma = t.fit(returns)
                for i in range(num_simulations):
                    prices = [last_price]
                    for d in range(days_forward):
                        ret = t.rvs(df, loc=mu, scale=sigma)
                        prices.append(prices[-1] * (1 + ret))
                    simulations[i] = prices[1:]
                    
            elif model_type == 'garch':
                # GARCH(1,1) model
                garch = arch_model(returns * 100, vol='Garch', p=1, q=1).fit(disp='off')
                omega = garch.params['omega']
                alpha = garch.params['alpha[1]']
                beta = garch.params['beta[1]']
                
                for i in range(num_simulations):
                    prices = [last_price]
                    vol = garch.conditional_volatility[-1]
                    for d in range(days_forward):
                        # Update volatility
                        resid = np.random.normal(0, vol)
                        vol = np.sqrt(omega + alpha * resid**2 + beta * vol**2)
                        # Generate return
                        ret = resid / 100  # Scale back returns
                        prices.append(prices[-1] * (1 + ret))
                    simulations[i] = prices[1:]
                    
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Calculate risk metrics
            final_returns = (simulations[:, -1] / last_price) - 1
            var_95 = np.percentile(final_returns, 5)
            es_95 = final_returns[final_returns <= var_95].mean()
            
            return {
                'simulations': simulations,
                'model_type': model_type,
                'var_95': var_95,
                'expected_shortfall_95': es_95,
                'probability_of_loss': np.mean(final_returns < 0),
                'return_distribution': {
                    'mean': np.mean(final_returns),
                    'std': np.std(final_returns),
                    'skewness': pd.Series(final_returns).skew(),
                    'kurtosis': pd.Series(final_returns).kurtosis()
                }
            }
        except Exception as e:
            logger.error(f"Error in probabilistic risk modeling: {e}")
            return {}

    def stress_testing(self,
                      model: PricePredictor,
                      historical_data: pd.DataFrame,
                      scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Perform stress testing with extreme market scenarios.
        
        Args:
            model: Price prediction model
            historical_data: Historical price data
            scenarios: Dictionary of stress scenarios:
                Example:
                {
                    'market_crash': {'shock_size': -0.3, 'volatility_increase': 3.0},
                    'liquidity_crisis': {'spread_increase': 5.0, 'volume_drop': 0.8}
                }
                
        Returns:
            Dictionary of stress test results
        """
        try:
            results = {}
            for scenario_name, params in scenarios.items():
                # Backup original parameters
                original_commission = self.commission_rate
                original_slippage = self.slippage_rate
                
                try:
                    # Apply stress parameters
                    self.commission_rate *= params.get('commission_increase', 1)
                    self.slippage_rate *= params.get('slippage_increase', 1)
                    
                    # Apply market shock
                    shocked_data = historical_data.copy()
                    if 'shock_size' in params:
                        shocked_data['close'] *= (1 + params['shock_size'])
                    
                    # Run backtest with stressed parameters
                    result = self.run_backtest(model, shocked_data)
                    metrics = self.calculate_risk_adjusted_metrics(result['portfolio_history']['returns'])
                    
                    results[scenario_name] = {
                        'metrics': metrics,
                        'drawdown': result['max_drawdown'],
                        'shock_size': params.get('shock_size', 0),
                        'liquidity_impact': result['transaction_costs']['total'] / self.initial_capital
                    }
                    
                finally:
                    # Restore original parameters
                    self.commission_rate = original_commission
                    self.slippage_rate = original_slippage
                    
            return results
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}

    def regime_switching_model(self,
                              returns: np.ndarray,
                              num_regimes: int = 2,
                              num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Implement Markov regime-switching model for risk analysis.
        
        Args:
            returns: Historical returns array
            num_regimes: Number of market regimes (2 or 3)
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary containing regime analysis results
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            
            # Fit HMM model
            returns = returns.reshape(-1, 1)
            model = GaussianHMM(n_components=num_regimes, covariance_type="diag", n_iter=1000)
            model.fit(returns)
            
            # Get regime parameters
            means = model.means_.flatten()
            variances = np.array([np.diag(cov) for cov in model.covars_]).flatten()
            transmat = model.transmat_
            
            # Generate regime-conditional simulations
            simulations = []
            for _ in range(num_simulations):
                regime = np.random.choice(num_regimes, p=model.startprob_)
                sim_returns = []
                for _ in range(252):  # 1 year simulation
                    regime = np.random.choice(num_regimes, p=transmat[regime])
                    ret = np.random.normal(means[regime], np.sqrt(variances[regime]))
                    sim_returns.append(ret)
                simulations.append(sim_returns)
            
            # Calculate risk metrics
            final_values = (1 + np.array(simulations)).cumprod(axis=1)[:, -1]
            var_95 = np.percentile(final_values, 5)
            es_95 = final_values[final_values <= var_95].mean()
            
            return {
                'regime_means': means.tolist(),
                'regime_variances': variances.tolist(),
                'transition_matrix': transmat.tolist(),
                'simulations': np.array(simulations),
                'var_95': var_95,
                'expected_shortfall_95': es_95,
                'regime_stationary': self._calculate_stationary_distribution(transmat)
            }
        except Exception as e:
            logger.error(f"Error in regime-switching model: {e}")
            return {}

    def _calculate_stationary_distribution(self, transmat: np.ndarray) -> np.ndarray:
        """Calculate stationary distribution of Markov chain"""
        try:
            eigvals, eigvecs = np.linalg.eig(transmat.T)
            stationary = eigvecs[:, np.argmin(np.abs(eigvals - 1))]
            stationary = stationary / stationary.sum()
            return stationary.real
        except Exception as e:
            logger.error(f"Error calculating stationary distribution: {e}")
            return np.array([])

    def plot_regime_analysis(self, regime_results: Dict) -> None:
        """Visualize regime-switching model results"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10))
            
            # Regime Means and Variances
            plt.subplot(2, 2, 1)
            x = np.arange(len(regime_results['regime_means']))
            plt.bar(x - 0.2, regime_results['regime_means'], 0.4, label='Means')
            plt.bar(x + 0.2, regime_results['regime_variances'], 0.4, label='Variances')
            plt.title('Regime Parameters')
            plt.xlabel('Regime')
            plt.legend()
            
            # Transition Matrix
            plt.subplot(2, 2, 2)
            plt.imshow(regime_results['transition_matrix'], cmap='Blues')
            plt.colorbar()
            plt.title('Transition Matrix')
            plt.xlabel('To Regime')
            plt.ylabel('From Regime')
            
            # Stationary Distribution
            plt.subplot(2, 2, 3)
            plt.pie(regime_results['regime_stationary'],
                   labels=[f'Regime {i+1}' for i in range(len(regime_results['regime_stationary']))],
                   autopct='%1.1f%%')
            plt.title('Stationary Distribution')
            
            # Simulation Paths
            plt.subplot(2, 2, 4)
            for i in range(min(50, len(regime_results['simulations']))):
                plt.plot(regime_results['simulations'][i], alpha=0.1)
            plt.title('Regime-Switching Simulations')
            plt.xlabel('Days')
            plt.ylabel('Return')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting regime analysis: {e}")

    def generate_risk_report(self, strategy_name: str) -> str:
        """
        Generate detailed risk-adjusted performance report.
        
        Args:
            strategy_name: Name of strategy to analyze
            
        Returns:
            Formatted report string
        """
        try:
            if strategy_name not in self.results or 'metrics' not in self.results[strategy_name]:
                return f"No data available for strategy: {strategy_name}"
            
            metrics = self.results[strategy_name]['metrics']
            report = [
                f"Risk-Adjusted Performance Report - {strategy_name}",
                "="*50,
                f"Annualized Return: {metrics['annualized_return']:.2%}",
                f"Annualized Volatility: {metrics['annualized_volatility']:.2%}",
                f"Maximum Drawdown: {metrics['max_drawdown']:.2%}",
                "",
                "Risk-Adjusted Ratios:",
                f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}",
                f"Sortino Ratio: {metrics.get('sortino_ratio', 'N/A'):.2f}",
                f"Calmar Ratio: {metrics.get('calmar_ratio', 'N/A'):.2f}",
                f"Omega Ratio: {metrics.get('omega_ratio', 'N/A'):.2f}",
                f"Information Ratio: {metrics.get('information_ratio', 'N/A'):.2f}",
                "",
                "Downside Risk Metrics:",
                f"Downside Deviation: {metrics.get('downside_deviation', 'N/A'):.2%}",
                f"Value at Risk (95%): {metrics.get('value_at_risk_95', 'N/A'):.2%}",
                f"Conditional VaR (95%): {metrics.get('conditional_var_95', 'N/A'):.2%}",
                f"Tail Ratio: {metrics.get('tail_ratio', 'N/A'):.2f}",
                "",
                "Advanced Metrics:",
                f"Gain to Pain Ratio: {metrics.get('gain_to_pain_ratio', 'N/A'):.2f}",
                f"K-Ratio: {metrics.get('k_ratio', 'N/A'):.2f}",
                f"Ulcer Index: {metrics.get('ulcer_index', 'N/A'):.2f}",
                "",
                "Advanced Risk Metrics:",
                f"Upside Potential Ratio: {metrics.get('upside_potential_ratio', 'N/A'):.2f}",
                f"Sterling Ratio: {metrics.get('sterling_ratio', 'N/A'):.2f}",
                f"Probabilistic VaR (95%): {metrics.get('probabilistic_var', 'N/A'):.2%}",
                f"Expected Shortfall (95%): {metrics.get('expected_shortfall', 'N/A'):.2%}"
            ]
            
            return "\n".join(report)
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return "Error generating risk report"

    def _calculate_alert_statistics(self, alerts: List[Dict]) -> Dict:
        """Calculate alert statistics"""
        stats = {
            'total_alerts': len(alerts),
            'alert_counts': {},
            'priority_distribution': {'low': 0, 'medium': 0, 'high': 0}
        }
        
        for alert in alerts:
            stats['alert_counts'][alert['name']] = stats['alert_counts'].get(alert['name'], 0) + 1
            stats['priority_distribution'][alert['priority']] += 1
            
        return stats

    def add_price_alert(self, 
                       name: str, 
                       threshold: float, 
                       direction: str = 'below',
                       priority: str = 'medium') -> None:
        """
        Add price-based alert.
        
        Args:
            name: Alert name
            threshold: Price threshold
            direction: 'above' or 'below'
            priority: Alert priority
        """
        def condition(state):
            if direction == 'below':
                return state['price'] < threshold
            return state['price'] > threshold
            
        message = f"Price {direction} {threshold:.2f} at {{timestamp}}"
        self.alert_system.add_alert(name, condition, message, priority)
        
    def add_technical_alert(self,
                            name: str,
                            indicator: str,
                            condition: str,
                            value: float,
                            lookback: int = 14,
                            priority: str = 'medium') -> None:
        """
        Enhanced technical indicator-based alert.
        """
        def check_condition(state):
            # Implement indicator calculations
            prices = state['data']['close']
            if indicator == 'RSI':
                delta = prices.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(lookback).mean()
                avg_loss = loss.rolling(lookback).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi[-1]
                return self._check_condition(current_rsi, condition, value)
                
            elif indicator == 'MACD':
                ema12 = prices.ewm(span=12).mean()
                ema26 = prices.ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                current_macd = macd[-1]
                current_signal = signal[-1]
                if condition == 'cross_above':
                    return current_macd > current_signal and macd[-2] <= signal[-2]
                elif condition == 'cross_below':
                    return current_macd < current_signal and macd[-2] >= signal[-2]
                else:
                    return self._check_condition(current_macd, condition, value)
                    
            # Add other indicators here...
            
            return False
            
        message = f"{indicator} {condition} {value} at {{timestamp}}"
        self.alert_system.add_alert(name, check_condition, message, priority)

    def _check_condition(self, current_value: float, condition: str, threshold: float) -> bool:
        """Helper to evaluate condition"""
        if condition == '>':
            return current_value > threshold
        elif condition == '<':
            return current_value < threshold
        elif condition == 'cross_above':
            return current_value > threshold
        elif condition == 'cross_below':
            return current_value < threshold
        return False

    def add_risk_alert(self,
                     name: str,
                     metric: str,
                     threshold: float,
                     priority: str = 'high') -> None:
        """
        Add risk metric-based alert.
        
        Args:
            name: Alert name
            metric: 'drawdown', 'volatility', 'position_size' etc.
            threshold: Threshold value
            priority: Alert priority
        """
        def check_condition(state):
            # ... risk metric calculation logic ...
            return False  # Implementation depends on metric
            
        message = f"{metric} exceeded {threshold} at {{timestamp}}"
        self.alert_system.add_alert(name, check_condition, message, priority)

    def plot_alerts(self, results: Dict) -> None:
        """
        Visualize alerts on price chart.
        
        Args:
            results: Backtest results containing alerts
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            prices = results['portfolio_history']['prices']
            alerts = results['alerts']['all_alerts']
            
            plt.plot(prices, label='Price')
            
            for alert in alerts:
                plt.axvline(x=alert['timestamp'], color=self._alert_color(alert['priority']), alpha=0.3)
                
            # Create legend
            handles = [
                plt.Line2D([0], [0], color=self._alert_color('high'), label='High Priority'),
                plt.Line2D([0], [0], color=self._alert_color('medium'), label='Medium Priority'),
                plt.Line2D([0], [0], color=self._alert_color('low'), label='Low Priority')
            ]
            plt.legend(handles=handles)
            plt.title('Price Chart with Alerts')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting alerts: {e}")

    def _alert_color(self, priority: str) -> str:
        """Get color for alert priority"""
        return {
            'high': 'red',
            'medium': 'orange',
            'low': 'yellow'
        }.get(priority, 'gray')

    def add_dependent_alert(self,
                           name: str,
                           dependencies: List[Tuple[str, str]],
                           operator: str = 'AND',
                           priority: str = 'high') -> None:
        """
        Add alert that depends on other alerts.
        
        Args:
            name: Alert name
            dependencies: List of (alert_name, status) tuples
            operator: 'AND' or 'OR' for dependency logic
            priority: Alert priority
        """
        def condition(state):
            # Check dependency conditions
            conditions_met = []
            for alert_name, required_status in dependencies:
                # Check if parent alert is in recent triggers
                recent_triggers = [a for a in self.alert_system.triggered_alerts[-10:] 
                                  if a['name'] == alert_name]
                conditions_met.append(len(recent_triggers) > 0)
                
            if operator == 'AND':
                return all(conditions_met)
            return any(conditions_met)
            
        message = f"Dependent alert triggered ({operator} of {[d[0] for d in dependencies]}) at {{timestamp}}"
        self.alert_system.add_alert(name, condition, message, priority)

    def plot_patterns(self, results: Dict, pattern_name: str) -> None:
        """
        Visualize detected price patterns.
        
        Args:
            results: Backtest results containing alerts
            pattern_name: Name of pattern to visualize
        """
        try:
            import matplotlib.pyplot as plt
            
            prices = results['portfolio_history']['prices']
            pattern_alerts = [a for a in results['alerts']['all_alerts'] 
                             if a['name'] == pattern_name]
            
            plt.figure(figsize=(12, 6))
            plt.plot(prices, label='Price')
            
            for alert in pattern_alerts:
                plt.plot(alert['data']['close'], marker='o', markersize=8,
                        label=f"{pattern_name} at {alert['timestamp']}")
                
            plt.title(f"{pattern_name} Detections")
            plt.legend()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting patterns: {e}")

    def visualize_alert_dependencies(self) -> None:
        """Generate visualization of alert dependency tree"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.DiGraph()
            for alert in self.alert_system.alerts.values():
                if 'dependencies' in alert:
                    for dep in alert['dependencies']:
                        G.add_edge(dep[0], alert['name'])
            
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=2000, 
                   node_color='lightblue', font_size=10)
            plt.title("Alert Dependency Tree")
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing dependencies: {e}") 

    def add_sequential_alert(self,
                            name: str,
                            sequence: List[str],
                            time_window: int = 10,
                            priority: str = 'high') -> None:
        """
        Add alert that requires a specific sequence of previous alerts.
        
        Args:
            name: Alert name
            sequence: Ordered list of alert names that must trigger in order
            time_window: Maximum allowed time between alerts in sequence
            priority: Alert priority
        """
        def condition(state):
            triggered = self.alert_system.triggered_alerts
            if len(triggered) < len(sequence):
                return False
                
            # Check for sequence in recent triggers
            sequence_pos = 0
            for alert in reversed(triggered):
                if alert['name'] == sequence[-(sequence_pos+1)]:
                    sequence_pos += 1
                    if sequence_pos == len(sequence):
                        time_diff = (state['timestamp'] - alert['timestamp']).total_seconds()
                        return time_diff <= time_window * 86400  # Convert days to seconds
            return False
            
        message = f"Sequence {'→'.join(sequence)} completed at {{timestamp}}"
        self.alert_system.add_alert(name, condition, message, priority)

    def visualize_temporal_dependencies(self) -> None:
        """Visualize alert sequences with temporal relationships"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.DiGraph()
            for alert in self.alert_system.alerts.values():
                if 'sequence' in alert:
                    prev = None
                    for step in alert['sequence']:
                        if prev:
                            G.add_edge(prev, step)
                        prev = step
                    G.add_edge(prev, alert['name'])
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=2500, 
                   node_color='lightgreen', font_size=10, arrowsize=20)
            plt.title("Temporal Alert Dependencies")
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing temporal dependencies: {e}")

    def add_probabilistic_sequence_alert(self,
                                        name: str,
                                        sequence: List[Tuple[str, float]],
                                        time_window: int = 10,
                                        confidence: float = 0.8,
                                        priority: str = 'high') -> None:
        """
        Add probabilistic sequence alert with confidence thresholds.
        
        Args:
            name: Alert name
            sequence: List of (alert_name, probability) tuples
            time_window: Maximum time window for sequence
            confidence: Required confidence level (0-1)
            priority: Alert priority
        """
        def condition(state):
            triggered = self.alert_system.triggered_alerts
            if len(triggered) < len(sequence):
                return False
                
            # Calculate sequence probability using Bayesian inference
            prob = 1.0
            for alert_name, req_prob in sequence:
                count = sum(1 for a in triggered if a['name'] == alert_name)
                total = sum(1 for a in triggered)
                likelihood = count / total if total > 0 else 0
                prob *= (req_prob * likelihood) / (req_prob * likelihood + (1 - req_prob) * (1 - likelihood))
                
            return prob >= confidence
            
        message = f"Probabilistic sequence {sequence} (≥{confidence}) detected at {{timestamp}}"
        self.alert_system.add_alert(name, condition, message, priority)

    def add_markov_sequence_alert(self,
                                 name: str,
                                 transition_matrix: Dict[str, Dict[str, float]],
                                 initial_state: str,
                                 absorbing_state: str,
                                 priority: str = 'high') -> None:
        """
        Add Markov chain-based sequence alert.
        
        Args:
            name: Alert name
            transition_matrix: State transition probabilities
            initial_state: Starting state
            absorbing_state: Target state to trigger alert
            priority: Alert priority
        """
        def condition(state):
            triggered = self.alert_system.triggered_alerts
            current_state = initial_state
            for alert in triggered[-10:]:  # Check last 10 alerts
                current_state = transition_matrix.get(current_state, {}).get(alert['name'], initial_state)
                if current_state == absorbing_state:
                    return True
            return False
            
        message = f"Markov sequence reached {absorbing_state} at {{timestamp}}"
        self.alert_system.add_alert(name, condition, message, priority)

    def visualize_probabilistic_dependencies(self) -> None:
        """Visualize alert probabilities and transitions"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            G = nx.DiGraph()
            for alert in self.alert_system.alerts.values():
                if 'transitions' in alert:
                    for target, prob in alert['transitions'].items():
                        G.add_edge(alert['name'], target, weight=prob)
            
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=3000,
                   node_color='lightpink', font_size=10)
            edge_labels = {(u, v): f"{d['weight']:.2f}" 
                          for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            plt.title("Probabilistic Alert Transitions")
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing probabilistic dependencies: {e}")

    def hierarchical_optimization(self, historical_data: pd.DataFrame) -> None:
        """Run hierarchical reinforcement learning optimization"""
        for idx, (_, row) in enumerate(historical_data.iterrows()):
            market_state = self._get_market_state(row)
            
            # Manager sets goal
            goal = self.hrl_manager.set_goal(market_state)
            
            # Worker takes action
            action = self.hrl_worker.take_action(goal, market_state)
            
            # Execute action
            self._execute_hrl_action(action, market_state)
            
            # Update Q-tables (would include reward calculation in real implementation)
            self._update_hrl_q_tables(goal, action, market_state)

    def _execute_hrl_action(self, action: str, state: Dict) -> None:
        """Execute HRL action on the trading system"""
        if action == 'adjust_stop_loss':
            new_sl = state['price'] * 0.95
            self.alert_system.add_price_alert(f'SL_{datetime.now()}', new_sl, 'below')
        elif action == 'modify_position_size':
            # Adjust position sizing logic
            pass
        # ... other action implementations ...

    def visualize_wave_patterns(self, results: Dict) -> None:
        """Visualize Elliott Wave patterns with annotations"""
        try:
            import matplotlib.pyplot as plt
            
            prices = results['portfolio_history']['prices']
            wave_alerts = [a for a in results['alerts']['all_alerts'] 
                          if 'wave' in a['name']]
            
            plt.figure(figsize=(15, 8))
            plt.plot(prices, label='Price')
            
            for alert in wave_alerts:
                waves = alert['data']['swing_points']
                labels = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']
                for i, (x, y) in enumerate(waves):
                    plt.plot(x, y, 'o', markersize=8)
                    plt.text(x, y, labels[i], fontsize=8)
                
            plt.title("Elliott Wave Patterns")
            plt.legend()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing wave patterns: {e}")

    def neural_optimization(self, historical_data: pd.DataFrame) -> None:
        """Run neural hierarchical RL optimization"""
        self.neural_hrl.optimize(historical_data)
        
    def visualize_advanced_patterns(self, results: Dict) -> None:
        """Visualize advanced harmonic patterns"""
        try:
            import matplotlib.pyplot as plt
            
            prices = results['portfolio_history']['prices']
            pattern_alerts = [a for a in results['alerts']['all_alerts'] 
                            if 'harmonic' in a['name']]
            
            plt.figure(figsize=(15, 10))
            plt.plot(prices, label='Price')
            
            for alert in pattern_alerts:
                points = alert['data']['swing_points']
                labels = ['X', 'A', 'B', 'C', 'D']
                for i, (x, y) in enumerate(points):
                    plt.plot(x, y, 'o', markersize=10)
                    plt.text(x, y, labels[i], fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.8))
                
                # Draw trendlines
                plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], '--')
                plt.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], '--')
                
            plt.title("Advanced Harmonic Patterns")
            plt.legend()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error visualizing patterns: {e}")

    def pattern_confidence_report(self, results: Dict) -> pd.DataFrame:
        """Generate report of pattern confidence scores"""
        report_data = []
        for alert in results['alerts']['all_alerts']:
            if 'pattern_confidence' in alert['data']:
                report_data.append({
                    'pattern': alert['name'],
                    'timestamp': alert['timestamp'],
                    'confidence': alert['data']['pattern_confidence'],
                    'price': alert['data']['price']
                })
        return pd.DataFrame(report_data).sort_values('confidence', ascending=False)

    def plot_confidence_histogram(self, results: Dict) -> None:
        """Visualize pattern confidence distribution"""
        try:
            import seaborn as sns
            report = self.pattern_confidence_report(results)
            plt.figure(figsize=(10, 6))
            sns.histplot(data=report, x='confidence', bins=20, kde=True)
            plt.title("Pattern Confidence Distribution")
            plt.xlabel("Confidence Score")
            plt.ylabel("Count")
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting confidence histogram: {e}")

    def attention_analysis(self, historical_data: pd.DataFrame) -> None:
        """Run and visualize attention mechanism analysis"""
        self.neural_hrl.visualize_attention(historical_data)

    def adaptive_confidence_thresholding(self, results: Dict) -> None:
        """Dynamically adjust confidence thresholds based on performance"""
        report = self.pattern_confidence_report(results)
        successful = report[report['price'].pct_change() > 0]
        if not successful.empty:
            new_threshold = successful['confidence'].quantile(0.25)
            logger.info(f"Adjusting confidence threshold to {new_threshold:.2f}")
            for alert in self.alert_system.alerts.values():
                if 'confidence_threshold' in alert:
                    alert['confidence_threshold'] = max(0.5, new_threshold)

    def visualize_xl_attention(self, historical_data: pd.DataFrame) -> None:
        """Visualize Transformer-XL attention patterns"""
        import matplotlib.pyplot as plt
        
        with torch.no_grad():
            states = self.neural_hrl._preprocess_data(historical_data)
            _, mems = self.neural_hrl.manager.transformer(states)
            
        plt.figure(figsize=(18, 10))
        for i, mem in enumerate(mems[:3]):
            plt.subplot(3, 1, i+1)
            plt.imshow(mem.squeeze().cpu().numpy(),
                      cmap='plasma', aspect='auto')
            plt.title(f"Transformer-XL Memory Layer {i+1}")
            plt.colorbar()
        plt.tight_layout()
        plt.show()

    def analyze_attention_compression(self, historical_data: pd.DataFrame) -> None:
        """Analyze compressed vs local attention patterns"""
        self.compressed_hrl.visualize_compressed_attention(historical_data)