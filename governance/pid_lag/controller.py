"""
PID Controller with Lag

This module implements a PID (Proportional-Integral-Derivative) controller
with time lag to model governance responses to technological acceleration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass


@dataclass
class PIDParams:
    """Parameters for PID controller."""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.1  # Integral gain
    kd: float = 0.5  # Derivative gain
    setpoint: float = 0.0  # Target value
    windup_guard: float = 20.0  # Anti-windup guard limit
    sample_time: float = 1.0  # Sample time in time units
    
    def __post_init__(self):
        """Validate parameters."""
        if self.kp < 0:
            raise ValueError("Proportional gain must be non-negative")
        if self.ki < 0:
            raise ValueError("Integral gain must be non-negative")
        if self.kd < 0:
            raise ValueError("Derivative gain must be non-negative")
        if self.sample_time <= 0:
            raise ValueError("Sample time must be positive")
        if self.windup_guard < 0:
            raise ValueError("Anti-windup guard must be non-negative")


@dataclass
class LagParams:
    """Parameters for time lag model."""
    response_delay: int = 2  # Time steps of delay before response begins
    response_time: int = 5  # Time steps to reach full response
    decay_rate: float = 0.1  # Rate of decay for governance effectiveness
    effectiveness_cap: float = 0.9  # Maximum effectiveness of governance
    
    def __post_init__(self):
        """Validate parameters."""
        if self.response_delay < 0:
            raise ValueError("Response delay must be non-negative")
        if self.response_time <= 0:
            raise ValueError("Response time must be positive")
        if self.decay_rate < 0 or self.decay_rate > 1:
            raise ValueError("Decay rate must be between 0 and 1")
        if self.effectiveness_cap <= 0 or self.effectiveness_cap > 1:
            raise ValueError("Effectiveness cap must be between 0 and 1")


class PIDLagController:
    """
    PID controller with time lag for governance modeling.
    
    This class implements a PID controller with time lag to simulate
    governance responses to technological acceleration and jolts.
    """
    
    def __init__(
        self,
        pid_params: Optional[PIDParams] = None,
        lag_params: Optional[LagParams] = None
    ):
        """
        Initialize the PID controller with lag.
        
        Parameters
        ----------
        pid_params : PIDParams, optional
            Parameters for the PID controller
        lag_params : LagParams, optional
            Parameters for the time lag model
        """
        self.pid_params = pid_params or PIDParams()
        self.lag_params = lag_params or LagParams()
        
        # PID controller state
        self.last_error = 0.0
        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.last_time = 0.0
        self.output = 0.0
        
        # Lag model state
        self.response_queue = []  # Queue of pending responses
        self.current_effectiveness = self.lag_params.effectiveness_cap
        self.time_since_last_action = 0
    
    def reset(self):
        """Reset the controller state."""
        self.last_error = 0.0
        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.last_time = 0.0
        self.output = 0.0
        self.response_queue = []
        self.current_effectiveness = self.lag_params.effectiveness_cap
        self.time_since_last_action = 0
    
    def _compute_pid(self, error: float, current_time: float) -> float:
        """
        Compute PID control value based on error.
        
        Parameters
        ----------
        error : float
            Current error (setpoint - process_variable)
        current_time : float
            Current time
            
        Returns
        -------
        float
            PID control output
        """
        # Calculate time delta
        delta_time = current_time - self.last_time
        if delta_time < self.pid_params.sample_time:
            # Not enough time has passed since last update
            return self.output
        
        # Proportional term
        self.proportional = self.pid_params.kp * error
        
        # Integral term
        self.integral += self.pid_params.ki * error * delta_time
        
        # Apply anti-windup guard to integral term
        if self.integral > self.pid_params.windup_guard:
            self.integral = self.pid_params.windup_guard
        elif self.integral < -self.pid_params.windup_guard:
            self.integral = -self.pid_params.windup_guard
        
        # Derivative term (on measurement, not error)
        if delta_time > 0:
            self.derivative = self.pid_params.kd * (error - self.last_error) / delta_time
        else:
            self.derivative = 0.0
        
        # Calculate output
        self.output = self.proportional + self.integral + self.derivative
        
        # Update state
        self.last_error = error
        self.last_time = current_time
        
        return self.output
    
    def _apply_lag(self, control_action: float) -> float:
        """
        Apply time lag to control action.
        
        Parameters
        ----------
        control_action : float
            Raw control action from PID
            
        Returns
        -------
        float
            Actual control action after lag
        """
        # Add new control action to response queue with delay
        self.response_queue.append({
            'action': control_action,
            'delay_remaining': self.lag_params.response_delay,
            'response_remaining': self.lag_params.response_time,
            'total_response': self.lag_params.response_time,
            'magnitude': control_action
        })
        
        # Process response queue
        active_responses = []
        current_response = 0.0
        
        for response in self.response_queue:
            if response['delay_remaining'] > 0:
                # Still in delay phase
                response['delay_remaining'] -= 1
                active_responses.append(response)
            elif response['response_remaining'] > 0:
                # In response phase
                response_fraction = (response['total_response'] - response['response_remaining']) / response['total_response']
                current_response += response['magnitude'] * response_fraction
                response['response_remaining'] -= 1
                active_responses.append(response)
        
        # Update response queue
        self.response_queue = active_responses
        
        # Apply effectiveness decay if no recent actions
        self.time_since_last_action += 1
        if control_action == 0 and self.time_since_last_action > self.lag_params.response_time:
            self.current_effectiveness *= (1 - self.lag_params.decay_rate)
        else:
            # Reset decay when new action is taken
            self.current_effectiveness = self.lag_params.effectiveness_cap
            self.time_since_last_action = 0
        
        # Apply effectiveness cap
        return current_response * self.current_effectiveness
    
    def update(self, process_variable: float, current_time: float) -> float:
        """
        Update the controller with a new process variable value.
        
        Parameters
        ----------
        process_variable : float
            Current value of the process variable
        current_time : float
            Current time
            
        Returns
        -------
        float
            Control action after lag
        """
        # Calculate error
        error = self.pid_params.setpoint - process_variable
        
        # Compute PID control action
        control_action = self._compute_pid(error, current_time)
        
        # Apply lag
        actual_action = self._apply_lag(control_action)
        
        return actual_action
    
    def get_state(self) -> Dict:
        """
        Get the current state of the controller.
        
        Returns
        -------
        Dict
            Dictionary containing the current state
        """
        return {
            'proportional': self.proportional,
            'integral': self.integral,
            'derivative': self.derivative,
            'output': self.output,
            'last_error': self.last_error,
            'last_time': self.last_time,
            'effectiveness': self.current_effectiveness,
            'response_queue_length': len(self.response_queue),
            'time_since_last_action': self.time_since_last_action
        }


def simulate_pid_lag_response(
    process_variable_func: Callable[[float], float],
    controller: PIDLagController,
    time_range: Tuple[float, float],
    time_step: float = 1.0,
    disturbance_func: Optional[Callable[[float], float]] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate a PID controller with lag response to a process.
    
    Parameters
    ----------
    process_variable_func : Callable[[float], float]
        Function that returns the process variable value at a given time
    controller : PIDLagController
        PID controller with lag
    time_range : Tuple[float, float]
        Start and end times for simulation
    time_step : float, default=1.0
        Time step for simulation
    disturbance_func : Callable[[float], float], optional
        Function that returns a disturbance value at a given time
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing simulation results:
        - 'time': Time points
        - 'process_variable': Process variable values
        - 'control_action': Control actions
        - 'setpoint': Setpoint values
        - 'error': Error values
        - 'effectiveness': Controller effectiveness
    """
    # Initialize simulation
    start_time, end_time = time_range
    time_points = np.arange(start_time, end_time + time_step, time_step)
    n_points = len(time_points)
    
    # Initialize result arrays
    process_variable = np.zeros(n_points)
    control_action = np.zeros(n_points)
    setpoint = np.zeros(n_points)
    error = np.zeros(n_points)
    effectiveness = np.zeros(n_points)
    
    # Reset controller
    controller.reset()
    
    # Run simulation
    for i, t in enumerate(time_points):
        # Get process variable
        pv = process_variable_func(t)
        
        # Add disturbance if provided
        if disturbance_func is not None:
            pv += disturbance_func(t)
        
        # Update controller
        action = controller.update(pv, t)
        
        # Store results
        process_variable[i] = pv
        control_action[i] = action
        setpoint[i] = controller.pid_params.setpoint
        error[i] = controller.last_error
        effectiveness[i] = controller.current_effectiveness
    
    # Return results
    return {
        'time': time_points,
        'process_variable': process_variable,
        'control_action': control_action,
        'setpoint': setpoint,
        'error': error,
        'effectiveness': effectiveness
    }
