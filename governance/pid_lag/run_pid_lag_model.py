#!/usr/bin/env python
"""
Run Governance PID-Lag Toy Model

This script runs simulations of the Governance PID-Lag Toy Model
to explore governance responses to technological acceleration and jolts.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from controller import PIDParams, LagParams, PIDLagController, simulate_pid_lag_response
from technology import TechProgressParams, simulate_tech_progress


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Governance PID-Lag Toy Model')
    
    # Simulation parameters
    parser.add_argument('--start-time', type=float, default=0.0,
                        help='Start time for simulation')
    parser.add_argument('--end-time', type=float, default=100.0,
                        help='End time for simulation')
    parser.add_argument('--time-step', type=float, default=1.0,
                        help='Time step for simulation')
    
    # Technology parameters
    parser.add_argument('--base-growth-rate', type=float, default=0.05,
                        help='Base exponential growth rate')
    parser.add_argument('--jolt-time', type=float, default=50.0,
                        help='Time of jolt (-1 for no jolt)')
    parser.add_argument('--jolt-magnitude', type=float, default=0.2,
                        help='Magnitude of jolt')
    parser.add_argument('--jolt-duration', type=float, default=5.0,
                        help='Duration of jolt effect')
    parser.add_argument('--jolt-decay', type=float, default=0.5,
                        help='Decay rate of jolt effect')
    parser.add_argument('--noise-std', type=float, default=0.01,
                        help='Standard deviation of noise')
    parser.add_argument('--initial-value', type=float, default=1.0,
                        help='Initial technology level')
    
    # PID parameters
    parser.add_argument('--kp', type=float, default=0.5,
                        help='Proportional gain')
    parser.add_argument('--ki', type=float, default=0.1,
                        help='Integral gain')
    parser.add_argument('--kd', type=float, default=0.2,
                        help='Derivative gain')
    parser.add_argument('--setpoint', type=float, default=0.2,
                        help='Risk setpoint (target risk level)')
    
    # Lag parameters
    parser.add_argument('--response-delay', type=int, default=5,
                        help='Time steps of delay before response begins')
    parser.add_argument('--response-time', type=int, default=10,
                        help='Time steps to reach full response')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='Rate of decay for governance effectiveness')
    parser.add_argument('--effectiveness-cap', type=float, default=0.8,
                        help='Maximum effectiveness of governance')
    
    # Risk parameters
    parser.add_argument('--risk-threshold', type=float, default=10.0,
                        help='Technology level at which risk is 0.5')
    parser.add_argument('--risk-steepness', type=float, default=2.0,
                        help='Steepness of the risk curve')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for output files')
    parser.add_argument('--output-prefix', type=str, default='pid_lag',
                        help='Prefix for output files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def run_simulation(args):
    """Run the simulation with the given arguments."""
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'data'), exist_ok=True)
    
    # Create technology parameters
    tech_params = TechProgressParams(
        base_growth_rate=args.base_growth_rate,
        jolt_time=args.jolt_time if args.jolt_time >= 0 else None,
        jolt_magnitude=args.jolt_magnitude,
        jolt_duration=args.jolt_duration,
        jolt_decay=args.jolt_decay,
        noise_std=args.noise_std,
        initial_value=args.initial_value
    )
    
    # Create PID parameters
    pid_params = PIDParams(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        setpoint=args.setpoint,
        windup_guard=20.0,
        sample_time=args.time_step
    )
    
    # Create lag parameters
    lag_params = LagParams(
        response_delay=args.response_delay,
        response_time=args.response_time,
        decay_rate=args.decay_rate,
        effectiveness_cap=args.effectiveness_cap
    )
    
    # Create controller
    controller = PIDLagController(pid_params, lag_params)
    
    # Run simulation without governance
    print("Running simulation without governance...")
    results_no_gov = simulate_tech_progress(
        tech_params,
        governance_func=None,
        time_range=(args.start_time, args.end_time),
        time_step=args.time_step,
        risk_threshold=args.risk_threshold,
        risk_steepness=args.risk_steepness,
        seed=args.seed
    )
    
    # Run simulation with governance
    print("Running simulation with governance...")
    
    # Create governance function that uses the PID controller
    def governance_func(t, tech_level):
        # Calculate risk
        risk = 1 / (1 + np.exp(-args.risk_steepness * (tech_level - args.risk_threshold)))
        # Update controller with risk as process variable
        return controller.update(risk, t)
    
    results_with_gov = simulate_tech_progress(
        tech_params,
        governance_func=governance_func,
        time_range=(args.start_time, args.end_time),
        time_step=args.time_step,
        risk_threshold=args.risk_threshold,
        risk_steepness=args.risk_steepness,
        seed=args.seed
    )
    
    # Save results
    save_results(args, results_no_gov, results_with_gov)
    
    # Create visualizations
    create_visualizations(args, results_no_gov, results_with_gov)
    
    return results_no_gov, results_with_gov


def save_results(args, results_no_gov, results_with_gov):
    """Save simulation results to CSV files."""
    # Create DataFrames
    df_no_gov = pd.DataFrame({
        'time': results_no_gov['time'],
        'tech_level': results_no_gov['tech_level'],
        'jolt_effect': results_no_gov['jolt_effect'],
        'governance_effect': results_no_gov['governance_effect'],
        'risk': results_no_gov['risk']
    })
    
    df_with_gov = pd.DataFrame({
        'time': results_with_gov['time'],
        'tech_level': results_with_gov['tech_level'],
        'jolt_effect': results_with_gov['jolt_effect'],
        'governance_effect': results_with_gov['governance_effect'],
        'risk': results_with_gov['risk']
    })
    
    # Save to CSV
    df_no_gov.to_csv(os.path.join(args.output_dir, 'data', f'{args.output_prefix}_no_governance.csv'), index=False)
    df_with_gov.to_csv(os.path.join(args.output_dir, 'data', f'{args.output_prefix}_with_governance.csv'), index=False)
    
    print(f"Results saved to {args.output_dir}/data/")


def create_visualizations(args, results_no_gov, results_with_gov):
    """Create visualizations of simulation results."""
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.rcParams['font.size'] = 12
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot technology level
    axs[0].plot(results_no_gov['time'], results_no_gov['tech_level'], 
                label='Without Governance', color='blue', linewidth=2)
    axs[0].plot(results_with_gov['time'], results_with_gov['tech_level'], 
                label='With Governance', color='green', linewidth=2)
    
    # Add jolt marker if applicable
    if args.jolt_time >= 0:
        axs[0].axvline(x=args.jolt_time, color='red', linestyle='--', 
                      label='Jolt Occurs', alpha=0.7)
    
    axs[0].set_ylabel('Technology Level')
    axs[0].set_title('Technology Progress with and without Governance')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot risk
    axs[1].plot(results_no_gov['time'], results_no_gov['risk'], 
                label='Without Governance', color='blue', linewidth=2)
    axs[1].plot(results_with_gov['time'], results_with_gov['risk'], 
                label='With Governance', color='green', linewidth=2)
    axs[1].axhline(y=args.setpoint, color='black', linestyle='--', 
                  label='Risk Setpoint', alpha=0.7)
    
    if args.jolt_time >= 0:
        axs[1].axvline(x=args.jolt_time, color='red', linestyle='--', alpha=0.7)
    
    axs[1].set_ylabel('Risk Level')
    axs[1].set_title('Risk Level with and without Governance')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot governance effect
    axs[2].plot(results_with_gov['time'], results_with_gov['governance_effect'], 
                label='Governance Effect', color='purple', linewidth=2)
    axs[2].plot(results_with_gov['time'], results_with_gov['jolt_effect'], 
                label='Jolt Effect', color='red', linewidth=2)
    
    if args.jolt_time >= 0:
        axs[2].axvline(x=args.jolt_time, color='red', linestyle='--', alpha=0.7)
    
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Effect Magnitude')
    axs[2].set_title('Governance and Jolt Effects')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Governance PID-Lag Model Simulation\nKp={args.kp}, Ki={args.ki}, Kd={args.kd}, Delay={args.response_delay}, Response Time={args.response_time}', 
                fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    plt.savefig(os.path.join(args.output_dir, 'figures', f'{args.output_prefix}_simulation.png'), dpi=300)
    print(f"Visualization saved to {args.output_dir}/figures/{args.output_prefix}_simulation.png")
    
    # Create additional visualization for parameter exploration
    create_parameter_exploration_plot(args)


def create_parameter_exploration_plot(args):
    """Create a plot exploring different parameter settings."""
    # Set up parameters to explore
    response_delays = [2, 5, 10, 20]
    response_times = [5, 10, 20]
    
    # Set up plot
    fig, axs = plt.subplots(len(response_delays), len(response_times), 
                           figsize=(15, 12), sharex=True, sharey=True)
    
    # Run simulations for each parameter combination
    for i, delay in enumerate(response_delays):
        for j, resp_time in enumerate(response_times):
            # Update lag parameters
            lag_params = LagParams(
                response_delay=delay,
                response_time=resp_time,
                decay_rate=args.decay_rate,
                effectiveness_cap=args.effectiveness_cap
            )
            
            # Create controller
            pid_params = PIDParams(
                kp=args.kp,
                ki=args.ki,
                kd=args.kd,
                setpoint=args.setpoint,
                windup_guard=20.0,
                sample_time=args.time_step
            )
            controller = PIDLagController(pid_params, lag_params)
            
            # Create tech parameters
            tech_params = TechProgressParams(
                base_growth_rate=args.base_growth_rate,
                jolt_time=args.jolt_time if args.jolt_time >= 0 else None,
                jolt_magnitude=args.jolt_magnitude,
                jolt_duration=args.jolt_duration,
                jolt_decay=args.jolt_decay,
                noise_std=args.noise_std,
                initial_value=args.initial_value
            )
            
            # Create governance function
            def governance_func(t, tech_level):
                risk = 1 / (1 + np.exp(-args.risk_steepness * (tech_level - args.risk_threshold)))
                return controller.update(risk, t)
            
            # Run simulation
            results = simulate_tech_progress(
                tech_params,
                governance_func=governance_func,
                time_range=(args.start_time, args.end_time),
                time_step=args.time_step,
                risk_threshold=args.risk_threshold,
                risk_steepness=args.risk_steepness,
                seed=args.seed
            )
            
            # Plot results
            axs[i, j].plot(results['time'], results['risk'], color='green', linewidth=1.5)
            axs[i, j].axhline(y=args.setpoint, color='black', linestyle='--', alpha=0.5)
            
            if args.jolt_time >= 0:
                axs[i, j].axvline(x=args.jolt_time, color='red', linestyle='--', alpha=0.5)
            
            axs[i, j].set_title(f'Delay={delay}, Response Time={resp_time}')
            axs[i, j].grid(True, alpha=0.3)
    
    # Add labels
    for i in range(len(response_delays)):
        axs[i, 0].set_ylabel('Risk Level')
    
    for j in range(len(response_times)):
        axs[-1, j].set_xlabel('Time')
    
    plt.suptitle('Parameter Exploration: Effect of Response Delay and Response Time on Risk Control', 
                fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    plt.savefig(os.path.join(args.output_dir, 'figures', f'{args.output_prefix}_parameter_exploration.png'), dpi=300)
    print(f"Parameter exploration plot saved to {args.output_dir}/figures/{args.output_prefix}_parameter_exploration.png")


def create_latex_table(args, results_no_gov, results_with_gov):
    """Create a LaTeX table summarizing the simulation results."""
    # Calculate key metrics
    max_tech_no_gov = np.max(results_no_gov['tech_level'])
    max_tech_with_gov = np.max(results_with_gov['tech_level'])
    
    max_risk_no_gov = np.max(results_no_gov['risk'])
    max_risk_with_gov = np.max(results_with_gov['risk'])
    
    avg_risk_no_gov = np.mean(results_no_gov['risk'])
    avg_risk_with_gov = np.mean(results_with_gov['risk'])
    
    # Find time when risk exceeds 0.5
    risk_threshold_time_no_gov = None
    for i, risk in enumerate(results_no_gov['risk']):
        if risk > 0.5:
            risk_threshold_time_no_gov = results_no_gov['time'][i]
            break
    
    risk_threshold_time_with_gov = None
    for i, risk in enumerate(results_with_gov['risk']):
        if risk > 0.5:
            risk_threshold_time_with_gov = results_with_gov['time'][i]
            break
    
    # Calculate time delay in response to jolt
    if args.jolt_time >= 0:
        # Find peak governance response after jolt
        jolt_idx = np.argmin(np.abs(results_with_gov['time'] - args.jolt_time))
        gov_response = results_with_gov['governance_effect'][jolt_idx:]
        peak_response_idx = np.argmax(gov_response) + jolt_idx
        peak_response_time = results_with_gov['time'][peak_response_idx]
        response_delay = peak_response_time - args.jolt_time
    else:
        response_delay = "N/A"
    
    # Create LaTeX table
    latex_content = r"""
\begin{table}[ht]
\centering
\caption{Governance PID-Lag Model Simulation Results}
\label{tab:pid_lag_results}
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{Without Governance} & \textbf{With Governance} \\
\midrule
Maximum Technology Level & %.2f & %.2f \\
Maximum Risk Level & %.2f & %.2f \\
Average Risk Level & %.2f & %.2f \\
Time to Risk > 0.5 & %s & %s \\
\midrule
\multicolumn{3}{l}{\textbf{Governance Parameters}} \\
PID Gains (Kp, Ki, Kd) & \multicolumn{2}{r}{%.2f, %.2f, %.2f} \\
Response Delay & \multicolumn{2}{r}{%d time steps} \\
Response Time & \multicolumn{2}{r}{%d time steps} \\
Effectiveness Cap & \multicolumn{2}{r}{%.2f} \\
Jolt Response Delay & \multicolumn{2}{r}{%s} \\
\bottomrule
\end{tabular}
\end{table}
""" % (
    max_tech_no_gov, max_tech_with_gov,
    max_risk_no_gov, max_risk_with_gov,
    avg_risk_no_gov, avg_risk_with_gov,
    str(risk_threshold_time_no_gov) if risk_threshold_time_no_gov is not None else "N/A",
    str(risk_threshold_time_with_gov) if risk_threshold_time_with_gov is not None else "N/A",
    args.kp, args.ki, args.kd,
    args.response_delay,
    args.response_time,
    args.effectiveness_cap,
    str(response_delay) if isinstance(response_delay, (int, float)) else response_delay
)
    
    # Save to file
    with open(os.path.join(args.output_dir, f'{args.output_prefix}_results.tex'), 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to {args.output_dir}/{args.output_prefix}_results.tex")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Print parameters
    print("Running Governance PID-Lag Toy Model with parameters:")
    print(f"  Technology: growth_rate={args.base_growth_rate}, jolt_time={args.jolt_time}, jolt_magnitude={args.jolt_magnitude}")
    print(f"  PID Controller: Kp={args.kp}, Ki={args.ki}, Kd={args.kd}, setpoint={args.setpoint}")
    print(f"  Lag Model: response_delay={args.response_delay}, response_time={args.response_time}, effectiveness_cap={args.effectiveness_cap}")
    
    # Run simulation
    results_no_gov, results_with_gov = run_simulation(args)
    
    # Create LaTeX table
    create_latex_table(args, results_no_gov, results_with_gov)
    
    print("Simulation complete!")


if __name__ == "__main__":
    main()
