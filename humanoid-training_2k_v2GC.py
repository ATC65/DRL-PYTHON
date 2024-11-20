# Instalar las bibliotecas necesarias
!pip install gymnasium
!pip install gymnasium[mujoco]
!pip install stable-baselines3[extra]

#Verificación de instalación
import gymnasium as gym
print(gym.__version__)  # Esto debería imprimir la versión de gymnasium

#Verificación de uso de GPU
import tensorflow as tf

if tf.test.gpu_device_name():
    print('GPU encontrada:', tf.test.gpu_device_name())
else:
    print('No se encontró una GPU. Asegúrate de haberla habilitado en la configuración del entorno de ejecución.')

# Optimizar las importaciones e instalaciones
import os
from typing import Dict, Any
!pip install gymnasium[mujoco] stable-baselines3[extra] --quiet
# Optimizar las importaciones e instalaciones
import os
from typing import Dict, Any
!pip install gymnasium[mujoco] stable-baselines3[extra] --quiet


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

class ExtendedMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ExtendedMetricsCallback, self).__init__(verbose)
        self.metrics = {
            'rewards': [],
            'x_velocities': [],
            'timesteps': [],
            'stability_metric': [],
            'cumulative_reward': 0
        }
        self.episode_count = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]

        self.metrics['cumulative_reward'] += reward
        self.metrics['timesteps'].append(self.num_timesteps)
        self.metrics['rewards'].append(reward)

        if 'x_velocity' in info:
            self.metrics['x_velocities'].append(info['x_velocity'])

        return True

    def _on_rollout_end(self):
        if len(self.metrics['x_velocities']) > 0:
            stability = 1.0 / (1.0 + np.std(self.metrics['x_velocities']))
            self.metrics['stability_metric'].append(stability)

        self.episode_count += 1
        self.metrics['cumulative_reward'] = 0

        return True

def create_env():
    env = gym.make('Humanoid-v4',
                   forward_reward_weight=1.25,
                   ctrl_cost_weight=0.099,
                   render_mode=None)
    env = Monitor(env, './logs')
    return env
################################################################
# ASIGNACIÓN DE timesteps PARA ENTRENAMIENTO DE LAS UNIDADES
def train_and_compare_agents(total_timesteps=1000):
    agents = {
        'SAC': (SAC, {
            'learning_rate': 3e-4,
            'batch_size': 128,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 10000
        }),
        'TD3': (TD3, {
            'learning_rate': 3e-4,
            'batch_size': 128,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 10000
        }),
        'DDPG': (DDPG, {
            'learning_rate': 3e-4,
            'batch_size': 128,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 10000
        }),
        'PPO': (PPO, {
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 128,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01
        })
    }
#########################################################################
    results = {}

    for agent_name, (agent_class, params) in agents.items():
        print(f"\nEntrenando {agent_name}...")
        env = DummyVecEnv([lambda: create_env()])

        model = agent_class(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./tensorboard_logs/{agent_name}",
            **params
        )

        callback = ExtendedMetricsCallback()
        model.learn(total_timesteps=total_timesteps, callback=callback)

        model.save(f"humanoid_{agent_name.lower()}")
        results[agent_name] = callback.metrics

    return results

def cumulative_average(data):
    """Calcula el promedio acumulado de una lista."""
    return np.cumsum(data) / np.arange(1, len(data) + 1)

def plot_comparative_metrics(results):
    metrics_to_plot = {
        'rewards': 'Recompensa Final',
        'x_velocities': 'Velocidad X Promedio',
        'stability_metric': 'Estabilidad Final'
    }

    plt.style.use('default')

    for metric_key, metric_name in metrics_to_plot.items():
        plt.figure(figsize=(12, 6))

        for agent_name, metrics in results.items():
            if metric_key in metrics and len(metrics[metric_key]) > 0:
                data = cumulative_average(metrics[metric_key])
                timesteps = metrics['timesteps'][:len(data)]

                plt.plot(timesteps, data, label=agent_name, alpha=0.8)

        plt.title(f'Comparación de Promedios Acumulados de {metric_name} entre Agentes')
        plt.xlabel('Timestep')
        plt.ylabel(f'Promedio Acumulado de {metric_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'cumulative_average_{metric_key}.png')
        plt.show()

def generate_comparison_report(results):
    report = pd.DataFrame()

    for agent_name, metrics in results.items():
        final_metrics = {
            'Recompensa Final': np.mean(metrics['rewards'][-100:]) if len(metrics['rewards']) > 0 else 0,
            'Velocidad X Promedio': np.mean(metrics['x_velocities']) if len(metrics['x_velocities']) > 0 else 0,
            'Estabilidad Final': np.mean(metrics['stability_metric'][-100:]) if len(metrics['stability_metric']) > 0 else 0
        }
        report[agent_name] = pd.Series(final_metrics)

    report.to_csv('comparison_report.csv')
    return report

if __name__ == "__main__":
    results = train_and_compare_agents()
    plot_comparative_metrics(results)
    report = generate_comparison_report(results)
    print("\nReporte de Comparación Final:")
    print(report)
