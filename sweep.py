import optuna
from optuna.samplers import TPESampler
import os
import yaml
import argparse

from train import main as train_main

def objective(trial) -> float:
    # 1. Hyperparameter Space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    num_diffusion_steps = trial.suggest_categorical("num_diffusion_steps", [25, 50, 100])
    corruption_prob = trial.suggest_float("corruption_prob", 0.1, 0.3)
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2])
    
    base_args = {
        "output_dir": os.path.join(os.getcwd(), "sweep_results"),
        "do_train": True,
        "do_eval": True,
        "logging_steps": 50,
        "save_steps": 5000,
        "eval_steps": 500,
        # "max_train_samples": 5000,
        "max_seq_length": 128,
        "num_train_epochs": 1, # Keep epochs low for sweeping
        "report_to": "wandb",  # or "none"
        # "run_name": f"sweep_trial_{tune.get_context().get_trial_id()}",
        "auto_naming": True,
        
        # Disable heavy disk saving during sweeps
        "save_total_limit": 1,
    }

    # 2. Load Config
    # Robustly find config.yaml relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_config_path = os.path.join(script_dir, "configs", "config.yaml")
    
    if not os.path.exists(base_config_path):
        base_config_path = os.path.join(os.getcwd(), "config.yaml")

    with open(base_config_path, "r") as f:
        args = yaml.safe_load(f)
        
    args.update(base_args)

    # 3. Apply Overrides
    args["learning_rate"] = learning_rate
    args["num_diffusion_steps"] = num_diffusion_steps
    args["corruption_prob"] = corruption_prob
    args["gradient_accumulation_steps"] = gradient_accumulation_steps
    args["max_steps"] = 5000 # Limit steps for faster sweeps
    args["warmup_steps"] = 100
    args["lr_scheduler_kwargs"] = {"num_decay_steps": 1000}
    
    # Unique names and cleanup
    args["run_name"] = f"optuna_trial_{trial.number}"
    args["output_dir"] = os.path.join(os.getcwd(), "sweep_results", args["run_name"])
    args["do_train"] = True
    args["do_eval"] = True
    args["save_total_limit"] = 1
    args["report_to"] = "wandb" 

    # 4. Run Training
    try:
        eval_loss = train_main(override_args=args)
        return eval_loss
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float("inf")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials to run on this worker")
    args = parser.parse_args()
    
    # Create the study storage (shared by all workers)
    study_name = "diffusion_sweep_v1"
    storage_url = "sqlite:///db.sqlite3"
    
    sampler = TPESampler(multivariate=True, n_startup_trials=10)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage_url,
        sampler=sampler,
        load_if_exists=True 
    )
    
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')
    print(f"Worker started on GPU {gpu_id}. Running {args.n_trials} trials.")
    
    # Run 10 trials per worker (Total = 8 GPUs * 10 = 80 trials)
    study.optimize(objective, n_trials=args.n_trials)