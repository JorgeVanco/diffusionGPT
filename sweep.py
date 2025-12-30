import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import yaml
import torch

from train import main as train_main

def train_wrapper(config) -> None:
    """
    Wraps the train_main function to translate Ray Tune config 
    into the dictionary format expected by HfArgumentParser.
    """
    
    # Base arguments (defaults)
    base_args = {
        "output_dir": os.path.join(os.getcwd(), "sweep_results"),
        "do_train": True,
        "do_eval": True,
        "logging_steps": 50,
        "save_steps": 500,
        "eval_steps": 500,
        "max_train_samples": 5000,
        "per_device_train_batch_size": 16, # Adjust based on GPU
        "per_device_eval_batch_size": 16,
        "max_seq_length": 128,
        "num_train_epochs": 1, # Keep epochs low for sweeping
        "report_to": "wandb",  # or "none"
        "run_name": f"sweep_trial_{tune.get_context().get_trial_id()}",
        
        # Disable heavy disk saving during sweeps
        "save_total_limit": 1,
    }
    
    # Load base config from file
    base_config_path = os.path.join(os.environ["TUNE_ORIG_WORKING_DIR"], "configs/config.yaml")
    with open(base_config_path, "r") as f:
        args = yaml.safe_load(f)
        
    args.update(base_args)

    # Update with values from Ray Tune
    args.update(config)
    print(f"Starting trial with config: {args}")

    # Call the training function
    # We catch errors to prevent one bad trial from killing the whole sweep
    try:
        eval_loss = train_main(override_args=args)
        
        # Report metric to Ray Tune
        tune.report(eval_loss=eval_loss)
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        raise e

if __name__ == "__main__":
    # Initialize Ray
    # If you have a cluster, point this to the head node. 
    # For local, just ray.init()
    project_dir = os.path.abspath(os.path.join(os.getcwd()))
    ray.init(
        num_cpus=4,
        num_gpus=1,
        runtime_env={
            "working_dir": project_dir,
            "excludes": ["output", "wandb", "ray_results", ".git"], # Prevent copying huge files to workers
            "env_vars": os.environ.copy() # Pass current environment variables
        }
    ) 

    # ray.init()
    print("Starting Hyperparameter Sweep...")

    # Define the Search Space
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "num_diffusion_steps": tune.choice([25, 50, 100]),
        "corruption_prob": tune.uniform(0.1, 0.3),
        "gradient_accumulation_steps": tune.choice([1, 2]),
    }

    # Define the Scheduler (Early Stopping)
    # ASHA is great for aggressive early stopping of bad trials
    scheduler = ASHAScheduler(
        metric="eval_loss",
        mode="min",
        max_t=5000, # Max training iterations (steps)
        grace_period=100, # Run at least this many steps before stopping
        reduction_factor=2
    )

    analysis = tune.run(
        train_wrapper,
        config=search_space,
        num_samples=10, # Number of different hyperparam combinations to try
        scheduler=scheduler,
        resources_per_trial={"cpu": 2, "gpu": 0.5}, # 0.5 allows running 2 trials on 1 GPU (if memory allows!)
        storage_path=os.path.abspath("./ray_results"),
        keep_checkpoints_num=1,
    )

    print("Best hyperparameters found were: ", analysis.best_config)