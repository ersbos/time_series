import argparse
import yaml
import wandb
import torch
from model import get_model
from utilities import train_model, get_dataloaders

def main():

    config_path = "config.yaml"  # update this path if needed
    with open(config_path, 'r') as file:
     config = yaml.safe_load(file)

    # Unpack configuration sections
    wandb_config    = config.get("wandb", {})
    wandb_host= wandb_config.get("host")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Used device is:",device)
    if wandb_config:
        wandb_api_key = wandb_config.get("api_key")
        if wandb_api_key:
            wandb.login(key=wandb_api_key,host=wandb_host
                       )
            # Optionally, initialize the wandb run
            wandb.init(
                project=wandb_config.get("project", "default_project"),
                entity=wandb_config.get("entity", None),
                config=config,
                notes= wandb_config.get("note", "")
            )
            print('Wandb run name: ', wandb.run.name)
    else:
        print('[WARNING] Wandb login failed. Continuing without wandb.')
    # Build model based on settings from config
    model = get_model(config).to(device)
    print("Model:{}".format(model))
    # Get training and validation data loaders
    train_loader, val_loader = get_dataloaders(config)

    # Begin training
    train_model(model, train_loader, val_loader, config, device)

if __name__ == "__main__":
    main()