import jax
import jax.numpy as jnp
import haiku as hk
import sys
import os
import numpy as np

# Add the parent directory to the system path to resolve the import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.architecture import VishwamAILLM
from src.training.trainer import VishwamAITrainer
from src.data.dataset import create_dataset
from transformers import AutoTokenizer
import yaml

def main():
    # Load configuration
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs/default_config.yaml'))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    config['pad_token_id'] = config.get('pad_token_id', tokenizer.pad_token_id)

    # Add mathematical symbols to tokenizer
    math_symbols = ['+', '-', '*', '/', '=', '(', ')', '^', 'sqrt', 'pi', 'e']
    tokenizer.add_tokens(math_symbols)

    # Create datasets
    train_dataset = create_dataset(config['train_file'], tokenizer, config['batch_size'], config['max_seq_length'])
    eval_dataset = create_dataset(config['eval_file'], tokenizer, config['batch_size'], config['max_seq_length'])

    # Initialize model
    def model_fn(inputs):
        model = VishwamAILLM(config)
        return model(inputs, is_training=True)

    model = hk.transform(model_fn)

    # Initialize trainer
    trainer = VishwamAITrainer(model, config)

    # Train model
    rng_key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config['max_seq_length']), dtype=jnp.int32)
    params = model.init(rng_key, dummy_input)
    trained_params = trainer.train(params, train_dataset, eval_dataset, config['num_epochs'])

    # Save trained parameters
    model.save_pretrained('/home/ubuntu/chat-agent/VishwamAI-main/saved_models')

    # Save checkpoint after each epoch
    checkpoint_dir = '/home/ubuntu/chat-agent/VishwamAI-main/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.npy')
    np.save(checkpoint_path, trained_params)

if __name__ == "__main__":
    main()
