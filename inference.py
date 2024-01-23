import argparse
import yaml
import os
from main import load_trained_pipeline

def config_2_args(path):
    with open(path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="Generate args from yaml")
    for key, value in yaml_data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args([])
    return args

def run_inference(config_path: str, prompt_postfix: str, num_inference_steps=5, guidance_scale=1.0, save_images=False, **kwargs):
    args = config_2_args(config_path)
    
    lora_path = os.path.join(model_path, f"checkpoint-{args.checkpointing_steps * args.num_train_epochs}")
    loop = 2
    model_path = os.path.join(args.output_dir, args.character_name, str(loop))

    pipe = load_trained_pipeline(
        argparse.Namespace(
            pretrained_model_name_or_path=model_path,
            lcm_lora=True,
        ), 
        load_lora=True,
        lora_path=lora_path,
    )


    image_postfix = prompt_postfix.replace(" ", "_")
    
    # remember to use the place holder here
    prompt = f"A photo of {args.placeholder_token}{prompt_postfix}."
    images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, **kwargs).images
    if save_images:
        output_folder = f"./inference_results/{args.character_name}"
        os.makedirs(output_folder, exist_ok=True)
        for i, image in enumerate(images):
            image.save(os.path.join(output_folder, f"{args.character_name}_{image_postfix}_{i}.png"))
    return images

if __name__ == "__main__":
    run_inference("config/theChosenOne.yaml", " sitting on a rocket.")
