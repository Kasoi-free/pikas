import json
from diffusers import DiffusionPipeline, TCDScheduler, AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
import torch
from interactions import Client, Intents, listen, slash_command, SlashContext, OptionType, slash_option, File
from huggingface_hub import hf_hub_download
from googletrans import Translator
from safetensors.torch import load_file
from diffusers.utils import export_to_gif

# bot secrets
with open('config.json') as config_file:
    config = json.load(config_file)
    serv_id = config['serv_id']
    bot_token = config['bot_token']
bot = Client(intents=Intents.DEFAULT)

# translator preps
translator = Translator()

# generative model preps
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-1step-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Use TCD scheduler to achieve better image quality
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
# Lower eta results in more detail for multi-steps inference
eta = 1.0

prompt = "A majestic lion jumping from a big stone at night"

# Animation model setup
device = "cuda"
dtype = torch.float16

step = 8  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_8step_diffusers.safetensors"
base = "emilianJR/epiCRealism"  # Choose your favorite base model.

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
anim_pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
anim_pipe.scheduler = EulerDiscreteScheduler.from_config(anim_pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

# State tracking variable
last_function_called = None

# Slash command 1
@slash_command(name="genpic",
               description="Generate a picture",
               # scopes=serv_id
               )
@slash_option(
    name="text",
    description="prompt",
    required=True,
    opt_type=OptionType.STRING
)
async def pic_gen(ctx: SlashContext, text: str):
    await ctx.defer()

    filename, new_text = generating_image(text)
    files = [File(filename)]

    # Including new_text in the message
    await ctx.send(content=f"Prompt: {new_text}", files=files)

# Slash command 2
@slash_command(name="animate",
               description="Generate an animation",
               # scopes=serv_id
               )
@slash_option(
    name="text",
    description="prompt",
    required=True,
    opt_type=OptionType.STRING
)
async def animate(ctx: SlashContext, text: str):
    await ctx.defer()

    filename, new_text = generating_animation(text)
    files = [File(filename)]

    # Including new_text in the message
    await ctx.send(content=f"Prompt: {new_text}", files=files)

# GENERATING IMAGE
def generating_image(prompt):
    global last_function_called

    try:
        translated_text = translator.translate(prompt, dest='en')
        new_text = str(translated_text.text)
    except:
        translated_text = prompt
        new_text = str(prompt)

    image = pipe(prompt=new_text, num_inference_steps=8, guidance_scale=0, eta=eta).images[0]
    filename = "pic.png"
    image.save(filename)

    # Conditionally flush CUDA cache
    if last_function_called == "generating_animation":
        torch.cuda.empty_cache()

    last_function_called = "generating_image"
    return filename, new_text

# GENERATING ANIMATION
def generating_animation(prompt):
    global last_function_called

    try:
        translated_text = translator.translate(prompt, dest='en')
        new_text = str(translated_text.text)
    except:
        translated_text = prompt
        new_text = str(prompt)

    output = anim_pipe(prompt=new_text, guidance_scale=1.0, num_inference_steps=step)
    filename = "animation.gif"
    export_to_gif(output.frames[0], filename)

    # Conditionally flush CUDA cache
    if last_function_called == "generating_image":
        torch.cuda.empty_cache()

    last_function_called = "generating_animation"
    return filename, new_text

bot.start(bot_token)
