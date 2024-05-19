Requirements:
Python 3
Cuda-compatible GPU with > 20 GB VRAM

Installation:
pip install -U torch googletrans peft discord-py-interactions accelerate transformers diffusers

Create config file with secrets:
touch config.json

{
    "serv_id": "server ID goes here",
    "bot_token": "Bot token goes there"
}