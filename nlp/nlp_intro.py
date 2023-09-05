"""
    The library needs to be configured with your account's secret key which is available 
    on the website. 
    Either set it as the OPENAI_API_KEY environment variable before using the library
"""

import openai
openai.api_key = "sk-..."

# list models
models = openai.Model.list()

# print the first model's id
print(models.data[0].id)

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])

# print the chat completion
print(chat_completion.choices[0].message.content)