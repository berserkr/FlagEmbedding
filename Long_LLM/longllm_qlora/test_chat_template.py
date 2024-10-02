from transformers import AutoTokenizer
from src.chat import apply_chat_template

messages = [ 
        {'role' : 'user', 'content' : 'Hello, how are you?'}, {'role' : 'assistant', 'content' : 'I am good thank you, and you?'},
        {'role' : 'user', 'content' : 'Hello, how are you?'}, {'role' : 'assistant', 'content' : 'I am so so. Thanks.'},
]

messages = messages[:2] # + messages + messages
print(messages)

tokenizer=AutoTokenizer.from_pretrained('/proj/checkpoints/stallone/models/preview/granite-3b-instruct-preview-4k-r240917a')

print(100*'-')

tokens=tokenizer.apply_chat_template(messages)
print(tokens)

print(tokenizer.decode(tokens))

#print(100*'-') 

#res = apply_chat_template('llama-3', messages, system_message=None, tokenizer=tokenizer, return_labels=True)
#print(res)
#print(tokenizer.decode([59, 3860, 4644, 17715, 844, 30, 461, 844, 44823, 110, 87, 366, 81, 314, 28318]))

print(100*'-')  

res = apply_chat_template('granite', messages, system_message=None, tokenizer=tokenizer, return_labels=True)
print(res)

print(100*'-')  

print(tokenizer.decode([59, 3860, 4644, 17715, 844, 30, 461, 844, 49, 0, 203]))
print(tokenizer.decode([ 59, 3860, 1259, 1259, 32, 5647, 32, 0]))

