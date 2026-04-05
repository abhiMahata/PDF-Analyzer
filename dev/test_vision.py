import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import base64



load_dotenv(override=True)

try:
    llm = ChatGroq(model="llama-3.2-11b-vision-preview")
    

    msg = HumanMessage(content="Hello, are you available?")
    res = llm.invoke([msg])
    print("SUCCESS: ", res.content)
except Exception as e:
    print("ERROR: ", e)
