import os
from dotenv import load_dotenv
from util.constants import LLM_MODE
import lmstudio as lms
from openai import OpenAI
from util.groq_llm_client import GROQLLM

# Load environment variables from the .env file
load_dotenv()

class LLMClient:
    def __init__(self, json_format=False, temp=None):
        self.temp = temp
        self.json_format = json_format
        
        if LLM_MODE == 'openai':
            # Support custom base_url for local vLLM or OpenAI-compatible services
            openai_kwargs = {}
            base_url = os.getenv('OPENAI_BASE_URL')
            api_key = os.getenv('OPENAI_API_KEY', 'sk-local-vllm')
            if base_url:
                openai_kwargs['base_url'] = base_url
            openai_kwargs['api_key'] = api_key
            self.client = OpenAI(**openai_kwargs)
        elif LLM_MODE == 'groq':
            self.client = GROQLLM(response_format_json=json_format, temp=temp)
        else:
            self.model = os.getenv("LLM_OFFLINE_MODEL")
            if not self.model:
                raise ValueError("LLM_OFFLINE_MODEL is not set in the environment variables.")
            
            self.client = lms.llm(self.model)

    

    def generate(self, system_prompt: str, user_prompt: str):
        if LLM_MODE == 'openai':
            # Model name configurable via VERSIONRAG_LLM_MODEL env var
            model_name = os.getenv('VERSIONRAG_LLM_MODEL', 'gpt-4o-mini')
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }

            if self.temp is not None:
                kwargs["temperature"] = self.temp

            if self.json_format:
                kwargs["response_format"] = {"type": "json_object"}
                
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        elif LLM_MODE == 'groq':
            response = self.client.invoke(system_instruction=system_prompt, input=user_prompt)
            return response.content
        else:
            config = {}
            if self.temp is not None:
                config["temperature"] = self.temp

            if self.json_format:
                config["response_format"] = {"type": "json_object"}
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.client.respond({"messages": messages}, config=config)
            return response.content