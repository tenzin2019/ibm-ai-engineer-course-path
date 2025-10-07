import ibm_watsonx_ai

client = ibm_watsonx_ai.Client(
    project_id="your_project_id",
    api_key="your_api_key"
)

client.list_models()

client.get_model(model_id="your_model_id")

from langchain_ibm import WatsonxLLM

model = WatsonxLLM(model=model)

model.generate("What is the capital of France?")



