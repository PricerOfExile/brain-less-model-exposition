import importlib.util
import os.path

import torch
import json

from fastapi import FastAPI, Request

from generic_pytorch_model import NNCOnfig, ConfigurableNN


app = FastAPI()

model = None
transofm_json_input_to_list = None

@app.on_event("startup")
def load_model():
    global model, transofm_json_input_to_list

    model_architecture_path = 'model_architecture.json'
    if os.path.exists(model_architecture_path):
        with open(model_architecture_path, 'r') as f:
            model_config = NNCOnfig.from_json(json.load(f))
    else:
        print(f"Model architecture file not found at {model_architecture_path}")

    model_path = 'model.pth'
    if os.path.exists(model_path):
        model = ConfigurableNN(model_config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print(f"Model file not found at {model_path}")

    transfomer_path = 'transformer.py'
    if os.path.exists(transfomer_path):
        spec = importlib.util.spec_from_file_location("transformer", transfomer_path)
        transfomer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transfomer_module)
        transfomer_json_input_to_list = transfomer_module.transform
    else:
        print(f"Transformer file not found at {transfomer_path}")


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    result = None

    input_tensor = None
    output = None

    if transofm_json_input_to_list is not None:
        input_list = transfomer_json_input_to_list(data)
        input_tensor = torch.tensor(input_list, dtype=torch.float32)
    else:
        print("Transformer function not loaded yet")

    if model is not None:
        with torch.no_grad():
            output = model(input_tensor)
            result = output.numpy().tolist()
    else:
        print("Model not loaded yet")

    return {"prediction": result if result is not None else "Model not loaded yet"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)