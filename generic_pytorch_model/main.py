import importlib.util
import os.path

import torch
import json

import logging
import uuid
import contextvars
from fastapi import FastAPI, Request, HTTPException

from generic_pytorch_model import NNCOnfig, ConfigurableNN

correlation_id_var = contextvars.ContextVar("correlation_id", default=None)


app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("generic_pytorch_model")


model = None
transform_json_input_to_list = None
response_translator = None


@app.middleware("http")
async def log_requests_response(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    correlation_id_var.set(correlation_id)
    logger.info(f"Incoming Request: {request.method} {request.url}", extra={"correlation_id": correlation_id})

    response = await call_next(request)

    response.headers['X-Correlation-ID'] = correlation_id
    logger.info(f"Completed Request: {response.status_code}", extra={"correlation_id": correlation_id})

    return response


@app.on_event("startup")
def load_model():
    global model, transform_json_input_to_list, response_translator
    current_correlation_id = correlation_id_var.get()


    model_architecture_path = 'model_architecture.json'
    if os.path.exists(model_architecture_path):
        logger.info("Model architecture found... building the model config.", extra={"correlation_id": current_correlation_id})
        with open(model_architecture_path, 'r') as f:
            model_config = NNCOnfig.from_json(json.load(f))
    else:
        logger.warning(f"Model architecture file not found at {model_architecture_path}.", extra={"correlation_id": current_correlation_id})

    model_path = 'model.pth'
    if os.path.exists(model_path):
        logger.info("Model found... loading the config into the model.", extra={"correlation_id": current_correlation_id})
        model = ConfigurableNN(model_config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        logger.warning(f"Model file not found at {model_path}.", extra={"correlation_id": current_correlation_id})

    transfomer_path = 'transformer.py'
    if os.path.exists(transfomer_path):
        logger.info("Transformer file found... loading the transformer function.", extra={"correlation_id": current_correlation_id})
        spec = importlib.util.spec_from_file_location("transformer", transfomer_path)
        transfomer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transfomer_module)
        transform_json_input_to_list = transfomer_module.transform
    else:
        logger.warning(f"Transformer file not found at {transfomer_path}.", extra={"correlation_id": current_correlation_id})

    translator_path = 'translator.py'
    if os.path.exists(translator_path):
        logger.info("Translator file found... loading the translator function.", extra={"correlation_id": current_correlation_id})
        translator_spec = importlib.util.spec_from_file_location("translator", translator_path)
        translator_module = importlib.util.module_from_spec(translator_spec)
        translator_spec.loader.exec_module(translator_module)
        response_translator = translator_module.translate
    else:
        logger.warning(f"Translator file not found at {translator_path}", extra={"correlation_id": current_correlation_id})


@app.post("/predict")
async def predict(request: Request):
    global model, transform_json_input_to_list, response_translator
    current_correlation_id = correlation_id_var.get()


    data = await request.json()
    result = None

    input_tensor = None
    output = None

    if transform_json_input_to_list is not None:
        input_list = transform_json_input_to_list(data)
        input_tensor = torch.tensor(input_list, dtype=torch.float32).view(1, -1)
    else:
        logger.error("Transformer function not loaded. Unable to transform input into list for tensor.", extra={"correlation_id": current_correlation_id})
        raise HTTPException(
            status_code=500,
            detail={
                "message": "transformer function not loaded"
            })

    if model is not None:
        with torch.no_grad():
            output = model(input_tensor)
            result = output.numpy().tolist()
    else:
        logger.error("Model not loaded. Unable to make prediction.", extra={"correlation_id": current_correlation_id})
        raise HTTPException(
            status_code=500,
            detail={
                "message": "model not loaded"
            }
        )

    if response_translator is not None:
        return {"prediction": response_translator(*result)}
    else:
        logger.error("Translator function not loaded. Unable to interpret result", extra={"correlation_id": current_correlation_id})
        raise HTTPException(
            status_code=500,
            detail={
                "message": "translator function not loaded"
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)