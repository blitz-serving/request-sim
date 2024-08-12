import os 
from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.inputs import TokensPrompt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import uuid
from transformers import AutoTokenizer
import json 
import time

# http接口服务
app=FastAPI()

# vLLM参数
model_dir="/nvme/huggingface/hub/opt-1.3b"
dtype='float16'

# vLLM模型加载
def load_vllm():
    global tokenizer,engine    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    args=AsyncEngineArgs(model_dir)
    args.worker_use_ray=False
    args.engine_use_ray=False
    args.tokenizer=model_dir
    args.dtype=dtype
    args.max_num_seqs=20 
    engine=AsyncLLMEngine.from_engine_args(args)
    return tokenizer,engine

tokenizer,engine=load_vllm()

@app.post("/chat")
async def chat(request: Request):
    request=await request.json()
    input_tokens=request.get('tokens',None)
    max_tokens=request.get('max_tokens', 10)
    print(f"max tokens: {max_tokens}")
    history=request.get('history',[])
    user_stop_words=[]
    
    if input_tokens is None:
        return Response(status_code=502,content='input_tokens is empty')
        
    sampling_params=SamplingParams(top_p=0.9,
                                    top_k=-1,
                                    temperature=0.1,
                                    max_tokens=max_tokens,
                                    min_tokens=max_tokens,
                                    )
    request_id=str(uuid.uuid4().hex)
    results_iter=engine.generate(inputs = TokensPrompt(prompt_token_ids=input_tokens), sampling_params=sampling_params,request_id=request_id)

    start = time.perf_counter()
    lags = []
    all_text = ""
    async for result in results_iter:
        now = time.perf_counter()
        lags.append(now-start)
        start = now
    response = {
        "output": "haha",
    }
    if len(lags) == 0:
        lags.append(0)
    headers = {"x-first-token-time": str(lags[0]), 
               "x-inference-time": str((sum(lags[1:])if len(lags) > 1 else 0) + lags[0]),
               "x-max-time-between-tokens": str(max(lags[1:]) if len(lags) > 1 else -1)
               }
    return JSONResponse(content=response, headers=headers)

if __name__=='__main__':
    uvicorn.run(app,
                host="127.0.0.1",
                port=5060,
                log_level="debug")