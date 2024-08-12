import os 
from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.inputs import TokensPrompt
from modelscope import AutoTokenizer, GenerationConfig,snapshot_download
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from prompt_utils import _build_prompt,remove_stop_words
import uuid
import json 
import time

# http接口服务
app=FastAPI()

# vLLM参数
model_dir="/nvme/huggingface/hub/opt-1.3b"
dtype='float16'

# vLLM模型加载
def load_vllm():
    global generation_config,tokenizer,stop_words_ids,engine    
    # 模型下载
    snapshot_download(model_dir)
    # 模型基础配置
    generation_config=GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
    # 加载分词器
    tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    tokenizer.eos_token_id=generation_config.eos_token_id
    # 推理终止词
    stop_words_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_token_id]
    # vLLM基础配置
    args=AsyncEngineArgs(model_dir)
    args.worker_use_ray=False
    args.engine_use_ray=False
    args.tokenizer=model_dir
    args.trust_remote_code=True
    args.dtype=dtype
    args.max_num_seqs=20    # batch最大20条样本
    # 加载模型
    os.environ['VLLM_USE_MODELSCOPE']='True'
    engine=AsyncLLMEngine.from_engine_args(args)
    return generation_config,tokenizer,stop_words_ids,engine

generation_config,tokenizer,stop_words_ids,engine=load_vllm()

# 用户停止句匹配
def match_user_stop_words(response_token_ids,user_stop_tokens):
    for stop_tokens in user_stop_tokens:
        if len(response_token_ids)<len(stop_tokens):
            continue 
        if response_token_ids[-len(stop_tokens):]==stop_tokens:
            return True  # 命中停止句, 返回True
    return False

# chat对话接口
@app.post("/chat")
async def chat(request: Request):
    request=await request.json()
    input_tokens=request.get('tokens',None)
    max_tokens=request.get('max_tokens', 10)
    history=request.get('history',[])
    system=request.get('system','You are a helpful assistant.')
    stream=False
    user_stop_words=[]
    
    if input_tokens is None:
        return Response(status_code=502,content='input_tokens is empty')

    # 用户停止词
    user_stop_tokens=[]
    for words in user_stop_words:
        user_stop_tokens.append(tokenizer.encode(words))
        
    # vLLM请求配置
    sampling_params=SamplingParams(top_p=0.9,
                                    top_k=-1,
                                    temperature=0.1,
                                    max_tokens=max_tokens,
                                    min_tokens=max_tokens,
                                    ignore_eof=Ture,
                                    )
    # vLLM异步推理（在独立线程中阻塞执行推理，主线程异步等待完成通知）
    request_id=str(uuid.uuid4().hex)
    results_iter=engine.generate(inputs = TokensPrompt(prompt_token_ids=input_tokens), sampling_params=sampling_params,request_id=request_id)
    
    # 流式返回，即迭代transformer的每一步推理结果并反复返回
    if False and stream:
        async def streaming_resp():
            start = time.perf_counter()
            lags = []
            all_text = ""
            async for result in results_iter:
                now = time.perf_counter()
                lags.append(now-start)
                start = now
                # 移除im_end,eos等系统停止词
                token_ids=remove_stop_words(result.outputs[0].token_ids,stop_words_ids)
                # 返回截止目前的tokens输出                
                text=tokenizer.decode(token_ids)
                all_text += text
                # yield (json.dumps({'text':text})+'\0').encode('utf-8')
                # 匹配用户停止词,终止推理
                if match_user_stop_words(token_ids,user_stop_tokens):
                    await engine.abort(request_id)   # 终止vllm后续推理
                    break
            response = {
                "output": all_text,
                "ttft": lags[0] if len(lags) > 0 else -1,
                "max-tbt": max(lags[1:]) if len(lags) > 1 else -1
            }
            return JSONResponse(response)
        return StreamingResponse(streaming_resp())

    # 整体一次性返回模式
    start = time.perf_counter()
    lags = []
    all_text = ""
    async for result in results_iter:
        now = time.perf_counter()
        lags.append(now-start)
        start = now
        # 移除im_end,eos等系统停止词
        token_ids=remove_stop_words(result.outputs[0].token_ids,stop_words_ids)
        # 返回截止目前的tokens输出                
        text=tokenizer.decode(token_ids)
        all_text += text
        # yield (json.dumps({'text':text})+'\0').encode('utf-8')
        # 匹配用户停止词,终止推理
        if match_user_stop_words(token_ids,user_stop_tokens):
            await engine.abort(request_id)   # 终止vllm后续推理
            break
    response = {
        "output": all_text,
    }
    headers = {"x-first-token-time": lags[0] if len(lags) > 0 else -1, 
               "x-inference-time": lags[-1]-lags[0] if len(lags) > 1 else -1,
               "x-max-time-between-tokens": max(lags[1:]) if len(lags) > 1 else -1
               }
    return JSONResponse(content=response, headers=headers)

if __name__=='__main__':
    uvicorn.run(app,
                host=127.0.0.1,
                port=5060,
                log_level="debug")