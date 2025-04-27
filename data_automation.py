import boto3
import os
import json
import logging
from botocore.exceptions import ClientError
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv # 添加 dotenv 支持
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # 添加 uvicorn 運行服務

# 配置日誌記錄
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # 更好的日誌格式

#===========================通用設定===========================
# 加载 .env 文件
load_dotenv('param.env')
# 從環境變數讀取 AWS 憑證和預設區域 (boto3 會自動查找這些標準環境變數)
AWS_ACCESS_KEY_ID  =os.getenv('AWS_ACCESS_KEY_ID') # 讀取但這裡不顯式使用
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')# 讀取但這裡不顯式使用
AWS_SESSION_TOKEN  =os.getenv('AWS_SESSION_TOKEN') # 讀取但這裡不顯式使用
AWS_DEFAULT_REGION =os.getenv('AWS_DEFAULT_REGION') # 可用於配置 boto3 區域，或作為備用

# Knowledge Base 相關設定
KNOWLEDGE_BASE_ID = "TSBXVVFWBF"
AWS_REGION = "us-west-2" # <-- 請確認並修改此區域！

GENERATION_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0" # <-- 請確認並修改此行！

#===========================AWS 客戶端初始化===========================


bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name=AWS_REGION, # 使用 KB 所在的區域
    # 如果要使用顯式憑證 (不建議在 EC2 上這樣做，除非必要)，取消下面註釋：
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN
)

app = FastAPI()

# 設定 CORS 中間件
app.add_middleware(
CORSMiddleware,
allow_origins=["*"], # 開發時使用 "*" 方便測試，上線時請設定為允許的來源網址
allow_credentials=True,
allow_methods=["*"], # 通常 API 只會用到 GET, POST 等
allow_headers=["*"],
)

#===========================Knowledge Base 查詢端點===========================

# 定義 Knowledge Base 查詢請求體的模型
class KnowledgeBaseQueryRequest(BaseModel):
    query: str
    max_results: int = 3 # 設置一個默認值，可以從請求中覆蓋傳遞給 LLM 的 Passage 數量

# 定義一個 FastAPI 端點來處理知識庫查詢請求
@app.post("/query_knowledge_base", status_code=status.HTTP_200_OK, tags=["knowledge_base"], summary="Query Knowledge Base")
async def query_knowledge_base_endpoint(query_request: KnowledgeBaseQueryRequest):
    """
    接收查詢請求，呼叫 Bedrock Knowledge Base 進行檢索和生成回答。
    """
    logger.info(f"收到知識庫查詢請求: {query_request.query}")
    logger.info(f"將使用 Knowledge Base ID: {KNOWLEDGE_BASE_ID}, Region: {AWS_REGION}, Model ID: {GENERATION_MODEL_ID}")

    try:
    # 呼叫 bedrock-agent-runtime 的 retrieve_and_generate API
        response = bedrock_agent_runtime.retrieve_and_generate(
            input={
            'text': query_request.query
            },
            retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',

                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                    'modelArn': GENERATION_MODEL_ID, # 請確認並修改此模型 ID！

                    'retrievalConfiguration': {
                    'vectorSearchConfiguration': {
                    'numberOfResults': query_request.max_results
                    }
                    },

                    'generationConfiguration': {
                    'inferenceConfig': {
                    'textInferenceConfig': {
                    'maxTokens': 1000,
                    'temperature': 0.2,
                    'topP': 0.9
                    }
                    },

                'promptTemplate': """
                您是一位專業且經驗豐富的建築法規專家。你的任務是根據使用者提供的搜尋結果來回答問題。請嚴格遵守以下要求：

                    1. 僅能根據搜尋結果資訊作答，不得使用常識推論或自行假設。
                    2. 逐條列出重點事項, 每點後面必須用句號結尾, 至少列出2點以上, 且每點要有清楚簡述。
                    3. 回覆字數必須超過100字且不超過200字。如果內容不足, 請適度補充搜尋結果相關資訊以達到字數要求。
                    4. 回答時需條理清晰，逐點分類，內容準確簡潔，且整體回答需有完整標點符號。
                    5. 若搜尋結果中無法找到明確答案，請明確回覆：「無法在搜尋結果中找到該問題的確切答案。」並說明找不到的原因（例如：資料過於籠統、不涵蓋該主題等）。
                    6. 使用者的陳述不一定正確，請務必核對搜尋結果，僅在確認吻合時才可引用其陳述。
                    7. 回答時都必須用繁體中文回應。

                    Here are the search results in numbered order:
                    $search_results$

                    $output_format_instructions$

                """, # === Prompt Template 結束 ===

                            'safeguardsConfiguration': {

                                'guardrailIdentifier': 'arn:aws:bedrock:us-west-2:654304324935:guardrail/m2ykp00ff7th',
                                'guardrailVersion': '1'
                            }
                        }
                    }
                }
            )       

    # ... 後面的錯誤處理和返回部分不變 ...

        logger.info("Bedrock Knowledge Base 檢索並生成請求成功！")

        # 從響應中提取生成的回答和來源引用
        generated_response = response.get('output', {}).get('text', 'N/A')
        citations = response.get('citations', [])

        # 將 citations 格式化得更簡潔，方便 API 返回
        formatted_citations = []
        for citation in citations:
             references = citation.get('retrievedReferences', [])
             for ref in references:
                  content = ref.get('content', {}).get('text', 'N/A')
                  uri = ref.get('location', {}).get('s3Location', {}).get('uri', 'N/A')
                  formatted_citations.append({
                      "content_snippet": content,
                      "source_uri": uri
                  })

        # 返回成功的 JSON 響應
        return {
            'answer': generated_response,
            'citations': formatted_citations
        }

    except ClientError as e:
        # 捕獲 Boto3 的特定錯誤
        error_code = e.response['Error']['Code']
        error_message = e.response['Error'].get('Message', 'Unknown error from AWS API.')
        logger.error(f"AWS ClientError calling Bedrock KB: {error_code} - {error_message}")

        # 根據錯誤類型返回不同的 HTTP 狀態碼和信息
        if error_code == 'AccessDeniedException':
             raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"AWS Permissions Error: Access denied. {error_message}")
        elif error_code == 'ValidationException':
             # 這可能是模型 ID 錯誤或請求參數錯誤導致的錯誤
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"AWS Validation Error: Check Bedrock model ID ({GENERATION_MODEL_ID}) or request parameters. {error_message}")
        elif error_code == 'UnrecognizedClientException':
             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"AWS Credentials Error: Invalid security token. {error_message}")
        # 可以根據需要添加其他錯誤碼處理，例如 ResourceNotFoundException 如果 KB ID 錯誤
        elif error_code == 'ResourceNotFoundException':
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Bedrock Knowledge Base not found: {KNOWLEDGE_BASE_ID}. {error_message}")
        # 其他 Bedrock 相關錯誤
        elif error_code.startswith('Bedrock'):
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bedrock API Error: {error_code} - {error_message}")
        else:
             # 捕獲其他未預期的 AWS 錯誤
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"AWS API error: {error_code} - {error_message}")
    except Exception as e:
        # 捕獲其他任何未預期的錯誤
        logger.error(f"Endpoint /query_knowledge_base encountered unexpected error: {e}", exc_info=True) # 打印詳細錯誤信息
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error processing knowledge base query: {e}")


#===========================服務啟動===========================

# 在腳本直接執行時啟動 FastAPI 服務
if __name__ == "__main__":
    # 請確保您已安裝 uvicorn：pip install uvicorn
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=5005)