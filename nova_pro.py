import boto3
import os
import base64
import io
import json
import logging
import magic
from PIL import Image
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv  # 添加 dotenv 支持
from fastapi.middleware.cors import CORSMiddleware


#===========================S3 Setting===========================
# 加载 .env 文件
load_dotenv('param.env')
AWS_DEFAULT_REGION   =os.getenv('AWS_DEFAULT_REGION')
AWS_ACCESS_KEY_ID    =os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN    =os.getenv('AWS_SESSION_TOKEN')  
#===========================S3 Setting===========================

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 上線時要設定安全一點
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# input: encoded image
# output: S3 path (designed image)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ImageError(Exception):
    "Custom exception for errors returned by Amazon Nova Canvas"

    def __init__(self, message):
        self.message = message

# Item Class 繼承 BaseModel
class Item_design(BaseModel):  
    file_path_s3: str

# 上傳檔案到 S3
def upload_file_to_s3(bucket_name, local_file_path, s3_file_key):
    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_key)
        print(f"成功上傳 {local_file_path} 到 s3://{bucket_name}/{s3_file_key}")
    except Exception as e:
        print("上傳失敗：", e)

# 從 S3 下載檔案到本地
def download_file_from_s3(bucket_name, s3_file_key, local_file_path):
    try:
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"成功從 s3://{bucket_name}/{s3_file_key} 下載到 {local_file_path}")
    except Exception as e:
        print("下載失敗：", e)


# 載入並編碼圖片
# def load_and_encode_image(image_path):
#     with open(image_path, 'rb') as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

@app.post("/image_gen_text", 
         status_code = status.HTTP_200_OK, 
         tags=["nova_pro"],
         summary="generate text",
         description="To generate the text of the image.",
         response_description="recommendation")

def nova_pro(item: Item_design):
    """
    Entrypoint for Amazon Nova Pro example.
    """
    try:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s: %(message)s")
        
        # 指定你的 S3 bucket 名稱
        bucket_name = os.getenv('bucket_name')
        bucket_name_new = os.getenv('bucket_name_new')

        picture_filename = item.file_path_s3
        # pure_filename = os.path.basename(picture_filename).split('.')[0]
        pure_filename = picture_filename.split('.')[0]
        
        download_file_from_s3(bucket_name, picture_filename, picture_filename)

        model_id = 'us.amazon.nova-pro-v1:0'
        # model_id = 'anthropic.claude-3-7-sonnet-20250219-v1:0'

        # client = AnthropicBedrock(
        #     aws_access_key=AWS_ACCESS_KEY_ID,
        #     aws_secret_key=AWS_SECRET_ACCESS_KEY,
        #     aws_region="us-east-1"
        # )
        
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            config=Config(retries={'max_attempts': 3, 'mode': 'standard'}), 
            region_name="us-east-1"
        )

        # 準備輸入圖片
        # filename_list = ['frame_0.png', 'frame_520.png', 'frame_1040.png']  # 替換為您的圖片路徑
        # input_image = [load_and_encode_image(path) for path in filename_list]
        
        # # Read image from file and encode it as base64 string.
        with open(picture_filename, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode('utf8')

        # image_type = magic.from_buffer(input_image, mime=True)

        body = json.dumps({"messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "text": "Describe this house design in detail, and suggest the layout of the entire space."
                                    }
                                ]
                            }],
                            "inferenceConfig": {
                                "temperature": 0.2,
                                "maxTokens": 1000
                            }
                        })



        # recipe_content = message.content[0].text
        # print(recipe_content)
        # with open("output.txt", "w", encoding="utf-8") as f:
        #     f.write("這是第一行文字。\n")

        # 呼叫 Nova Pro 模型
        # response_body = bedrock.invoke_model(
        #     modelId=model_id,
        #     contentType="application/json",
        #     accept="application/json",
        #     body=payload
        # )

        accept = "application/json"
        content_type = "application/json"

        response = bedrock.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=content_type
        )

        # response_body = json.loads(response.get("body").read())
        # print(response_body)

        # 回传结果解析
        result = json.loads(response.get("body").read())
        # result.get("text", "No description generated")
        result2 = result["output"]["message"]["content"][0]["text"]
        print(type(result2))
        text_filename = pure_filename+".txt"
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(result2)
        
        # upload file to S3
        upload_file_to_s3(bucket_name_new, text_filename, text_filename)

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " +
              format(message))
    except ImageError as err:
        logger.error(err.message)
        print(err.message)

    else:
        print(
            f"Finished generating image with Amazon Nova Canvas  model {model_id}.")
