import boto3
import os
import base64
import io
import json
import logging
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


def generate_image(model_id, body):
    """
    Generate an image using Amazon Nova Canvas  model on demand.
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    Returns:
        image_bytes (bytes): The image generated by the model.
    """

    logger.info(
        "Generating image with Amazon Nova Canvas model %s", model_id)

    bedrock = boto3.client(
        service_name='bedrock-runtime',
        config=Config(read_timeout=300), 
        region_name="us-east-1"
    )

    accept = "application/json"
    content_type = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())

    finish_reason = response_body.get("error")

    if finish_reason is not None:
        raise ImageError(f"Image generation error. Error is {finish_reason}")

    logger.info(
        "Successfully generated image with Amazon Nova Canvas model %s", model_id)
   

    return response_body

# 載入並編碼圖片
# def load_and_encode_image(image_path):
#     with open(image_path, 'rb') as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

@app.post("/style_convert", 
         status_code = status.HTTP_200_OK, 
         tags=["nova_canvas"],
         summary="style convertion",
         description="To convert the style of the image.",
         response_description="designed picture")

def nova_canvas(item: Item_design):
    """
    Entrypoint for Amazon Nova Canvas  example.
    """
    try:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s: %(message)s")
        
        # 指定你的 S3 bucket 名稱
        bucket_name = os.getenv('bucket_name')
        bucket_name_new = os.getenv('bucket_name')

        picture_filename = item.file_path_s3
        pure_filename = os.path.basename(picture_filename).split('.')[0]
        
        download_file_from_s3(bucket_name, picture_filename, picture_filename)

        model_id = 'amazon.nova-canvas-v1:0'

        # 準備輸入圖片
        # filename_list = ['frame_0.png', 'frame_520.png', 'frame_1040.png']  # 替換為您的圖片路徑
        # input_image = [load_and_encode_image(path) for path in filename_list]
        
        # # Read image from file and encode it as base64 string.
        with open(picture_filename, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode('utf8')

        body = json.dumps({
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": "Modernize the house, photo-realistic, 8k, hdr",
                "negativeText": "bad quality, low resolution, cartoon",
                "images": [input_image],
                "similarityStrength": 0.95,  # Range: 0.2 to 1.0
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 512,
                "width": 512,
                "cfgScale": 8.0,
                "seed": 300
            }
        })

        response_body = generate_image(model_id=model_id, body=body)

        for i in range(len(response_body.get("images"))):
            base64_image = response_body.get("images")[i]
            base64_bytes = base64_image.encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)

            design_image = Image.open(io.BytesIO(image_bytes))
            design_image.show()
            picture_filename_new = pure_filename+'_designed.jpg'
            design_image.save(picture_filename_new)

            # upload file to S3
            upload_file_to_s3(bucket_name_new, picture_filename_new, picture_filename_new)

            file_url = "https://testviedo-gen.s3.us-west-2.amazonaws.com/"+picture_filename_new
            return_json = json.dumps({"file_url":file_url})

        return return_json

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
