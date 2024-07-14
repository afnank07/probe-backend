import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List
from dotenv import load_dotenv
import requests
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import anthropic
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS").split(',')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Allows the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the GraphQL endpoint
SANTIMENT_URL = 'https://api.santiment.net/graphql'

# Define request model
class MetricRequest(BaseModel):
    metrics: List[str]
    token: str
    startDate: str
    endDate: str
    interval: str

    @validator('metrics')
    def check_metrics(cls, v):
        if not v:
            raise ValueError('At least one metric name must be provided')
        return v

    # @validator('startDate', 'endDate')
    # def check_date_format(cls, v):
    #     try:
    #         datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
    #     except ValueError:
    #         raise ValueError('Incorrect date format, should be YYYY-MM-DDTHH:MM:SSZ')
    #     return v

    @validator('interval')
    def check_interval(cls, v):
        valid_intervals = ['5m', '15m', '30m', '1h', '4h', '1d']
        if v not in valid_intervals:
            raise ValueError(f'Invalid interval. Must be one of: {", ".join(valid_intervals)}')
        return v

def create_metric_query(metric_name: str, token: str, startDate: str, endDate: str, interval: str):
    return f"""
    {{
  getMetric(metric: "{metric_name}") {{
    timeseriesData(
      slug: "{token}"
      from: "{startDate}"
      to: "{endDate}"
      interval: "{interval}"
    ) {{
      value
      datetime
    }}
  }}
}}
    """

async def fetch_santiment_data(query: str):
    headers = {
        'Content-Type': 'application/graphql',
        'Authorization': f'Apikey {os.getenv("SANTIMENT_API_KEY")}',
    }
    response = requests.post(SANTIMENT_URL, data=query, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from Santiment")

def plot_data(result_data, metric_names, token):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()  # Create a second y-axis
    
    axes = [ax1, ax2]
    colors = ['blue', 'orange']
    
    for i, metric_name in enumerate(metric_names):
        if metric_name in result_data:
            data = result_data[metric_name]
            df = pd.DataFrame(data['data']['getMetric']['timeseriesData'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            axes[i].plot(df['datetime'], df['value'], label=metric_name, color=colors[i])
            axes[i].set_ylabel(metric_name, color=colors[i])
            axes[i].tick_params(axis='y', labelcolor=colors[i])
    
    # print("before base64")
    ax1.set_xlabel('Date')
    
    plt.title(f'{token.capitalize()} - Multiple Metrics')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig('crypto_metrics_combined.png')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

async def get_claude_analysis(image_base64, metric_names, token):
    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_SK"))
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are an expert in crypto technical analysis, analyze the values of the metrics and make your own decisions.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Analyze the following metrics for {token}: {', '.join(metric_names)}. Provide your insights and conclusions, comparing the metrics where relevant."
                    }
                ]
            }
        ]
    )
    return message.content[0].text

@app.post("/analyze")
async def analyze_crypto(request: MetricRequest):
    try:
        startDate = datetime.strptime(request.startDate, '%Y-%m-%d').strftime('%Y-%m-%dT%H:%M:%SZ')
        endDate = datetime.strptime(request.endDate, '%Y-%m-%d').strftime('%Y-%m-%dT%H:%M:%SZ')
        queries = [create_metric_query(
            metric_name,
            request.token,
            startDate,
            endDate,
            request.interval
        ) for metric_name in request.metrics]
        
        result_data = {}
        for metric_name, query in zip(request.metrics, queries):
            data = await fetch_santiment_data(query)
            result_data[metric_name] = data
        
        image_base64 = plot_data(result_data, request.metrics, request.token)
        analysis = await get_claude_analysis(image_base64, request.metrics, request.token)
        
        return JSONResponse(content={
            # "image": image_base64,
            "analysis": analysis
        })

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)