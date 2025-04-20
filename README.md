# Financial Insights API

A FastAPI application that enables financial professionals to get insights about companies using a multi-LLM pipeline. The app processes financial documents from the Quartr API, gets real-time web information, and uses advanced AI models to answer queries.

## Features

- Accept JSON payloads with company name and query
- Fetch financial documents from Quartr API using company IDs
- Store and serve documents through AWS S3
- Three-step LLM chain for comprehensive answers:
  1. **Document Analysis**: Gemini 2.0 Flash analyzes company documents
  2. **Web Search**: Perplexity API fetches current information
  3. **Final Synthesis**: Claude 3.7 Sonnet combines both sources for comprehensive answers
- Parallel processing for optimal performance
- Conversation context management for follow-up questions

## Architecture

The application uses a sophisticated architecture with multiple components:

- **FastAPI Backend**: RESTful API interface
- **Supabase Integration**: Stores and retrieves company data with Quartr IDs
- **AWS S3 Storage**: Securely stores and serves financial documents with proper content types
- **Quartr API Integration**: Fetches financial documents for selected companies
- **Document Processing**: Converts and standardizes documents (especially transcripts)
- **Multi-LLM Pipeline**:
  - **Gemini AI**: Analyzes company documents 
  - **Perplexity API**: Searches the web for current information
  - **Claude AI**: Synthesizes all information into comprehensive responses
- **Asynchronous Processing**: Runs tasks in parallel for optimal performance
- **Conversation Management**: Maintains context for follow-up questions

## API Endpoints

### POST /api/insights

Accepts a JSON payload with company name and query, returns financial insights.

**Request:**

```json
{
  "company_name": "Apple Inc.",
  "query": "What were the key highlights from the last earnings call?",
  "conversation_context": []  // Optional for follow-up questions
}
```

**Response:**

```json
{
  "answer": "Detailed answer from the multi-LLM pipeline...",
  "processing_time": 15.32,
  "sources": [
    {
      "type": "document",
      "title": "Q2 2023 Earnings Call Transcript",
      "url": "https://alpineinsights.s3.eu-central-2.amazonaws.com/apple_inc/transcript/apple_inc_20230502_transcript.pdf"
    },
    {
      "type": "web",
      "title": "Apple Reports Second Quarter Results",
      "url": "https://www.apple.com/newsroom/2023/05/apple-reports-second-quarter-results/"
    }
  ]
}
```

### GET /health

Health check endpoint to verify the API is running.

## Prerequisites

- Python 3.9+
- Quartr API key
- Google Gemini API key
- Perplexity API key
- Anthropic Claude API key
- AWS account with S3 bucket
- Supabase account with table for company universe data

## Local Setup

1. Clone the repository
2. Install dependencies
```
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys
```
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application
```
hypercorn main:app --reload
```

## Environment Variables

```
# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
AWS_BUCKET_NAME=your_s3_bucket_name

# API Keys
QUARTR_API_KEY=your_quartr_api_key
CLAUDE_API_KEY=your_claude_api_key
GEMINI_API_KEY=your_gemini_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key

# Supabase (for company data)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Deployment on Railway

1. Fork this repository
2. Create a new Railway project from the GitHub repository
3. Add environment variables in Railway dashboard
4. Deploy the application

## Using with N8N

1. Add an HTTP Request node in N8N
2. Configure it with the following settings:
   - Method: POST
   - URL: https://[your-railway-app].up.railway.app/api/insights
   - Authentication: None
   - Request Content Type: JSON
   - Request Body:
   ```json
   {
     "company_name": "{{$node.previous_node.json.company_name}}",
     "query": "{{$node.previous_node.json.query}}"
   }
   ```
3. Add appropriate error handling and output processing in your N8N workflow

## Project Structure
```
financial-insights-api/
├── main.py                  # FastAPI application
├── supabase_client.py       # Supabase integration for company data
├── utils.py                 # Utility classes for API, AWS S3, and document processing
├── utils_helper.py          # Helper functions to resolve circular imports
├── logger.py                # Logger instance
├── requirements.txt         # Python dependencies
├── .env.example             # Template for environment variables
├── railway.json             # Railway deployment configuration
└── README.md                # Project documentation
```
