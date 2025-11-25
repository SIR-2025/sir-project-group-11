from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

# List all usage records
usage = client.usage.records.list(limit=10)
for record in usage.data:
    print(record)
