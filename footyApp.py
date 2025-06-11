import json
import uuid
import boto3


def lambda_handler(event, context):

    request_id = str(uuid.uuid4())
    lambda_client = boto3.client('lambda')

    lambda_client.invoke(
        FunctionName='footyAppBackground',
        InvocationType='Event',
        Payload=json.dumps({'request_id': request_id})
    )

    html = f"""
    <html>
      <head><meta charset="UTF-8"><title>Teams Maker</title><body>
        <h1>⏳ Making the teams...</h1>
        <h2>This may take up to one minute.</h2>
        <div id="result"></div>

        <script>
          const checkStatus = async () => {{
            const res = await fetch("https://67pbjzngapadhysqtyjb2sjsr40qnyya.lambda-url.eu-north-1.on.aws/?request_id={request_id}");
            const data = await res.json();

            console.log("Got status response:", data);

            if (data.status === 'done') {{
              document.body.innerHTML = data.result;
            }} else {{
              setTimeout(checkStatus, 3000);
            }}
          }};
          checkStatus();
        </script>

      </body>
    </html>
    """

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html; charset=UTF-8"
        },
        "body": html
    }

