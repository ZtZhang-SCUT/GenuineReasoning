curl --request POST \
    --url http://152.136.41.186:30131/v1/chat/completions \
    --header 'Authorization: Bearer sk-kQKMTKyEQA7X6eZ_AR72Cqvb2o7NgskyrKG9UdPqUPD8zu-_HeVs5Nie0_M' \
    -H "Content-Type: application/json" \
    --data '{
      "model": "deepseek-v3.1t",
      "messages": [
          {
              "role": "user",
              "content": "hi~"
          }
      ],
  }'